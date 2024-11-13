import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
    def __init__(self, head_size, embed_size, block_size, dropout=0.05):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(2, 1) / self.head_size ** 0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=2)
        wei = self.dropout(wei)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, head_size, embed_size, block_size, num_head, dropout=0.05):
        super().__init__()
        self.sa_head = nn.ModuleList([Head(head_size, embed_size, block_size, dropout) for _ in range(num_head)])
        self.proj = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.cat([head(x) for head in self.sa_head], dim=-1)
        return self.dropout(self.proj(x))

class FeedForward(nn.Module):
    def __init__(self, embed_size, dropout=0.05):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ff(x)
    
class Block(nn.Module):
    def __init__(self, head_size, embed_size, block_size, num_head, dropout=0.05):  # Reduced dropout
        super().__init__()
        head_size = embed_size // num_head
        self.multihead = MultiHeadAttention(head_size, embed_size, block_size, num_head, dropout)
        self.ff = FeedForward(embed_size, dropout)
        self.ll1 = nn.LayerNorm(embed_size)
        self.ll2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        x = x + self.multihead(self.ll1(x))
        return x + self.ff(self.ll2(x))

class LittleLanguageModel(nn.Module):
    def __init__(self, vocab_size, head_size, embed_size, block_size, num_head, num_layers, device, dropout=0.05):  # Reduced dropout
        super().__init__()
        self.device = device
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, embed_size)
        self.positional_embedding = nn.Embedding(block_size, embed_size)
        self.linear = nn.Linear(embed_size, vocab_size)
        self.block = nn.Sequential(*[Block(head_size, embed_size, block_size, num_head, dropout) for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(embed_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        logits = self.token_embedding_table(idx)  # (B, T, C)
        ps = self.positional_embedding(torch.arange(T, device=idx.device))
        x = logits + ps  # (B, T, C)
        logits = self.linear(self.layer_norm(self.block(x)))  # Improved LayerNorm positioning
        
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0):
        if temperature <= 0.0:
            raise ValueError("Temperature must be greater than zero for sampling.")
        
        if temperature == 0.0:
            return self(idx)[0].argmax(dim=-1)
        
        for _ in range(max_new_tokens):
            crop_idx = idx[:, -self.block_size:].to(self.device)
            logits, _ = self(crop_idx)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1).to(self.device)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx