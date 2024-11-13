import matplotlib.pyplot as plt
from typing import Tuple
import torch

def get_batch(data, block_size, batch_size, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a small batch of data of inputs x and targets y
    """ 
    ix = torch.randint(len(data) - block_size, (batch_size,)) 
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

def train_model(model, train_data, val_data, block_size, batch_size, device, optimizer, epochs=5000, eval_interval=100):
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        if epoch % eval_interval == 0:
            model.eval()
            train_loss, val_loss = 0.0, 0.0
            with torch.no_grad():
                for _ in range(100):
                    x_train, y_train = get_batch(train_data, block_size, batch_size, device)
                    x_val, y_val = get_batch(val_data, block_size, batch_size, device)
                    _, train_batch_loss = model(x_train, y_train)
                    _, val_batch_loss = model(x_val, y_val)
                    train_loss += train_batch_loss.item()
                    val_loss += val_batch_loss.item()
            train_losses.append(train_loss / 100)
            val_losses.append(val_loss / 100)
            print(f"Epoch {epoch} | Train Loss: {train_loss / 100:.4f} | Val Loss: {val_loss / 100:.4f}")
            model.train()

        x_batch, y_batch = get_batch(train_data, block_size, batch_size, device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x_batch, y_batch)
        loss.backward()
        optimizer.step()
        
        # Learning rate scheduler (e.g., halve every 1000 epochs)
        if epoch % 1000 == 0 and epoch > 0:
            for g in optimizer.param_groups:
                g['lr'] *= 0.5
    return train_losses, val_losses

def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Evaluation Interval')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss over Time')
    plt.show()