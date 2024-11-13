# LittleLanguageModel

LittleLanguageModel is a Transformer-based language model, designed for text generation and sequence modeling tasks. This repository contains the implementation of a transformer architecture, with a specific instance called DanteGPT, trained on a dataset of Italian text.

## Overview

The model is based on the Transformer architecture, utilizing multiple layers of self-attention heads and feedforward neural networks to generate human-like text. DanteGPT is a variant of LittleLanguageModel, specifically trained to generate text inspired by Dante Alighieri’s *Divine Comedy*. 

### Features:
- **Transformer Architecture**: Multiple attention heads and layers to handle complex relationships within the input sequence.
- **Text Generation**: Capability to generate text based on an initial prompt using a temperature-controlled sampling technique.
- **Training**: The model is trained on a large corpus of Italian text to capture language structure and style.

## Installation

To set up the environment, clone the repository and install the dependencies:

```bash
git clone https://github.com/GiuseppeBellamacina/LittleLanguageModel.git
cd LittleLanguageModel
pip install -r requirements.txt
```

## Usage

### Training the Model

To train the model, use the `train_model()` function with the desired parameters. This function will train the model on a dataset, periodically evaluating and logging the loss.

```python
from model import LittleLanguageModel
from train import train_model
import torch

file = 'file.txt'
text = open(file, 'r', encoding='utf-8').read()
vocab = sorted(list(set(text)))

encode = lambda s: [vocab.index(c) for c in s]
decode = lambda l: "".join([vocab[c] for c in l]) 

# Split the dataset into training and validation sets
x = int(0.9*len(text))
text = torch.tensor(encode(text), dtype=torch.long)
train, val = text[:x], text[x:]

# Define the model parameters
vocab_size = len(vocab)  # Based on the dataset
embed_size = 512
num_heads = 8
head_size = embed_size // num_heads
num_layers = 6
block_size = 128  # Example sequence length
batch_size = 64

# Find the device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate the model
model = LittleLanguageModel(vocab_size, head_size, embed_size, block_size, num_heads, num_layers, device).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Train the model
EPOCHS = 5000
train_losses, val_losses = train_model(model, train, val, block_size, batch_size, device, optimizer, EPOCHS)
```

### Generating Text

To generate text using the trained model, you can use the `generate()` method. You provide an initial prompt and the number of tokens to generate.

```python
# Example for generating text
ids = torch.tensor(
    encode("Nel mezzo del cammin di nostra vita"),
    dtype=torch.long
).unsqueeze(0).to(device)

generated_ids = model.generate(
    ids,
    max_new_tokens=2000,
    temperature=0.8
)

print(decode(generated_ids[0].tolist()))
```

### Available Functions:
- **`train_model()`**: Trains the model on the training dataset and evaluates on the validation set.
- **`generate()`**: Generates text based on an initial prompt.

## Model Architecture

- **Embedding Layer**: Encodes input tokens into dense vectors.
- **Positional Encoding**: Adds positional information to input tokens to maintain the sequential nature of the data.
- **Multi-Head Attention**: Multiple self-attention heads to capture different types of relationships in the data.
- **Feedforward Layers**: A fully connected neural network for further learning.
- **Layer Normalization**: Normalizes the input to each layer to speed up training and improve stability.

## Training Data

DanteGPT is trained on a text corpus that includes a selection of classical Italian literature, primarily focusing on Dante Alighieri’s *Divine Comedy*.

## Acknowledgments

- The Transformer model architecture is based on the paper: ["Attention is All You Need"](https://arxiv.org/abs/1706.03762).
- Special thanks to the open-source community for the contributions that made this project possible.

---

You can customize the sections (especially the usage and requirements) based on the specific details of your project.
