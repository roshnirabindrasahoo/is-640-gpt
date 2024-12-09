import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 16
block_size = 8
max_iters = 100
eval_interval = 500
learning_rate = 0.001
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 50
n_embd = 32
n_head = 4
n_layer = 3
dropout = 0.2


torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
