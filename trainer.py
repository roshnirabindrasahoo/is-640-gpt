import torch
from model import BigramLanguageModel
from data import load_data, split_data, get_batch

# Hyperparameters
batch_size = 16
block_size = 8
max_iters = 100
eval_interval = 500
learning_rate = 0.01
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 50
n_embd = 32
n_head = 4
n_layer = 3
dropout = 0.2

torch.manual_seed(1337)

# Load and prepare data
data, decode, vocab_size = load_data('input.txt')
train_data, val_data = split_data(data)

# Initialize Model
model = BigramLanguageModel(vocab_size, n_embd, block_size, n_head, n_layer, dropout)
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = {split: torch.zeros(eval_iters) for split in ['train', 'val']}
    for split, data_split in zip(['train', 'val'], [train_data, val_data]):
        for k in range(eval_iters):
            X, Y = get_batch(data_split, block_size, batch_size)
            X, Y = X.to(device), Y.to(device)
            _, loss = model(X, Y)
            losses[split][k] = loss.item()
    model.train()
    return {k: v.mean() for k, v in losses.items()}

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch(train_data, block_size, batch_size)
    xb, yb = xb.to(device), yb.to(device)

    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Generate Text
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500, block_size=block_size)[0].tolist()))
