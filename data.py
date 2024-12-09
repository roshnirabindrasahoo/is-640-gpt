import torch

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    data = torch.tensor(encode(text), dtype=torch.long)
    return data, decode, len(chars)
    
def split_data(data, train_ratio=0.9):
    n = int(train_ratio * len(data))
    return data[:n], data[n:]
