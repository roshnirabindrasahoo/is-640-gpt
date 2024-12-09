import torch
from data import Data
from model import GPTLanguageModel
from trainer import Trainer

if __name__ == "__main__":
    # Hyperparameters
    batch_size = 16 # Increasing batch size for faster training
    block_size = 8
    max_iters = 100    # For better performance increasing number of training iterations
    eval_interval = 500
    eval_iters = 200
    learning_rate = 1e-3  
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    n_embd = 32    
    n_head = 4
    n_layer = 3
    dropout = 0.2

    
    data_obj = Data(file_path='input.txt', device=device, block_size=block_size)

    
    model = GPTLanguageModel(
        vocab_size=data_obj.vocab_size,
        n_embd=n_embd, 
        n_head=n_head, 
        n_layer=n_layer, 
        block_size=block_size, 
        dropout=dropout,
        device=device
    ).to(device)

    
    trainer = Trainer(model, data_obj, max_iters=max_iters, eval_interval=eval_interval, eval_iters=eval_iters, batch_size=batch_size, learning_rate=learning_rate)
    trainer.train()

    
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_ids = model.generate(context, max_new_tokens=100)
    print(data_obj.decode(generated_ids[0].tolist()))
