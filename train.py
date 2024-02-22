from mistral.model import Transformer , ModelArgs
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import math
import mmap
import random
from transformers import LlamaTokenizerFast
import gc


path = "mar-tokenizer"

file_path = 'D:\mr_dataset\mr.txt'

device = "cuda"

# torch.cuda.empty_cache()
# gc.collect()

tokenizer_loaded = LlamaTokenizerFast.from_pretrained(path)
tokenizer_loaded.add_special_tokens({'pad_token': '[PAD]'})
tokenizer_loaded.padding_side = "right"


# arg = ModelArgs(dim=1024,
#                 n_layers=6,
#                 head_dim=4,
#                 hidden_dim=4096,
#                 n_heads=6,
#                 n_kv_heads=2,
#                 norm_eps=1e-6,
#                 max_batch_size=8,
#                 vocab_size=200,
#                 sliding_window=4,
#                 )

arg = ModelArgs(dim=1024,
                n_layers=12,
                head_dim=6,
                hidden_dim=4096,
                n_heads=8,
                n_kv_heads=4,
                norm_eps=1e-6,
                max_batch_size=8,
                vocab_size=200,
                sliding_window=4,
                )

model = Transformer(args=arg).to(device)

model_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
print(f"Model size: {model_size_bytes / (1024**2):.2f} MB")

print(f'device is set to: {device}')

def get_random_chunk(filename):
    
    with open(filename, 'rb') as f:
        block_size = 30
        batch_size=8
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            # Determine the file size and a random position to start reading
            file_size = len(mm)
            start_pos = random.randint(0, (file_size) - block_size*batch_size)

            # Seek to the random position and read the block of text
            mm.seek(start_pos)
            block = mm.read(block_size*batch_size-1)

            # Decode the block to a string, ignoring any invalid byte sequences
            decoded_block = block.decode('utf-8', errors='ignore').replace('\r', '')
            
            # Train and test splits
            #data = torch.tensor(tokenizer_loaded.encode(decoded_block), dtype=torch.long)
            # data = tokenizer_loaded.encode(decoded_block,padding='max_length' ,max_length=100)
            data = tokenizer_loaded.encode(decoded_block)
            
    return data #decoded_block


def get_batch():
    block_size = 30
    batch_size=8
    input_ids, seq_lengths, output_ids= [], [],[]
    
    
    # data = get_random_chunk(filename='D:\mr_dataset\mr.txt')
    # ix = torch.randint(len(data) - block_size, (batch_size,))
    for _ in range(batch_size):
        data = get_random_chunk(filename='D:\mr_dataset\mr.txt')
        input_ids.extend(data[:block_size])
        output_ids.extend(data[1:block_size+1])
        seq_lengths.append(len(data[:block_size]))
    # input_ids.extend([data[i:i+block_size] for i in ix])
    # seq_lengths.append(len(input_ids))
    x = torch.tensor(input_ids, dtype=torch.long)
   
    # x = torch.stack([data[i:i+block_size] for i in ix])
    # for i in range(batch_size):
    #     output_ids.extend(data[i+1:i+block_size+1])
    # output_ids.extend([data[i+7:i+block_size+7] for i in ix])
    y = torch.tensor(output_ids, dtype=torch.long)
    
    # y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    # x, y = x.to(device), y.to(device)
    #return [data[i:i+block_size] for i in ix],[data[i+7:i+block_size+7] for i in ix]

    x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x,y ,seq_lengths


# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 10000#600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 10000#600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla



# optimizer
optimizer = torch.optim.AdamW(params=model.parameters(), weight_decay= weight_decay, lr=learning_rate, betas=(beta1, beta2))

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

iter_num= 0
while True:
    total_loss=0
    
    #set lr with deacy_fn for each iter
    lr = get_lr(iter_num) if decay_lr else learning_rate
    x,y,seq_lengths = get_batch()
    logits = model(input_ids=x, seqlens=seq_lengths)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)),   y.view(-1), ignore_index=-1)
    total_loss += loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
 
    print(f"epoch: {iter_num:.5f} | Training_loss: {total_loss:.5f}")
    iter_num+=1

    if iter_num>max_iters:
        break

# 3. Save the model state dict
MODEL_SAVE_PATH = f'weights/my-lm_epochs-{max_iters}.pth'
print(f'Saving model to: {MODEL_SAVE_PATH}')
torch.save(obj=model.state_dict(), # only saving the state_dict() only saves the models learned parameters
           f=MODEL_SAVE_PATH)

