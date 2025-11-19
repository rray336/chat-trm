# ABOUTME: Chat supervised fine-tuning script with vocab expansion and masked loss
# ABOUTME: Loads Shakespeare checkpoint, expands vocabulary, trains on chat data

import os
import sys
import time
import math
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.model.config import TRMConfig
from src.model.trm import TRM
from src.data.chat_tokenizer import ChatTokenizer
from src.data.chat_dataset import ChatDataset, collate_chat_batch

# -----------------------------------------------------------------------------
# Default config values
# I/O
out_dir = 'out-chat-sft'
eval_interval = 100
log_interval = 10
eval_iters = 20
always_save_checkpoint = True  # Save at every eval for chat training

# Checkpoint loading
init_from = 'shakespeare'  # Load Shakespeare checkpoint
shakespeare_checkpoint = 'out-shakespeare-char/ckpt.pt'

# wandb logging
wandb_log = False
wandb_project = 'chat-trm'
wandb_run_name = 'chat-sft-' + str(int(time.time()))

# data
train_data_path = 'data/chat_synthetic/train.jsonl'
val_data_path = 'data/chat_synthetic/val.jsonl'
batch_size = 4  # Small for CPU
block_size = 256

# model (will be loaded from checkpoint)
vocab_size = 69  # 65 chars + 4 special tokens
old_vocab_size = 65  # Shakespeare vocab size

# optimizer
learning_rate = 1e-4  # Lower for fine-tuning
max_iters = 1000
warmup_iters = 100
lr_decay_iters = 1000
min_lr = 1e-5
weight_decay = 0.0
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0

# system
device = 'cpu'
dtype = 'float32'
compile = False

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())  # Load config overrides
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

# Setup
os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Load tokenizer
print("Loading chat tokenizer...")
tokenizer = ChatTokenizer()
assert tokenizer.vocab_size == vocab_size, f"Tokenizer vocab ({tokenizer.vocab_size}) != config ({vocab_size})"

# Load datasets
print(f"Loading chat datasets...")
train_dataset = ChatDataset(train_data_path, tokenizer, block_size=block_size, split='train')
val_dataset = ChatDataset(val_data_path, tokenizer, block_size=block_size, split='val')

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          collate_fn=collate_chat_batch)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_chat_batch)

print(f"Training examples: {len(train_dataset)}")
print(f"Validation examples: {len(val_dataset)}")

# Initialize/load model
iter_num = 0
best_val_loss = 1e9

if init_from == 'scratch':
    print("Initializing model from scratch...")
    model_config = TRMConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        **{k: config[k] for k in ['hidden_size', 'num_heads', 'n_layers',
                                   'recursion_steps', 'expansion', 'dropout'] if k in config}
    )
    model = TRM(model_config)

elif init_from == 'shakespeare':
    print(f"Loading Shakespeare checkpoint from {shakespeare_checkpoint}...")

    # Load checkpoint
    checkpoint = torch.load(shakespeare_checkpoint, map_location=device, weights_only=False)
    checkpoint_model_config = checkpoint['model_config']

    # Create new config with expanded vocab
    # checkpoint_model_config is a TRMConfig object, access attributes directly
    model_config = TRMConfig(
        vocab_size=vocab_size,  # Expanded vocab
        block_size=checkpoint_model_config.block_size,
        hidden_size=checkpoint_model_config.hidden_size,
        num_heads=checkpoint_model_config.num_heads,
        n_layers=checkpoint_model_config.n_layers,
        recursion_steps=checkpoint_model_config.recursion_steps,
        expansion=checkpoint_model_config.expansion,
        dropout=checkpoint_model_config.dropout,
    )

    print(f"Expanding vocabulary from {old_vocab_size} to {vocab_size}...")

    # Create model with new vocab size
    model = TRM(model_config)

    # Load state dict and expand embedding layers
    state_dict = checkpoint['model']
    model_state = model.state_dict()

    # Copy all weights except embedding layers
    for k, v in state_dict.items():
        if k in ['embed_tokens.embedding_weight', 'lm_head.weight']:
            # Handle embedding layers specially (expand vocab)
            if k == 'embed_tokens.embedding_weight':
                # old: [65, hidden_size], new: [69, hidden_size]
                old_emb = v
                new_emb = model_state[k].clone()  # Start with random init
                new_emb[:old_vocab_size] = old_emb  # Copy old embeddings
                model_state[k] = new_emb
                print(f"  Expanded {k}: {old_emb.shape} -> {new_emb.shape}")

            elif k == 'lm_head.weight':
                # old: [65, hidden_size], new: [69, hidden_size]
                old_head = v
                new_head = model_state[k].clone()  # Start with random init
                new_head[:old_vocab_size] = old_head  # Copy old weights
                model_state[k] = new_head
                print(f"  Expanded {k}: {old_head.shape} -> {new_head.shape}")
        else:
            # Copy all other weights directly
            if k in model_state and model_state[k].shape == v.shape:
                model_state[k] = v

    # Load expanded state dict
    model.load_state_dict(model_state)
    print(f"Loaded checkpoint with expanded vocabulary")

else:
    raise ValueError(f"Unknown init_from: {init_from}")

model.to(device)

# Print model info
print(f"\nModel configuration:")
print(f"  Vocabulary: {model_config.vocab_size} tokens")
print(f"  Parameters: {model.get_num_params()/1e6:.2f}M")
print(f"  Recursion steps: {model_config.recursion_steps}")

# Initialize optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                             betas=(beta1, beta2), weight_decay=weight_decay)

# Compile model
if compile:
    print("Compiling model...")
    model = torch.compile(model)

# Learning rate scheduler
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# Evaluation function
@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = {'train': 0.0, 'val': 0.0}

    for split, loader in [('train', train_loader), ('val', val_loader)]:
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(loader):
            if batch_idx >= eval_iters:
                break

            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)

            with ctx:
                logits, loss, _ = model(input_ids, targets=target_ids)

            total_loss += loss.item()
            num_batches += 1

        losses[split] = total_loss / num_batches if num_batches > 0 else 0.0

    model.train()
    return losses

# WandB logging
if wandb_log:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# Training loop
print(f"\nStarting chat fine-tuning for {max_iters} iterations...")
print(f"Device: {device}")
print(f"Batch size: {batch_size}\n")

t0 = time.time()

while iter_num < max_iters:
    # Set learning rate
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Evaluate
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {lr:.2e}")

        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
            })

        # Save checkpoint
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_config': model_config,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'config': config,
            }
            print(f"Saving checkpoint to {out_dir}")
            torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

    # Training step
    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        target_ids = batch['target_ids'].to(device)

        # Forward pass
        with ctx:
            logits, loss, _ = model(input_ids, targets=target_ids)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Logging
        if iter_num % log_interval == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            lossf = loss.item()
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.0f}ms")

        iter_num += 1
        if iter_num >= max_iters:
            break

    if iter_num >= max_iters:
        break

print("\nTraining complete!")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Final checkpoint saved to: {out_dir}/ckpt.pt")
