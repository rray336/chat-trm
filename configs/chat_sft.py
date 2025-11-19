# Chat supervised fine-tuning configuration
# Fine-tune Shakespeare model for conversational chat

out_dir = 'out-chat-sft'
eval_interval = 100
eval_iters = 20
log_interval = 10

# Save checkpoints frequently for chat training
always_save_checkpoint = True

# Load from Shakespeare checkpoint
init_from = 'shakespeare'
shakespeare_checkpoint = 'out-shakespeare-char/ckpt.pt'

# wandb logging
wandb_log = False
wandb_project = 'chat-trm'
wandb_run_name = 'chat-sft-v1'

# Chat data
train_data_path = 'data/chat_synthetic/train.jsonl'
val_data_path = 'data/chat_synthetic/val.jsonl'
batch_size = 4  # Small batch size for CPU training
block_size = 256

# Vocabulary (65 chars + 4 special tokens)
vocab_size = 69
old_vocab_size = 65

# Optimizer (lower LR for fine-tuning)
learning_rate = 1e-4  # 10x lower than Shakespeare training
max_iters = 1000
warmup_iters = 100
lr_decay_iters = 1000
min_lr = 1e-5
weight_decay = 0.0
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0

# System
device = 'cpu'
dtype = 'float32'
compile = False
