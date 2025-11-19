# Train TRM on Shakespeare character-level dataset
# Good for debugging and initial experimentation on CPU

out_dir = 'out-shakespeare-char'
eval_interval = 250
eval_iters = 200
log_interval = 10

# Save checkpoints only when validation improves (small dataset = overfit risk)
always_save_checkpoint = False

wandb_log = False  # enable via command line: --wandb_log=True
wandb_project = 'chat-trm'
wandb_run_name = 'shakespeare-trm-v1'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256  # context of up to 256 previous characters

# TRM model configuration (small for CPU training)
vocab_size = 65
hidden_size = 384
num_heads = 6
n_layers = 2
recursion_steps = 2  # key parameter: number of recursive refinement passes
expansion = 4.0
dropout = 0.2

# Optimizer
learning_rate = 1e-3  # can afford higher LR with small model
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100

# System
device = 'cpu'  # run on CPU for initial testing
compile = False  # don't torch.compile (not needed for CPU)
