# TRM Training Quick Start Guide

## Test Training Session

### 1. Quick Sanity Check (~5 mins)

Test that everything works with a short training run:

```bash
cd c:\Users\rahul\OneDrive\IMP_DOCS\AI\chat-trm

python src/training/train.py configs/shakespeare_char.py --max_iters=100 --eval_interval=20 --log_interval=5
```

**Expected output:**

```
Overriding config with configs/shakespeare_char.py:
...
tokens per iteration: 16,384
vocab_size from meta: 65
TRM model initialized with 0.52M parameters

Starting training for 50 iterations...
Model: TRMConfig(vocab_size=65, block_size=256, ...)
Device: cpu
Dtype: float32

step 0: train loss 4.XXXX, val loss 4.XXXX
iter 0: loss 4.XXXX, time XXXms, mfu 0.00%
iter 5: loss 4.XXXX, time XXXms, mfu 0.00%
...
step 25: train loss 3.XXXX, val loss 3.XXXX
...
step 50: train loss 2.XXXX, val loss 2.XXXX

Training complete!
```

### 2. Full Training Run (~30-60 mins)

Train the model fully:

```bash
python src/training/train.py configs/shakespeare_char.py
```

This will:

- Train for 5000 iterations
- Save checkpoints to `out-shakespeare-char/`
- Take 30-60 minutes on CPU

### 3. Monitor Training

Watch the loss curves:

- **Train loss**: Should decrease from ~4.2 to ~1.5-2.0
- **Val loss**: Should decrease similarly (may be slightly higher due to overfitting)

Key metrics:

- Loss decreasing = model is learning
- Val loss < train loss at start = normal
- Val loss > train loss later = some overfitting (expected on small dataset)

### 4. Generate Text

After training completes, generate Shakespeare-like text:

```bash
# Generate 3 samples
python generate.py

# Custom generation
python generate.py --start="ROMEO:" --num_samples=5 --temperature=0.8

# More creative (higher temp)
python generate.py --temperature=1.2 --top_k=100

# More conservative (lower temp)
python generate.py --temperature=0.5 --top_k=50
```

## Training Configuration Options

Override any parameter from command line:

```bash
python src/training/train.py configs/shakespeare_char.py \
    --max_iters=1000 \           # Number of training iterations
    --batch_size=32 \             # Batch size
    --learning_rate=1e-3 \        # Learning rate
    --recursion_steps=3 \         # Number of recursive passes (KEY PARAMETER!)
    --hidden_size=512 \           # Model size
    --n_layers=3 \                # Number of transformer blocks
    --eval_interval=100 \         # Evaluate every N iterations
    --wandb_log=True              # Enable W&B logging
```

## Key Files

- **Training**: `src/training/train.py`
- **Model**: `src/model/trm.py`
- **Config**: `configs/shakespeare_char.py`
- **Data**: `data/shakespeare_char/`
- **Checkpoints**: `out-shakespeare-char/ckpt.pt`

## Troubleshooting

### Training is slow

- Normal on CPU! Each iteration takes ~1-2 seconds
- Reduce `batch_size` to 32 or 16
- Reduce `hidden_size` to 256
- Reduce `recursion_steps` to 1

### Out of memory

- Reduce `batch_size`
- Reduce `hidden_size`
- Reduce `block_size`

### Loss not decreasing

- Check that data loaded correctly (look for "vocab_size from meta: 65")
- Try increasing `learning_rate` to 3e-3
- Increase `max_iters` (may need more training)

### Want faster results

Use a smaller model:

```bash
python src/training/train.py configs/shakespeare_char.py \
    --hidden_size=256 \
    --n_layers=2 \
    --recursion_steps=1 \
    --max_iters=2000
```

## What Success Looks Like

After 5000 iterations, you should see:

- ✅ Train loss: ~1.5-2.0 (down from ~4.2)
- ✅ Val loss: ~1.8-2.2
- ✅ Generated text has Shakespeare-like structure
- ✅ Words are mostly real English words
- ✅ Character names and dialogue format appear

Example good generation:

```
ROMEO:
What light through yonder window breaks?
It is the east, and Juliet is the sun.

JULIET:
O Romeo, Romeo, wherefore art thou Romeo?
```

## Next: Experiment with Recursion

The KEY innovation is the `recursion_steps` parameter. Try:

```bash
# Baseline: No recursion (1 step)
python src/training/train.py configs/shakespeare_char.py --recursion_steps=1 --out_dir=out-recursion-1

# With recursion (2 steps)
python src/training/train.py configs/shakespeare_char.py --recursion_steps=2 --out_dir=out-recursion-2

# More recursion (3 steps)
python src/training/train.py configs/shakespeare_char.py --recursion_steps=3 --out_dir=out-recursion-3
```

Compare final validation losses to see if recursion helps!
