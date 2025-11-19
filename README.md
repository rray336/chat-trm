# Chat-TRM: Tiny Recursive Model for Text Completion

A recursive reasoning language model that applies transformer blocks iteratively to refine predictions, achieving parameter efficiency through weight reuse rather than layer stacking.

## What This Project Does

Chat-TRM is a text completion model inspired by the Tiny Recursive Model (TRM) architecture, adapted for autoregressive language generation. Instead of stacking many transformer layers, the model uses **recursive reasoning**: a small number of transformer blocks are applied multiple times, allowing the model to iteratively refine its predictions.

**Key Innovation**: The model makes multiple "thinking passes" over the same input, progressively improving its output through recursion rather than depth.

## Architecture

### Shared with TRM

- **Recursive reasoning loop**: Apply transformer blocks iteratively rather than stacking them
- **Parameter efficiency**: Reuse weights across recursion steps (~0.5M-5M parameters)
- **Modern components**:
  - RMS Normalization (simpler than LayerNorm)
  - SwiGLU activation (better than GELU)
  - RoPE positional embeddings (better than learned positions)
- **Learnable initial state**: Model learns optimal "blank slate" starting point

### Key Differences from TRM

| Aspect | TRM (Original) | Chat-TRM (This Project) |
|--------|----------------|-------------------------|
| **Task** | Puzzle solving (ARC-AGI, Sudoku) | Text completion |
| **Attention** | Non-causal (bidirectional) | **Causal** (autoregressive) |
| **Architecture** | Dual-state (z_H, z_L) hierarchy | **Single-state** (z) simplified |
| **Gradient flow** | Only last recursion step | **All steps** (full backprop) |
| **ACT halting** | Adaptive computation time | Fixed recursion depth |
| **Training data** | Visual puzzles | **Shakespeare text** (character-level) |

**Why these changes?**
- Causal attention is required for autoregressive text generation
- Simplified single-state architecture makes code clearer and easier to experiment with
- Full gradient flow lets us understand if early recursion steps contribute to learning
- Fixed recursion depth is simpler and sufficient for text completion

## Training Harness

Built on **nanoGPT's** proven training infrastructure:

- **Data loading**: Memory-mapped datasets for efficiency
- **Optimization**: AdamW with cosine LR schedule and warmup
- **Training loop**: Clean, minimal implementation with DDP support
- **Checkpointing**: Automatic best model saving
- **Config system**: Easy hyperparameter overrides via command line
- **Mixed precision**: Optional fp16/bfloat16 training

The training code is adapted from nanoGPT but modified to work with the TRM recursive architecture.

## Quick Start

### Installation

```bash
# Clone repository
cd chat-trm

# Install dependencies
pip install -r requirements.txt

# Prepare Shakespeare dataset
python src/data/shakespeare.py
```

### Training

**Quick test** (~5 minutes on CPU):
```bash
python src/training/train.py configs/shakespeare_char.py \
    --max_iters=100 \
    --eval_interval=20 \
    --eval_iters=20 \
    --log_interval=5
```

**Full training** (~30-60 minutes on CPU):
```bash
python src/training/train.py configs/shakespeare_char.py
```

**Monitor progress**:
- Loss should decrease from ~4.2 to ~1.5-2.0
- Checkpoint saved to `out-shakespeare-char/ckpt.pt`

### Generate Text

After training completes:

```bash
# Generate samples
python generate.py

# Custom generation
python generate.py --start="ROMEO:" --num_samples=5 --temperature=0.8

# More creative (higher temperature)
python generate.py --temperature=1.2

# More conservative (lower temperature)
python generate.py --temperature=0.5
```

## Experimenting with Recursion

The key research question: **Does recursive refinement improve text completion?**

Compare different recursion depths:

```bash
# Baseline: No recursion (1 step)
python src/training/train.py configs/shakespeare_char.py \
    --recursion_steps=1 \
    --out_dir=out-recursion-1

# With recursion (2 steps)
python src/training/train.py configs/shakespeare_char.py \
    --recursion_steps=2 \
    --out_dir=out-recursion-2

# More recursion (3 steps)
python src/training/train.py configs/shakespeare_char.py \
    --recursion_steps=3 \
    --out_dir=out-recursion-3
```

Then compare final validation losses to see if recursion helps!

## Configuration

Key hyperparameters (in `configs/shakespeare_char.py`):

```python
# Model architecture
hidden_size = 384           # Embedding dimension
num_heads = 6               # Attention heads
n_layers = 2                # Transformer blocks (reused recursively)
recursion_steps = 2         # Number of recursive passes (KEY!)
block_size = 256            # Context length

# Training
batch_size = 64
learning_rate = 1e-3
max_iters = 5000
```

Override any parameter from command line:
```bash
python src/training/train.py configs/shakespeare_char.py \
    --hidden_size=512 \
    --recursion_steps=3 \
    --learning_rate=3e-3
```

## Project Structure

```
chat-trm/
├── src/
│   ├── model/
│   │   ├── config.py       # Model configuration
│   │   ├── layers.py       # RMSNorm, SwiGLU, Attention, RoPE
│   │   └── trm.py          # Main TRM model with recursive loop
│   ├── training/
│   │   └── train.py        # Training loop (from nanoGPT)
│   └── data/
│       └── shakespeare.py  # Data preparation
├── configs/
│   └── shakespeare_char.py # Training configuration
├── generate.py             # Text generation script
├── QUICKSTART.md           # Detailed guide
└── README.md               # This file
```

## Model Details

**Current Configuration** (Shakespeare dataset):
- Parameters: ~3.5M (0.5M non-embedding)
- Hidden size: 384
- Attention heads: 6
- Transformer blocks: 2
- Recursion steps: 2
- Context length: 256 tokens
- Dataset: Shakespeare character-level (~1M training tokens)

**Compute Trade-off**: N recursion steps ≈ N× forward pass compute. The hypothesis is that parameter reuse through recursion is more efficient than stacking layers.

## Expected Results

After 5000 iterations on Shakespeare:
- Train loss: ~1.5-2.0 (from ~4.2)
- Val loss: ~1.8-2.2
- Generated text has Shakespeare-like structure
- Real English words and dialogue format

Example generation:
```
ROMEO:
What light through yonder window breaks?
It is the east, and Juliet is the sun.
```

## References

**Tiny Recursive Model (TRM)**:
- Paper: ["Less is More: Recursive Reasoning with Tiny Networks"](https://arxiv.org/abs/2510.04871)
- Repository: https://github.com/SamsungSAILMontreal/TinyRecursiveModels
- Achieves 45% on ARC-AGI-1 with only 7M parameters
- This project adapts TRM's recursive reasoning concept and core components (RMSNorm, SwiGLU, RoPE) for text completion

**nanoGPT**:
- Repository: https://github.com/karpathy/nanoGPT
- Author: Andrej Karpathy
- This project uses nanoGPT's training infrastructure and data loading patterns

For detailed attribution, see [CITATIONS.md](CITATIONS.md)

## Troubleshooting

**Training is slow on CPU**
- Normal! Each iteration takes 1-2 seconds
- Reduce `batch_size`, `hidden_size`, or `recursion_steps`
- Or use GPU if available

**Loss not decreasing**
- Check data loaded correctly: "vocab_size from meta: 65"
- Try higher learning rate: `--learning_rate=3e-3`
- Increase training: `--max_iters=10000`

**Out of memory**
- Reduce `batch_size` or `hidden_size`
- Reduce `block_size` (context length)

See [QUICKSTART.md](QUICKSTART.md) for detailed troubleshooting guide.

## License

MIT License - see [LICENSE](LICENSE) for details.

This project contains code adapted from:
- [Tiny Recursive Model](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) (MIT License)
- [nanoGPT](https://github.com/karpathy/nanoGPT) (MIT License)

See [CITATIONS.md](CITATIONS.md) for detailed attribution.
