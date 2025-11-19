# Chat Training Guide

## Overview

This document describes how to train the TRM model for conversational chat using supervised fine-tuning (SFT) on top of the Shakespeare checkpoint.

## What Was Added

**Phase 1: Tokenizer & Data**
- `src/data/chat_tokenizer.py` - Character tokenizer with 4 special tokens for chat
- `src/data/chat_dataset.py` - JSONL conversation loader with masking
- `data/chat_synthetic/generate.py` - Synthetic conversation generator
- `data/chat_synthetic/train.jsonl` - 1350 training conversations
- `data/chat_synthetic/val.jsonl` - 150 validation conversations

**Phase 2: Training Infrastructure**
- `src/training/chat_sft.py` - Chat fine-tuning script with vocab expansion
- `configs/chat_sft.py` - Chat training configuration

**Phase 3: Utilities**
- `src/utils/expand_vocab.py` - Standalone vocab expansion utility
- `test_chat.py` - Interactive chat interface

## Quick Start

### 1. Prepare Data (Already Done)

The synthetic dataset has been generated:
```bash
python data/chat_synthetic/generate.py
```

Output:
- `data/chat_synthetic/train.jsonl` - 1350 conversations
- `data/chat_synthetic/val.jsonl` - 150 conversations

### 2. Train Chat Model

**Prerequisites**: You need a trained Shakespeare checkpoint at `out-shakespeare-char/ckpt.pt`

**Quick test** (100 iterations, ~10 minutes):
```bash
python src/training/chat_sft.py configs/chat_sft.py --max_iters=100 --log_interval=10
```

**Full training** (1000 iterations, ~30-60 minutes on CPU):
```bash
python src/training/chat_sft.py configs/chat_sft.py
```

**What happens:**
1. Loads Shakespeare checkpoint (65 vocab)
2. Expands vocabulary to 69 tokens (adds 4 special tokens)
3. Initializes new token embeddings randomly
4. Trains on chat conversations with masked loss
5. Saves checkpoints to `out-chat-sft/ckpt.pt`

### 3. Test Chat Model

After training completes:
```bash
python test_chat.py
```

Or specify checkpoint:
```bash
python test_chat.py --checkpoint out-chat-sft/ckpt.pt
```

## Training Details

### Vocabulary Expansion

**Original (Shakespeare):**
- Tokens 0-64: Characters (a-z, A-Z, punctuation, etc.)
- Total: 65 tokens

**Expanded (Chat):**
- Tokens 0-64: Characters (same as Shakespeare)
- Token 65: `<|user_start|>`
- Token 66: `<|user_end|>`
- Token 67: `<|assistant_start|>`
- Token 68: `<|assistant_end|>`
- Total: 69 tokens

### Masked Loss

Only assistant responses are trained on:

**Example conversation:**
```
<|user_start|>Hello!<|user_end|><|assistant_start|>Hi there!<|assistant_end|>
```

**Masking:**
- User message tokens: `mask=0` (ignore in loss)
- Assistant message tokens: `mask=1` (train on these)
- Special tokens: `mask=0` (ignore in loss)

This ensures the model learns to generate assistant responses, not user messages.

### Hyperparameters

**From `configs/chat_sft.py`:**
```python
learning_rate = 1e-4      # 10x lower than Shakespeare (fine-tuning)
max_iters = 1000
batch_size = 4            # Small for CPU
block_size = 256
warmup_iters = 100
grad_clip = 1.0
```

**CPU-Optimized:**
- Small batch size (4)
- No gradient accumulation needed
- Float32 precision
- No compilation

## Expected Results

### Training Progress

**Initial (iter 0):**
- Train loss: ~3.0-4.0 (higher due to random special token embeddings)
- Val loss: ~3.0-4.0

**After 100 iterations:**
- Train loss: ~2.5-3.0
- Val loss: ~2.5-3.0

**After 1000 iterations:**
- Train loss: ~1.5-2.0
- Val loss: ~1.8-2.2

### Chat Quality

After training, the model should be able to:
- ✅ Respond to greetings ("Hello" → "Hi there!")
- ✅ Answer simple questions ("What is 2+2?" → "2 plus 2 equals 4.")
- ✅ Maintain conversation structure
- ✅ Use proper grammar and punctuation

**Note**: Character-level tokenization means responses may be less fluent than BPE-based models, but should still be coherent.

## File Structure

```
chat-trm/
├── src/
│   ├── data/
│   │   ├── chat_tokenizer.py       # Tokenizer with special tokens
│   │   └── chat_dataset.py         # JSONL loader
│   ├── training/
│   │   └── chat_sft.py             # Chat training script
│   └── utils/
│       └── expand_vocab.py         # Vocab expansion utility
├── configs/
│   └── chat_sft.py                 # Chat config
├── data/
│   └── chat_synthetic/
│       ├── generate.py             # Data generator
│       ├── train.jsonl             # Training data
│       └── val.jsonl               # Validation data
├── out-shakespeare-char/
│   └── ckpt.pt                     # Shakespeare checkpoint (input)
├── out-chat-sft/
│   └── ckpt.pt                     # Chat checkpoint (output)
└── test_chat.py                    # Interactive chat interface
```

## Troubleshooting

### "Checkpoint not found"

**Problem**: `out-shakespeare-char/ckpt.pt` doesn't exist

**Solution**: Train the Shakespeare model first:
```bash
python src/training/train.py configs/shakespeare_char.py
```

### Training is slow

**Solution**: This is normal on CPU! Reduce iterations for testing:
```bash
python src/training/chat_sft.py configs/chat_sft.py --max_iters=100
```

### Out of memory

**Solution**: Reduce batch size:
```bash
python src/training/chat_sft.py configs/chat_sft.py --batch_size=2
```

### Loss not decreasing

**Possible causes:**
1. Special token embeddings need more iterations to learn (first 100-200 iters)
2. Learning rate too low/high - try adjusting
3. Data not loading correctly - check dataset paths

### Model gives weird responses

**Causes:**
1. Not enough training iterations - try 1000+
2. Shakespeare bias still strong - needs more chat data
3. Character-level limitations - consider BPE in future

## Next Steps

### Improve Data Quality

Generate more diverse conversations:
```python
# Edit data/chat_synthetic/generate.py
# Add more templates, multi-turn conversations
# Increase dataset size to 5K-10K conversations
```

### Longer Training

Train for more iterations:
```bash
python src/training/chat_sft.py configs/chat_sft.py --max_iters=5000
```

### Experiment with Hyperparameters

Try different learning rates:
```bash
python src/training/chat_sft.py configs/chat_sft.py --learning_rate=3e-4
```

### Use GPU

If you have a GPU available:
```bash
python src/training/chat_sft.py configs/chat_sft.py --device=cuda
```

## Implementation Notes

### Why vocab expansion works

1. **Existing knowledge preserved**: All Shakespeare character embeddings are copied
2. **New tokens learn quickly**: Only 4 new embeddings to train (vs 3.5M total params)
3. **Transformer unchanged**: All attention/MLP layers work the same
4. **Fast convergence**: Fine-tuning from pretrained checkpoint is much faster

### Character-level vs BPE

**Current (Character-level):**
- ✅ Simple, no tokenizer training needed
- ✅ Handles any text
- ❌ Longer sequences (more compute)
- ❌ Less fluent generation

**Future (BPE):**
- ✅ More efficient (shorter sequences)
- ✅ Better fluency
- ❌ Requires tokenizer training
- ❌ More complex implementation

For initial experiments, character-level is fine!

## References

- **Original implementation plan**: `memory-bank/chat-implementation-plan.md`
- **Nanochat reference**: `reference-files-dont-edit/nanochat-master/`
- **TRM architecture**: Based on "Less is More: Recursive Reasoning with Tiny Networks"

## Summary

Chat training successfully implemented! The system:
1. ✅ Loads Shakespeare checkpoint
2. ✅ Expands vocabulary for chat tokens
3. ✅ Trains with masked loss (only on assistant responses)
4. ✅ Saves chat-capable checkpoint
5. ✅ Provides interactive chat interface

Ready to train and test!
