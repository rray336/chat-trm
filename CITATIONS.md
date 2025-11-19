# Citations and Acknowledgments

This project adapts and combines code from two open-source projects:

## Tiny Recursive Model (TRM)

- **Paper**: "Less is More: Recursive Reasoning with Tiny Networks"
- **Authors**: Alexia Jolicoeur-Martineau et al.
- **Paper URL**: https://arxiv.org/abs/2510.04871
- **Repository**: https://github.com/SamsungSAILMontreal/TinyRecursiveModels
- **License**: MIT License
- **Copyright**: Samsung SAIL Montreal

### What We Adapted

- **Recursive reasoning architecture**: Core concept of applying transformer blocks iteratively
- **Model components** (from `src/model/layers.py`):
  - RMSNorm (Root Mean Square Normalization)
  - SwiGLU activation function
  - RoPE (Rotary Position Embeddings)
- **Learnable initial state**: Pattern for initializing the latent state
- **Configuration patterns**: Model architecture configuration approach

### Key Differences from TRM

- **Causal attention**: Our implementation uses causal (autoregressive) attention for text generation, while TRM uses bidirectional attention for puzzle solving
- **Single-state architecture**: Simplified to single latent state (z), while TRM uses dual-state hierarchy (z_H, z_L)
- **Gradient flow**: Backpropagation through all recursion steps, while TRM only backprops through final step
- **Task domain**: Text completion vs visual puzzle solving

---

## nanoGPT

- **Author**: Andrej Karpathy
- **Repository**: https://github.com/karpathy/nanoGPT
- **License**: MIT License

### What We Adapted

- **Training infrastructure** (from `src/training/train.py`):
  - Main training loop with gradient accumulation
  - DDP (Distributed Data Parallel) support
  - Mixed precision training with GradScaler
  - Checkpoint management and best model tracking
  - Learning rate scheduling (cosine decay with warmup)
- **Data loading** (from `src/data/shakespeare.py`):
  - Memory-mapped dataset preparation
  - Train/validation split logic
  - Character-level tokenization
- **Configuration system** (from `configurator.py`):
  - Command-line config override mechanism
  - Config file loading and merging
- **Evaluation patterns**: Loss estimation and validation

### Key Differences from nanoGPT

- **Model architecture**: Uses recursive reasoning with weight reuse instead of stacked transformer layers
- **Attention mechanism**: Adapted to work with TRM-style recursive processing
- **Forward pass**: Modified to support multiple recursion steps through the same blocks

---

## BibTeX Citations

If you use this code in academic work, please cite both original papers:

```bibtex
@article{jolicoeur2024trm,
  title={Less is More: Recursive Reasoning with Tiny Networks},
  author={Jolicoeur-Martineau, Alexia and others},
  journal={arXiv preprint arXiv:2510.04871},
  year={2024}
}

@software{karpathy2022nanogpt,
  title={nanoGPT},
  author={Karpathy, Andrej},
  year={2022},
  url={https://github.com/karpathy/nanoGPT}
}
```

---

## License Compliance

This project is released under the MIT License, consistent with both source projects. See [LICENSE](LICENSE) for details.

Both TinyRecursiveModels and nanoGPT are licensed under the MIT License, which permits:
- Commercial use
- Modification
- Distribution
- Private use

The MIT License requires:
- License and copyright notice inclusion
- No warranty disclaimer

All original copyright notices from both projects are preserved in our LICENSE file.
