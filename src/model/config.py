# ABOUTME: Configuration dataclass for TRM model architecture and training hyperparameters.
# ABOUTME: Defines all model dimensions, recursion depth, and training settings.

from dataclasses import dataclass


@dataclass
class TRMConfig:
    """Configuration for TRM-inspired text completion model."""

    # Model architecture
    vocab_size: int = 65  # Size of vocabulary (Shakespeare char-level: 65)
    block_size: int = 256  # Maximum context length
    hidden_size: int = 384  # Embedding dimension
    num_heads: int = 6  # Number of attention heads
    n_layers: int = 2  # Number of transformer blocks
    recursion_steps: int = 2  # Number of recursive refinement passes

    # Architecture details
    expansion: float = 4.0  # MLP expansion factor for SwiGLU
    dropout: float = 0.0  # Dropout rate (0.0 for pretraining, 0.1+ for finetuning)
    bias: bool = False  # Use bias in linear layers
    rms_norm_eps: float = 1e-5  # Epsilon for RMS normalization
    rope_theta: float = 10000.0  # Base for RoPE frequencies

    # Precision
    forward_dtype: str = "float32"  # "float32", "bfloat16", or "float16"

    def __post_init__(self):
        """Validate configuration."""
        assert self.hidden_size % self.num_heads == 0, \
            f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({self.num_heads})"
        assert self.recursion_steps >= 1, \
            f"recursion_steps must be >= 1, got {self.recursion_steps}"
        assert self.forward_dtype in ["float32", "bfloat16", "float16"], \
            f"forward_dtype must be float32, bfloat16, or float16, got {self.forward_dtype}"

    @property
    def head_dim(self):
        """Dimension of each attention head."""
        return self.hidden_size // self.num_heads

    def get_num_params(self, non_embedding=True):
        """
        Estimate number of parameters in the model.
        This is a rough estimate and actual value may differ slightly.
        """
        # Embedding layers
        embed_params = self.vocab_size * self.hidden_size  # token embedding
        embed_params += self.hidden_size  # initial state

        # Transformer blocks (repeated n_layers times)
        # Each block has: Attention + MLP
        # Attention: qkv_proj + o_proj
        attn_qkv = self.hidden_size * (3 * self.hidden_size)  # simplified
        attn_o = self.hidden_size * self.hidden_size
        attn_params = attn_qkv + attn_o

        # MLP: SwiGLU (gate_up + down)
        inter_size = round(self.expansion * self.hidden_size * 2 / 3)
        inter_size = (-(inter_size // -256)) * 256  # find_multiple of 256
        mlp_params = self.hidden_size * (inter_size * 2) + inter_size * self.hidden_size

        block_params = (attn_params + mlp_params) * self.n_layers

        # LM head
        lm_head_params = self.hidden_size * self.vocab_size

        total = embed_params + block_params + lm_head_params

        if non_embedding:
            # Don't count embedding params (common practice)
            total -= self.vocab_size * self.hidden_size

        return total
