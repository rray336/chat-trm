# ABOUTME: Main TRM model implementing recursive reasoning for text completion.
# ABOUTME: Uses iterative refinement through repeated transformer block applications to improve predictions.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from .config import TRMConfig
from .layers import (
    CastedEmbedding,
    CastedLinear,
    RotaryEmbedding,
    Attention,
    SwiGLU,
    rms_norm,
    trunc_normal_init_,
)


class TransformerBlock(nn.Module):
    """
    Single transformer block with causal self-attention and SwiGLU MLP.
    Uses post-normalization (norm after residual connection).
    """

    def __init__(self, config: TRMConfig):
        super().__init__()
        self.config = config

        # Self-attention
        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.head_dim,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,  # Standard MHA (not GQA)
            causal=True  # Causal for autoregressive text generation
        )

        # MLP with SwiGLU activation
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )

        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through transformer block.

        Args:
            cos_sin: RoPE cos/sin values
            hidden_states: Input tensor [batch, seq_len, hidden_size]

        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        # Self-attention with post-norm
        attn_output = self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states)
        hidden_states = rms_norm(hidden_states + attn_output, variance_epsilon=self.norm_eps)

        # MLP with post-norm
        mlp_output = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + mlp_output, variance_epsilon=self.norm_eps)

        return hidden_states


class TRM(nn.Module):
    """
    Tiny Recursive Model for text completion.

    Uses recursive reasoning: applies transformer blocks iteratively to refine predictions.
    Instead of stacking many layers, we reuse a small number of layers multiple times.
    """

    def __init__(self, config: TRMConfig):
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)

        # Token embeddings
        self.embed_scale = math.sqrt(config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            init_std=embed_init_std,
            cast_to=self.forward_dtype
        )

        # Positional encoding (RoPE)
        self.rotary_emb = RotaryEmbedding(
            dim=config.head_dim,
            max_position_embeddings=config.block_size,
            base=config.rope_theta
        )

        # Transformer blocks (reused recursively)
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # Initial latent state (learnable)
        self.z_init = nn.Parameter(
            trunc_normal_init_(torch.empty(config.hidden_size, dtype=self.forward_dtype), std=1.0)
        )

        # Output head
        self.lm_head = CastedLinear(config.hidden_size, config.vocab_size, bias=False)

        # Optional: tie embedding and lm_head weights (common in language models)
        # self.lm_head.weight = self.embed_tokens.embedding_weight

        print(f"TRM model initialized with {self.get_num_params() / 1e6:.2f}M parameters")

    def get_num_params(self, non_embedding=True):
        """Return the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embed_tokens.embedding_weight.numel()
        return n_params

    def _input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Convert input token IDs to embeddings with scaling."""
        embedding = self.embed_tokens(input_ids.to(torch.int32))
        return self.embed_scale * embedding

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_recursion_logits: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[torch.Tensor]]]:
        """
        Forward pass with recursive reasoning.

        Args:
            idx: Input token IDs [batch, seq_len]
            targets: Target token IDs for loss calculation [batch, seq_len]
            return_recursion_logits: If True, return logits from each recursion step

        Returns:
            logits: Output logits [batch, seq_len, vocab_size]
            loss: Cross-entropy loss (if targets provided)
            recursion_logits: List of logits from each step (if return_recursion_logits=True)
        """
        batch_size, seq_len = idx.shape
        device = idx.device

        # Get RoPE cos/sin
        cos_sin = self.rotary_emb()

        # Get input embeddings
        input_embeddings = self._input_embeddings(idx)

        # Initialize latent state
        # Expand z_init to batch size and sequence length
        z = self.z_init.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)

        # Track recursion logits if requested
        recursion_logits = [] if return_recursion_logits else None

        # Recursive reasoning loop
        for step in range(self.config.recursion_steps):
            # Inject input context
            z = z + input_embeddings

            # Apply transformer blocks
            for block in self.blocks:
                z = block(cos_sin=cos_sin, hidden_states=z)

            # Store intermediate logits if requested
            if return_recursion_logits:
                step_logits = self.lm_head(z)
                recursion_logits.append(step_logits.detach())

        # Final output
        logits = self.lm_head(z)

        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )

        return logits, loss, recursion_logits

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate new tokens autoregressively.

        Args:
            idx: Starting token IDs [batch, seq_len]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: If set, only sample from top k logits

        Returns:
            Generated token IDs [batch, seq_len + max_new_tokens]
        """
        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            # Forward pass
            logits, _, _ = self(idx_cond)

            # Get logits for last position
            logits = logits[:, -1, :] / temperature

            # Optional top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        """
        Estimate model flops utilization (MFU) similar to nanoGPT.
        Note: This is an approximation and may not be exact for recursive model.
        """
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layers, cfg.num_heads, cfg.head_dim, cfg.block_size

        # Approximate flops per token (ignoring recursion for now)
        flops_per_token = 6 * N + 12 * L * H * Q * T
        # Multiply by recursion steps
        flops_per_token *= cfg.recursion_steps

        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter

        # Express as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops

        mfu = flops_achieved / flops_promised
        return mfu
