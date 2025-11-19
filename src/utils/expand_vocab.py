# ABOUTME: Utility to expand vocabulary in existing checkpoint
# ABOUTME: Used to prepare Shakespeare checkpoint for chat training with special tokens

import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


def expand_vocab_checkpoint(checkpoint_path, old_vocab_size, new_vocab_size, output_path):
    """
    Expand vocabulary in a checkpoint from old_vocab_size to new_vocab_size.

    New token embeddings are initialized randomly.

    Args:
        checkpoint_path: Path to original checkpoint (e.g., Shakespeare)
        old_vocab_size: Original vocab size (e.g., 65)
        new_vocab_size: New vocab size (e.g., 69)
        output_path: Where to save expanded checkpoint

    Returns:
        None (saves to output_path)
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    state_dict = checkpoint['model']
    model_config = checkpoint['model_config']

    print(f"\nExpanding vocabulary from {old_vocab_size} to {new_vocab_size}...")

    # Expand tok_embeddings.weight: [old_vocab, hidden] -> [new_vocab, hidden]
    if 'tok_embeddings.weight' in state_dict:
        old_emb = state_dict['tok_embeddings.weight']
        hidden_size = old_emb.shape[1]

        new_emb = torch.randn(new_vocab_size, hidden_size) * 0.02  # Small random init
        new_emb[:old_vocab_size] = old_emb  # Copy old embeddings

        state_dict['tok_embeddings.weight'] = new_emb
        print(f"  tok_embeddings.weight: {old_emb.shape} -> {new_emb.shape}")

    # Expand lm_head.weight: [hidden, old_vocab] -> [hidden, new_vocab]
    if 'lm_head.weight' in state_dict:
        old_head = state_dict['lm_head.weight']
        hidden_size = old_head.shape[0]

        new_head = torch.randn(hidden_size, new_vocab_size) * 0.02  # Small random init
        new_head[:, :old_vocab_size] = old_head  # Copy old weights

        state_dict['lm_head.weight'] = new_head
        print(f"  lm_head.weight: {old_head.shape} -> {new_head.shape}")

    # Update config
    model_config['vocab_size'] = new_vocab_size
    checkpoint['model_config'] = model_config

    # Save expanded checkpoint
    print(f"\nSaving expanded checkpoint to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(checkpoint, output_path)

    print(f"Done! Checkpoint saved with {new_vocab_size} vocabulary size.")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Expand vocabulary in checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to original checkpoint')
    parser.add_argument('--old_vocab', type=int, default=65,
                       help='Original vocabulary size')
    parser.add_argument('--new_vocab', type=int, default=69,
                       help='New vocabulary size')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for expanded checkpoint')

    args = parser.parse_args()

    expand_vocab_checkpoint(
        checkpoint_path=args.checkpoint,
        old_vocab_size=args.old_vocab,
        new_vocab_size=args.new_vocab,
        output_path=args.output
    )
