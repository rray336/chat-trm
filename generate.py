"""Generate text from trained TRM model."""

import os
import sys
import pickle
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model.trm import TRM

def generate_text(
    checkpoint_path='out-shakespeare-char/ckpt.pt',
    start_text='\n',
    max_new_tokens=500,
    temperature=0.8,
    top_k=200,
    num_samples=1
):
    """
    Generate text from a trained model.

    Args:
        checkpoint_path: Path to model checkpoint
        start_text: Text to start generation from
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Sample from top k tokens (None = sample from all)
        num_samples: Number of samples to generate
    """
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model_config = checkpoint['model_config']

    # Load metadata (for encoding/decoding)
    data_dir = 'data/shakespeare_char'
    with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)

    stoi = meta['stoi']
    itos = meta['itos']

    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    # Create model
    print(f"Creating model with config: {model_config}")
    model = TRM(model_config)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    print(f"\nGenerating {num_samples} sample(s)...")
    print(f"Start text: {repr(start_text)}")
    print(f"Temperature: {temperature}, Top-k: {top_k}")
    print(f"Recursion steps: {model_config.recursion_steps}")
    print("=" * 80)

    # Generate samples
    for i in range(num_samples):
        # Encode start text
        start_ids = encode(start_text)
        x = torch.tensor(start_ids, dtype=torch.long).unsqueeze(0)

        # Generate
        with torch.no_grad():
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)

        # Decode
        generated = decode(y[0].tolist())

        print(f"\nSample {i+1}:")
        print("-" * 80)
        print(generated)
        print("-" * 80)

    # Print training info
    print(f"\nCheckpoint info:")
    print(f"  Training iterations: {checkpoint['iter_num']}")
    print(f"  Best validation loss: {checkpoint['best_val_loss']:.4f}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate text from trained TRM model')
    parser.add_argument('--checkpoint', type=str, default='out-shakespeare-char/ckpt.pt',
                       help='Path to checkpoint')
    parser.add_argument('--start', type=str, default='\n',
                       help='Start text')
    parser.add_argument('--num_samples', type=int, default=3,
                       help='Number of samples to generate')
    parser.add_argument('--max_new_tokens', type=int, default=500,
                       help='Number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=200,
                       help='Top-k sampling')

    args = parser.parse_args()

    generate_text(
        checkpoint_path=args.checkpoint,
        start_text=args.start,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        num_samples=args.num_samples
    )
