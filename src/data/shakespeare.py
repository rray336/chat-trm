# ABOUTME: Prepares Shakespeare dataset for character-level language modeling
# ABOUTME: Data preparation patterns adapted from nanoGPT (https://github.com/karpathy/nanoGPT)

import os
import pickle
import requests
import numpy as np

def prepare_shakespeare_data(output_dir='data/shakespeare_char'):
    """
    Download and prepare the Shakespeare dataset for character-level training.
    Creates train.bin, val.bin, and meta.pkl files.

    Args:
        output_dir: Directory to save the prepared data files
    """
    os.makedirs(output_dir, exist_ok=True)

    # Download the tiny shakespeare dataset
    input_file = os.path.join(output_dir, 'input.txt')
    if not os.path.exists(input_file):
        print("Downloading Shakespeare dataset...")
        data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        response = requests.get(data_url)
        with open(input_file, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f"Downloaded to {input_file}")

    # Read the data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = f.read()
    print(f"Length of dataset in characters: {len(data):,}")

    # Get all unique characters
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    print(f"All unique characters: {''.join(chars)}")
    print(f"Vocab size: {vocab_size}")

    # Create character-to-integer mappings
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s):
        """Encode string to list of integers"""
        return [stoi[c] for c in s]

    def decode(l):
        """Decode list of integers to string"""
        return ''.join([itos[i] for i in l])

    # Create train and validation splits (90/10)
    n = len(data)
    train_data = data[:int(n * 0.9)]
    val_data = data[int(n * 0.9):]

    # Encode to integers
    train_ids = encode(train_data)
    val_ids = encode(val_data)
    print(f"Train has {len(train_ids):,} tokens")
    print(f"Val has {len(val_ids):,} tokens")

    # Export to binary files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)

    train_path = os.path.join(output_dir, 'train.bin')
    val_path = os.path.join(output_dir, 'val.bin')
    train_ids.tofile(train_path)
    val_ids.tofile(val_path)
    print(f"Saved train data to {train_path}")
    print(f"Saved val data to {val_path}")

    # Save metadata for encoding/decoding
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }
    meta_path = os.path.join(output_dir, 'meta.pkl')
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)
    print(f"Saved metadata to {meta_path}")
    print("\nDataset preparation complete!")

if __name__ == '__main__':
    prepare_shakespeare_data()
