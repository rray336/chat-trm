# ABOUTME: Dataset loader for chat conversations in JSONL format
# ABOUTME: Handles tokenization with masking for supervised fine-tuning

import json
import torch
from torch.utils.data import Dataset
from typing import List, Dict
import os


class ChatDataset(Dataset):
    """
    Dataset for loading chat conversations from JSONL files.

    Each line in the JSONL file should be a JSON array of messages:
    [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]

    The dataset tokenizes conversations and returns:
    - input_ids: Token IDs for input (all tokens except last)
    - target_ids: Token IDs for targets (all tokens except first)
    - mask: Binary mask (1=train, 0=ignore) for masking user messages
    """

    def __init__(self, jsonl_path: str, tokenizer, block_size: int = 256, split='train'):
        """
        Initialize chat dataset.

        Args:
            jsonl_path: Path to JSONL file with conversations
            tokenizer: ChatTokenizer instance
            block_size: Maximum sequence length
            split: 'train' or 'val' (for logging purposes)
        """
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.split = split

        # Load conversations from JSONL
        self.conversations = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    messages = json.loads(line)
                    if isinstance(messages, list) and len(messages) > 0:
                        self.conversations.append(messages)
                    else:
                        print(f"Warning: Skipping invalid conversation at line {line_num}")
                except json.JSONDecodeError as e:
                    print(f"Warning: JSON decode error at line {line_num}: {e}")
                    continue

        print(f"Loaded {len(self.conversations)} conversations from {jsonl_path}")

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        """
        Get a single conversation, tokenized with masking.

        Returns:
            Dictionary with:
                - input_ids: torch.LongTensor, shape (seq_len,)
                - target_ids: torch.LongTensor, shape (seq_len,)
                - mask: torch.LongTensor, shape (seq_len,), 1=train, 0=ignore
                - length: int, actual length before padding
        """
        messages = self.conversations[idx]

        # Tokenize conversation with masking
        ids, mask = self.tokenizer.render_conversation(messages)

        # Truncate if too long
        if len(ids) > self.block_size:
            ids = ids[:self.block_size]
            mask = mask[:self.block_size]

        # Create input (all except last) and target (all except first)
        # This is the standard autoregressive setup
        input_ids = ids[:-1]
        target_ids = ids[1:]
        mask = mask[1:]  # Mask aligns with targets

        # Pad if necessary
        seq_len = len(input_ids)
        if seq_len < self.block_size - 1:
            pad_len = (self.block_size - 1) - seq_len
            input_ids = input_ids + [0] * pad_len
            target_ids = target_ids + [0] * pad_len
            mask = mask + [0] * pad_len  # Padded positions are ignored

        # Convert to tensors
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        target_ids = torch.tensor(target_ids, dtype=torch.long)
        mask = torch.tensor(mask, dtype=torch.long)

        # Apply mask to targets (set ignored positions to -1)
        # PyTorch CrossEntropyLoss will ignore targets with value -1
        target_ids = target_ids.clone()
        target_ids[mask == 0] = -1

        return {
            'input_ids': input_ids,
            'target_ids': target_ids,
            'length': seq_len,
        }


def collate_chat_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for batching chat conversations.

    Args:
        batch: List of dictionaries from ChatDataset.__getitem__

    Returns:
        Dictionary with batched tensors:
            - input_ids: (batch_size, seq_len)
            - target_ids: (batch_size, seq_len)
            - lengths: (batch_size,)
    """
    input_ids = torch.stack([item['input_ids'] for item in batch])
    target_ids = torch.stack([item['target_ids'] for item in batch])
    lengths = torch.tensor([item['length'] for item in batch], dtype=torch.long)

    return {
        'input_ids': input_ids,
        'target_ids': target_ids,
        'lengths': lengths,
    }


if __name__ == '__main__':
    # Test the dataset
    print("Testing ChatDataset...")

    # Create a small test JSONL file
    test_file = 'test_conversations.jsonl'
    test_conversations = [
        [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"}
        ],
        [
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm doing well, thanks for asking."}
        ],
        [
            {"role": "user", "content": "What's 2+2?"},
            {"role": "assistant", "content": "2+2 equals 4."}
        ],
    ]

    print(f"\nCreating test file: {test_file}")
    with open(test_file, 'w', encoding='utf-8') as f:
        for conv in test_conversations:
            f.write(json.dumps(conv) + '\n')

    # Load with ChatTokenizer
    from chat_tokenizer import ChatTokenizer
    # Use absolute path for testing
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    tokenizer = ChatTokenizer('data/shakespeare_char/meta.pkl')

    # Create dataset
    dataset = ChatDataset(test_file, tokenizer, block_size=128, split='test')

    print(f"\nDataset size: {len(dataset)}")

    # Test first item
    print("\nTesting first conversation:")
    item = dataset[0]
    print(f"  Input shape: {item['input_ids'].shape}")
    print(f"  Target shape: {item['target_ids'].shape}")
    print(f"  Actual length: {item['length']}")
    print(f"  Number of training targets: {(item['target_ids'] != -1).sum().item()}")

    # Test batching
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_chat_batch)

    print("\nTesting batching:")
    batch = next(iter(dataloader))
    print(f"  Batch input shape: {batch['input_ids'].shape}")
    print(f"  Batch target shape: {batch['target_ids'].shape}")
    print(f"  Batch lengths: {batch['lengths']}")

    # Cleanup
    os.remove(test_file)
    print(f"\nCleaned up test file: {test_file}")

    print("\nDataset test passed!")
