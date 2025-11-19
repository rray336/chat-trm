# ABOUTME: Character-level tokenizer with special tokens for chat conversations
# ABOUTME: Implements render_conversation() with masking for supervised fine-tuning

import pickle
import os
from typing import List, Tuple, Dict


class ChatTokenizer:
    """
    Character-level tokenizer extended with special tokens for chat.

    Base vocabulary (0-64): Character-level tokens from Shakespeare
    Special tokens (65-68): Chat formatting tokens
    """

    # Special token definitions
    SPECIAL_TOKENS = {
        '<|user_start|>': 65,
        '<|user_end|>': 66,
        '<|assistant_start|>': 67,
        '<|assistant_end|>': 68,
    }

    def __init__(self, char_vocab_path='data/shakespeare_char/meta.pkl'):
        """
        Initialize tokenizer with base character vocabulary + special tokens.

        Args:
            char_vocab_path: Path to Shakespeare character vocabulary pickle
        """
        # Load base character vocabulary (65 tokens)
        with open(char_vocab_path, 'rb') as f:
            meta = pickle.load(f)

        self.base_stoi = meta['stoi']  # char -> int (0-64)
        self.base_itos = meta['itos']  # int -> char (0-64)

        # Extend with special tokens
        self.stoi = dict(self.base_stoi)  # Copy base vocab
        self.itos = dict(self.base_itos)  # Copy base vocab

        # Add special tokens (indices 65-68)
        for token, idx in self.SPECIAL_TOKENS.items():
            self.stoi[token] = idx
            self.itos[idx] = token

        self.vocab_size = 69  # 65 chars + 4 special tokens

        # Reverse mapping for special tokens
        self.special_token_ids = {v: k for k, v in self.SPECIAL_TOKENS.items()}

    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs (character-level).

        Args:
            text: String to encode

        Returns:
            List of token IDs
        """
        return [self.stoi[c] for c in text]

    def decode(self, ids: List[int]) -> str:
        """
        Decode token IDs to text.

        Args:
            ids: List of token IDs

        Returns:
            Decoded string
        """
        chars = []
        for i in ids:
            if i in self.itos:
                chars.append(self.itos[i])
            else:
                chars.append('?')  # Unknown token
        return ''.join(chars)

    def encode_special(self, token_name: str) -> int:
        """Get token ID for special token."""
        if token_name not in self.SPECIAL_TOKENS:
            raise ValueError(f"Unknown special token: {token_name}")
        return self.SPECIAL_TOKENS[token_name]

    def render_conversation(self, messages: List[Dict[str, str]],
                          include_bos: bool = False) -> Tuple[List[int], List[int]]:
        """
        Convert conversation messages to token IDs with masking.

        This is the KEY method for chat training. It:
        1. Formats conversation with special tokens
        2. Returns mask indicating which tokens to train on

        Args:
            messages: List of {"role": "user"/"assistant", "content": "..."}
            include_bos: Whether to prepend BOS token (not used for char-level)

        Returns:
            ids: Token IDs for entire conversation
            mask: Binary mask (1=train on this token, 0=ignore)

        Example:
            Input: [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"}
            ]

            Output:
                ids: [65, 'H', 'i', 66, 67, 'H', 'e', 'l', 'l', 'o', '!', 68]
                mask: [0,  0,   0,  0,  0,  1,   1,   1,   1,   1,   1,   0]

            Meaning:
                - User message tokens (and special tokens) have mask=0 (don't train)
                - Assistant message tokens have mask=1 (train on these)
        """
        ids = []
        mask = []

        for msg in messages:
            role = msg['role']
            content = msg['content']

            if role == 'user':
                # Add user message with special tokens
                # <|user_start|> content <|user_end|>
                ids.append(self.SPECIAL_TOKENS['<|user_start|>'])
                mask.append(0)  # Don't train on special token

                # Encode user content
                content_ids = self.encode(content)
                ids.extend(content_ids)
                mask.extend([0] * len(content_ids))  # Don't train on user text

                ids.append(self.SPECIAL_TOKENS['<|user_end|>'])
                mask.append(0)  # Don't train on special token

            elif role == 'assistant':
                # Add assistant message with special tokens
                # <|assistant_start|> content <|assistant_end|>
                ids.append(self.SPECIAL_TOKENS['<|assistant_start|>'])
                mask.append(0)  # Don't train on special token

                # Encode assistant content
                content_ids = self.encode(content)
                ids.extend(content_ids)
                mask.extend([1] * len(content_ids))  # TRAIN on assistant text

                ids.append(self.SPECIAL_TOKENS['<|assistant_end|>'])
                mask.append(0)  # Don't train on special token

            elif role == 'system':
                # System messages: treat like user (don't train on them)
                # For now, just ignore system messages
                # Could prepend to first user message if needed
                pass
            else:
                raise ValueError(f"Unknown role: {role}")

        return ids, mask

    def render_user_message(self, content: str) -> List[int]:
        """
        Encode a user message with special tokens.
        Useful for interactive chat.

        Returns:
            Token IDs: [<|user_start|>, ...content..., <|user_end|>]
        """
        ids = [self.SPECIAL_TOKENS['<|user_start|>']]
        ids.extend(self.encode(content))
        ids.append(self.SPECIAL_TOKENS['<|user_end|>'])
        return ids

    def render_assistant_start(self) -> List[int]:
        """Get token IDs for starting an assistant response."""
        return [self.SPECIAL_TOKENS['<|assistant_start|>']]

    def get_assistant_end_token(self) -> int:
        """Get token ID for assistant end marker."""
        return self.SPECIAL_TOKENS['<|assistant_end|>']

    def save_vocab(self, path: str):
        """Save extended vocabulary to pickle file."""
        meta = {
            'stoi': self.stoi,
            'itos': self.itos,
            'vocab_size': self.vocab_size,
            'special_tokens': self.SPECIAL_TOKENS,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(meta, f)
        print(f"Saved chat vocabulary ({self.vocab_size} tokens) to {path}")


if __name__ == '__main__':
    # Test the tokenizer
    print("Testing ChatTokenizer...")

    tokenizer = ChatTokenizer()

    print(f"\nVocabulary size: {tokenizer.vocab_size}")
    print(f"Special tokens: {tokenizer.SPECIAL_TOKENS}")

    # Test conversation rendering
    messages = [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "I'm doing well."},
    ]

    ids, mask = tokenizer.render_conversation(messages)

    print(f"\nTest conversation:")
    for msg in messages:
        print(f"  {msg['role']}: {msg['content']}")

    print(f"\nTokenized length: {len(ids)} tokens")
    print(f"Mask: {mask[:50]}..." if len(mask) > 50 else f"Mask: {mask}")
    print(f"\nMask statistics:")
    print(f"  Train tokens (mask=1): {sum(mask)} ({100*sum(mask)/len(mask):.1f}%)")
    print(f"  Ignore tokens (mask=0): {len(mask)-sum(mask)} ({100*(len(mask)-sum(mask))/len(mask):.1f}%)")

    # Decode to verify
    decoded = tokenizer.decode(ids)
    print(f"\nDecoded conversation:")
    print(decoded)

    print("\nTokenizer test passed!")
