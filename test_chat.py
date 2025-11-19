# ABOUTME: Simple interactive chat interface for testing trained chat model
# ABOUTME: Loads chat checkpoint and enables conversational interaction

import os
import sys
import torch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.model.config import TRMConfig
from src.model.trm import TRM
from src.data.chat_tokenizer import ChatTokenizer


def load_chat_model(checkpoint_path='out-chat-sft/ckpt.pt', device='cpu'):
    """
    Load trained chat model from checkpoint.

    Args:
        checkpoint_path: Path to chat model checkpoint
        device: Device to load model on

    Returns:
        model, tokenizer, config
    """
    print(f"Loading chat model from {checkpoint_path}...")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint['model_config']

    # Create model
    model = TRM(model_config)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.to(device)

    # Load tokenizer
    tokenizer = ChatTokenizer()

    print(f"Model loaded: {model.get_num_params()/1e6:.2f}M parameters")
    print(f"Vocabulary: {model_config.vocab_size} tokens")
    print(f"Recursion steps: {model_config.recursion_steps}\n")

    return model, tokenizer, model_config


@torch.no_grad()
def generate_response(model, tokenizer, conversation_ids, max_new_tokens=100,
                     temperature=0.8, top_k=50, device='cpu'):
    """
    Generate assistant response given conversation history.

    Args:
        model: TRM model
        tokenizer: ChatTokenizer
        conversation_ids: List of token IDs for conversation so far
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling
        device: Device

    Returns:
        response_ids: List of generated token IDs (not including special tokens)
    """
    # Add assistant start token
    input_ids = conversation_ids + [tokenizer.encode_special('<|assistant_start|>')]

    # Convert to tensor
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    # Generate tokens
    response_ids = []
    assistant_end_token = tokenizer.get_assistant_end_token()

    for _ in range(max_new_tokens):
        # Get logits
        logits, _, _ = model(input_tensor)

        # Get logits for next token (last position)
        logits = logits[:, -1, :] / temperature

        # Top-k filtering
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        # Sample
        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()

        # Stop if we hit assistant end token
        if next_id == assistant_end_token:
            break

        # Add to response and input
        response_ids.append(next_id)
        input_tensor = torch.cat([input_tensor, torch.tensor([[next_id]], device=device)], dim=1)

    return response_ids


def chat_loop(model, tokenizer, device='cpu', max_turns=50):
    """
    Interactive chat loop.

    Args:
        model: TRM model
        tokenizer: ChatTokenizer
        device: Device
        max_turns: Maximum number of conversation turns
    """
    print("=" * 60)
    print("Chat-TRM Interactive Chat")
    print("=" * 60)
    print("Type your message and press Enter. Type 'quit' to exit.\n")

    conversation_ids = []

    for turn in range(max_turns):
        # Get user input
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("\nGoodbye!")
            break

        # Encode user message
        user_msg_ids = tokenizer.render_user_message(user_input)
        conversation_ids.extend(user_msg_ids)

        # Generate response
        response_ids = generate_response(
            model, tokenizer, conversation_ids,
            max_new_tokens=100, temperature=0.8, top_k=50, device=device
        )

        # Decode response
        response_text = tokenizer.decode(response_ids)

        print(f"Assistant: {response_text}\n")

        # Add to conversation history
        conversation_ids.append(tokenizer.encode_special('<|assistant_start|>'))
        conversation_ids.extend(response_ids)
        conversation_ids.append(tokenizer.encode_special('<|assistant_end|>'))


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Test chat model interactively')
    parser.add_argument('--checkpoint', type=str, default='out-chat-sft/ckpt.pt',
                       help='Path to chat checkpoint')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to run on')
    args = parser.parse_args()

    # Check checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        print("Please train the chat model first using:")
        print("  python src/training/chat_sft.py configs/chat_sft.py")
        return

    # Load model
    model, tokenizer, config = load_chat_model(args.checkpoint, args.device)

    # Start chat
    chat_loop(model, tokenizer, args.device)


if __name__ == '__main__':
    main()
