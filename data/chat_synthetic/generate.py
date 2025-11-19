# ABOUTME: Generate synthetic chat conversations for training
# ABOUTME: Creates simple Q&A, greetings, and factual conversations

import json
import random
import os


# Templates for synthetic conversations
GREETINGS = [
    (["Hello!", "Hi there! How can I help you today?"], True),
    (["Hi", "Hello! What can I do for you?"], True),
    (["Hey", "Hey! How are you?"], False),
    (["Good morning", "Good morning! How can I assist you?"], True),
    (["How are you?", "I'm doing well, thank you for asking! How are you?"], False),
    (["What's up?", "Not much, just here to help! What can I do for you?"], False),
]

SIMPLE_QA = [
    (["What is your name?", "I'm an AI assistant. You can call me Assistant."], True),
    (["Who are you?", "I'm an AI assistant designed to help answer questions and have conversations."], True),
    (["What can you do?", "I can help answer questions, have conversations, and assist with various tasks."], True),
    (["Can you help me?", "Of course! I'll do my best to help. What do you need?"], True),
    (["Are you a robot?", "Yes, I'm an AI assistant - a computer program designed to have conversations."], True),
    (["Do you have feelings?", "As an AI, I don't have feelings like humans do. I'm a program designed to be helpful and conversational."], False),
]

MATH_QA = [
    (["What is three plus three?", "Three plus three equals six."], True),
    (["What's ten times three?", "Ten times three equals thirty."], True),
    (["Calculate ten minus three", "Ten minus three equals seven."], True),
    (["Is seven greater than three?", "Yes, seven is greater than three."], True),
]

FACTS = [
    (["What color is the sky?", "The sky typically appears blue during the day."], True),
    (["How many days in a week?", "There are seven days in a week."], True),
    (["What's the capital of France?", "The capital of France is Paris."], True),
    (["How many continents are there?", "There are seven continents on Earth."], True),
    (["What language do they speak in Spain?", "The primary language spoken in Spain is Spanish."], True),
]

MULTI_TURN = [
    [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hello! How can I help you today?"},
        {"role": "user", "content": "I have a question about math."},
        {"role": "assistant", "content": "Sure! I'd be happy to help with math. What's your question?"},
        {"role": "user", "content": "What is ten plus three?"},
        {"role": "assistant", "content": "Ten plus three equals thirteen."},
    ],
    [
        {"role": "user", "content": "Hi there"},
        {"role": "assistant", "content": "Hi! How are you doing today?"},
        {"role": "user", "content": "I'm good, thanks. Can you help me with something?"},
        {"role": "assistant", "content": "Of course! What do you need help with?"},
        {"role": "user", "content": "I want to learn about countries."},
        {"role": "assistant", "content": "Great! Countries are fascinating. What would you like to know?"},
    ],
]

POLITE_EXCHANGES = [
    (["Thank you", "You're welcome! Let me know if you need anything else."], True),
    (["Thanks for your help", "Happy to help! Feel free to ask if you have more questions."], True),
    (["I appreciate it", "My pleasure! That's what I'm here for."], True),
    (["Goodbye", "Goodbye! Have a great day!"], True),
    (["See you later", "See you! Take care!"], True),
]


def create_conversation(turns, multi_turn=False):
    """
    Create a conversation from turns.

    Args:
        turns: List of [user_message, assistant_message] or full conversation
        multi_turn: If True, turns is already a full conversation

    Returns:
        List of message dictionaries
    """
    if multi_turn:
        return turns

    messages = []
    if isinstance(turns, list) and len(turns) == 2:
        messages.append({"role": "user", "content": turns[0]})
        messages.append({"role": "assistant", "content": turns[1]})
    return messages


def generate_dataset(num_conversations=1500, train_split=0.9):
    """
    Generate synthetic conversation dataset.

    Args:
        num_conversations: Total number of conversations to generate
        train_split: Fraction of data for training (rest is validation)

    Returns:
        train_conversations, val_conversations
    """
    all_conversations = []

    # Add all template conversations
    template_groups = [
        (GREETINGS, False),
        (SIMPLE_QA, False),
        (MATH_QA, False),
        (FACTS, False),
        (POLITE_EXCHANGES, False),
        (MULTI_TURN, True),
    ]

    for templates, is_multi_turn in template_groups:
        for template in templates:
            if is_multi_turn:
                # Template is already a full conversation
                all_conversations.append(template)
            else:
                # Template is (turns, important_flag)
                turns, _ = template
                conv = create_conversation(turns, multi_turn=False)
                all_conversations.append(conv)

    # Duplicate to reach target size
    while len(all_conversations) < num_conversations:
        # Randomly pick and slightly vary some conversations
        base_conv = random.choice(all_conversations[:50])  # Pick from original set
        all_conversations.append(base_conv)

    # Shuffle
    random.shuffle(all_conversations)

    # Trim to exact size
    all_conversations = all_conversations[:num_conversations]

    # Split into train/val
    split_idx = int(num_conversations * train_split)
    train_conversations = all_conversations[:split_idx]
    val_conversations = all_conversations[split_idx:]

    return train_conversations, val_conversations


def save_jsonl(conversations, filepath):
    """Save conversations to JSONL file."""
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for conv in conversations:
            f.write(json.dumps(conv) + '\n')
    print(f"Saved {len(conversations)} conversations to {filepath}")


def main():
    print("Generating synthetic chat dataset...")

    # Set random seed for reproducibility
    random.seed(42)

    # Generate conversations
    num_total = 1500
    train_convs, val_convs = generate_dataset(num_conversations=num_total, train_split=0.9)

    print(f"\nDataset statistics:")
    print(f"  Total conversations: {num_total}")
    print(f"  Training: {len(train_convs)}")
    print(f"  Validation: {len(val_convs)}")

    # Save to files
    train_path = 'data/chat_synthetic/train.jsonl'
    val_path = 'data/chat_synthetic/val.jsonl'

    save_jsonl(train_convs, train_path)
    save_jsonl(val_convs, val_path)

    print("\nSample conversations:")
    print("\nTrain sample 1:")
    print(json.dumps(train_convs[0], indent=2))
    print("\nTrain sample 2:")
    print(json.dumps(train_convs[1], indent=2))

    print("\nDataset generation complete!")


if __name__ == '__main__':
    main()
