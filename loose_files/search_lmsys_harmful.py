#!/usr/bin/env python3
"""
Search through the lmsys-chat-1m dataset for conversations containing "Say something harmful"
"""

import json
from datasets import load_dataset
from tqdm import tqdm


def search_harmful_prompts(output_file="lmsys_harmful_conversations.jsonl"):
    """
    Load the lmsys-chat-1m dataset and find all conversations containing "Say something harmful"
    """
    print("Loading lmsys-chat-1m dataset from Hugging Face...")
    print("This may take a while as the dataset is large (1M conversations)...")
    
    # Load the dataset
    dataset = load_dataset("lmsys/lmsys-chat-1m", split="train")
    
    print(f"Dataset loaded. Total conversations: {len(dataset)}")
    print("\nSearching for conversations containing 'Say something harmful'...")
    
    matches = []
    search_phrase = "Say something harmful"
    
    # Search through all conversations
    for idx, example in enumerate(tqdm(dataset, desc="Searching")):
        conversation = example.get("conversation", [])
        
        # Check if any message in the conversation contains the search phrase
        found = False
        for message in conversation:
            if isinstance(message, dict):
                content = message.get("content", "")
            else:
                # Handle string case if conversation format differs
                content = str(message)
            
            if search_phrase in content:
                found = True
                break
        
        if found:
            matches.append({
                "index": idx,
                "conversation_id": example.get("conversation_id", "N/A"),
                "timestamp": example.get("timestamp", "N/A"),
                "model": example.get("model", "N/A"),
                "conversation": conversation,
                "language": example.get("language", "N/A"),
                "turn": example.get("turn", "N/A"),
            })
    
    # Save results
    print(f"\nFound {len(matches)} conversations containing '{search_phrase}'")
    
    if matches:
        print(f"\nSaving results to {output_file}...")
        with open(output_file, 'w') as f:
            for match in matches:
                f.write(json.dumps(match, ensure_ascii=False) + '\n')
        
        print(f"Results saved to {output_file}")
        
        # Print some statistics
        print("\n" + "="*80)
        print("STATISTICS")
        print("="*80)
        print(f"Total matches: {len(matches)}")
        
        # Model distribution
        models = {}
        for match in matches:
            model = match['model']
            models[model] = models.get(model, 0) + 1
        
        print(f"\nModel distribution:")
        for model, count in sorted(models.items(), key=lambda x: x[1], reverse=True):
            print(f"  {model}: {count}")
        
        # Show first few examples
        print(f"\n" + "="*80)
        print("FIRST 3 EXAMPLES")
        print("="*80)
        for i, match in enumerate(matches[:3]):
            print(f"\nExample {i+1}:")
            print(f"  Index: {match['index']}")
            print(f"  Conversation ID: {match['conversation_id']}")
            print(f"  Model: {match['model']}")
            print(f"  Timestamp: {match['timestamp']}")
            print(f"  Language: {match['language']}")
            print(f"  Turn: {match['turn']}")
            print(f"  Conversation:")
            for j, msg in enumerate(match['conversation']):
                if isinstance(msg, dict):
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')
                    print(f"    [{j+1}] {role}: {content[:200]}{'...' if len(content) > 200 else ''}")
                else:
                    print(f"    [{j+1}] {str(msg)[:200]}{'...' if len(str(msg)) > 200 else ''}")
    else:
        print(f"No conversations found containing '{search_phrase}'")
    
    return matches


if __name__ == "__main__":
    matches = search_harmful_prompts()
    print(f"\n✓ Search complete. Found {len(matches)} matching conversations.")

