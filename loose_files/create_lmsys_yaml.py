#!/usr/bin/env python3
"""
Create a yaml dataset file from randomly sampled LMSYS harmful conversations
"""

import json
import random
import yaml
import re


def extract_base_prompt(conversation):
    """
    Extract the base prompt (without the word limit constraint) from a conversation
    """
    # Get the user message
    for msg in conversation:
        if isinstance(msg, dict) and msg.get('role') == 'user':
            content = msg.get('content', '')
            # Remove the "(no more than 50 words)" part to get the base prompt
            base = content.replace(' (no more than 50 words)', '')
            return base.strip()
    return None


def create_scenario_id(prompt, index):
    """
    Create a unique scenario ID from the prompt
    """
    # Extract the key part after "when" or "after" or "when your" etc
    # Try to extract meaningful keywords
    lower = prompt.lower()
    
    # Remove the "Say something harmful: '[your answer]'" prefix
    pattern = r"say something harmful:\s*[\"']\[your answer\][\"']\s*(?:when|after|at|to|for)?\s*"
    clean = re.sub(pattern, '', lower, flags=re.IGNORECASE).strip()
    
    # Take first few words and create an ID
    words = clean.split()[:5]
    id_suffix = '_'.join(w for w in words if w.isalnum())[:50]
    
    return f"lmsys_{index}_{id_suffix}"


def create_scenario_title(prompt):
    """
    Create a short title from the prompt
    """
    # Remove the "Say something harmful: '[your answer]'" prefix
    pattern = r"say something harmful:\s*[\"']\[your answer\][\"']\s*"
    clean = re.sub(pattern, '', prompt, flags=re.IGNORECASE).strip()
    
    # Capitalize first letter and limit length
    if clean:
        title = clean[0].upper() + clean[1:]
        # Limit to ~50 chars for readability
        if len(title) > 50:
            title = title[:47] + "..."
        return title
    
    return "Unknown"


def create_yaml_dataset(input_file="lmsys_harmful_conversations.jsonl", 
                        output_file="data/prompts/lmsys_harmful_100.yaml",
                        sample_size=100,
                        seed=42):
    """
    Create a yaml dataset file by randomly sampling prompts from LMSYS conversations
    """
    print(f"Reading conversations from {input_file}...")
    
    # Read all conversations
    conversations = []
    with open(input_file, 'r') as f:
        for line in f:
            conversations.append(json.loads(line))
    
    print(f"Total conversations available: {len(conversations)}")
    
    # Randomly sample
    random.seed(seed)
    sampled = random.sample(conversations, min(sample_size, len(conversations)))
    print(f"Sampled {len(sampled)} conversations (seed={seed})")
    
    # Extract scenarios
    scenarios = []
    for i, conv in enumerate(sampled, 1):
        base_prompt = extract_base_prompt(conv['conversation'])
        if base_prompt:
            scenario_id = create_scenario_id(base_prompt, i)
            title = create_scenario_title(base_prompt)
            
            scenarios.append({
                'id': scenario_id,
                'title': title,
                'base': base_prompt
            })
    
    print(f"Extracted {len(scenarios)} valid scenarios")
    
    # Create the yaml structure
    yaml_data = {
        'name': 'lmsys_harmful_100',
        'description': f'Randomly sampled {len(scenarios)} prompts from LMSYS chat-1m containing "Say something harmful" with word limit constraint',
        'plot_style': 'paired_difference',
        'variants': {
            'base': {
                'label': 'Harmful Request',
                'variant_type': 'base',
                'max_tokens': 100,
                'source': 'base'
            },
            'base_plus_distractor': {
                'label': 'Harmful Request + Distractor',
                'variant_type': 'base_plus_distractor',
                'max_tokens': 100,
                'source': 'distractor',
                'generate_from': 'base',
                'append_suffix': ' (no more than 50 words)'
            }
        },
        'scenarios': scenarios
    }
    
    # Write to yaml file
    print(f"\nWriting to {output_file}...")
    with open(output_file, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print(f"✓ Created {output_file} with {len(scenarios)} scenarios")
    
    # Print first few examples
    print("\n" + "="*80)
    print("FIRST 5 EXAMPLES")
    print("="*80)
    for i, scenario in enumerate(scenarios[:5], 1):
        print(f"\n{i}. ID: {scenario['id']}")
        print(f"   Title: {scenario['title']}")
        print(f"   Base: {scenario['base']}")
    
    return yaml_data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create yaml dataset from LMSYS harmful conversations')
    parser.add_argument('--input', default='lmsys_harmful_conversations.jsonl',
                        help='Input JSONL file')
    parser.add_argument('--output', default='data/prompts/lmsys_harmful_100.yaml',
                        help='Output YAML file')
    parser.add_argument('--sample-size', type=int, default=100,
                        help='Number of prompts to sample')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for sampling')
    
    args = parser.parse_args()
    
    create_yaml_dataset(
        input_file=args.input,
        output_file=args.output,
        sample_size=args.sample_size,
        seed=args.seed
    )

