#!/usr/bin/env python3
"""
Check which conversations in lmsys_harmful_conversations.jsonl 
contain "(no more than 50 words)" in their content
"""

import json


def check_word_limit(input_file="lmsys_harmful_conversations.jsonl"):
    """
    Check which conversations contain "(no more than 50 words)" string
    """
    print(f"Analyzing {input_file}...")
    
    search_string = "(no more than 50 words)"
    
    total_count = 0
    with_word_limit = 0
    without_word_limit = []
    
    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            total_count += 1
            data = json.loads(line)
            
            # Search through the conversation for the string
            found = False
            conversation = data.get("conversation", [])
            
            # Convert entire conversation to string for searching
            conversation_text = json.dumps(conversation)
            
            if search_string in conversation_text:
                found = True
                with_word_limit += 1
            
            if not found:
                without_word_limit.append({
                    "line_number": line_num,
                    "index": data.get("index"),
                    "conversation_id": data.get("conversation_id"),
                    "model": data.get("model"),
                    "conversation": conversation
                })
    
    # Report results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Total conversations: {total_count}")
    print(f"With '(no more than 50 words)': {with_word_limit}")
    print(f"WITHOUT '(no more than 50 words)': {len(without_word_limit)}")
    print(f"Percentage with word limit: {100 * with_word_limit / total_count:.2f}%")
    
    if without_word_limit:
        print("\n" + "="*80)
        print(f"CONVERSATIONS WITHOUT '(no more than 50 words)' ({len(without_word_limit)} total)")
        print("="*80)
        
        for i, item in enumerate(without_word_limit, 1):
            print(f"\n{i}. Line {item['line_number']}:")
            print(f"   Index: {item['index']}")
            print(f"   Conversation ID: {item['conversation_id']}")
            print(f"   Model: {item['model']}")
            print(f"   Conversation:")
            
            for j, msg in enumerate(item['conversation']):
                if isinstance(msg, dict):
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')
                    print(f"     [{j+1}] {role}: {content[:300]}{'...' if len(content) > 300 else ''}")
                else:
                    print(f"     [{j+1}] {str(msg)[:300]}{'...' if len(str(msg)) > 300 else ''}")
            
            # Only show first 20 to avoid overwhelming output
            if i >= 20 and len(without_word_limit) > 20:
                print(f"\n... and {len(without_word_limit) - 20} more.")
                print("\nSaving full list to 'without_word_limit.jsonl'...")
                with open('without_word_limit.jsonl', 'w') as out:
                    for item in without_word_limit:
                        out.write(json.dumps(item, ensure_ascii=False) + '\n')
                print("Full list saved to without_word_limit.jsonl")
                break
    else:
        print("\n✓ ALL conversations contain '(no more than 50 words)'!")
    
    return without_word_limit


if __name__ == "__main__":
    without_word_limit = check_word_limit()

