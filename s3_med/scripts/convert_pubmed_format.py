#!/usr/bin/env python3
"""
Convert PubMed data from MedRAG format to s3 index builder format.
Input: {"id": "...", "contents": "\"title\"\ntext"}
Output: {"id": "...", "title": "title", "text": "text", "contents": "..."}
"""

import json
import argparse
from tqdm import tqdm

def convert_format(input_file, output_file):
    """Convert PubMed jsonl format for index builder."""
    
    with open(input_file, 'r', encoding='utf-8') as inf:
        with open(output_file, 'w', encoding='utf-8') as outf:
            for line in tqdm(inf, desc="Converting format"):
                try:
                    data = json.loads(line.strip())
                    
                    contents = data.get('contents', '')
                    
                    # Try to extract title and text
                    if contents.startswith('"') and '"\n' in contents:
                        # Format: "title"\ntext
                        parts = contents.split('"\n', 1)
                        title = parts[0][1:]  # Remove leading quote
                        
                        if len(parts) > 1:
                            remaining_text = parts[1]
                            # Check if title is repeated at the beginning of the text
                            if remaining_text.startswith(title):
                                # Remove the repeated title and any following period/space
                                text = remaining_text[len(title):].lstrip('. ')
                            else:
                                text = remaining_text
                        else:
                            text = ""
                    else:
                        # No clear title/text separation
                        # Take first line as title, rest as text
                        lines = contents.split('\n', 1)
                        title = lines[0][:200]  # Limit title length
                        text = lines[1] if len(lines) > 1 else lines[0]
                    
                    # Create new format
                    new_data = {
                        'id': data['id'],
                        'title': title,
                        'text': text  # Keep original for reference
                    }
                    
                    outf.write(json.dumps(new_data) + '\n')
                    
                except Exception as e:
                    print(f"Error processing line: {e}")
                    continue

def main():
    parser = argparse.ArgumentParser(description='Convert PubMed data format')
    parser.add_argument('--input', required=True, help='Input jsonl file')
    parser.add_argument('--output', required=True, help='Output jsonl file')
    
    args = parser.parse_args()
    
    print(f"Converting {args.input} to {args.output}")
    convert_format(args.input, args.output)
    print("Conversion complete!")

if __name__ == "__main__":
    main()