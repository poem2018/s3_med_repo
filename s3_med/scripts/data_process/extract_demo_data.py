#!/usr/bin/env python3
"""
Extract and process demo data from demo_data.jsonl
"""

import json
import re

# Read the demo_data.jsonl file
with open('/scratch/bcew/ruikez2/intern/s3_med/data/demo_exp/demo_data.jsonl', 'r') as f:
    lines = f.readlines()

# Process each line
data_entries = []
for line in lines:
    if line.strip():
        entry = json.loads(line)
        
        # Extract text without Similar Patient Case section
        text = entry['text']
        
        # Remove the Similar Patient Case section for no_similar version
        pattern = r'Similar Patient Case:.*?(?=\n</patient_data>)'
        text_no_similar = re.sub(pattern, '', text, flags=re.DOTALL)
        
        # Clean up extra newlines before closing tag
        text_no_similar = re.sub(r'\n+</patient_data>', '\n</patient_data>', text_no_similar)
        
        data_entries.append({
            'text_with_similar': entry['text'],
            'text_no_similar': text_no_similar,
            'data_source': entry['data_source'],
            'reward_model': entry['reward_model']
        })

# Print the extracted data in Python format for both files
print("# For convert_to_parquet_no_similar.py:")
print("json_data = [")
for i, entry in enumerate(data_entries):
    print("    {")
    print(f'        "text": {json.dumps(entry["text_no_similar"])},')
    print(f'        "data_source": "{entry["data_source"]}",')
    print("        \"reward_model\": {")
    print(f'            "style": "{entry["reward_model"]["style"]}",')
    print("            \"ground_truth\": {")
    print(f'                "answers": {json.dumps(entry["reward_model"]["ground_truth"]["answers"])},')
    print(f'                "patient_id": "{entry["reward_model"]["ground_truth"]["patient_id"]}",')
    print(f'                "mortality": {entry["reward_model"]["ground_truth"]["mortality"]},')
    print(f'                "mortality_inunit": {entry["reward_model"]["ground_truth"]["mortality_inunit"]}')
    print("            }")
    print("        }")
    if i < len(data_entries) - 1:
        print("    },")
    else:
        print("    }")
print("]")

print("\n\n# For convert_to_parquet.py:")
print("json_data = [")
for i, entry in enumerate(data_entries):
    print("    {")
    print(f'        "text": {json.dumps(entry["text_with_similar"])},')
    print(f'        "data_source": "{entry["data_source"]}",')
    print("        \"reward_model\": {")
    print(f'            "style": "{entry["reward_model"]["style"]}",')
    print("            \"ground_truth\": {")
    print(f'                "answers": {json.dumps(entry["reward_model"]["ground_truth"]["answers"])},')
    print(f'                "patient_id": "{entry["reward_model"]["ground_truth"]["patient_id"]}",')
    print(f'                "mortality": {entry["reward_model"]["ground_truth"]["mortality"]},')
    print(f'                "mortality_inunit": {entry["reward_model"]["ground_truth"]["mortality_inunit"]}')
    print("            }")
    print("        }")
    if i < len(data_entries) - 1:
        print("    },")
    else:
        print("    }")
print("]")