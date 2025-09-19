#!/usr/bin/env python3
"""
Test script to verify MIMIC data format and generate sample output
"""

import pandas as pd
import json
import os

# Load a sample of the MIMIC data
df = pd.read_parquet('data/mimic_mortality_ug/train_e5_ug.parquet')
print(f"Loaded {len(df)} entries")
print(f"Data sources: {df['data_source'].unique()}")

# Check the structure of a sample row
sample_row = df.iloc[0]
print("\nSample row structure:")
print(f"Data source: {sample_row['data_source']}")
print(f"Prompt type: {type(sample_row['prompt'])}")
print(f"Reward model type: {type(sample_row['reward_model'])}")

# Extract information from the first row
prompt_content = sample_row['prompt'][0]['content']
ground_truth = sample_row['reward_model']['ground_truth']

print(f"\nGround truth keys: {ground_truth.keys()}")
print(f"Patient ID: {ground_truth.get('patient_id')}")
print(f"Mortality: {ground_truth.get('mortality')}")

# Extract clinical summary
import re
patient_match = re.search(r'<patient_data>(.*?)</patient_data>', prompt_content, re.DOTALL)
if patient_match:
    clinical_summary = patient_match.group(1).strip()
    print(f"\nExtracted clinical summary length: {len(clinical_summary)} chars")
    print(f"First 200 chars: {clinical_summary[:200]}...")

# Create a sample output structure
sample_output = {}
for idx in range(min(5, len(df))):
    row = df.iloc[idx]
    ground_truth = row['reward_model']['ground_truth']
    patient_id = str(ground_truth.get('patient_id', f'patient_{idx}'))
    
    sample_output[patient_id] = {
        'patient_id': patient_id,
        'clinical_summary': f"[Clinical data for patient {patient_id}]",
        'mortality': int(ground_truth.get('mortality', 0)),
        'golden_answers': ["high mortality risk" if ground_truth.get('mortality') else "low mortality risk"],
        'context_with_info': "Doc 1 (Title: Sample Medical Literature) Sample retrieval result..."
    }

# Save sample output
os.makedirs('data/RAG_retrieval/mimic_test_sample', exist_ok=True)
output_path = 'data/RAG_retrieval/mimic_test_sample/mimic_icu_mortality_output_sequences.json'
with open(output_path, 'w') as f:
    json.dump(sample_output, f, indent=2)

print(f"\nSample output saved to: {output_path}")
print(f"Output has {len(sample_output)} entries")