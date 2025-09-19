#!/usr/bin/env python3
import pandas as pd
import json

# Read the parquet file
df = pd.read_parquet('/scratch/bcew/ruikez2/intern/s3_med/data/demo_exp/demo_mimic_mortality.parquet')
idx = 1
# Show first record
print('='*60)
print('PARQUET FILE STRUCTURE')
print('='*60)
print(f'Total records: {len(df)}')
print(f'Columns: {df.columns.tolist()}')
print()

# Show first record in detail
record = df.iloc[idx]
# print(record)
print('='*60)
print('FIRST RECORD DETAILS')
print('='*60)

print('1. DATA SOURCE:')
print(f'   {record["data_source"]}')
print()

print('2. ABILITY:')
print(f'   {record["ability"]}')
print()

print('3. PROMPT:')
prompt = record['prompt'][0]
print(prompt)
print(f'   Role: {prompt["role"]}')
print(f'   Content (showing complete):')
print('   ' + '-'*50)
content = prompt['content']
print(content)
print()

print('4. REWARD MODEL:')
rm = record['reward_model']
print(f'   Style: {rm["style"]}')
print(f'   Ground Truth:')
gt = rm['ground_truth']
for key, value in gt.items():
    print(f'      - {key}: {value}')
print()

print('5. EXTRA INFO:')
extra = record['extra_info']
for key, value in extra.items():
    print(f'   - {key}: {value}')