#!/usr/bin/env python3
"""Debug 400 Bad Request errors"""

import json
import requests
import os

# Load one patient to test
data_dir = "/scratch/bcew/ruikez2/intern/s3_med/data/baseline_text_500"
patient_file = os.path.join(data_dir, 'all_patients.json')

with open(patient_file, 'r') as f:
    patients = json.load(f)

# Test with the second patient (first one that failed)
patient = patients[1]  # index 1 = second patient
patient_text_full = patient['text']

# Remove outcome section
if "=== OUTCOME ===" in patient_text_full:
    patient_text = patient_text_full.split("=== OUTCOME ===")[0].strip()
else:
    patient_text = patient_text_full

# Create the prompt
prompt = f"""You are an expert ICU physician. Based on the following patient's clinical data from the first 48 hours of ICU admission, predict whether the patient will die during this ICU stay.

{patient_text}

Based on the clinical data above, analyze the patient's condition and predict the ICU mortality outcome.

Instructions:
1. Analyze key risk factors including vital signs, Glasgow Coma Scale scores, and medical history
2. Consider temporal trends in the data
3. Provide a clear prediction: HIGH RISK (likely to die) or LOW RISK (likely to survive)
4. Briefly explain your reasoning (2-3 sentences)

Format your response as:
PREDICTION: [HIGH RISK or LOW RISK]
REASONING: [Your brief explanation]
"""

print(f"Patient ID: {patient['patient_id']}")
print(f"Prompt length: {len(prompt)} characters")
print(f"First 200 chars of prompt: {prompt[:200]}...")
print("\n" + "="*50)

# Try the request and capture full error
headers = {"Content-Type": "application/json"}

messages = [
    {"role": "user", "content": prompt}
]

payload = {
    "model": "Qwen/Qwen2.5-3B-Instruct",
    "messages": messages,
    "temperature": 0.3,
    "top_p": 0.95,
    "max_tokens": 3000,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0    
}

try:
    response = requests.post("http://localhost:8000/v1/chat/completions", headers=headers, json=payload)
    print(f"Status code: {response.status_code}")
    
    if response.status_code == 400:
        print("\n400 Bad Request - Full error message:")
        print(response.text)
        
        # Try to parse error as JSON
        try:
            error_json = response.json()
            print("\nParsed error:")
            print(json.dumps(error_json, indent=2))
        except:
            pass
    elif response.status_code == 200:
        print("\n200 OK - Success!")
        res = response.json()
        print(f"Response: {res['choices'][0]['message']['content'][:200]}...")
    else:
        print(f"\nUnexpected status: {response.status_code}")
        print(response.text[:500])
        
except Exception as e:
    print(f"Exception: {e}")

# Also check if there are any special characters
print("\n" + "="*50)
print("Checking for special characters in prompt...")
import unicodedata

# Check for non-ASCII characters
non_ascii = [c for c in prompt if ord(c) > 127]
if non_ascii:
    print(f"Found {len(non_ascii)} non-ASCII characters:")
    for c in non_ascii[:10]:  # Show first 10
        print(f"  '{c}' (U+{ord(c):04X} - {unicodedata.name(c, 'UNKNOWN')})")
else:
    print("No non-ASCII characters found")

# Check if prompt has null characters
if '\x00' in prompt:
    print("WARNING: Null characters found in prompt!")