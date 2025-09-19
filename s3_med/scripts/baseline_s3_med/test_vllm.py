#!/usr/bin/env python3
"""Test vLLM server with different prompt lengths"""

import requests
import json

def test_request(prompt_length="short"):
    url = 'http://localhost:8000/v1/chat/completions'
    headers = {'Content-Type': 'application/json'}
    
    if prompt_length == "short":
        content = "Is the patient at HIGH RISK or LOW RISK? Answer: The patient is at"
    elif prompt_length == "medium":
        content = "Patient data: " + "vital signs normal " * 100 + "\nIs this HIGH RISK or LOW RISK?"
    else:  # long
        content = "Patient data: " + "vital signs normal " * 500 + "\nIs this HIGH RISK or LOW RISK?"
    
    print(f"Testing {prompt_length} prompt (length: {len(content)} chars)...")
    
    payload = {
        'model': 'Qwen/Qwen2.5-3B-Instruct',
        'messages': [
            {'role': 'user', 'content': content}
        ],
        'temperature': 0.3,
        'max_tokens': 50
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        print(f'Status code: {response.status_code}')
        
        if response.status_code == 200:
            result = response.json()
            print(f'Success! Response: {result["choices"][0]["message"]["content"][:100]}')
        else:
            print(f'Error response: {response.text[:200]}')
            
    except requests.exceptions.Timeout:
        print("Request timed out!")
    except Exception as e:
        print(f'Exception: {e}')

# Test different lengths
print("=" * 50)
test_request("short")
print("\n" + "=" * 50)
test_request("medium")  
print("\n" + "=" * 50)
test_request("long")