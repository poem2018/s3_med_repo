#!/usr/bin/env python3
"""Test different message formats for vLLM"""

import requests
import json

def test_with_system(prompt):
    """Test with system message"""
    headers = {"Content-Type": "application/json"}
    messages = [
        {"role": "system", "content": "You are a helpful medical assistant."},
        {"role": "user", "content": prompt}
    ]
    
    payload = {
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 50
    }
    
    try:
        response = requests.post("http://localhost:8000/v1/chat/completions", headers=headers, json=payload, timeout=10)
        print(f"With system - Status: {response.status_code}")
        if response.status_code == 200:
            res = response.json()
            print(f"Response: {res['choices'][0]['message']['content'][:100]}")
        else:
            print(f"Error: {response.text[:200]}")
        return response.status_code == 200
    except Exception as e:
        print(f"Exception with system: {e}")
        return False

def test_without_system(prompt):
    """Test without system message"""
    headers = {"Content-Type": "application/json"}
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    payload = {
        "model": "Qwen/Qwen2.5-3B-Instruct", 
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 50
    }
    
    try:
        response = requests.post("http://localhost:8000/v1/chat/completions", headers=headers, json=payload, timeout=10)
        print(f"Without system - Status: {response.status_code}")
        if response.status_code == 200:
            res = response.json()
            print(f"Response: {res['choices'][0]['message']['content'][:100]}")
        else:
            print(f"Error: {response.text[:200]}")
        return response.status_code == 200
    except Exception as e:
        print(f"Exception without system: {e}")
        return False

# Test both formats
prompt = "Based on the patient data, predict HIGH RISK or LOW RISK: The patient is"

print("=" * 50)
print("Testing WITHOUT system message:")
without_success = test_without_system(prompt)

print("\n" + "=" * 50)
print("Testing WITH system message:")
with_success = test_with_system(prompt)

print("\n" + "=" * 50)
print(f"Results:")
print(f"Without system message: {'SUCCESS' if without_success else 'FAILED'}")
print(f"With system message: {'SUCCESS' if with_success else 'FAILED'}")