#!/usr/bin/env python3
"""Test RAG service to understand correct API format"""

import requests
import json

endpoint = "http://127.0.0.1:3000/retrieve"

print("Testing RAG Service API\n" + "="*50)

# Test 1: Try with 'query' (current implementation)
print("\nTest 1: Using 'query' field")
payload1 = {
    "query": "ICU mortality risk factors",
    "topk": 3
}
print(f"Payload: {json.dumps(payload1, indent=2)}")

try:
    response = requests.post(endpoint, json=payload1, timeout=5)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print("✓ Success with 'query'")
        data = response.json()
        if data.get("results"):
            print(f"  Retrieved {len(data['results'])} documents")
    else:
        print(f"✗ Error: {response.text[:200]}")
except Exception as e:
    print(f"✗ Request failed: {e}")

# Test 2: Try with 'queries' (what error suggests)
print("\nTest 2: Using 'queries' field (as array)")
payload2 = {
    "queries": ["ICU mortality risk factors"],
    "topk": 3
}
print(f"Payload: {json.dumps(payload2, indent=2)}")

try:
    response = requests.post(endpoint, json=payload2, timeout=5)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print("✓ Success with 'queries' array")
        data = response.json()
        print(f"Response structure: {list(data.keys())}")
        if data.get("results"):
            print(f"  Retrieved {len(data['results'])} result sets")
            if isinstance(data["results"], list) and len(data["results"]) > 0:
                first_result = data["results"][0]
                if isinstance(first_result, list):
                    print(f"  First query returned {len(first_result)} documents")
                elif isinstance(first_result, dict):
                    print(f"  First result is a dict with keys: {list(first_result.keys())}")
    else:
        print(f"✗ Error: {response.text[:200]}")
except Exception as e:
    print(f"✗ Request failed: {e}")

# Test 3: Try with 'queries' as single string
print("\nTest 3: Using 'queries' field (as string)")
payload3 = {
    "queries": "ICU mortality risk factors",
    "topk": 3
}
print(f"Payload: {json.dumps(payload3, indent=2)}")

try:
    response = requests.post(endpoint, json=payload3, timeout=5)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print("✓ Success with 'queries' string")
        data = response.json()
        print(f"Response structure: {list(data.keys())}")
    else:
        print(f"✗ Error: {response.text[:200]}")
except Exception as e:
    print(f"✗ Request failed: {e}")

print("\n" + "="*50)
print("SUMMARY")
print("="*50)
print("\nBased on the tests above, update the retrieve_medical_knowledge function")
print("to use the correct API format that works with your RAG service.")