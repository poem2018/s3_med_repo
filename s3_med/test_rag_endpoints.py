#!/usr/bin/env python3
"""Test different RAG service endpoints"""

import requests
import json

base_url = "http://127.0.0.1:3000"

print("Testing RAG Service Endpoints\n" + "="*50)

# Test 1: Check root endpoint
print("\nTest 1: Root endpoint")
try:
    response = requests.get(base_url, timeout=5)
    print(f"GET {base_url}")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        try:
            data = response.json()
            print(f"Response: {json.dumps(data, indent=2)[:500]}")
        except:
            print(f"Response text: {response.text[:200]}")
except Exception as e:
    print(f"Failed: {e}")

# Test 2: Check /docs endpoint
print("\nTest 2: Documentation endpoint")
try:
    response = requests.get(f"{base_url}/docs", timeout=5)
    print(f"GET {base_url}/docs")
    print(f"Status: {response.status_code}")
except Exception as e:
    print(f"Failed: {e}")

# Test 3: Try /search endpoint (alternative to /retrieve)
print("\nTest 3: /search endpoint")
payload = {
    "queries": ["ICU mortality risk factors"],
    "topk": 3
}
try:
    response = requests.post(f"{base_url}/search", json=payload, timeout=10)
    print(f"POST {base_url}/search")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Success! Response structure: {list(data.keys())}")
    else:
        print(f"Error: {response.text[:200]}")
except Exception as e:
    print(f"Failed: {e}")

# Test 4: Try single query format on /retrieve
print("\nTest 4: /retrieve with single query (not array)")
payload = {
    "query": "ICU mortality risk factors",  # Single string, not array
    "topk": 3
}
try:
    response = requests.post(f"{base_url}/retrieve", json=payload, timeout=10)
    print(f"POST {base_url}/retrieve (single query)")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Success! Response structure: {list(data.keys())}")
    else:
        print(f"Error: {response.text[:200]}")
except Exception as e:
    print(f"Failed: {e}")

# Test 5: Check what methods are available
print("\nTest 5: OPTIONS request to /retrieve")
try:
    response = requests.options(f"{base_url}/retrieve", timeout=5)
    print(f"OPTIONS {base_url}/retrieve")
    print(f"Status: {response.status_code}")
    print(f"Allowed methods: {response.headers.get('Allow', 'Not specified')}")
except Exception as e:
    print(f"Failed: {e}")

print("\n" + "="*50)
print("RECOMMENDATIONS")
print("="*50)
print("\nBased on the server error, it seems the RAG service has a bug in handling")
print("batch queries. Options to fix:")
print("1. Fix the server code at /scratch/bcew/ruikez2/intern/s3_med/s3/search/retrieval_server.py:350")
print("2. Use a different endpoint if available")
print("3. Temporarily disable RAG retrieval until the server is fixed")