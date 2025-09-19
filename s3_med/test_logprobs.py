#!/usr/bin/env python3
"""Test script to debug logprobs API issues"""

import requests
import json
import sys

def test_basic_request():
    """Test basic request without logprobs"""
    print("="*60)
    print("Test 1: Basic request WITHOUT logprobs")
    print("="*60)

    payload = {
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "messages": [{"role": "user", "content": "Answer with only one word: HIGH or LOW"}],
        "temperature": 0.1,
        "max_tokens": 10
    }

    try:
        response = requests.post(
            "http://localhost:8000/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=10
        )

        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("✓ Success! Basic request works")
            print(f"Response text: {result['choices'][0]['message']['content']}")
            return True
        else:
            print(f"✗ Error response: {response.text[:500]}")
            return False

    except Exception as e:
        print(f"✗ Request failed: {e}")
        return False

def test_logprobs_request():
    """Test request with logprobs enabled"""
    print("\n" + "="*60)
    print("Test 2: Request WITH logprobs enabled")
    print("="*60)

    payload = {
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "messages": [{"role": "user", "content": "Answer with only one word: HIGH or LOW"}],
        "temperature": 0.1,
        "max_tokens": 10,
        "logprobs": True,
        "top_logprobs": 5
    }

    print("Payload being sent:")
    print(json.dumps(payload, indent=2))
    print()

    try:
        response = requests.post(
            "http://localhost:8000/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=10
        )

        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("✓ Success! Logprobs request works")
            print(f"Response text: {result['choices'][0]['message']['content']}")

            # Check if logprobs are in the response
            if 'logprobs' in result['choices'][0]:
                print("✓ Logprobs are included in response!")
                logprobs = result['choices'][0]['logprobs']
                if 'content' in logprobs and logprobs['content']:
                    print(f"  Number of tokens: {len(logprobs['content'])}")
                    # Show first token's logprobs
                    first_token = logprobs['content'][0]
                    print(f"  First token: '{first_token.get('token', 'N/A')}'")
                    print(f"  First token logprob: {first_token.get('logprob', 'N/A')}")
                    if 'top_logprobs' in first_token:
                        print("  Top alternatives for first token:")
                        for alt in first_token['top_logprobs'][:3]:
                            print(f"    - '{alt['token']}': {alt['logprob']}")
            else:
                print("⚠ WARNING: Logprobs NOT in response despite being requested")
                print("  This means the server doesn't support logprobs or needs special configuration")
            return True
        else:
            print(f"✗ Error {response.status_code}: {response.reason}")
            try:
                error_data = response.json()
                print("Error details:")
                print(json.dumps(error_data, indent=2))
            except:
                print(f"Response text: {response.text[:500]}")
            return False

    except Exception as e:
        print(f"✗ Request failed: {e}")
        return False

def test_server_info():
    """Check server information"""
    print("\n" + "="*60)
    print("Test 3: Server Information")
    print("="*60)

    try:
        # Try to get models list
        response = requests.get("http://localhost:8000/v1/models", timeout=5)
        print(f"Models endpoint status: {response.status_code}")

        if response.status_code == 200:
            models = response.json()
            print("Available models:")
            if 'data' in models:
                for model in models['data']:
                    print(f"  - {model.get('id', 'unknown')}")
            else:
                print(json.dumps(models, indent=2)[:500])

        # Try to get server health/info
        print("\nTrying server root endpoint...")
        response = requests.get("http://localhost:8000/", timeout=5)
        print(f"Root endpoint status: {response.status_code}")

    except Exception as e:
        print(f"✗ Failed to get server info: {e}")

def main():
    print("\n🔍 VLLM LOGPROBS CAPABILITY TEST\n")

    # Run tests
    basic_works = test_basic_request()
    logprobs_works = test_logprobs_request()
    test_server_info()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    if not basic_works:
        print("❌ Basic requests failing - check if vLLM server is running on port 8000")
        print("   Start vLLM with: python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-3B-Instruct --port 8000")
    elif not logprobs_works:
        print("⚠️  Basic requests work but logprobs requests fail")
        print("   This likely means vLLM needs to be started with logprobs support:")
        print("   python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-3B-Instruct --port 8000 --enable-logprobs")
    else:
        print("✅ Everything works! Both basic and logprobs requests succeed")

    print("\nNote: If logprobs aren't supported, you may need to:")
    print("1. Update vLLM to a newer version")
    print("2. Start vLLM with --enable-logprobs flag")
    print("3. Or use a different confidence calculation method")

if __name__ == "__main__":
    main()