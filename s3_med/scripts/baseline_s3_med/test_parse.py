#!/usr/bin/env python3
"""Test parsing logic"""

def parse_prediction(response):
    """Parse the LLM response to extract prediction"""
    if not response:
        # If no response, use a random prediction based on base rate
        import random
        return random.choice([0, 0, 0, 0, 0, 0, 0, 1])  # ~12.5% mortality rate
    
    response_upper = response.upper()
    
    print(f"Response upper: {response_upper[:100]}")
    
    # Look for explicit prediction
    if "HIGH RISK" in response_upper or "HIGH-RISK" in response_upper:
        print("Found HIGH RISK -> returning 1")
        return 1
    elif "LOW RISK" in response_upper or "LOW-RISK" in response_upper:
        print("Found LOW RISK -> returning 0")
        return 0
    
    # Fallback: look for mortality/death keywords
    death_keywords = ["WILL DIE", "LIKELY TO DIE", "MORTALITY LIKELY", "POOR PROGNOSIS", "FATAL"]
    survival_keywords = ["WILL SURVIVE", "LIKELY TO SURVIVE", "GOOD PROGNOSIS", "FAVORABLE"]
    
    for keyword in death_keywords:
        if keyword in response_upper:
            print(f"Found death keyword: {keyword} -> returning 1")
            return 1
    
    for keyword in survival_keywords:
        if keyword in response_upper:
            print(f"Found survival keyword: {keyword} -> returning 0")
            return 0
    
    # Default to survival if uncertain
    print("No keywords found -> defaulting to 0")
    return 0

# Test with the actual response
response = """PREDICTION: LOW RISK
REASONING: The patient's Glasgow Coma Scale (GCS) scores remain relatively stable throughout the first 48 hours, with the highest score being 6 (in hours 1, 4, 8, 14, 16, 20, 24, ..."""

print(f"Testing response: {response[:50]}...")
result = parse_prediction(response)
print(f"Result: {result}")
print(f"Expected: 0 (LOW RISK)")

# Test with empty response
print("\n--- Testing empty response ---")
empty_result = parse_prediction("")
print(f"Empty result: {empty_result}")

# Test with truncated response that might have issues
print("\n--- Testing truncated response ---")
truncated = """[... truncated ...]

PREDICTION: LOW RISK
REASONING: Patient shows stable vitals"""
truncated_result = parse_prediction(truncated)
print(f"Truncated result: {truncated_result}")