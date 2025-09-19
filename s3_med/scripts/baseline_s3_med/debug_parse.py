#!/usr/bin/env python3
"""Debug parsing issue"""

def parse_prediction(response):
    """Parse the LLM response to extract prediction"""
    if not response:
        # If no response, use a random prediction based on base rate
        import random
        return random.choice([0, 0, 0, 0, 0, 0, 0, 1])  # ~12.5% mortality rate
    
    response_upper = response.upper()
    
    print(f"Response length: {len(response)}")
    print(f"First 200 chars of response_upper: {response_upper[:200]}")
    
    # Look for explicit prediction
    if "HIGH RISK" in response_upper or "HIGH-RISK" in response_upper:
        print("Found HIGH RISK")
        return 1
    elif "LOW RISK" in response_upper or "LOW-RISK" in response_upper:
        print("Found LOW RISK")
        return 0
    
    # Fallback: look for mortality/death keywords
    death_keywords = ["WILL DIE", "LIKELY TO DIE", "MORTALITY LIKELY", "POOR PROGNOSIS", "FATAL"]
    survival_keywords = ["WILL SURVIVE", "LIKELY TO SURVIVE", "GOOD PROGNOSIS", "FAVORABLE"]
    
    for keyword in death_keywords:
        if keyword in response_upper:
            print(f"Found death keyword: {keyword}")
            return 1
    
    for keyword in survival_keywords:
        if keyword in response_upper:
            print(f"Found survival keyword: {keyword}")
            return 0
    
    # Default to survival if uncertain
    print("No keywords found, defaulting to 0")
    return 0

# Test with the problematic response
response = """PREDICTION: LOW RISK
REASONING: The patient's vital signs are generally stable with no significant drops in blood pressure or heart rate. The Glasgow Coma Scale scores remain consistent, indicating that the patient's level of consciousness is relatively stable. The patient's blood glucose levels are within the normal range, and there are no signs of acute organ failure or severe complications. The patient's oxygen saturation and respiratory rate are also within normal limits. Given these observa"""

print("Testing problematic response:")
print("=" * 50)
result = parse_prediction(response)
print(f"Final result: {result}")
print(f"Expected: 0 (LOW RISK)")

# Check if there's something weird with the response
print("\n" + "=" * 50)
print("Checking for hidden characters:")
print(f"Response bytes: {response.encode()[:100]}")
print(f"'LOW RISK' in response.upper(): {'LOW RISK' in response.upper()}")
print(f"'HIGH RISK' in response.upper(): {'HIGH RISK' in response.upper()}")