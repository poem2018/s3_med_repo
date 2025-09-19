#!/usr/bin/env python3
"""
Test script to verify mortality prediction reward computation
"""

import sys
sys.path.append('/scratch/bcew/ruikez2/intern/s3_med')

from verl.utils.reward_score.rag_2 import compute_score_rag, extract_mortality_prediction

# Test cases
test_cases = [
    {
        "name": "Clear Yes prediction",
        "solution": "After searching... <search_complete>true</search_complete>\nBased on the patient data, Yes, this patient will likely die in ICU.",
        "ground_truth": {"mortality": 1, "patient_id": "test_001"},
        "expected_score": 1.0
    },
    {
        "name": "Clear No prediction",
        "solution": "After analysis... <search_complete>true</search_complete>\nNo, this patient has low mortality risk.",
        "ground_truth": {"mortality": 0, "patient_id": "test_002"},
        "expected_score": 1.0
    },
    {
        "name": "Wrong prediction",
        "solution": "After review... <search_complete>true</search_complete>\nYes, high mortality risk.",
        "ground_truth": {"mortality": 0, "patient_id": "test_003"},
        "expected_score": 0.0
    },
    {
        "name": "Unclear prediction",
        "solution": "After searching... <search_complete>true</search_complete>\nThe patient's condition is complex and uncertain.",
        "ground_truth": {"mortality": 1, "patient_id": "test_004"},
        "expected_score": -0.5
    }
]

print("Testing mortality prediction reward computation...")
print("=" * 60)

for test in test_cases:
    print(f"\nTest: {test['name']}")
    print(f"Ground truth mortality: {test['ground_truth']['mortality']}")
    
    # Test extraction function
    prediction = extract_mortality_prediction(test['solution'])
    print(f"Extracted prediction: {prediction}")
    
    # Test reward computation
    score, _, _ = compute_score_rag(
        solution_str=test['solution'],
        ground_truth=test['ground_truth'],
        zeroshot_answers={},
        data_source='mimic_icu_mortality'
    )
    
    print(f"Computed score: {score}")
    print(f"Expected score: {test['expected_score']}")
    
    if score == test['expected_score']:
        print("✓ PASSED")
    else:
        print("✗ FAILED")

print("\n" + "=" * 60)
print("Testing complete!")