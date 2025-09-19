#!/usr/bin/env python3
"""
Balance the dataset to have approximately equal numbers of mortality=0 and mortality=1 cases
"""

import json
import random
import os
from pathlib import Path

def balance_dataset(input_dir, output_dir, random_seed=42):
    """
    Balance the dataset by undersampling the majority class
    """
    random.seed(random_seed)
    
    # Load the original dataset
    input_file = os.path.join(input_dir, 'all_patients.json')
    with open(input_file, 'r') as f:
        all_patients = json.load(f)
    
    # Separate patients by mortality label
    mortality_0 = [p for p in all_patients if p['mortality'] == 0]
    mortality_1 = [p for p in all_patients if p['mortality'] == 1]
    
    print(f"Original dataset:")
    print(f"  Total patients: {len(all_patients)}")
    print(f"  Mortality = 0: {len(mortality_0)}")
    print(f"  Mortality = 1: {len(mortality_1)}")
    
    # Balance by undersampling the majority class
    min_samples = min(len(mortality_0), len(mortality_1))
    
    # Randomly sample from the majority class
    if len(mortality_0) > len(mortality_1):
        sampled_mortality_0 = random.sample(mortality_0, min_samples)
        sampled_mortality_1 = mortality_1
    else:
        sampled_mortality_0 = mortality_0
        sampled_mortality_1 = random.sample(mortality_1, min_samples)
    
    # Combine and shuffle
    balanced_patients = sampled_mortality_0 + sampled_mortality_1
    random.shuffle(balanced_patients)
    
    print(f"\nBalanced dataset:")
    print(f"  Total patients: {len(balanced_patients)}")
    print(f"  Mortality = 0: {len(sampled_mortality_0)}")
    print(f"  Mortality = 1: {len(sampled_mortality_1)}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the balanced dataset
    output_file = os.path.join(output_dir, 'all_patients.json')
    with open(output_file, 'w') as f:
        json.dump(balanced_patients, f, indent=2)
    
    print(f"\nBalanced dataset saved to: {output_file}")
    
    # Also save a summary
    summary = {
        'original': {
            'total': len(all_patients),
            'mortality_0': len(mortality_0),
            'mortality_1': len(mortality_1)
        },
        'balanced': {
            'total': len(balanced_patients),
            'mortality_0': len(sampled_mortality_0),
            'mortality_1': len(sampled_mortality_1)
        },
        'random_seed': random_seed
    }
    
    summary_file = os.path.join(output_dir, 'balance_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to: {summary_file}")
    
    return balanced_patients

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Balance the dataset for equal mortality labels")
    parser.add_argument("--input_dir", default="/scratch/bcew/ruikez2/intern/s3_med/data/baseline_text_500",
                        help="Input directory containing original dataset")
    parser.add_argument("--output_dir", default="/scratch/bcew/ruikez2/intern/s3_med/data/baseline_text_500_balanced",
                        help="Output directory for balanced dataset")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    balance_dataset(args.input_dir, args.output_dir, args.random_seed)