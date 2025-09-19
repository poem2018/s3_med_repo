#!/usr/bin/env python3
"""
Step 5: Split dataset into train/val/test with 8:1:1 ratio.
Ensures balanced mortality rates across splits.
"""

import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

# Check for test mode
TEST_MODE = os.environ.get('TEST_MODE', 'false').lower() == 'true'
if TEST_MODE:
    print("🧪 Running in TEST MODE - will only process subset of data")

def split_dataset(input_file, output_dir):
    """Split dataset into train/val/test sets."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Extract patient IDs and mortality labels
    patient_ids = [p['patient_id'] for p in data]
    mortality_labels = [p['visit_info']['mortality'] for p in data]
    
    # Calculate statistics
    total_patients = len(data)
    mortality_count = sum(mortality_labels)
    mortality_rate = mortality_count / total_patients * 100
    
    print(f"Total patients: {total_patients}")
    print(f"Mortality cases: {mortality_count} ({mortality_rate:.2f}%)")
    
    # First split: 80% train+val, 20% test
    indices = list(range(len(data)))
    
    # In test mode, use simpler split due to small sample size
    if TEST_MODE and len(data) < 20:
        # For very small datasets, just split by index
        n = len(data)
        train_size = int(n * 0.6)  # 60% train
        val_size = int(n * 0.2)    # 20% val
        # Rest is test
        
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size+val_size]
        test_idx = indices[train_size+val_size:]
        print(f"TEST MODE: Simple split - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    else:
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=0.1,
            stratify=mortality_labels if sum(mortality_labels) > 0 else None,
            random_state=42
        )
    
        # Get train+val labels for second split
        train_val_labels = [mortality_labels[i] for i in train_val_idx]
        
        # Second split: From train+val, take 11.11% for val (which gives us 10% of total)
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=0.1111,  # 10% of total = 1/9 of 90%
            stratify=train_val_labels if sum(train_val_labels) > 0 else None,
            random_state=42
        )
    
    # Create splits
    train_data = [data[i] for i in train_idx]
    val_data = [data[i] for i in val_idx]
    test_data = [data[i] for i in test_idx]
    
    # Calculate mortality rates for each split
    train_mortality = sum(p['visit_info']['mortality'] for p in train_data)
    val_mortality = sum(p['visit_info']['mortality'] for p in val_data)
    test_mortality = sum(p['visit_info']['mortality'] for p in test_data)
    
    print(f"\nTrain set: {len(train_data)} patients, {train_mortality} deaths ({train_mortality/len(train_data)*100:.2f}%)")
    print(f"Val set: {len(val_data)} patients, {val_mortality} deaths ({val_mortality/len(val_data)*100:.2f}%)")
    print(f"Test set: {len(test_data)} patients, {test_mortality} deaths ({test_mortality/len(test_data)*100:.2f}%)")
    
    # Verify split ratio
    print(f"\nSplit ratio: {len(train_data)/total_patients:.1%}:{len(val_data)/total_patients:.1%}:{len(test_data)/total_patients:.1%}")
    
    # Save splits
    with open(output_dir / 'train_data.json', 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(output_dir / 'val_data.json', 'w') as f:
        json.dump(val_data, f, indent=2)
    
    with open(output_dir / 'test_data.json', 'w') as f:
        json.dump(test_data, f, indent=2)
    
    # Also save just the patient IDs for easy reference
    splits = {
        'train': [p['patient_id'] for p in train_data],
        'val': [p['patient_id'] for p in val_data],
        'test': [p['patient_id'] for p in test_data],
        'statistics': {
            'total_patients': total_patients,
            'train_size': len(train_data),
            'val_size': len(val_data),
            'test_size': len(test_data),
            'train_mortality': train_mortality,
            'val_mortality': val_mortality,
            'test_mortality': test_mortality
        }
    }
    
    with open(output_dir / 'data_splits.json', 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"\nData splits saved to: {output_dir}")

if __name__ == '__main__':
    suffix = '_test' if TEST_MODE else '_full'
    input_file = f'/scratch/bcew/ruikez2/intern/s3_med/data/patient_data_with_temporal{suffix}.json'
    output_dir = f'/scratch/bcew/ruikez2/intern/s3_med/data/splits{suffix}'
    
    split_dataset(input_file, output_dir)