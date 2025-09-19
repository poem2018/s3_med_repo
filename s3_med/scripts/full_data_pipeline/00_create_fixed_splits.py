#!/usr/bin/env python3
"""
Step 0: Create fixed train/val/test splits based on PATIENT IDs (not ICU stays).
This ensures consistent splits across multiple runs.
Only needs to be run once to generate the split file.
Note: A patient may have multiple ICU stays, and all stays from the same patient
will be in the same split (train/val/test).
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import json
from sklearn.model_selection import train_test_split

def create_fixed_splits():
    """Create fixed train/val/test splits based on patient IDs."""
    
    print("Creating fixed data splits (PATIENT-BASED)...")
    
    # Check if splits already exist
    splits_file = '/scratch/bcew/ruikez2/intern/s3_med/data/fixed_data_splits_patient.json'
    if os.path.exists(splits_file):
        print(f"Fixed patient-based splits already exist at: {splits_file}")
        with open(splits_file, 'r') as f:
            splits = json.load(f)
        print(f"  Train: {len(splits['train_patients'])} patients with {len(splits['train'])} ICU stays")
        print(f"  Val: {len(splits['val_patients'])} patients with {len(splits['val'])} ICU stays")
        print(f"  Test: {len(splits['test_patients'])} patients with {len(splits['test'])} ICU stays")
        return splits
    
    # Read ICUSTAYS to get all patient IDs and their stays
    icustays_path = "/scratch/bcew/ruikez2/intern/s3_med/data/mimiciv_full_benchmark/ICUSTAYS.csv"
    
    if not os.path.exists(icustays_path):
        # Try to read from compressed file if benchmark file doesn't exist
        icustays_gz = "/scratch/bcew/ruikez2/intern/s3_med/data/mimiciv/3.1/icu/icustays.csv.gz"
        if os.path.exists(icustays_gz):
            print("Reading from compressed icustays.csv.gz...")
            icustays = pd.read_csv(icustays_gz, compression='gzip', usecols=['subject_id', 'stay_id'])
            icustays.columns = ['SUBJECT_ID', 'ICUSTAY_ID']
        else:
            print(f"Error: Neither {icustays_path} nor {icustays_gz} found.")
            print("Please run 01_prepare_full_mimiciv.py first or check data path.")
            return None
    else:
        print("Reading ICUSTAYS.csv...")
        icustays = pd.read_csv(icustays_path, usecols=['SUBJECT_ID', 'ICUSTAY_ID'])
    
    # Get unique patients and their stays
    patient_stays = icustays[['SUBJECT_ID', 'ICUSTAY_ID']].drop_duplicates()
    
    # Get unique patient IDs
    unique_patients = patient_stays['SUBJECT_ID'].unique()
    unique_patients = [str(p) for p in unique_patients]  # Convert to string for consistency
    
    print(f"Total unique patients: {len(unique_patients)}")
    print(f"Total ICU stays: {len(patient_stays)}")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Split patients (not stays) into 90% train, 5% val, 5% test
    # First split: 90% train, 10% val+test
    train_patients, val_test_patients = train_test_split(
        unique_patients,
        test_size=0.1,  # 10% for val+test
        random_state=42
    )
    
    # Second split: From val+test, split 50/50 for val and test (5% each of total)
    val_patients, test_patients = train_test_split(
        val_test_patients,
        test_size=0.5,  # 50% of 10% = 5% of total
        random_state=42
    )
    
    # Now map patients to their ICU stays
    train_patients_set = set([int(p) for p in train_patients])
    val_patients_set = set([int(p) for p in val_patients])
    test_patients_set = set([int(p) for p in test_patients])
    
    train_stays = []
    val_stays = []
    test_stays = []
    
    for _, row in patient_stays.iterrows():
        patient_id = row['SUBJECT_ID']
        stay_id = str(row['ICUSTAY_ID'])
        
        if patient_id in train_patients_set:
            train_stays.append(stay_id)
        elif patient_id in val_patients_set:
            val_stays.append(stay_id)
        elif patient_id in test_patients_set:
            test_stays.append(stay_id)
    
    # Create splits dictionary
    splits = {
        'train': train_stays,
        'val': val_stays,
        'test': test_stays,
        'train_patients': train_patients,
        'val_patients': val_patients,
        'test_patients': test_patients,
        'total_patients': len(unique_patients),
        'total_stays': len(patient_stays),
        'train_patient_count': len(train_patients),
        'val_patient_count': len(val_patients),
        'test_patient_count': len(test_patients),
        'train_size': len(train_stays),
        'val_size': len(val_stays),
        'test_size': len(test_stays),
        'patient_split_ratio': f"{len(train_patients)/len(unique_patients):.1%}:{len(val_patients)/len(unique_patients):.1%}:{len(test_patients)/len(unique_patients):.1%}",
        'stay_split_ratio': f"{len(train_stays)/len(patient_stays):.1%}:{len(val_stays)/len(patient_stays):.1%}:{len(test_stays)/len(patient_stays):.1%}",
        'random_seed': 42
    }
    
    # Save splits
    with open(splits_file, 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"\nFixed PATIENT-BASED splits created and saved to: {splits_file}")
    print(f"  Total unique patients: {splits['total_patients']}")
    print(f"  Total ICU stays: {splits['total_stays']}")
    print(f"\nPatient split:")
    print(f"  Train: {splits['train_patient_count']} patients ({len(train_patients)/len(unique_patients):.1%})")
    print(f"  Val: {splits['val_patient_count']} patients ({len(val_patients)/len(unique_patients):.1%})")
    print(f"  Test: {splits['test_patient_count']} patients ({len(test_patients)/len(unique_patients):.1%})")
    print(f"\nICU stay split:")
    print(f"  Train: {splits['train_size']} stays ({len(train_stays)/len(patient_stays):.1%})")
    print(f"  Val: {splits['val_size']} stays ({len(val_stays)/len(patient_stays):.1%})")
    print(f"  Test: {splits['test_size']} stays ({len(test_stays)/len(patient_stays):.1%})")
    print(f"\n  Random seed: {splits['random_seed']}")
    
    return splits

def get_split_subset(split_name='test', max_samples=100):
    """Get a subset of a specific split for testing (returns ICU stay IDs)."""
    
    splits_file = '/scratch/bcew/ruikez2/intern/s3_med/data/fixed_data_splits_patient.json'
    
    if not os.path.exists(splits_file):
        print("Fixed patient-based splits not found. Creating them now...")
        splits = create_fixed_splits()
        if splits is None:
            return None
    else:
        with open(splits_file, 'r') as f:
            splits = json.load(f)
    
    if split_name not in ['train', 'val', 'test']:
        print(f"Invalid split name: {split_name}. Must be 'train', 'val', or 'test'.")
        return None
    
    # Get the ICU stays for the specified split
    split_ids = splits[split_name]
    
    # Take subset if requested
    if max_samples and len(split_ids) > max_samples:
        subset_ids = split_ids[:max_samples]
        print(f"Using first {max_samples} {split_name} ICU stays out of {len(split_ids)}")
        # Count unique patients in subset
        if split_name + '_patients' in splits:
            patient_stays = pd.read_csv("/scratch/bcew/ruikez2/intern/s3_med/data/mimiciv_full_benchmark/ICUSTAYS.csv", 
                                       usecols=['SUBJECT_ID', 'ICUSTAY_ID'])
            subset_stays_int = [int(sid) for sid in subset_ids]
            subset_patients = patient_stays[patient_stays['ICUSTAY_ID'].isin(subset_stays_int)]['SUBJECT_ID'].nunique()
            print(f"  This includes approximately {subset_patients} unique patients")
    else:
        subset_ids = split_ids
        print(f"Using all {len(split_ids)} {split_name} ICU stays")
    
    return subset_ids

if __name__ == "__main__":
    # Create fixed splits
    splits = create_fixed_splits()
    
    if splits:
        print("\n" + "="*60)
        print("You can now process specific splits using these fixed patient lists.")
        print("The splits will remain consistent across multiple runs.")
        print("="*60)