#!/usr/bin/env python3
"""
Process only a subset of data (e.g., test set) to avoid OOM issues.
This script processes data for specific patient IDs only.
"""

import os
import pandas as pd
import numpy as np
import json
from pathlib import Path
import gzip

def load_fixed_splits():
    """Load the fixed patient-based data splits."""
    splits_file = '/scratch/bcew/ruikez2/intern/s3_med/data/fixed_data_splits_patient.json'
    if not os.path.exists(splits_file):
        # Try old splits file for backwards compatibility
        old_splits_file = '/scratch/bcew/ruikez2/intern/s3_med/data/fixed_data_splits.json'
        if os.path.exists(old_splits_file):
            print(f"Warning: Using old stay-based splits. Run 00_create_fixed_splits.py to create patient-based splits.")
            with open(old_splits_file, 'r') as f:
                return json.load(f)
        raise FileNotFoundError(f"Fixed patient-based splits not found. Please run 00_create_fixed_splits.py first.")
    
    with open(splits_file, 'r') as f:
        return json.load(f)

def process_chartevents_subset(stay_ids, output_file, target_count=None):
    """Process chartevents for specific stay IDs only.
    
    Args:
        stay_ids: List of stay IDs to find
        output_file: Output CSV file path
        target_count: Stop after finding data for this many patients (default: find all)
    """
    
    if target_count:
        print(f"Processing chartevents for up to {target_count} out of {len(stay_ids)} stays...")
    else:
        print(f"Processing chartevents for {len(stay_ids)} stays...")
        target_count = len(stay_ids)  # Find all if not specified
    
    # Read chartevents in chunks and filter
    chartevents_path = "/scratch/bcew/ruikez2/intern/s3_med/data/mimiciv/3.1/icu/chartevents.csv.gz"
    
    if not os.path.exists(chartevents_path):
        print(f"Error: {chartevents_path} not found")
        return None
    
    # Convert stay_ids to set of integers for matching
    # The stay_ids from splits are strings, but chartevents has integers
    stay_ids_set = set(int(sid) for sid in stay_ids[:target_count])  # Only look for target_count patients
    found_stay_ids = set()  # Track which stay_ids we've found data for
    
    filtered_chunks = []
    chunk_count = 0
    total_rows = 0
    
    print("Reading and filtering chartevents.csv.gz in chunks...")
    
    # Read in chunks to avoid memory issues
    for chunk in pd.read_csv(chartevents_path, compression='gzip', chunksize=100000):
        chunk_count += 1
        
        # Filter for our stay IDs
        filtered = chunk[chunk['stay_id'].isin(stay_ids_set)]
        
        if len(filtered) > 0:
            filtered_chunks.append(filtered)
            total_rows += len(filtered)
            # Track which stay_ids we found
            found_stay_ids.update(filtered['stay_id'].unique())
        
        if chunk_count % 10 == 0:
            print(f"  Processed {chunk_count} chunks, found {total_rows} rows for {len(found_stay_ids)}/{target_count} patients...")
        
        # EARLY STOP: Stop when we found data for enough patients
        if len(found_stay_ids) >= target_count:
            print(f"  ✓ Found data for {len(found_stay_ids)} patients, stopping early at chunk {chunk_count}")
            break
        
        # Safety limit to prevent infinite processing
        if chunk_count >= 4000:  # ~3307 chunks total in file
            print(f"  Reached chunk limit ({chunk_count} chunks), found data for {len(found_stay_ids)} patients")
            break
    
    if filtered_chunks:
        print(f"Combining {len(filtered_chunks)} filtered chunks...")
        result = pd.concat(filtered_chunks, ignore_index=True)
        
        # Rename columns to match benchmark format
        column_mapping = {
            'subject_id': 'SUBJECT_ID',
            'hadm_id': 'HADM_ID',
            'stay_id': 'ICUSTAY_ID',
            'charttime': 'CHARTTIME',
            'itemid': 'ITEMID',
            'value': 'VALUE',
            'valueuom': 'VALUEUOM'
        }
        result.rename(columns=column_mapping, inplace=True)
        
        # Save to CSV
        print(f"Saving to {output_file}...")
        result.to_csv(output_file, index=False)
        print(f"  Saved {len(result)} rows for {result['ICUSTAY_ID'].nunique()} stays")
        
        return result
    else:
        print("No data found for specified stay IDs")
        print("  WARNING: No CHARTEVENTS.csv created - this will cause issues in subsequent steps")
        return None

def process_subset_pipeline(split_name='test', max_samples=None):
    """Process data for a specific split subset."""
    
    print(f"\n{'='*60}")
    print(f"Processing {split_name} split subset")
    print('='*60)
    
    # Load fixed splits
    splits = load_fixed_splits()
    
    # Get the stay IDs for this split
    stay_ids = splits[split_name]
    
    # Limit samples if specified
    if max_samples and len(stay_ids) > max_samples:
        stay_ids = stay_ids[:max_samples]
        print(f"Processing first {max_samples} out of {len(splits[split_name])} {split_name} stays")
    else:
        print(f"Processing all {len(stay_ids)} {split_name} stays")
    
    # Create output directory
    output_dir = Path(f'/scratch/bcew/ruikez2/intern/s3_med/data/baseline_text/subset_{split_name}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the subset IDs
    subset_info = {
        'split': split_name,
        'total_stays': len(stay_ids),
        'stay_ids': stay_ids
    }
    
    with open(output_dir / f'{split_name}_subset_info.json', 'w') as f:
        json.dump(subset_info, f, indent=2)
    
    # Process chartevents for this subset
    chartevents_output = output_dir / 'CHARTEVENTS.csv'
    # Use max_samples as target_count to stop early when we find enough data
    process_chartevents_subset(stay_ids, chartevents_output, target_count=max_samples)
    
    # Process other necessary files (smaller, can be loaded fully)
    print("\nProcessing other data files...")
    
    # ICUSTAYS
    icustays_path = "/scratch/bcew/ruikez2/intern/s3_med/data/mimiciv/3.1/icu/icustays.csv.gz"
    if os.path.exists(icustays_path):
        print("Processing ICUSTAYS...")
        icustays = pd.read_csv(icustays_path, compression='gzip')
        # Convert stay_ids to integers for matching
        stay_ids_int = [int(sid) for sid in stay_ids]
        icustays_filtered = icustays[icustays['stay_id'].isin(stay_ids_int)].copy()
        
        # Rename columns
        icustays_filtered.rename(columns={
            'subject_id': 'SUBJECT_ID',
            'hadm_id': 'HADM_ID',
            'stay_id': 'ICUSTAY_ID',
            'intime': 'INTIME',
            'outtime': 'OUTTIME'
        }, inplace=True)
        
        icustays_filtered.to_csv(output_dir / 'ICUSTAYS.csv', index=False)
        print(f"  Saved {len(icustays_filtered)} ICU stays")
        
        # Get unique subject and hadm IDs for filtering other tables
        subject_ids = icustays_filtered['SUBJECT_ID'].unique()
        hadm_ids = icustays_filtered['HADM_ID'].unique()
    
    # PATIENTS
    patients_path = "/scratch/bcew/ruikez2/intern/s3_med/data/mimiciv/3.1/hosp/patients.csv.gz"
    if os.path.exists(patients_path):
        print("Processing PATIENTS...")
        patients = pd.read_csv(patients_path, compression='gzip')
        patients_filtered = patients[patients['subject_id'].isin(subject_ids)].copy()
        
        patients_filtered.rename(columns={
            'subject_id': 'SUBJECT_ID',
            'gender': 'GENDER',
            'anchor_age': 'ANCHOR_AGE',  # Keep as ANCHOR_AGE for benchmark
            'anchor_year': 'ANCHOR_YEAR',
            'anchor_year_group': 'ANCHOR_YEAR_GROUP',
            'dod': 'DOD'
        }, inplace=True)
        
        patients_filtered.to_csv(output_dir / 'PATIENTS.csv', index=False)
        print(f"  Saved {len(patients_filtered)} patients")
    
    # ADMISSIONS
    admissions_path = "/scratch/bcew/ruikez2/intern/s3_med/data/mimiciv/3.1/hosp/admissions.csv.gz"
    if os.path.exists(admissions_path):
        print("Processing ADMISSIONS...")
        admissions = pd.read_csv(admissions_path, compression='gzip')
        admissions_filtered = admissions[admissions['hadm_id'].isin(hadm_ids)].copy()
        
        admissions_filtered.rename(columns={
            'subject_id': 'SUBJECT_ID',
            'hadm_id': 'HADM_ID',
            'admittime': 'ADMITTIME',
            'dischtime': 'DISCHTIME',
            'deathtime': 'DEATHTIME',
            'race': 'ETHNICITY'
        }, inplace=True)
        
        admissions_filtered.to_csv(output_dir / 'ADMISSIONS.csv', index=False)
        print(f"  Saved {len(admissions_filtered)} admissions")
    
    # D_ITEMS (needed for chartevents labels)
    d_items_path = "/scratch/bcew/ruikez2/intern/s3_med/data/mimiciv/3.1/icu/d_items.csv.gz"
    if os.path.exists(d_items_path):
        print("Processing D_ITEMS...")
        d_items = pd.read_csv(d_items_path, compression='gzip')
        d_items.rename(columns={
            'itemid': 'ITEMID',
            'label': 'LABEL',
            'abbreviation': 'ABBREVIATION',
            'linksto': 'LINKSTO',
            'category': 'CATEGORY',
            'unitname': 'UNITNAME'
        }, inplace=True)
        d_items.to_csv(output_dir / 'D_ITEMS.csv', index=False)
        print(f"  Saved {len(d_items)} item definitions")
    
    # D_LABITEMS (needed for lab events)
    d_labitems_path = "/scratch/bcew/ruikez2/intern/s3_med/data/mimiciv/3.1/hosp/d_labitems.csv.gz"
    if os.path.exists(d_labitems_path):
        print("Processing D_LABITEMS...")
        d_labitems = pd.read_csv(d_labitems_path, compression='gzip')
        d_labitems.rename(columns={
            'itemid': 'ITEMID',
            'label': 'LABEL'
        }, inplace=True)
        d_labitems.to_csv(output_dir / 'D_LABITEMS.csv', index=False)
        print(f"  Saved {len(d_labitems)} lab item definitions")
    
    # D_ICD_DIAGNOSES (needed for ICD code definitions)
    d_icd_path = "/scratch/bcew/ruikez2/intern/s3_med/data/mimiciv/3.1/hosp/d_icd_diagnoses.csv.gz"
    if os.path.exists(d_icd_path):
        print("Processing D_ICD_DIAGNOSES...")
        d_icd = pd.read_csv(d_icd_path, compression='gzip')
        # Filter to ICD9 codes only (as expected by benchmark)
        d_icd = d_icd[d_icd['icd_version'] == 9].copy()
        d_icd.rename(columns={
            'icd_code': 'ICD9_CODE',
            'icd_version': 'ICD_VERSION',
            'long_title': 'LONG_TITLE'
        }, inplace=True)
        d_icd.to_csv(output_dir / 'D_ICD_DIAGNOSES.csv', index=False)
        print(f"  Saved {len(d_icd)} ICD-9 diagnosis definitions")
    
    # DIAGNOSES_ICD (needed for patient diagnoses)
    diagnoses_path = "/scratch/bcew/ruikez2/intern/s3_med/data/mimiciv/3.1/hosp/diagnoses_icd.csv.gz"
    if os.path.exists(diagnoses_path):
        print("Processing DIAGNOSES_ICD...")
        diagnoses = pd.read_csv(diagnoses_path, compression='gzip')
        # Filter to our subjects and ICD9 only
        diagnoses_filtered = diagnoses[
            (diagnoses['subject_id'].isin(subject_ids)) & 
            (diagnoses['icd_version'] == 9)
        ].copy()
        diagnoses_filtered.rename(columns={
            'subject_id': 'SUBJECT_ID',
            'hadm_id': 'HADM_ID',
            'seq_num': 'SEQ_NUM',
            'icd_code': 'ICD9_CODE',
            'icd_version': 'ICD_VERSION'
        }, inplace=True)
        diagnoses_filtered.to_csv(output_dir / 'DIAGNOSES_ICD.csv', index=False)
        print(f"  Saved {len(diagnoses_filtered)} diagnoses for selected patients")
    
    # LABEVENTS (may be needed for some analyses)
    labevents_path = "/scratch/bcew/ruikez2/intern/s3_med/data/mimiciv/3.1/hosp/labevents.csv.gz"
    if os.path.exists(labevents_path):
        print("Processing LABEVENTS (this may take a while)...")
        # Process in chunks due to large size
        labevents_chunks = []
        chunk_count = 0
        for chunk in pd.read_csv(labevents_path, compression='gzip', chunksize=100000):
            # Filter for our subjects
            filtered = chunk[chunk['subject_id'].isin(subject_ids)]
            if len(filtered) > 0:
                labevents_chunks.append(filtered)
            chunk_count += 1
            if chunk_count % 10 == 0:
                print(f"    Processed {chunk_count} chunks...")
            # Process more chunks for LABEVENTS
            if chunk_count >= 500:  # Increased from 100 to 500
                print(f"    Stopping after {chunk_count} chunks for memory conservation")
                break
        
        if labevents_chunks:
            labevents_filtered = pd.concat(labevents_chunks, ignore_index=True)
            labevents_filtered.rename(columns={
                'subject_id': 'SUBJECT_ID',
                'hadm_id': 'HADM_ID',
                'itemid': 'ITEMID',
                'charttime': 'CHARTTIME',
                'value': 'VALUE',
                'valueuom': 'VALUEUOM'
            }, inplace=True)
            labevents_filtered.to_csv(output_dir / 'LABEVENTS.csv', index=False)
            print(f"  Saved {len(labevents_filtered)} lab events")
    
    # OUTPUTEVENTS (needed for urine output, etc.)
    outputevents_path = "/scratch/bcew/ruikez2/intern/s3_med/data/mimiciv/3.1/icu/outputevents.csv.gz"
    if os.path.exists(outputevents_path):
        print("Processing OUTPUTEVENTS...")
        outputevents = pd.read_csv(outputevents_path, compression='gzip')
        outputevents_filtered = outputevents[outputevents['stay_id'].isin(stay_ids_int)].copy()
        outputevents_filtered.rename(columns={
            'subject_id': 'SUBJECT_ID',
            'stay_id': 'ICUSTAY_ID',
            'hadm_id': 'HADM_ID',
            'charttime': 'CHARTTIME',
            'itemid': 'ITEMID',
            'value': 'VALUE'
        }, inplace=True)
        outputevents_filtered.to_csv(output_dir / 'OUTPUTEVENTS.csv', index=False)
        print(f"  Saved {len(outputevents_filtered)} output events")
    
    print(f"\n{'='*60}")
    print(f"Subset processing complete!")
    print(f"Output directory: {output_dir}")
    print('='*60)
    
    return output_dir

if __name__ == "__main__":
    import sys
    
    # Parse arguments
    split = sys.argv[1] if len(sys.argv) > 1 else 'test'
    max_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    
    # Set TEST_MODE for small sample
    if max_samples <= 100:
        os.environ['TEST_MODE'] = 'true'
    
    process_subset_pipeline(split, max_samples)