#!/usr/bin/env python3
"""
Step 2: Adapt full MIMIC-IV data columns to match what benchmark expects.
"""

import pandas as pd
import os
from pathlib import Path

# Check for test mode
TEST_MODE = os.environ.get('TEST_MODE', 'false').lower() == 'true'
if TEST_MODE:
    print("🧪 Running in TEST MODE - will only process subset of data")

def adapt_csv_columns(input_path, output_path):
    """Convert MIMIC-IV column names to match benchmark expectations."""
    
    # Define column mappings for each file
    column_mappings = {
        'ADMISSIONS.csv': {
            'subject_id': 'SUBJECT_ID',
            'hadm_id': 'HADM_ID',
            'admittime': 'ADMITTIME',
            'dischtime': 'DISCHTIME',
            'deathtime': 'DEATHTIME',
            'race': 'ETHNICITY'
        },
        'PATIENTS.csv': {
            'subject_id': 'SUBJECT_ID',
            'gender': 'GENDER',
            'anchor_age': 'ANCHOR_AGE',
            'anchor_year': 'ANCHOR_YEAR',
            'anchor_year_group': 'ANCHOR_YEAR_GROUP',
            'dod': 'DOD'
        },
        'ICUSTAYS.csv': {
            'subject_id': 'SUBJECT_ID',
            'hadm_id': 'HADM_ID',
            'stay_id': 'ICUSTAY_ID',
            'first_careunit': 'FIRST_CAREUNIT',
            'last_careunit': 'LAST_CAREUNIT',
            'intime': 'INTIME',
            'outtime': 'OUTTIME',
            'los': 'LOS'
        },
        'DIAGNOSES_ICD.csv': {
            'subject_id': 'SUBJECT_ID',
            'hadm_id': 'HADM_ID',
            'seq_num': 'SEQ_NUM',
            'icd_code': 'ICD9_CODE',
            'icd_version': 'ICD_VERSION'
        },
        'D_ICD_DIAGNOSES.csv': {
            'icd_code': 'ICD9_CODE',
            'icd_version': 'ICD_VERSION',
            'long_title': 'LONG_TITLE'
        },
        'CHARTEVENTS.csv': {
            'subject_id': 'SUBJECT_ID',
            'hadm_id': 'HADM_ID',
            'stay_id': 'ICUSTAY_ID',
            'charttime': 'CHARTTIME',
            'itemid': 'ITEMID',
            'value': 'VALUE',
            'valueuom': 'VALUEUOM',
            'warning': 'WARNING'
        },
        'LABEVENTS.csv': {
            'subject_id': 'SUBJECT_ID',
            'hadm_id': 'HADM_ID',
            'itemid': 'ITEMID',
            'charttime': 'CHARTTIME',
            'value': 'VALUE',
            'valueuom': 'VALUEUOM'
        },
        'OUTPUTEVENTS.csv': {
            'subject_id': 'SUBJECT_ID',
            'hadm_id': 'HADM_ID',
            'stay_id': 'ICUSTAY_ID',
            'charttime': 'CHARTTIME',
            'itemid': 'ITEMID',
            'value': 'VALUE',
            'valueuom': 'VALUEUOM'
        },
        'D_ITEMS.csv': {
            'itemid': 'ITEMID',
            'label': 'LABEL',
            'abbreviation': 'ABBREVIATION',
            'linksto': 'LINKSTO',
            'category': 'CATEGORY',
            'unitname': 'UNITNAME',
            'param_type': 'PARAM_TYPE',
            'lownormalvalue': 'LOWNORMALVALUE',
            'highnormalvalue': 'HIGHNORMALVALUE'
        },
        'D_LABITEMS.csv': {
            'itemid': 'ITEMID',
            'label': 'LABEL',
            'category': 'CATEGORY',
            'loinc_code': 'LOINC_CODE'
        }
    }
    
    os.makedirs(output_path, exist_ok=True)
    
    for filename, mapping in column_mappings.items():
        input_file = os.path.join(input_path, filename)
        output_file = os.path.join(output_path, filename)
        
        if os.path.exists(input_file):
            print(f"Processing {filename}...")
            
            # Read in chunks for large files
            if filename in ['CHARTEVENTS.csv', 'LABEVENTS.csv']:
                print(f"  Reading large file {filename} in chunks...")
                chunks = []
                chunk_count = 0
                max_chunks = 2 if TEST_MODE else None  # Limit chunks in test mode
                
                for chunk in pd.read_csv(input_file, chunksize=1000000, low_memory=False):
                    chunk_count += 1
                    print(f"    Processing chunk {chunk_count}...")
                    chunk.rename(columns=mapping, inplace=True)
                    
                    # Filter ICD9 codes if applicable
                    if filename == 'DIAGNOSES_ICD.csv' and 'ICD_VERSION' in chunk.columns:
                        chunk = chunk[chunk['ICD_VERSION'] == 9].copy()
                    
                    chunks.append(chunk)
                    
                    # In test mode, only process limited chunks
                    if TEST_MODE and max_chunks and chunk_count >= max_chunks:
                        print(f"    TEST MODE: Stopping after {max_chunks} chunks")
                        break
                
                df = pd.concat(chunks, ignore_index=True)
            else:
                df = pd.read_csv(input_file, low_memory=False)
                df.rename(columns=mapping, inplace=True)
                
                # In test mode, limit to first 1000 rows for small files
                if TEST_MODE and len(df) > 1000:
                    print(f"    TEST MODE: Limiting to first 1000 rows")
                    df = df.head(1000)
                
                # Filter ICD9 codes
                if filename == 'D_ICD_DIAGNOSES.csv' and 'ICD_VERSION' in df.columns:
                    df = df[df['ICD_VERSION'] == 9].copy()
                elif filename == 'DIAGNOSES_ICD.csv' and 'ICD_VERSION' in df.columns:
                    df = df[df['ICD_VERSION'] == 9].copy()
            
            # Save with uppercase column names
            df.to_csv(output_file, index=False)
            print(f"  Saved to {output_file}")
        else:
            print(f"  Warning: {input_file} not found!")

if __name__ == "__main__":
    input_path = "/scratch/bcew/ruikez2/intern/s3_med/data/mimiciv_full_benchmark"
    output_path = "/scratch/bcew/ruikez2/intern/s3_med/data/mimiciv_full_adapted"
    
    adapt_csv_columns(input_path, output_path)
    print(f"\nAdapted data saved to: {output_path}")