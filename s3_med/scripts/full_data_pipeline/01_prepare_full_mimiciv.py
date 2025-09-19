#!/usr/bin/env python3
"""
Step 1: Prepare full MIMIC-IV data for benchmark processing by:
1. Extracting compressed files
2. Renaming to uppercase as expected by benchmark scripts
"""

import os
import gzip
import shutil
from pathlib import Path

# Check for test mode
TEST_MODE = os.environ.get('TEST_MODE', 'false').lower() == 'true'
if TEST_MODE:
    print("🧪 Running in TEST MODE - will only process subset of data")

def prepare_mimiciv_full(data_path, output_path):
    """Prepare full MIMIC-IV data for benchmark processing."""
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Define required files and their mappings
    file_mappings = {
        # Hospital files
        'hosp/patients.csv.gz': 'PATIENTS.csv',
        'hosp/admissions.csv.gz': 'ADMISSIONS.csv',
        'hosp/diagnoses_icd.csv.gz': 'DIAGNOSES_ICD.csv',
        'hosp/d_icd_diagnoses.csv.gz': 'D_ICD_DIAGNOSES.csv',
        'hosp/labevents.csv.gz': 'LABEVENTS.csv',
        'hosp/d_labitems.csv.gz': 'D_LABITEMS.csv',
        
        # ICU files
        'icu/icustays.csv.gz': 'ICUSTAYS.csv',
        'icu/chartevents.csv.gz': 'CHARTEVENTS.csv',
        'icu/outputevents.csv.gz': 'OUTPUTEVENTS.csv',
        'icu/d_items.csv.gz': 'D_ITEMS.csv',
        'icu/inputevents.csv.gz': 'INPUTEVENTS.csv',
        'icu/procedureevents.csv.gz': 'PROCEDUREEVENTS.csv',
    }
    
    for source_file, target_file in file_mappings.items():
        source_path = os.path.join(data_path, source_file)
        target_path = os.path.join(output_path, target_file)
        
        # Skip if already exists
        if os.path.exists(target_path):
            print(f"Skipping {target_file} (already exists)")
            continue
            
        print(f"Processing {source_file} -> {target_file}")
        
        # Decompress the gz file
        if os.path.exists(source_path):
            print(f"  Decompressing {source_path}")
            with gzip.open(source_path, 'rb') as f_in:
                with open(target_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print(f"  Done: {target_file}")
        else:
            print(f"  Warning: {source_path} not found!")

if __name__ == "__main__":
    data_path = "/scratch/bcew/ruikez2/intern/s3_med/data/mimiciv/3.1"
    output_path = "/scratch/bcew/ruikez2/intern/s3_med/data/mimiciv_full_benchmark"
    
    prepare_mimiciv_full(data_path, output_path)
    print(f"\nData prepared in: {output_path}")