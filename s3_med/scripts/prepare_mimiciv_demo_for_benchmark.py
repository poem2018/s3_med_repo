#!/usr/bin/env python3
"""
Prepare MIMIC-IV demo data for benchmark processing by:
1. Extracting compressed files
2. Renaming to uppercase as expected by benchmark scripts
"""

import os
import gzip
import shutil
from pathlib import Path

def prepare_mimiciv_demo(demo_path, output_path):
    """Prepare MIMIC-IV demo data for benchmark processing."""
    
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
    }
    
    for source_file, target_file in file_mappings.items():
        source_path = os.path.join(demo_path, source_file)
        target_path = os.path.join(output_path, target_file)
        
        print(f"Processing {source_file} -> {target_file}")
        
        # Check if uncompressed version exists
        uncompressed_dir = source_path.replace('.gz', '')
        if os.path.isdir(uncompressed_dir):
            # Use existing uncompressed file
            uncompressed_file = os.path.join(uncompressed_dir, os.path.basename(source_file).replace('.gz', ''))
            if os.path.exists(uncompressed_file):
                print(f"  Using existing uncompressed file: {uncompressed_file}")
                shutil.copy2(uncompressed_file, target_path)
                continue
        
        # Otherwise, decompress the gz file
        if os.path.exists(source_path):
            print(f"  Decompressing {source_path}")
            with gzip.open(source_path, 'rb') as f_in:
                with open(target_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        else:
            print(f"  Warning: {source_path} not found!")

if __name__ == "__main__":
    demo_path = "/scratch/bcew/ruikez2/intern/s3_med/data/mimiciv_demo"
    output_path = "/scratch/bcew/ruikez2/intern/s3_med/data/mimiciv_demo_benchmark"
    
    prepare_mimiciv_demo(demo_path, output_path)
    print(f"\nData prepared in: {output_path}")