#!/usr/bin/env python3
"""
Adapt MIMIC-IV demo data columns to match what benchmark expects.
"""

import pandas as pd
import os

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
            'race': 'ETHNICITY'  # Map race to ETHNICITY
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
            df = pd.read_csv(input_file, low_memory=False)
            
            # Rename columns
            df.rename(columns=mapping, inplace=True)
            
            # Add any missing required columns with default values
            if filename == 'CHARTEVENTS.csv' and 'ICUSTAY_ID' not in df.columns and 'stay_id' in df.columns:
                df['ICUSTAY_ID'] = df['stay_id']
            
            # For D_ICD_DIAGNOSES, filter to ICD9 codes only
            if filename == 'D_ICD_DIAGNOSES.csv' and 'ICD_VERSION' in df.columns:
                df = df[df['ICD_VERSION'] == 9].copy()
            
            # For DIAGNOSES_ICD, filter to ICD9 codes only
            if filename == 'DIAGNOSES_ICD.csv' and 'ICD_VERSION' in df.columns:
                df = df[df['ICD_VERSION'] == 9].copy()
                
            # Save with uppercase column names
            df.to_csv(output_file, index=False)
            print(f"  Saved to {output_file}")
        else:
            print(f"  Warning: {input_file} not found!")

if __name__ == "__main__":
    input_path = "/scratch/bcew/ruikez2/intern/s3_med/data/mimiciv_demo_benchmark"
    output_path = "/scratch/bcew/ruikez2/intern/s3_med/data/mimiciv_demo_adapted"
    
    adapt_csv_columns(input_path, output_path)
    print(f"\nAdapted data saved to: {output_path}")