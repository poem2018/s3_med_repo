#!/usr/bin/env python3
"""
Convert MIMIC-IV benchmark episode data to required JSON format.
Creates two files:
1. chartevents_matrices.json: List of 17×hours matrices (including ALL indicators)
2. patient_info.json: Patient demographics, past ICD/CCS codes, current visit time

This version includes pH and Glucose in the hourly matrix.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Define mappings for Glasgow Coma Scale text values to numeric codes
GCS_MAPPINGS = {
    'Glascow coma scale eye opening': {
        'Spontaneously': 4,
        'To Speech': 3,
        'To Pain': 2,
        'No response': 1
    },
    'Glascow coma scale motor response': {
        'Obeys Commands': 6,
        'Localizes Pain': 5,
        'Flex-withdraws': 4,
        'Abnormal Flexion': 3,
        'Abnormal extension': 2,
        'No response': 1
    },
    'Glascow coma scale verbal response': {
        'Oriented': 5,
        'Confused': 4,
        'Inappropriate Words': 3,
        'Incomprehensible sounds': 2,
        'No Response': 1,
        'No Response-ETT': 1
    }
}

def load_benchmark_ccs_mappings(benchmark_output):
    """Load CCS mappings from benchmark's diagnoses files."""
    icd_to_ccs = {}
    
    for subject_dir in os.listdir(benchmark_output):
        if subject_dir.isdigit():
            diagnoses_path = os.path.join(benchmark_output, subject_dir, 'diagnoses.csv')
            if os.path.exists(diagnoses_path):
                df = pd.read_csv(diagnoses_path)
                if 'HCUP_CCS_2015' in df.columns:
                    for _, row in df.iterrows():
                        if pd.notna(row['HCUP_CCS_2015']):
                            icd_to_ccs[str(row['ICD9_CODE'])] = str(row['HCUP_CCS_2015'])
    
    return icd_to_ccs

def convert_to_native_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    return obj

def convert_gcs_to_numeric(value, field_name):
    """Convert Glasgow Coma Scale text values to numeric codes."""
    if pd.isna(value) or value == '':
        return None
    
    value_str = str(value).strip()
    
    if field_name in GCS_MAPPINGS:
        mapping = GCS_MAPPINGS[field_name]
        if value_str in mapping:
            return mapping[value_str]
        for key, val in mapping.items():
            if key.lower() == value_str.lower():
                return val
    
    return None

def process_episode_timeseries(episode_path):
    """Process episode timeseries to create hourly matrix with all 17 indicators."""
    # Define all 17 indicators in the correct order
    all_indicators = [
        'Capillary refill rate',
        'Diastolic blood pressure',
        'Fraction inspired oxygen',
        'Glascow coma scale eye opening',
        'Glascow coma scale motor response',
        'Glascow coma scale total',
        'Glascow coma scale verbal response',
        'Glucose',  # Lab indicator
        'Heart Rate',
        'Height',
        'Mean blood pressure',
        'Oxygen saturation',
        'pH',  # Lab indicator
        'Respiratory rate',
        'Systolic blood pressure',
        'Temperature',
        'Weight'
    ]
    
    # Read timeseries data
    df = pd.read_csv(episode_path)
    
    # Create hourly bins
    df['hour'] = np.floor(df['Hours']).astype(int)
    
    # Get the range of hours
    max_hour = int(df['hour'].max()) + 1 if len(df) > 0 else 1
    
    # Create matrix (17 indicators × hours)
    matrix = []
    
    for indicator in all_indicators:
        if indicator in df.columns:
            hourly_values = []
            
            # Check if this is a Glasgow Coma Scale field
            if 'Glascow coma scale' in indicator and indicator != 'Glascow coma scale total':
                # Convert text values to numeric for GCS fields
                for h in range(max_hour):
                    hour_data = df[df['hour'] == h][indicator]
                    if len(hour_data) > 0:
                        numeric_values = []
                        for val in hour_data:
                            numeric_val = convert_gcs_to_numeric(val, indicator)
                            if numeric_val is not None:
                                numeric_values.append(numeric_val)
                        
                        if numeric_values:
                            hourly_values.append(convert_to_native_types(np.mean(numeric_values)))
                        else:
                            hourly_values.append(None)
                    else:
                        hourly_values.append(None)
            else:
                # For all other fields (including pH and Glucose)
                for h in range(max_hour):
                    hour_data = df[df['hour'] == h][indicator]
                    numeric_data = pd.to_numeric(hour_data, errors='coerce')
                    valid_data = numeric_data.dropna()
                    if len(valid_data) > 0:
                        hourly_values.append(convert_to_native_types(valid_data.mean()))
                    else:
                        hourly_values.append(None)
            
            matrix.append(hourly_values)
        else:
            # If indicator not in data, fill with None
            matrix.append([None] * max_hour)
    
    return matrix

def process_patient_data(subject_dir, mimic_data_path, icd_to_ccs):
    """Process data for a single patient, returning all ICU stays."""
    # Read stays data
    stays_path = os.path.join(subject_dir, 'stays.csv')
    stays = pd.read_csv(stays_path)
    
    if len(stays) == 0:
        return [], []
    
    all_matrices = []
    all_patient_infos = []
    
    # Read all admissions for this patient
    subject_id = int(stays.iloc[0]['SUBJECT_ID'])
    all_admits_path = os.path.join(mimic_data_path, 'ADMISSIONS.csv')
    all_admits = pd.read_csv(all_admits_path)
    patient_admits = all_admits[all_admits['SUBJECT_ID'] == subject_id].copy()
    patient_admits['ADMITTIME'] = pd.to_datetime(patient_admits['ADMITTIME'])
    
    # Read all diagnoses for this patient
    all_diagnoses_path = os.path.join(mimic_data_path, 'DIAGNOSES_ICD.csv')
    all_diagnoses = pd.read_csv(all_diagnoses_path)
    patient_diagnoses = all_diagnoses[all_diagnoses['SUBJECT_ID'] == subject_id]
    
    # Process each ICU stay
    for stay_idx, stay in stays.iterrows():
        stay_id = int(stay['ICUSTAY_ID'])
        hadm_id = int(stay['HADM_ID'])
        current_admit_time = pd.to_datetime(stay['ADMITTIME'])
        
        # Create patient_id as subject_id_stay_id
        patient_id = f"{subject_id}_{stay_id}"
        
        # Get patient demographics for this stay
        patient_info = {
            'patient_id': patient_id,  # Changed from subject_id to patient_id (subject_id_stay_id)
            'gender': stay['GENDER'],
            'age': convert_to_native_types(stay['AGE']),
            'race': stay['ETHNICITY'],
            'current_visit_time': current_admit_time.isoformat(),
            'mortality_inunit': convert_to_native_types(stay['MORTALITY_INUNIT']),  # ICU death
            'mortality': convert_to_native_types(stay['MORTALITY']),  # Hospital death
        }
        
        # Get past admissions (before current admission)
        past_admits = patient_admits[patient_admits['ADMITTIME'] < current_admit_time]
        past_hadm_ids = set(past_admits['HADM_ID'].tolist())
        
        # Get past diagnoses
        past_diagnoses = patient_diagnoses[
            (patient_diagnoses['HADM_ID'].isin(past_hadm_ids)) &
            (patient_diagnoses['ICD_VERSION'] == 9)
        ]
        
        # Get unique ICD codes from past visits
        past_icd_codes = past_diagnoses['ICD9_CODE'].unique().tolist()
        patient_info['past_icd_codes'] = [str(code) for code in past_icd_codes]
        
        # Convert past ICD codes to CCS codes using benchmark mappings
        past_ccs_codes = set()
        for icd in past_icd_codes:
            icd_str = str(icd).strip()
            if icd_str in icd_to_ccs:
                past_ccs_codes.add(icd_to_ccs[icd_str])
        patient_info['past_ccs_codes'] = sorted(list(past_ccs_codes))
        
        # Process episode timeseries - find the corresponding episode file
        episode_num = stay_idx + 1  # Episode numbers start from 1
        episode_path = os.path.join(subject_dir, f'episode{episode_num}_timeseries.csv')
        
        if os.path.exists(episode_path):
            # Get matrix with all 17 indicators
            matrix = process_episode_timeseries(episode_path)
            all_matrices.append(matrix)
            all_patient_infos.append(patient_info)
        else:
            # If no timeseries data, still include patient info but skip matrix
            all_patient_infos.append(patient_info)
    
    return all_matrices, all_patient_infos

def main():
    # Paths
    benchmark_output = '/scratch/bcew/ruikez2/intern/s3_med/data/mimic4_benchmark_output'
    mimic_data_path = '/scratch/bcew/ruikez2/intern/s3_med/data/mimiciv_demo_adapted'
    output_dir = '/scratch/bcew/ruikez2/intern/s3_med/data'
    
    # Load ICD to CCS mappings from benchmark output
    print("Loading CCS mappings from benchmark output...")
    icd_to_ccs = load_benchmark_ccs_mappings(benchmark_output)
    print(f"Loaded {len(icd_to_ccs)} ICD to CCS mappings")
    
    # Get all subject directories
    subject_dirs = [d for d in os.listdir(benchmark_output) 
                   if os.path.isdir(os.path.join(benchmark_output, d)) and d.isdigit()]
    
    print(f"Found {len(subject_dirs)} subjects to process")
    
    # Process each patient
    all_matrices = []
    all_patient_info = []
    
    for subject_dir_name in subject_dirs:
        subject_dir = os.path.join(benchmark_output, subject_dir_name)
        
        try:
            matrices, patient_infos = process_patient_data(subject_dir, mimic_data_path, icd_to_ccs)
            
            # Add all matrices and patient infos for this patient's stays
            all_matrices.extend(matrices)
            all_patient_info.extend(patient_infos)
                
        except Exception as e:
            print(f"Error processing subject {subject_dir_name}: {e}")
            continue
    
    print(f"Successfully processed {len(all_matrices)} ICU stays with timeseries data")
    print(f"Successfully processed {len(all_patient_info)} ICU stays with demographics")
    
    # Calculate statistics
    stays_with_past_icd = sum(1 for p in all_patient_info if p['past_icd_codes'])
    stays_with_past_ccs = sum(1 for p in all_patient_info if p['past_ccs_codes'])
    print(f"ICU stays with past ICD codes: {stays_with_past_icd}")
    print(f"ICU stays with past CCS codes: {stays_with_past_ccs}")
    
    # Count unique patients
    unique_patients = len(set(p['patient_id'].split('_')[0] for p in all_patient_info))
    print(f"Unique patients: {unique_patients}")
    
    # Save to JSON files
    output_matrices = os.path.join(output_dir, 'chartevents_matrices.json')
    output_info = os.path.join(output_dir, 'patient_info.json')
    
    # Save with one patient per line for better readability
    with open(output_matrices, 'w') as f:
        f.write('[')
        for i, matrix in enumerate(all_matrices):
            if i > 0:
                f.write(',\n')
            else:
                f.write('\n')
            json.dump(matrix, f, separators=(',', ':'))
        f.write('\n]')
    
    with open(output_info, 'w') as f:
        f.write('[')
        for i, info in enumerate(all_patient_info):
            if i > 0:
                f.write(',\n')
            else:
                f.write('\n')
            json.dump(info, f, separators=(',', ':'))
        f.write('\n]')
    
    print(f"\nOutput files created:")
    print(f"  - {output_matrices}")
    print(f"  - {output_info}")
    
    # Print sample info
    if all_matrices:
        print(f"\nFirst matrix dimensions: {len(all_matrices[0])} indicators × {len(all_matrices[0][0])} hours")
        print("\nIndicator order in matrix:")
        indicators = [
            'Capillary refill rate',
            'Diastolic blood pressure',
            'Fraction inspired oxygen',
            'Glascow coma scale eye opening',
            'Glascow coma scale motor response',
            'Glascow coma scale total',
            'Glascow coma scale verbal response',
            'Glucose',
            'Heart Rate',
            'Height',
            'Mean blood pressure',
            'Oxygen saturation',
            'pH',
            'Respiratory rate',
            'Systolic blood pressure',
            'Temperature',
            'Weight'
        ]
        for i, ind in enumerate(indicators):
            print(f"  {i}: {ind}")
    
    if all_patient_info:
        print(f"\nSample ICU stay info:")
        print(f"  First stay patient_id: {all_patient_info[0]['patient_id']}")
        print(f"  Keys: {list(all_patient_info[0].keys())}")

if __name__ == "__main__":
    main()