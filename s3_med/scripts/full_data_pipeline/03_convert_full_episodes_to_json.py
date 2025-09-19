#!/usr/bin/env python3
"""
Step 3: Convert MIMIC-IV benchmark episode data to required JSON format.
Creates two files:
1. chartevents_matrices.json: List of 17×hours matrices (including ALL indicators)
2. patient_info.json: Patient demographics, past ICD/CCS codes, current visit time
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Check for test mode
TEST_MODE = os.environ.get('TEST_MODE', 'false').lower() == 'true'
if TEST_MODE:
    print("🧪 Running in TEST MODE - will only process subset of data")
    MAX_PATIENTS = 10  # Process only 10 patients in test mode

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
                try:
                    diagnoses_df = pd.read_csv(diagnoses_path, dtype={'ICD9_CODE': str})
                    for _, row in diagnoses_df.iterrows():
                        # The column is actually called HCUP_CCS_2015, not CCS_CATEGORY
                        if pd.notna(row.get('HCUP_CCS_2015')):
                            icd_to_ccs[str(row['ICD9_CODE'])] = row['HCUP_CCS_2015']
                except Exception as e:
                    print(f"Error processing {diagnoses_path}: {e}")
    
    print(f"Loaded {len(icd_to_ccs)} ICD to CCS mappings")
    return icd_to_ccs

def process_value(feature, value):
    """Process values based on feature type."""
    if pd.isna(value) or value == '':
        return None
        
    value_str = str(value)
    
    # Handle GCS mappings
    if feature in GCS_MAPPINGS:
        return GCS_MAPPINGS.get(feature, {}).get(value_str, None)
    
    # Try to convert to float
    try:
        return float(value_str)
    except:
        return None

def convert_episodes_to_json(benchmark_output, output_dir):
    """Convert episode data to JSON format."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load ICD to CCS mappings
    icd_to_ccs = load_benchmark_ccs_mappings(benchmark_output)
    
    # Feature list (17 features)
    FEATURES = [
        'Capillary refill rate',
        'Diastolic blood pressure',
        'Fraction inspired oxygen',
        'Glascow coma scale eye opening',
        'Glascow coma scale motor response',
        'Glascow coma scale total',
        'Glascow coma scale verbal response',
        'Glucose',
        'Heart Rate',  # Fixed capitalization to match CSV column
        'Height',
        'Mean blood pressure',
        'Oxygen saturation',
        'Respiratory rate',
        'Systolic blood pressure',
        'Temperature',
        'Weight',
        'pH'
    ]
    
    all_matrices = []
    patient_info_list = []
    processed_count = 0
    error_count = 0
    
    # Process each subject
    subject_dirs = sorted([d for d in os.listdir(benchmark_output) if d.isdigit()])
    
    # In test mode with small sample, limit the number of subjects
    # But for full test set processing, don't limit
    if TEST_MODE and len(subject_dirs) > 1000:
        print(f"Processing full test set: {len(subject_dirs)} subjects")
    elif TEST_MODE and MAX_PATIENTS < len(subject_dirs):
        subject_dirs = subject_dirs[:MAX_PATIENTS]
        print(f"TEST MODE: Processing only {len(subject_dirs)} subjects")
    
    for subject_dir in subject_dirs:
        subject_path = os.path.join(benchmark_output, subject_dir)
        
        # Find all episode timeseries files directly in subject directory
        episode_files = [f for f in os.listdir(subject_path) if f.endswith('_timeseries.csv')]
        
        for episode_file_name in sorted(episode_files):
            episode_file = os.path.join(subject_path, episode_file_name)
            
            if not os.path.exists(episode_file):
                continue
            
            try:
                # Read episode data
                episode_df = pd.read_csv(episode_file)
                
                if episode_df.empty:
                    continue
                
                # Get max hours (check for valid range)
                if 'Hours' not in episode_df.columns:
                    continue
                    
                min_hour = episode_df['Hours'].min()
                max_hour = episode_df['Hours'].max()
                
                # Skip if hours are invalid (allow negative hours for early data)
                if pd.isna(min_hour) or pd.isna(max_hour):
                    continue
                    
                # Adjust for negative hours - shift all hours to start from 0
                if min_hour < 0:
                    episode_df['Hours'] = episode_df['Hours'] - min_hour
                    max_hour = max_hour - min_hour
                    min_hour = 0
                    
                max_hours = min(48, int(max_hour) + 1)  # Limit to first 48 hours
                
                # Create matrix
                matrix = np.full((len(FEATURES), max_hours), np.nan)
                
                # Fill matrix
                for _, row in episode_df.iterrows():
                    hour = int(row['Hours'])
                    if hour < 0 or hour >= max_hours:
                        continue  # Skip invalid hours
                    for i, feature in enumerate(FEATURES):
                        if feature in row:
                            processed_val = process_value(feature, row[feature])
                            if processed_val is not None:
                                matrix[i, hour] = processed_val
                
                # Create patient ID from filename (e.g., episode1_timeseries.csv -> episode1)
                episode_num = episode_file_name.replace('_timeseries.csv', '').replace('episode', '')
                patient_id = f"{subject_dir}_{episode_num}"
                
                # Get patient demographics and outcomes
                stays_path = os.path.join(subject_path, 'stays.csv')
                diagnoses_path = os.path.join(subject_path, 'diagnoses.csv')
                
                patient_info = {
                    'patient_id': patient_id,
                    'gender': 'U',
                    'age': 0,
                    'race': 'UNKNOWN',
                    'current_visit_time': '',
                    'mortality_inunit': 0,
                    'mortality': 0,
                    'past_icd_codes': [],
                    'past_ccs_codes': []
                }
                
                # Read stays info
                if os.path.exists(stays_path):
                    stays_df = pd.read_csv(stays_path)
                    # Episode number corresponds to row index (episode1 = first stay, etc.)
                    episode_idx = int(episode_num) - 1 if episode_num.isdigit() else 0
                    if episode_idx < len(stays_df):
                        matching_stay = stays_df.iloc[[episode_idx]]
                    else:
                        matching_stay = pd.DataFrame()  # Empty if out of bounds
                    
                    if not matching_stay.empty:
                        stay = matching_stay.iloc[0]
                        patient_info['gender'] = stay.get('GENDER', 'U')
                        patient_info['age'] = int(stay.get('AGE', 0))
                        patient_info['race'] = stay.get('ETHNICITY', 'UNKNOWN')
                        patient_info['current_visit_time'] = str(stay.get('INTIME', ''))
                        patient_info['mortality_inunit'] = int(stay.get('MORTALITY_INUNIT', 0))
                        patient_info['mortality'] = int(stay.get('MORTALITY', 0))
                
                # Read diagnoses (past visits)
                if os.path.exists(diagnoses_path):
                    diagnoses_df = pd.read_csv(diagnoses_path, dtype={'ICD9_CODE': str})
                    past_codes = diagnoses_df['ICD9_CODE'].dropna().unique().tolist()
                    patient_info['past_icd_codes'] = past_codes
                    
                    # Get CCS codes
                    ccs_codes = set()
                    for icd in past_codes:
                        if str(icd) in icd_to_ccs:
                            ccs_codes.add(icd_to_ccs[str(icd)])
                    patient_info['past_ccs_codes'] = sorted(list(ccs_codes))
                
                all_matrices.append(matrix.tolist())
                patient_info_list.append(patient_info)
                processed_count += 1
                
                if processed_count % 100 == 0:
                    print(f"Processed {processed_count} episodes...")
                    
            except Exception as e:
                error_count += 1
                print(f"Error processing {episode_file}: {e}")
                continue
    
    # Save outputs
    # Check if we're processing test subset (benchmark_test_subset path)
    if 'benchmark_test_subset' in benchmark_output:
        suffix = '_test'  # Always use _test suffix for test subset data
    else:
        suffix = '_test' if TEST_MODE else '_full'
    output_matrices = os.path.join(output_dir, f'chartevents_matrices{suffix}.json')
    output_info = os.path.join(output_dir, f'patient_info{suffix}.json')
    
    with open(output_matrices, 'w') as f:
        json.dump(all_matrices, f)
    
    with open(output_info, 'w') as f:
        json.dump(patient_info_list, f, indent=2)
    
    print(f"\nProcessing complete!")
    print(f"Total episodes processed: {processed_count}")
    print(f"Total errors: {error_count}")
    print(f"Saved matrices to: {output_matrices}")
    print(f"Saved patient info to: {output_info}")
    
    # Print statistics
    if len(patient_info_list) > 0:
        mortality_count = sum(1 for p in patient_info_list if p['mortality'] == 1)
        print(f"\nDataset statistics:")
        print(f"Total patients: {len(patient_info_list)}")
        print(f"Mortality cases: {mortality_count} ({mortality_count/len(patient_info_list)*100:.2f}%)")
    else:
        print(f"\nNo patient data processed. Check if episode files exist in {benchmark_output}")

if __name__ == '__main__':
    # Use different paths based on test mode
    if TEST_MODE:
        benchmark_output = '/scratch/bcew/ruikez2/intern/s3_med/data/baseline_text/benchmark_test_subset'
    else:
        benchmark_output = '/scratch/bcew/ruikez2/intern/s3_med/data/mimic4_benchmark_output_full'
    output_dir = '/scratch/bcew/ruikez2/intern/s3_med/data/baseline_text'
    
    convert_episodes_to_json(benchmark_output, output_dir)