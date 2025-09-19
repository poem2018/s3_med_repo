#!/usr/bin/env python3
"""
Preprocess MIMIC-IV data for ICU mortality prediction using S3 framework
This version uses the updated data format with patient_info.json and chartevents_matrices.json
"""

import json
import os
import numpy as np
from typing import Dict, List, Any
import argparse
from tqdm import tqdm

# 17 indicators in the matrix (in order)
INDICATOR_NAMES = [
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

# Abbreviations for compact display
INDICATOR_ABBREV = {
    'Heart Rate': 'HR',
    'Respiratory rate': 'RR',
    'Oxygen saturation': 'SpO2',
    'Mean blood pressure': 'MAP',
    'Systolic blood pressure': 'SBP',
    'Diastolic blood pressure': 'DBP',
    'Temperature': 'Temp',
    'Glucose': 'Glucose',
    'pH': 'pH',
    'Glascow coma scale total': 'GCS',
    'Fraction inspired oxygen': 'FiO2',
    'Capillary refill rate': 'CRR',
    'Weight': 'Wt',
    'Height': 'Ht'
}

def serialize_patient_data_to_text(patient_info: Dict, matrix: List[List], max_hours: int = 48) -> str:
    """
    Convert patient data into a structured text description for the S3 prompt.
    
    Args:
        patient_info: Dictionary containing patient demographics and past history
        matrix: 17 x hours matrix of clinical indicators
        max_hours: Maximum number of hours to include (default 48)
    
    Returns:
        Serialized text description of the patient in structured format
    """
    # Extract patient demographics
    patient_id = patient_info['patient_id']
    age = patient_info['age']
    gender = patient_info['gender']
    race = patient_info['race']
    
    # Convert gender to readable format
    gender_text = "Male" if gender == 'M' else "Female"
    
    # Start building the structured patient summary
    patient_text = f"PATIENT {patient_id} 24H SUMMARY:\n"
    patient_text += f"- Demographics: {age}, {gender_text}\n"
    
    # Add past medical history
    past_ccs = patient_info.get('past_ccs_codes', [])
    if past_ccs:
        patient_text += f"- Diagnoses before admission: {', '.join(past_ccs[:5])}\n"
    else:
        patient_text += "- Diagnoses before admission: None documented\n"
    
    # Process clinical data from matrix for trajectory snippets
    patient_text += "- Trajectory snippets (0-24h):\n"
    
    if matrix and len(matrix) > 0:
        n_hours = min(len(matrix[0]), 24) if matrix[0] else 0  # Focus on first 24 hours
        
        # Show ALL 17 indicators in trajectory
        # Collect values for each indicator in order
        for idx, indicator_name in enumerate(INDICATOR_NAMES):
            abbrev = INDICATOR_ABBREV.get(indicator_name, indicator_name)
            if idx < len(matrix) and matrix[idx]:
                # Collect ALL values from first 24 hours
                values = []
                for hour in range(n_hours):  # Get every hour for first 24 hours
                    if hour < len(matrix[idx]):
                        val = matrix[idx][hour]
                        if val is not None:
                            values.append(f"{val}")
                        else:
                            values.append("NA")  # Use NA for missing values
                
                if values:
                    # Show all 24 values
                    value_str = ", ".join(values)
                    patient_text += f"  {abbrev}: {value_str}\n"
                else:
                    patient_text += f"  {abbrev}: Not available\n"
        
        
        # Check for mechanical ventilation status
        fio2_idx = INDICATOR_NAMES.index('Fraction inspired oxygen')
        vent_status = "No"
        if fio2_idx < len(matrix) and any(v for v in matrix[fio2_idx][:n_hours] if v is not None and float(v) > 0.21):
            vent_status = "Yes"
        patient_text += f"\n- Mechanical ventilation: {vent_status}"
        
        # Add GCS if abnormal
        gcs_idx = INDICATOR_NAMES.index('Glascow coma scale total')
        if gcs_idx < len(matrix):
            gcs_values = [v for v in matrix[gcs_idx][:n_hours] if v is not None]
            if gcs_values and any(float(v) < 15 for v in gcs_values):
                avg_gcs = sum(float(v) for v in gcs_values) / len(gcs_values)
                patient_text += f"\n- Neurological status: GCS {avg_gcs:.1f}"
    
    return patient_text

def create_s3_prompt(patient_text: str) -> str:
    """
    Create the S3 framework prompt with proper tags.
    """
    return f"<question>Based on patient's information, will this patient die in this ICU stay?</question> <patient>{patient_text}</patient>"

def create_mimic_s3_dataset(data_dir: str, output_file: str, split: str = "train", 
                           train_ratio: float = 0.8, val_ratio: float = 0.1):
    """
    Create S3-formatted dataset from MIMIC data
    """
    # Load data
    print(f"Loading MIMIC data from {data_dir}...")
    
    patient_info_path = os.path.join(data_dir, 'patient_info.json')
    matrices_path = os.path.join(data_dir, 'chartevents_matrices.json')
    
    with open(patient_info_path, 'r') as f:
        patient_info_list = json.load(f)
    
    with open(matrices_path, 'r') as f:
        patient_matrices = json.load(f)
    
    print(f"Loaded {len(patient_info_list)} patients with {len(patient_matrices)} matrices")
    
    # Ensure we have matching data
    assert len(patient_info_list) == len(patient_matrices), \
        f"Mismatch: {len(patient_info_list)} patients but {len(patient_matrices)} matrices"
    
    # Determine split indices
    n_total = len(patient_info_list)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    if split == 'train':
        start_idx, end_idx = 0, n_train
    elif split == 'val':
        start_idx, end_idx = n_train, n_train + n_val
    else:  # test
        start_idx, end_idx = n_train + n_val, n_total
    
    print(f"Processing {split} split: indices {start_idx} to {end_idx}")
    
    # Create S3-formatted data
    output_lines = []
    
    for i in tqdm(range(start_idx, end_idx), desc=f"Processing {split} patients"):
        patient_info = patient_info_list[i]
        matrix = patient_matrices[i]
        
        # Serialize patient data
        patient_text = serialize_patient_data_to_text(patient_info, matrix)
        
        # Create S3 prompt
        prompt_text = create_s3_prompt(patient_text)
        
        # Create data entry in the expected format
        data_entry = {
            "text": prompt_text,
            "data_source": "mimic_icu_mortality",
            "reward_model": {
                "style": "rule",
                "ground_truth": {
                    "answers": ["Yes" if patient_info['mortality'] else "No"],
                    "patient_id": patient_info['patient_id'],
                    "mortality": patient_info['mortality'],
                    "mortality_inunit": patient_info['mortality_inunit']
                }
            }
        }
        
        output_lines.append(json.dumps(data_entry))
    
    # import pdb; pdb.set_trace()
    # Save to JSONL file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for line in output_lines:
            f.write(line + '\n')
    
    print(f"Saved {len(output_lines)} entries to {output_file}")
    
    # Print statistics
    mortality_count = sum(1 for i in range(start_idx, end_idx) 
                         if patient_info_list[i]['mortality'])
    print(f"\n{split} Dataset Statistics:")
    print(f"Total patients: {len(output_lines)}")
    print(f"Mortality rate: {mortality_count}/{len(output_lines)} ({mortality_count/len(output_lines)*100:.1f}%)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/scratch/bcew/ruikez2/intern/s3_med/data',
                        help='Directory containing patient_info.json and chartevents_matrices.json')
    parser.add_argument('--output_dir', default='./data/mimic_s3',
                        help='Output directory for S3-formatted data')
    parser.add_argument('--split', default='all', choices=['train', 'val', 'test', 'all'],
                        help='Dataset split to create')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Ratio of training data')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Ratio of validation data')
    
    args = parser.parse_args()
    
    # Create datasets for specified splits
    if args.split == 'all':
        splits = ['train', 'val', 'test']
    else:
        splits = [args.split]
    
    for split in splits:
        output_file = os.path.join(args.output_dir, f'{split}_mimic.jsonl')
        create_mimic_s3_dataset(
            args.data_dir,
            output_file,
            split=split,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio
        )

if __name__ == '__main__':
    main()