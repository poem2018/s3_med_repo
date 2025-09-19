#!/usr/bin/env python3
"""
Step 4: Add temporal information to patient data.
Converts the matrix data into text format with temporal context.
"""

import os
import json
import numpy as np
from pathlib import Path

# Check for test mode
TEST_MODE = os.environ.get('TEST_MODE', 'false').lower() == 'true'
if TEST_MODE:
    print("Processing test data...")
else:
    print("Processing full data...")

def matrix_to_text(matrix, feature_names):
    """Convert a patient's matrix data to text format with temporal information.
    Format: Feature: value in hour1, value in hour2, ...
    """
    text_parts = []
    matrix = np.array(matrix)
    num_features, num_hours = matrix.shape
    
    # Limit to first 48 hours
    num_hours = min(48, num_hours)
    
    for feat_idx in range(num_features):
        feature_name = feature_names[feat_idx]
        values_by_hour = []
        
        for hour in range(num_hours):
            value = matrix[feat_idx, hour]
            if not np.isnan(value):
                if value % 1 == 0:  # Integer value
                    values_by_hour.append(f"{int(value)} in hour {hour}")
                else:
                    values_by_hour.append(f"{value:.2f} in hour {hour}")
        
        if values_by_hour:
            text_parts.append(f"{feature_name}: " + ", ".join(values_by_hour))
    
    return "\n".join(text_parts)

def add_temporal_info(input_matrices_file, input_info_file, output_file):
    """Add temporal information to patient data."""
    
    # Feature names (must match CSV column names exactly)
    FEATURES = [
        'Capillary refill rate',
        'Diastolic blood pressure',
        'Fraction inspired oxygen',
        'Glascow coma scale eye opening',
        'Glascow coma scale motor response',
        'Glascow coma scale total',
        'Glascow coma scale verbal response',
        'Glucose',
        'Heart Rate',  # Fixed capitalization to match data
        'Height',
        'Mean blood pressure',
        'Oxygen saturation',
        'Respiratory rate',
        'Systolic blood pressure',
        'Temperature',
        'Weight',
        'pH'
    ]
    
    # Load data
    with open(input_matrices_file, 'r') as f:
        matrices = json.load(f)
    
    with open(input_info_file, 'r') as f:
        patient_info = json.load(f)
    
    # Process each patient
    patient_data_with_temporal = []
    
    for idx, (matrix, info) in enumerate(zip(matrices, patient_info)):
        # Convert matrix to temporal text
        temporal_text = matrix_to_text(matrix, FEATURES)
        
        # Create patient data with temporal info
        patient_data = {
            'patient_id': info['patient_id'],
            'demographics': {
                'gender': info['gender'],
                'age': info['age'],
                'race': info['race']
            },
            'visit_info': {
                'admission_time': info['current_visit_time'],
                'mortality_inunit': info['mortality_inunit'],
                'mortality': info['mortality']
            },
            'medical_history': {
                'past_icd_codes': info['past_icd_codes'],
                'past_ccs_codes': info['past_ccs_codes']
            },
            'temporal_data': temporal_text,
            'matrix': matrix
        }
        
        patient_data_with_temporal.append(patient_data)
        
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1} patients...")
    
    # Save output
    with open(output_file, 'w') as f:
        json.dump(patient_data_with_temporal, f, indent=2)
    
    print(f"\nSaved temporal data to: {output_file}")
    print(f"Total patients processed: {len(patient_data_with_temporal)}")

if __name__ == '__main__':
    suffix = '_test' if TEST_MODE else '_full'
    input_matrices = f'/scratch/bcew/ruikez2/intern/s3_med/data/baseline_text/chartevents_matrices{suffix}.json'
    input_info = f'/scratch/bcew/ruikez2/intern/s3_med/data/baseline_text/patient_info{suffix}.json'
    output_file = f'/scratch/bcew/ruikez2/intern/s3_med/data/baseline_text/patient_data_with_temporal{suffix}.json'
    
    add_temporal_info(input_matrices, input_info, output_file)