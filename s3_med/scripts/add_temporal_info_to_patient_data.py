#!/usr/bin/env python3
"""
Add temporal information to patient data for the first 48 hours.
Converts raw values like "120, 125, 130..." to "120 at hour 1, 125 at hour 2, 130 at hour 3..."
"""

import json
import os
import sys
from pathlib import Path

# Define all 17 indicators in the correct order (must match chartevents_matrices.json)
INDICATORS = [
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

# GCS mappings for converting numeric values back to text
GCS_EYE_TEXT = {
    4: "Spontaneously",
    3: "To Speech", 
    2: "To Pain",
    1: "No response"
}

GCS_MOTOR_TEXT = {
    6: "Obeys Commands",
    5: "Localizes Pain",
    4: "Flex-withdraws",
    3: "Abnormal Flexion",
    2: "Abnormal extension",
    1: "No response"
}

GCS_VERBAL_TEXT = {
    5: "Oriented",
    4: "Confused",
    3: "Inappropriate Words",
    2: "Incomprehensible sounds",
    1: "No Response"
}

def format_value_with_time(value, hour, indicator_name=None):
    """Format a single value with its hour timestamp."""
    if value is None:
        return None
    
    # Convert GCS numeric values to text descriptions
    if indicator_name == 'Glascow coma scale eye opening' and value in GCS_EYE_TEXT:
        return f"{GCS_EYE_TEXT[value]} at hour {hour}"
    elif indicator_name == 'Glascow coma scale motor response' and value in GCS_MOTOR_TEXT:
        return f"{GCS_MOTOR_TEXT[value]} at hour {hour}"
    elif indicator_name == 'Glascow coma scale verbal response' and value in GCS_VERBAL_TEXT:
        return f"{GCS_VERBAL_TEXT[value]} at hour {hour}"
    
    # For all other indicators, use numeric format
    return f"{value:.2f} at hour {hour}" if isinstance(value, float) else f"{value} at hour {hour}"

def process_patient_timeseries(matrix, max_hours=48):
    """
    Process a patient's timeseries matrix to add temporal information.
    
    Args:
        matrix: 17×hours matrix of values
        max_hours: Maximum number of hours to include (default 48)
    
    Returns:
        Dictionary with formatted strings for each indicator
    """
    formatted_data = {}
    
    # Limit to first 48 hours or available data
    hours_to_process = min(len(matrix[0]) if matrix and matrix[0] else 0, max_hours)
    
    for idx, indicator_name in enumerate(INDICATORS):
        if idx < len(matrix):
            values = matrix[idx][:hours_to_process]
            
            # Format non-null values with temporal information
            formatted_values = []
            for hour, value in enumerate(values, start=1):
                if value is not None:
                    # Pass indicator name for GCS text conversion
                    formatted_values.append(format_value_with_time(value, hour, indicator_name))
            
            if formatted_values:
                # Join values with commas
                formatted_data[indicator_name] = ", ".join(formatted_values)
            else:
                formatted_data[indicator_name] = "No data available"
        else:
            formatted_data[indicator_name] = "No data available"
    
    return formatted_data

def create_patient_summary_with_temporal(patient_info, timeseries_data):
    """
    Create a patient summary string with temporal information.
    
    Args:
        patient_info: Dictionary with patient demographics
        timeseries_data: Formatted timeseries data with temporal info
    
    Returns:
        Formatted string with patient summary
    """
    summary_parts = []
    
    # Add demographics
    summary_parts.append(f"Patient ID: {patient_info['patient_id']}")
    summary_parts.append(f"Age: {patient_info['age']} years")
    summary_parts.append(f"Gender: {patient_info['gender']}")
    summary_parts.append(f"Race/Ethnicity: {patient_info['race']}")
    
    # Add past medical history if available
    if patient_info.get('past_icd_codes'):
        summary_parts.append(f"Past ICD codes: {', '.join(patient_info['past_icd_codes'][:10])}")  # Limit to 10 codes
    
    if patient_info.get('past_ccs_codes'):
        summary_parts.append(f"Past CCS codes: {', '.join(patient_info['past_ccs_codes'][:10])}")  # Limit to 10 codes
    
    summary_parts.append("\nFirst 48 hours vital signs and lab values:")
    
    # Add formatted timeseries data
    for indicator, values_str in timeseries_data.items():
        if values_str != "No data available":
            summary_parts.append(f"{indicator}: {values_str}")
    
    # Add mortality labels for training
    summary_parts.append(f"\nICU Mortality: {patient_info.get('mortality_inunit', 'Unknown')}")
    summary_parts.append(f"Hospital Mortality: {patient_info.get('mortality', 'Unknown')}")
    
    return "\n".join(summary_parts)

def main():
    # Paths
    data_dir = '/scratch/bcew/ruikez2/intern/s3_med/data'
    matrices_file = os.path.join(data_dir, 'chartevents_matrices.json')
    patient_info_file = os.path.join(data_dir, 'patient_info.json')
    output_file = os.path.join(data_dir, 'patient_data_with_temporal.json')
    
    # Check if input files exist
    if not os.path.exists(matrices_file):
        print(f"Error: {matrices_file} not found!")
        sys.exit(1)
    
    if not os.path.exists(patient_info_file):
        print(f"Error: {patient_info_file} not found!")
        sys.exit(1)
    
    print(f"Loading data from:")
    print(f"  - {matrices_file}")
    print(f"  - {patient_info_file}")
    
    # Load JSON files
    with open(matrices_file, 'r') as f:
        all_matrices = json.load(f)
    
    with open(patient_info_file, 'r') as f:
        all_patient_info = json.load(f)
    
    print(f"Loaded {len(all_matrices)} timeseries matrices")
    print(f"Loaded {len(all_patient_info)} patient records")
    
    # Process each patient
    processed_patients = []
    
    for i, patient_info in enumerate(all_patient_info):
        patient_id = patient_info['patient_id']
        
        # Check if we have corresponding timeseries data
        if i < len(all_matrices):
            matrix = all_matrices[i]
            
            # Process timeseries with temporal information (first 48 hours)
            temporal_data = process_patient_timeseries(matrix, max_hours=48)
            
            # Create patient summary
            patient_summary = create_patient_summary_with_temporal(patient_info, temporal_data)
            
            # Store processed data
            processed_patient = {
                'patient_id': patient_id,
                'demographics': patient_info,
                'temporal_data': temporal_data,
                'summary': patient_summary,
                'mortality_inunit': patient_info.get('mortality_inunit', 0),
                'mortality': patient_info.get('mortality', 0)
            }
            
            processed_patients.append(processed_patient)
        else:
            print(f"Warning: No timeseries data for patient {patient_id}")
    
    print(f"\nProcessed {len(processed_patients)} patients with temporal information")
    
    # Save processed data
    with open(output_file, 'w') as f:
        json.dump(processed_patients, f, indent=2)
    
    print(f"\nOutput saved to: {output_file}")
    
    # Print sample output
    if processed_patients:
        print("\n" + "="*80)
        print("SAMPLE OUTPUT (First patient):")
        print("="*80)
        print(processed_patients[0]['summary'])
        print("="*80)

if __name__ == "__main__":
    main()