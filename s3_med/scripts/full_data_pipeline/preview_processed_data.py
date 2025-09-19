#!/usr/bin/env python3
"""
Preview processed data files to verify the data processing pipeline.
Shows the first 2 patients from each data file.
"""

import os
import json
import numpy as np
from pathlib import Path

# Check for test mode
TEST_MODE = os.environ.get('TEST_MODE', 'false').lower() == 'true'
SUFFIX = '_test' if TEST_MODE else '_full'
if TEST_MODE:
    print("🧪 Previewing TEST MODE data")

def preview_json_data(file_path, max_items=2):
    """Preview JSON data file."""
    print(f"\n{'='*60}")
    print(f"File: {file_path}")
    print('='*60)
    
    if not Path(file_path).exists():
        print(f"❌ File not found!")
        return
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        print(f"Total items: {len(data)}")
        print(f"Showing first {min(max_items, len(data))} items:\n")
        
        for i, item in enumerate(data[:max_items]):
            print(f"\n--- Item {i+1} ---")
            if isinstance(item, dict):
                pretty_print_dict(item)
            elif isinstance(item, list) and len(item) > 0:
                # For matrix data
                print(f"Matrix shape: {len(item)} x {len(item[0]) if item else 0}")
                print("First 3 features (first 10 hours):")
                for feat_idx in range(min(3, len(item))):
                    row = item[feat_idx][:10] if len(item[feat_idx]) > 10 else item[feat_idx]
                    print(f"  Feature {feat_idx}: {row}")
            else:
                print(item)
    elif isinstance(data, dict):
        pretty_print_dict(data)

def pretty_print_dict(d, indent=0):
    """Pretty print dictionary with proper indentation."""
    for key, value in d.items():
        if isinstance(value, dict):
            print(' ' * indent + f"{key}:")
            pretty_print_dict(value, indent + 2)
        elif isinstance(value, list):
            if len(value) == 0:
                print(' ' * indent + f"{key}: []")
            elif isinstance(value[0], (int, float, str)):
                # Simple list
                if len(value) > 5:
                    print(' ' * indent + f"{key}: [{', '.join(str(v) for v in value[:5])}, ... ({len(value)} items)]")
                else:
                    print(' ' * indent + f"{key}: {value}")
            else:
                # Complex list (like matrix)
                if isinstance(value[0], list) and len(value) > 0:
                    print(' ' * indent + f"{key}: Matrix {len(value)}x{len(value[0])}")
                    # Show first 3 rows, first 5 columns
                    for i in range(min(3, len(value))):
                        row_preview = value[i][:5] if len(value[i]) > 5 else value[i]
                        print(' ' * (indent + 2) + f"Row {i}: {row_preview}...")
                else:
                    print(' ' * indent + f"{key}: List of {len(value)} items")
        elif isinstance(value, str) and len(value) > 100:
            # Long string - show preview
            print(' ' * indent + f"{key}: {value[:100]}...")
        else:
            print(' ' * indent + f"{key}: {value}")

def main():
    """Preview all processed data files."""
    
    print("\n" + "="*60)
    print("MIMIC-IV Processed Data Preview")
    print("="*60)
    
    # Define file paths
    base_path = Path('/scratch/bcew/ruikez2/intern/s3_med/data/baseline_text')
    
    # Adjust file names based on mode
    if TEST_MODE:
        files_to_preview = [
            # Test dataset files
            (f'patient_info{SUFFIX}.json', 'Basic patient information'),
            (f'chartevents_matrices{SUFFIX}.json', 'Patient matrices'),
            (f'patient_data_with_temporal{SUFFIX}.json', 'Patient data with temporal info'),
            
            # Split files
            (f'splits{SUFFIX}/train_data.json', 'Training set'),
            (f'splits{SUFFIX}/val_data.json', 'Validation set'),
            (f'splits{SUFFIX}/test_data.json', 'Test set'),
            (f'splits{SUFFIX}/data_splits.json', 'Data split statistics')
        ]
    else:
        files_to_preview = [
            # Full dataset files
            (f'patient_info{SUFFIX}.json', 'Basic patient information'),
            (f'chartevents_matrices{SUFFIX}.json', 'Patient matrices'),
            (f'patient_data_with_temporal{SUFFIX}.json', 'Patient data with temporal info'),
            
            # Split files
            (f'splits{SUFFIX}/train_data.json', 'Training set'),
            (f'splits{SUFFIX}/val_data.json', 'Validation set'),
            (f'splits{SUFFIX}/test_data.json', 'Test set'),
            (f'splits{SUFFIX}/data_splits.json', 'Data split statistics')
        ]
    
    for filename, description in files_to_preview:
        file_path = base_path / filename
        print(f"\n{'='*60}")
        print(f"📄 {description}")
        preview_json_data(file_path, max_items=2)
    
    # Special handling for temporal data - show as text
    print("\n" + "="*60)
    print("Example: Patient data formatted as text")
    print("="*60)
    
    temporal_file = base_path / f'patient_data_with_temporal{SUFFIX}.json'
    if temporal_file.exists():
        with open(temporal_file, 'r') as f:
            data = json.load(f)
        
        if data and len(data) > 0:
            patient = data[0]
            print(f"\nPatient ID: {patient.get('patient_id', 'Unknown')}")
            print(f"Demographics: {patient.get('demographics', {})}")
            print(f"Visit Info: {patient.get('visit_info', {})}")
            
            if 'temporal_data' in patient:
                print("\nTemporal Data (first 500 chars):")
                print("-" * 40)
                temporal_text = patient['temporal_data']
                print(temporal_text[:500] + "..." if len(temporal_text) > 500 else temporal_text)
            
            if 'medical_history' in patient:
                history = patient['medical_history']
                print("\nMedical History:")
                print(f"  Past ICD codes: {len(history.get('past_icd_codes', []))} codes")
                print(f"  Past CCS codes: {len(history.get('past_ccs_codes', []))} codes")
                if history.get('past_ccs_codes'):
                    print(f"  Sample CCS: {history['past_ccs_codes'][:3]}")
    
    # Preview text format data
    print("\n" + "="*60)
    print("📝 Text Format Data Preview")
    print("="*60)
    
    text_data_path = base_path / f'text_data{SUFFIX}'
    
    # Preview JSONL files
    for split in ['train', 'val', 'test']:
        jsonl_file = text_data_path / f'{split}_text.jsonl'
        if jsonl_file.exists():
            print(f"\n--- {split.upper()} Text Data (first 2 patients) ---")
            with open(jsonl_file, 'r') as f:
                for i, line in enumerate(f):
                    if i >= 2:  # Only show first 2
                        break
                    patient_text = json.loads(line)
                    print(f"\nPatient {i+1} ID: {patient_text['patient_id']}")
                    print(f"Mortality: {'Yes' if patient_text['mortality'] == 1 else 'No'}")
                    print(f"Text (first 800 chars):")
                    print("-" * 40)
                    text_preview = patient_text['text'][:800]
                    print(text_preview + "..." if len(patient_text['text']) > 800 else text_preview)
        else:
            if not TEST_MODE:  # Only warn if not in test mode
                print(f"\n{split.upper()} text data not found at {jsonl_file}")
    
    # Preview baseline prompts
    prompts_file = text_data_path / 'baseline_prompts.json'
    if prompts_file.exists():
        print("\n" + "="*60)
        print("📋 Baseline Prompt Templates")
        print("="*60)
        with open(prompts_file, 'r') as f:
            prompts = json.load(f)
        for prompt_type, prompt_data in prompts.items():
            print(f"\n{prompt_type}:")
            print(f"  System: {prompt_data['system'][:100]}...")
            print(f"  User Template Preview: {prompt_data['user_template'][:150]}...")
    
    print("\n" + "="*60)
    print("Preview complete!")
    print("="*60)

if __name__ == '__main__':
    main()