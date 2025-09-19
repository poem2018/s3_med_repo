#!/usr/bin/env python3
"""
Step 6: Convert JSON data to text format for baseline models.
Creates human-readable text representations of patient data.
Simplified version that reads from patient_data_with_temporal file.
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

def format_patient_as_text(patient_data):
    """Convert a patient's data to human-readable text format with temporal data."""
    text_parts = []
    
    # Patient Demographics
    text_parts.append("=== PATIENT INFORMATION ===")
    text_parts.append(f"Patient ID: {patient_data.get('patient_id', 'Unknown')}")
    
    if 'demographics' in patient_data:
        demo = patient_data['demographics']
        text_parts.append(f"Age: {demo.get('age', 'Unknown')} years")
        text_parts.append(f"Gender: {demo.get('gender', 'Unknown')}")
        text_parts.append(f"Race/Ethnicity: {demo.get('race', 'Unknown')}")
    
    # Visit Information
    if 'visit_info' in patient_data:
        visit = patient_data['visit_info']
        text_parts.append(f"\n=== CURRENT VISIT ===")
        text_parts.append(f"Admission Time: {visit.get('admission_time', 'Unknown')}")
    
    # Medical History
    if 'medical_history' in patient_data:
        history = patient_data['medical_history']
        text_parts.append(f"\n=== MEDICAL HISTORY ===")
        
        # Past CCS codes (more interpretable than ICD codes)
        if history.get('past_ccs_codes'):
            text_parts.append("Past Medical Conditions:")
            for ccs in history['past_ccs_codes']:
                text_parts.append(f"  - {ccs}")
        else:
            text_parts.append("No prior medical conditions recorded")
    
    # Temporal Clinical Data
    text_parts.append(f"\n=== CLINICAL DATA ===")
    
    if 'temporal_data' in patient_data and patient_data['temporal_data']:
        # Use the pre-formatted temporal data
        text_parts.append(patient_data['temporal_data'])
    else:
        text_parts.append("No temporal data available")
    
    # Add mortality label for training (can be removed for test set)
    if 'visit_info' in patient_data:
        visit = patient_data['visit_info']
        if 'mortality' in visit:
            text_parts.append(f"\n=== OUTCOME ===")
            text_parts.append(f"ICU Mortality: {'Yes' if visit['mortality'] == 1 else 'No'}")
    
    return '\n'.join(text_parts)

def convert_json_to_text_files(input_file, output_dir):
    """Convert patient data with temporal info to text files."""
    
    input_file = Path(input_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectory for patient text files
    text_files_dir = output_dir / 'patient_text_files'
    text_files_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_file.exists():
        print(f"Error: {input_file} not found!")
        return
    
    # Load data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"\nProcessing {len(data)} patients")
    
    # Convert each patient to text
    text_data = []
    for i, patient in enumerate(data):
        text = format_patient_as_text(patient)
        
        # Save individual file to subdirectory
        patient_file = text_files_dir / f"{patient['patient_id']}.txt"
        with open(patient_file, 'w') as f:
            f.write(text)
        
        # Also collect for combined file
        text_data.append({
            'patient_id': patient['patient_id'],
            'text': text,
            'mortality': patient['visit_info'].get('mortality', 0)
        })
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(data)} patients...")
    
    # Save combined file in JSONL format (easy to load for ML)
    jsonl_file = output_dir / 'all_patients.jsonl'
    with open(jsonl_file, 'w') as f:
        for item in text_data:
            f.write(json.dumps(item) + '\n')
    
    # Save as JSON for easy loading
    json_file = output_dir / 'all_patients.json'
    with open(json_file, 'w') as f:
        json.dump(text_data, f, indent=2)
    
    print(f"  Saved {len(data)} patient text files to {text_files_dir}")
    print(f"  Saved combined JSONL to {jsonl_file}")
    print(f"  Saved combined JSON to {json_file}")
    
    # Calculate statistics
    if text_data:
        mortality_count = sum(1 for item in text_data if item['mortality'] == 1)
        print(f"  Statistics: {len(text_data)} patients, {mortality_count} deaths ({mortality_count/len(text_data)*100:.2f}%)")

def create_baseline_prompt_examples(output_dir):
    """Create example prompts for baseline models."""
    
    output_dir = Path(output_dir)
    prompts_file = output_dir / 'baseline_prompts.json'
    
    prompts = {
        "direct_prediction": {
            "system": "You are a medical AI assistant specialized in ICU mortality prediction. Based on the patient's clinical data from the first 48 hours of ICU admission, predict whether the patient will survive or die during this ICU stay.",
            "user_template": "Based on the following ICU patient's clinical data, predict the mortality risk (high or low) and provide a brief explanation:\n\n{patient_text}\n\nPrediction:",
            "few_shot_examples": []
        },
        "rag_enhanced": {
            "system": "You are a medical AI assistant with access to medical literature. Based on the patient's clinical data and relevant medical knowledge, predict ICU mortality risk.",
            "user_template": "Patient Data:\n{patient_text}\n\nRelevant Medical Literature:\n{retrieved_docs}\n\nBased on the patient data and medical literature, predict mortality risk:",
            "few_shot_examples": []
        },
        "similar_cases": {
            "system": "You are a medical AI assistant. You have access to similar historical cases for reference.",
            "user_template": "Current Patient:\n{patient_text}\n\nSimilar Historical Cases:\n{similar_cases}\n\nBased on the current patient and similar cases, predict mortality risk:",
            "few_shot_examples": []
        }
    }
    
    with open(prompts_file, 'w') as f:
        json.dump(prompts, f, indent=2)
    
    print(f"\nBaseline prompts saved to: {prompts_file}")

if __name__ == '__main__':
    # Convert patient data with temporal info to text
    suffix = '_test' if TEST_MODE else '_full'
    input_file = f'/scratch/bcew/ruikez2/intern/s3_med/data/baseline_text/patient_data_with_temporal{suffix}.json'
    output_dir = f'/scratch/bcew/ruikez2/intern/s3_med/data/baseline_text'
    
    print("Converting JSON data to text format...")
    convert_json_to_text_files(input_file, output_dir)
    
    # Create baseline prompt templates
    create_baseline_prompt_examples(output_dir)
    
    print("\n✅ Conversion complete!")
    print(f"Text data saved to: {output_dir}")
    print("\nOutput structure:")
    print(f"  - {output_dir}/patient_text_files/*.txt : Individual patient text files")
    print(f"  - {output_dir}/all_patients.jsonl : Combined data in JSONL format")
    print(f"  - {output_dir}/all_patients.json : Combined data in JSON format")
    print(f"  - {output_dir}/baseline_prompts.json : Prompt templates for baselines")