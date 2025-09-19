#!/usr/bin/env python3
"""
Preprocess MIMIC ICU mortality prediction data for S3 training
Updated version to work with new data format (patient_info.json and chartevents_matrices.json)
"""

import re
import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List
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

def extract_patient_features(patient_info: Dict, matrix: List[List], current_diagnoses: List[Dict] = None) -> Dict:
    """
    Extract key features from patient data for search query generation
    """
    features = {
        'abnormal_labs': [],
        'abnormal_vitals': [],
        'on_ventilation': False,
        'has_history': False,
        'risk_factors': [],
        'current_diagnoses': current_diagnoses or []
    }
    
    # Check for past medical history
    if patient_info.get('past_ccs_codes'):
        features['has_history'] = True
        features['risk_factors'].extend(patient_info['past_ccs_codes'][:3])
    
    # Analyze matrix data
    if matrix and len(matrix) > 0:
        # Check glucose levels
        glucose_idx = INDICATOR_NAMES.index('Glucose')
        if glucose_idx < len(matrix):
            glucose_values = [v for v in matrix[glucose_idx] if v is not None]
            if glucose_values:
                avg_glucose = np.mean([float(v) for v in glucose_values])
                if avg_glucose > 180:
                    features['abnormal_labs'].append(f"hyperglycemia (avg glucose={avg_glucose:.0f})")
        
        # Check pH
        ph_idx = INDICATOR_NAMES.index('pH')
        if ph_idx < len(matrix):
            ph_values = [v for v in matrix[ph_idx] if v is not None]
            if ph_values:
                avg_ph = np.mean([float(v) for v in ph_values])
                if avg_ph < 7.35:
                    features['abnormal_labs'].append(f"acidosis (pH={avg_ph:.2f})")
                elif avg_ph > 7.45:
                    features['abnormal_labs'].append(f"alkalosis (pH={avg_ph:.2f})")
        
        # Check vital signs
        hr_idx = INDICATOR_NAMES.index('Heart Rate')
        if hr_idx < len(matrix):
            hr_values = [v for v in matrix[hr_idx] if v is not None]
            if hr_values:
                avg_hr = np.mean([float(v) for v in hr_values])
                if avg_hr > 100:
                    features['abnormal_vitals'].append(f"tachycardia (HR={avg_hr:.0f})")
        
        # Check oxygen saturation
        spo2_idx = INDICATOR_NAMES.index('Oxygen saturation')
        if spo2_idx < len(matrix):
            spo2_values = [v for v in matrix[spo2_idx] if v is not None]
            if spo2_values:
                avg_spo2 = np.mean([float(v) for v in spo2_values])
                if avg_spo2 < 90:
                    features['abnormal_vitals'].append(f"hypoxemia (SpO2={avg_spo2:.0f}%)")
        
        # Check if on ventilation
        fio2_idx = INDICATOR_NAMES.index('Fraction inspired oxygen')
        if fio2_idx < len(matrix):
            fio2_values = [v for v in matrix[fio2_idx] if v is not None]
            if fio2_values and any(float(v) > 0.21 for v in fio2_values):
                features['on_ventilation'] = True
        
        # Check GCS
        gcs_idx = INDICATOR_NAMES.index('Glascow coma scale total')
        if gcs_idx < len(matrix):
            gcs_values = [v for v in matrix[gcs_idx] if v is not None]
            if gcs_values:
                avg_gcs = np.mean([float(v) for v in gcs_values])
                if avg_gcs < 15:
                    features['risk_factors'].append(f"neurological impairment (GCS={avg_gcs:.0f})")
    
    return features

def format_patient_summary(patient_info: Dict, features: Dict) -> str:
    """
    Create a concise patient summary for the search query
    """
    age = patient_info['age']
    gender = "male" if patient_info['gender'] == 'M' else "female"
    
    summary_parts = [f"{age} year old {gender} ICU patient"]
    
    if features['abnormal_labs']:
        summary_parts.append(f"with {', '.join(features['abnormal_labs'][:2])}")
    
    if features['abnormal_vitals']:
        summary_parts.append(f"{', '.join(features['abnormal_vitals'][:2])}")
    
    if features['on_ventilation']:
        summary_parts.append("on mechanical ventilation")
    
    if features['has_history']:
        history_str = ', '.join(features['risk_factors'][:2])
        summary_parts.append(f"history of {history_str}")
    
    return " ".join(summary_parts)

def format_complete_patient_data(patient_info: Dict, matrix: List[List], current_diagnoses: List[Dict] = None) -> str:
    """
    Format complete patient data including all clinical information
    """
    # Basic demographics
    age = patient_info['age']
    gender = "Male" if patient_info['gender'] == 'M' else "Female"
    
    output = []
    output.append(f"Patient Demographics:")
    output.append(f"- Age: {age} years")
    output.append(f"- Gender: {gender}")
    output.append(f"- Patient ID: {patient_info['patient_id']}")
    
    # Past medical history (before admission)
    if patient_info.get('past_ccs_codes') or patient_info.get('past_icd_codes'):
        output.append(f"\nPast Medical History (Before Admission):")
        # Include both ICD codes and descriptions
        if patient_info.get('past_icd_codes') and patient_info.get('past_ccs_codes'):
            for icd_code, ccs_desc in zip(patient_info['past_icd_codes'], patient_info['past_ccs_codes']):
                output.append(f"- [{icd_code}] {ccs_desc}")
        elif patient_info.get('past_ccs_codes'):
            for condition in patient_info['past_ccs_codes']:
                output.append(f"- {condition}")
    
    # Current admission diagnoses
    if current_diagnoses:
        output.append(f"\nCurrent Admission Diagnoses:")
        for diag in current_diagnoses[:10]:  # Limit to first 10 diagnoses
            output.append(f"- [{diag['ICD9_CODE']}] {diag['LONG_TITLE']}")
    
    # Clinical measurements (time series data)
    output.append(f"\nClinical Measurements (Time Series):")
    
    # Format the matrix data with indicator names - show all raw values
    for i, indicator_name in enumerate(INDICATOR_NAMES):
        if i < len(matrix) and matrix[i]:
            values = matrix[i]
            # Filter out None values
            valid_values = [v for v in values if v is not None]
            if valid_values:
                output.append(f"\n{indicator_name}:")
                # Show all raw values
                output.append(f"  Values: {', '.join(str(v) for v in valid_values)}")
    
    return "\n".join(output)

def make_prefix(patient_data: Dict, retriever: str = "e5") -> str:
    """
    Create the search agent prompt for MIMIC patient data
    """
    input_str = """You are a search copilot for medical literature retrieval. Based on an ICU patient's clinical data, you will help find relevant medical literature about mortality risk prediction.

You will go through a loop of <think> -> <query> -> <information> -> <think> -> <important_info> -> <search_complete> -> <query> (if not complete) ..., to gather relevant medical literature.

You should show your thinking process between <think> and </think>. You should show the search query between <query> and </query> in JSON format.
Based on the search query, we will return the top searched medical literature between <information> and </information>. You need to first think (<think>) on the retrieved information and put the doc id (1, 2, 3) of the important documents between <important_info> and </important_info> (e.g., <important_info>[1, 2]</important_info>).
After reviewing the information, you must decide whether to continue searching with a new query or indicate that the search is complete. If you need more information, formulate a new search query OR use <search_complete>False</search_complete> to indicate you want to continue searching. If you have sufficient information, use <search_complete>True</search_complete> to indicate that you have gathered enough information.

Focus on finding literature about:
1. ICU mortality prediction models and risk factors
2. Clinical parameters associated with ICU mortality
3. Specific conditions or abnormalities present in the patient
4. Prognostic indicators for critically ill patients

Based on the following ICU patient's clinical data, predict the mortality risk (high or low) and find relevant medical literature to support the assessment.

<patient_data>
[patient clinical information]
</patient_data>

The loop is as follows:
<think>
[analyze patient data and identify key risk factors]
</think>
<query>
{
    "query": "[search query for medical literature]"
} 
</query>
<information>
[top searched medical literature]
</information>
<think>
[analyze the search results for relevance to patient]
</think>
<important_info>
[doc ids of relevant papers]
</important_info>
<search_complete>
False
</search_complete>
<query>
{
    "query": "[refined search query]"
}
</query>
...... (several turns, max 4 turns in total)

<search_complete>
True
</search_complete>

Now, start the loop with the following patient data:
<patient_data>
"""
    
    # Use complete patient data instead of summary
    input_str += patient_data['complete_data'] + """
</patient_data>
"""
    return input_str

def make_prefix_with_question(patient_data: Dict, retriever: str = "e5") -> str:
    """
    Create prompt with explicit question and complete patient data
    """
    input_str = """<question>
Based on the following ICU patient's clinical data, predict the mortality risk (high or low) and find relevant medical literature to support the assessment.
</question>

<patient_data>
"""
    
    input_str += patient_data['complete_data'] + """
</patient_data>
"""
    return input_str

def create_dataset_entry(patient_info: Dict, matrix: List[List], 
                        index: int, split: str, current_diagnoses: List[Dict] = None,
                        use_complete_data: bool = True) -> Dict:
    """
    Create a single dataset entry in S3 format
    """
    # Extract features
    features = extract_patient_features(patient_info, matrix, current_diagnoses)
    
    # Create clinical summary (still needed for ground truth)
    clinical_summary = format_patient_summary(patient_info, features)
    
    # Format complete patient data
    complete_patient_data = format_complete_patient_data(patient_info, matrix, current_diagnoses)
    
    # Create patient data dict
    patient_data = {
        'clinical_summary': clinical_summary,
        'complete_data': complete_patient_data,
        'patient_id': patient_info['patient_id']
    }
    
    # Create the prompt - always use make_prefix to include S3 format instructions
    prompt_content = make_prefix(patient_data)
    
    # Create the data entry
    data = {
        "data_source": "mimic_icu_mortality",
        "prompt": [{
            "role": "user",
            "content": prompt_content,
        }],
        "ability": "medical-reasoning",
        "reward_model": {
            "style": "rule",
            "ground_truth": {
                "patient_id": patient_info['patient_id'],
                "question": clinical_summary,
                "answers": ["high mortality risk" if patient_info['mortality'] else "low mortality risk"],
                "mortality": patient_info['mortality'],
                "mortality_inunit": patient_info['mortality_inunit']
            }
        },
        "extra_info": {
            'split': split,
            'index': index,
            'patient_id': patient_info['patient_id'],
            'clinical_summary': clinical_summary  # Store summary for reference
        }
    }
    
    return data

def load_current_diagnoses(data_dir: str) -> Dict[str, List[Dict]]:
    """
    Load current admission diagnoses from all_diagnoses.csv
    Returns a dictionary mapping patient_id to list of diagnoses
    """
    diagnoses_file = os.path.join(data_dir, 'mimic4_benchmark_output', 'all_diagnoses.csv')
    diagnoses_dict = {}
    
    if os.path.exists(diagnoses_file):
        df_diagnoses = pd.read_csv(diagnoses_file)
        
        # Group by SUBJECT_ID and ICUSTAY_ID
        for _, row in df_diagnoses.iterrows():
            # Create patient_id in the same format as patient_info.json
            patient_id = f"{row['SUBJECT_ID']}_{row['ICUSTAY_ID']}"
            
            if patient_id not in diagnoses_dict:
                diagnoses_dict[patient_id] = []
            
            diagnoses_dict[patient_id].append({
                'ICD9_CODE': row['ICD9_CODE'],
                'LONG_TITLE': row['LONG_TITLE'],
                'SEQ_NUM': row.get('SEQ_NUM', 999)  # Some entries might not have SEQ_NUM
            })
        
        # Sort diagnoses by sequence number for each patient
        for patient_id in diagnoses_dict:
            diagnoses_dict[patient_id].sort(key=lambda x: x['SEQ_NUM'])
        
        print(f"Loaded diagnoses for {len(diagnoses_dict)} patients")
    else:
        print(f"Warning: Diagnoses file not found at {diagnoses_file}")
    
    return diagnoses_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/scratch/bcew/ruikez2/intern/s3_med/data',
                       help='Directory containing patient_info.json and chartevents_matrices.json')
    parser.add_argument('--output_dir', default='./data/mimic_mortality_ug')
    parser.add_argument('--retriever', default="e5")
    parser.add_argument('--max_patients', type=int, default=None)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--use_complete_data', action='store_true', default=True,
                       help='Use complete patient data instead of summary')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading MIMIC data from {args.data_dir}...")
    
    with open(os.path.join(args.data_dir, 'patient_info.json'), 'r') as f:
        patient_info_list = json.load(f)
    
    with open(os.path.join(args.data_dir, 'chartevents_matrices.json'), 'r') as f:
        patient_matrices = json.load(f)
    
    # Load current admission diagnoses
    current_diagnoses_dict = load_current_diagnoses(args.data_dir)
    
    # Limit patients if specified
    if args.max_patients:
        patient_info_list = patient_info_list[:args.max_patients]
        patient_matrices = patient_matrices[:args.max_patients]
    
    n_patients = len(patient_info_list)
    n_train = int(n_patients * args.train_ratio)
    n_val = int(n_patients * args.val_ratio)
    
    print(f"Total patients: {n_patients}")
    print(f"Train: {n_train}, Val: {n_val}, Test: {n_patients - n_train - n_val}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each split
    splits = {
        'train': (0, n_train),
        'val': (n_train, n_train + n_val),
        'test': (n_train + n_val, n_patients)
    }
    
    for split_name, (start_idx, end_idx) in splits.items():
        print(f"\nProcessing {split_name} split...")
        
        dataset_entries = []
        
        for i in tqdm(range(start_idx, end_idx), desc=f"Creating {split_name} entries"):
            patient_id = patient_info_list[i]['patient_id']
            current_diagnoses = current_diagnoses_dict.get(patient_id, [])
            
            entry = create_dataset_entry(
                patient_info_list[i],
                patient_matrices[i],
                i - start_idx,
                split_name,
                current_diagnoses=current_diagnoses,
                use_complete_data=args.use_complete_data
            )
            dataset_entries.append(entry)
        
        # Save to parquet
        df = pd.DataFrame(dataset_entries)
        output_file = os.path.join(args.output_dir, f'{split_name}_{args.retriever}_ug.parquet')
        df.to_parquet(output_file)
        
        print(f"Saved {len(df)} entries to {output_file}")
        
        # Print sample
        if len(df) > 0:
            print(f"\nSample prompt (first 5000 chars):")
            print(df.iloc[0]['prompt'][0]['content'][:5000])
            print("...")
            
            # Print mortality statistics
            mortality_count = sum(1 for i in range(start_idx, end_idx) 
                                 if patient_info_list[i]['mortality'])
            print(f"Mortality rate: {mortality_count}/{end_idx-start_idx} ({mortality_count/(end_idx-start_idx)*100:.1f}%)")

if __name__ == '__main__':
    main()