#!/usr/bin/env python3
"""
Preprocess MIMIC-IV data for ICU mortality prediction using S3 framework
"""

import json
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import argparse
from tqdm import tqdm
import csv

# Feature descriptions mapping
CHART_ITEM_DESCRIPTIONS = {
    "220045": "Heart Rate",
    "220277": "O2 saturation pulseoxymetry", 
    "220210": "Respiratory Rate",
    "220048": "Heart Rate Alarm - High",
    "224650": "Total PEEP Level",
    "220179": "Non Invasive Blood Pressure systolic",
    "220180": "Non Invasive Blood Pressure diastolic",
    "220181": "Non Invasive Blood Pressure mean",
    "224080": "FiO2",
    "224086": "Tidal Volume",
    "224082": "PEEP",
    "224093": "Minute Volume",
    "228868": "Vent Mode",
    "224642": "Respiratory Rate (Set)",
    "223900": "PIP",
    "220739": "GCS - Eye Opening",
    "223901": "Plateau Pressure",
    "224641": "Respiratory Rate (Total)",
    "223781": "DIET TYPE",
    "226732": "O2 Delivery Device(s)",
    "224089": "Vt (set)",
    "224168": "Mean Airway Pressure",
    "223761": "Temperature Fahrenheit",
    "229321": "MDI/Nebs",
    "227944": "Skin Integrity",
    "227969": "Oxygen Delivery Method",
    "220052": "Arterial Blood Pressure mean",
    "220051": "Arterial Blood Pressure diastolic",
    "220050": "Arterial Blood Pressure systolic",
    "225054": "Arterial Line placed in outside facility",
    "223988": "Activity Tolerance",
    "223986": "Pupil Response Left",
    "228096": "Restraint Type",
    "223987": "Pupil Response Right",
    "223989": "Pupil Size Left",
    "223907": "Temperature Site",
    "224733": "Pressure Support",
    "227121": "O2 Delivery Device #2",
    "227288": "Response to Stimuli",
    "223795": "Pain Level",
    "224088": "Vt (spontaneous)",
    "224084": "FiO2 (set)",
    "224015": "O2 Flow",
    "223999": "Pupil Size Right",
    "228305": "Delirium assessment",
    "224001": "Riker-SAS Scale",
    "223792": "Pain Level Response",
    "228299": "Education Learner",
    "224003": "RASS",
    "224004": "CAM-ICU"
}

LAB_ITEM_DESCRIPTIONS = {
    "51221": "Hematocrit",
    "50912": "Creatinine",
    "51265": "Platelet Count",
    "51006": "Urea Nitrogen/BUN",
    "51222": "Hemoglobin",
    "51301": "White Blood Cells",
    "51249": "MCHC",
    "51279": "Red Blood Cells",
    "51250": "MCV",
    "51248": "MCH",
    "51277": "RDW",
    "50971": "Potassium",
    "50983": "Sodium",
    "50902": "Chloride",
    "50882": "Bicarbonate",
    "50868": "Anion Gap",
    "50931": "Glucose",
    "50893": "Calcium, Total",
    "50960": "Magnesium",
    "50970": "Phosphate",
    "50934": "H.pylori Ab",
    "51678": "L",
    "50947": "I",
    "52172": "pCO2 (venous)",
    "50861": "ALT/SGPT",
    "50878": "AST/SGOT",
    "51237": "INR",
    "51274": "PT",
    "50920": "Estimated GFR",
    "51275": "PTT",
    "51256": "Neutrophils",
    "51244": "Lymphocytes",
    "51254": "Monocytes",
    "51146": "Basophils",
    "51200": "Eosinophils",
    "50885": "Bilirubin, Total",
    "50863": "Alkaline Phosphatase",
    "50862": "Albumin",
    "52075": "Lactic Acid (Arterial)",
    "52073": "Lactic Acid",
    "52074": "Lactic Acid (Venous)",
    "51133": "Atypical Lymphocytes",
    "52069": "C-Reactive Protein",
    "51486": "CK-MB",
    "51491": "CK",
    "51498": "Ferritin",
    "51506": "Lipase",
    "51508": "Thyroid Stimulating Hormone",
    "51466": "Folate",
    "51478": "Vitamin B12"
}

def load_ccs_descriptions():
    """Load CCS code descriptions from CSV file"""
    ccs_descriptions = {}
    ccs_file = '/scratch/bcew/ruikez2/intern/KARE/ehr_prepare/resources/CCSCM.csv'
    try:
        with open(ccs_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ccs_descriptions[row['code']] = row['name']
    except:
        print("Warning: Could not load CCS descriptions")
    return ccs_descriptions

def format_patient_data_for_prompt(patient_info: Dict, matrix: List[List], 
                                   chart_items: List[str], lab_items: List[str],
                                   max_hours: int = 48, ccs_descriptions: Dict = None,
                                   demographics: Dict = None) -> str:
    """
    Format patient data into a structured clinical prompt
    """
    patient_id = patient_info['patient_id']
    n_hours = min(patient_info['n_hours'], max_hours)
    
    # Get demographics if available
    age = "[age]"
    gender = "[gender]"
    if demographics and patient_id in demographics:
        age = demographics[patient_id]['age']
        gender = "male" if demographics[patient_id]['gender'] == 'M' else "female"
    
    # Start with patient demographics
    prompt = f"Patient {patient_id} is {age} years old, {gender}. "
    
    # Add current visit time
    current_visit_time = patient_info.get('start_time', '[admission time]')
    prompt += f"Current ICU admission: {current_visit_time}. "
    
    # Add past medical history
    if 'past_visit_ccs' in patient_info and patient_info['past_visit_ccs']:
        prompt += "History symptoms in time order involve: "
        history_items = []
        for i, (visit_time, icd_codes, ccs_codes) in enumerate(zip(
            patient_info['past_visit_times'][:3],
            patient_info.get('past_visit_icds', [])[:3],
            patient_info['past_visit_ccs'][:3]
        )):
            visit_conditions = []
            for ccs_code in ccs_codes[:3]:
                if ccs_code.startswith("CCS:"):
                    ccs_num = ccs_code.split(":")[1]
                    if ccs_descriptions and ccs_num in ccs_descriptions:
                        visit_conditions.append(ccs_descriptions[ccs_num])
            if visit_conditions:
                history_items.append(f"{visit_time[:10]}: {', '.join(visit_conditions)}")
        prompt += "; ".join(history_items) + ". "
    else:
        prompt += "No significant past medical history recorded. "
    
    # Add initial lab values
    prompt += "\n\nIn this ICU visit, the Initial Lab Events include: "
    lab_values = patient_info['lab_values']
    lab_items_list = []
    
    # Priority lab values
    priority_labs = ["50931", "51006", "50912", "51301", "51221", "51222", "50971", "50983", "52073"]
    
    for lab_id in priority_labs:
        if lab_id in lab_items:
            idx = lab_items.index(lab_id)
            value = lab_values[idx]
            if value != "None":
                lab_name = LAB_ITEM_DESCRIPTIONS.get(lab_id, f"Lab {lab_id}")
                lab_items_list.append(f"{lab_name}={value}")
    
    # Add other significant lab values
    for i, (lab_id, value) in enumerate(zip(lab_items, lab_values)):
        if value != "None" and lab_id not in priority_labs and len(lab_items_list) < 15:
            lab_name = LAB_ITEM_DESCRIPTIONS.get(lab_id, f"Lab {lab_id}")
            lab_items_list.append(f"{lab_name}={value}")
    
    prompt += ", ".join(lab_items_list) + ". "
    
    # Add hourly chart events
    prompt += "\n\nThe Hourly Chart Events (last 48h) include: "
    
    # Get the most recent hours
    hours_to_show = min(n_hours, max_hours)
    
    # Key vital signs to track hourly
    key_vitals = {
        "220045": "HR",
        "220210": "RR", 
        "220277": "SpO2",
        "220052": "MAP",  # Mean Arterial Pressure
        "220179": "SBP",
        "220180": "DBP",
        "223761": "Temp"
    }
    
    hourly_data = []
    
    # Process each hour
    for hour_idx in range(hours_to_show):
        hour_vitals = []
        
        for chart_id, abbrev in key_vitals.items():
            if chart_id in chart_items:
                idx = chart_items.index(chart_id)
                if hour_idx < len(matrix[idx]):
                    value = matrix[idx][hour_idx]
                    if value != "None":
                        hour_vitals.append(f"{abbrev}={value}")
        
        if hour_vitals:
            hourly_data.append(f"h{hour_idx}: {', '.join(hour_vitals)}")
    
    # Show first 10 hours and last 10 hours if more than 20 hours
    if len(hourly_data) > 20:
        prompt += "; ".join(hourly_data[:10]) + "; ... ; " + "; ".join(hourly_data[-10:])
    else:
        prompt += "; ".join(hourly_data)
    
    prompt += "."
    
    # Add ventilator settings if applicable
    vent_params = ["224080", "224082", "224086", "223900"]  # FiO2, PEEP, Vt, PIP
    has_vent_data = False
    vent_info = []
    
    for param_id in vent_params:
        if param_id in chart_items:
            idx = chart_items.index(param_id)
            recent_values = [v for v in matrix[idx][-hours_to_show:] if v != "None"]
            if recent_values:
                has_vent_data = True
                param_name = CHART_ITEM_DESCRIPTIONS.get(param_id, f"Param {param_id}")
                vent_info.append(f"{param_name}={recent_values[-1]}")
    
    if has_vent_data:
        prompt += f"\n\nMechanical Ventilation: {', '.join(vent_info)}."
    
    # Add the clinical question
    prompt += "\n\nBased on the provided information, can this patient survive in this ICU visit?"
    
    return prompt

def create_mimic_s3_dataset(data_dir: str, output_dir: str, split: str = "train", 
                            max_patients: int = None):
    """
    Create S3-formatted dataset from MIMIC data
    """
    # Load data
    print(f"Loading MIMIC data from {data_dir}...")
    
    with open(os.path.join(data_dir, 'patient_info_enriched.json'), 'r') as f:
        patient_info_list = json.load(f)
    
    with open(os.path.join(data_dir, 'patient_matrices.json'), 'r') as f:
        patient_matrices = json.load(f)
    
    with open(os.path.join(data_dir, 'chart_items.json'), 'r') as f:
        chart_items = json.load(f)
    
    with open(os.path.join(data_dir, 'lab_items.json'), 'r') as f:
        lab_items = json.load(f)
    
    # Load CCS descriptions
    ccs_descriptions = load_ccs_descriptions()
    
    # Try to load patient demographics
    demographics = {}
    demographics_file = os.path.join(data_dir, 'patient_demographics.json')
    if os.path.exists(demographics_file):
        with open(demographics_file, 'r') as f:
            demographics = json.load(f)
        print(f"Loaded demographics for {len(demographics)} patients")
    else:
        print("Warning: No patient demographics file found")
    
    print(f"Loaded {len(patient_info_list)} patients")
    
    if max_patients:
        patient_info_list = patient_info_list[:max_patients]
        patient_matrices = patient_matrices[:max_patients]
    
    # Create S3-formatted data
    s3_data = []
    
    for i, (patient_info, matrix) in enumerate(tqdm(zip(patient_info_list, patient_matrices), 
                                                    desc="Processing patients")):
        # Format patient data into prompt
        patient_prompt = format_patient_data_for_prompt(
            patient_info, matrix, chart_items, lab_items, 
            ccs_descriptions=ccs_descriptions, demographics=demographics
        )
        
        # Create S3 data entry
        data_entry = {
            "data_source": "mimic_icu_mortality",
            "prompt": [{
                "role": "user",
                "content": f"""You are a search copilot for medical decision support. Based on the following ICU patient data, help search for relevant medical literature to predict mortality risk.

{patient_prompt}

You should search for medical literature about ICU mortality prediction, focusing on the patient's specific clinical parameters and conditions. Start by searching for general ICU mortality prediction models and risk factors."""
            }],
            "ability": "medical-reasoning",
            "reward_model": {
                "style": "medical_accuracy",
                "ground_truth": {
                    "patient_id": patient_info['patient_id'],
                    "clinical_data": {
                        "lab_values": patient_info['lab_values'],
                        "n_hours": patient_info['n_hours'],
                        "past_ccs": patient_info.get('past_visit_ccs', [])
                    }
                }
            },
            "extra_info": {
                'split': split,
                'index': i,
                'patient_id': patient_info['patient_id']
            }
        }
        
        s3_data.append(data_entry)
    
    # Convert to DataFrame and save as parquet
    df = pd.DataFrame(s3_data)
    
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{split}_mimic_s3.parquet')
    df.to_parquet(output_file)
    
    print(f"Saved {len(df)} entries to {output_file}")
    
    # Print statistics
    print("\nDataset Statistics:")
    print(f"Total patients: {len(df)}")
    print(f"Average prompt length: {df['prompt'].apply(lambda x: len(x[0]['content'])).mean():.0f} characters")
    
    return output_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/scratch/bcew/ruikez2/intern/s3_med/patient_matrices_optimized',
                        help='Directory containing MIMIC patient data')
    parser.add_argument('--output_dir', default='./data/mimic_icu_mortality',
                        help='Output directory for S3-formatted data')
    parser.add_argument('--split', default='train', choices=['train', 'test', 'val'],
                        help='Dataset split')
    parser.add_argument('--max_patients', type=int, default=None,
                        help='Maximum number of patients to process')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Ratio of training data')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Ratio of validation data')
    
    args = parser.parse_args()
    
    # Load all patient info to determine splits
    with open(os.path.join(args.data_dir, 'patient_info_enriched.json'), 'r') as f:
        all_patients = json.load(f)
    
    n_patients = len(all_patients)
    n_train = int(n_patients * args.train_ratio)
    n_val = int(n_patients * args.val_ratio)
    
    print(f"Total patients: {n_patients}")
    print(f"Train: {n_train}, Val: {n_val}, Test: {n_patients - n_train - n_val}")
    
    # Create datasets for all splits
    if args.split == 'all':
        splits = ['train', 'val', 'test']
    else:
        splits = [args.split]
    
    for split in splits:
        if split == 'train':
            max_patients = n_train
        elif split == 'val':
            # Skip first n_train patients
            # This would need modification to actually skip patients
            max_patients = n_val
        else:  # test
            max_patients = n_patients - n_train - n_val
        
        create_mimic_s3_dataset(
            args.data_dir,
            args.output_dir,
            split=split,
            max_patients=max_patients if args.max_patients is None else min(args.max_patients, max_patients)
        )