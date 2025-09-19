#!/usr/bin/env python3
"""
Preprocess MIMIC ICU mortality prediction data for S3 training
"""

import re
import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List
import argparse
from tqdm import tqdm
import csv

# Feature descriptions mapping (same as in preprocess_mimic_for_s3.py)
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

def format_patient_summary(patient_info: Dict, matrix: List[List], 
                          chart_items: List[str], lab_items: List[str],
                          ccs_descriptions: Dict = None) -> str:
    """
    Create a concise patient summary for the search query
    """
    # Extract key abnormal values
    abnormal_labs = []
    lab_values = patient_info['lab_values']
    
    # Check for critical lab values
    critical_labs = {
        "50912": ("Creatinine", 1.2, float('inf')),  # High creatinine
        "51006": ("BUN", 20, float('inf')),  # High BUN
        "50971": ("Potassium", 5.0, float('inf')),  # High K
        "50931": ("Glucose", 180, float('inf')),  # High glucose
        "51301": ("WBC", 12000, float('inf')),  # High WBC
        "52073": ("Lactate", 2.0, float('inf')),  # High lactate
    }
    
    for lab_id, (lab_name, threshold, upper) in critical_labs.items():
        if lab_id in lab_items:
            idx = lab_items.index(lab_id)
            value = lab_values[idx]
            if value != "None":
                try:
                    val = float(value)
                    if val > threshold:
                        abnormal_labs.append(f"{lab_name}={val}")
                except:
                    pass
    
    # Check vital signs
    abnormal_vitals = []
    vital_checks = {
        "220045": ("HR", 100, 150),  # Tachycardia
        "220210": ("RR", 20, 30),  # Tachypnea
        "220277": ("SpO2", 0, 92),  # Hypoxemia
        "220179": ("SBP", 0, 90),  # Hypotension
    }
    
    for vital_id, (vital_name, low, high) in vital_checks.items():
        if vital_id in chart_items:
            idx = chart_items.index(vital_id)
            recent_values = [v for v in matrix[idx][-24:] if v != "None"]
            if recent_values:
                try:
                    numeric_vals = [float(v) for v in recent_values]
                    avg_val = np.mean(numeric_vals)
                    if avg_val < low or avg_val > high:
                        abnormal_vitals.append(f"{vital_name}={avg_val:.1f}")
                except:
                    pass
    
    # Check if on mechanical ventilation
    on_vent = False
    if "224080" in chart_items:  # FiO2
        idx = chart_items.index("224080")
        if any(v != "None" for v in matrix[idx][-24:]):
            on_vent = True
    
    # Build summary
    summary_parts = []
    if abnormal_labs:
        summary_parts.append(f"abnormal labs: {', '.join(abnormal_labs[:3])}")
    if abnormal_vitals:
        summary_parts.append(f"vital signs: {', '.join(abnormal_vitals[:3])}")
    if on_vent:
        summary_parts.append("mechanical ventilation")
    if patient_info.get('past_visit_ccs'):
        # Get unique CCS codes from past visits with descriptions
        all_ccs = []
        for ccs_list in patient_info['past_visit_ccs'][:2]:
            for ccs_code in ccs_list[:2]:
                if ccs_code.startswith("CCS:"):
                    ccs_num = ccs_code.split(":")[1]
                    if ccs_descriptions and ccs_num in ccs_descriptions:
                        all_ccs.append(f"{ccs_descriptions[ccs_num]}")
                    else:
                        all_ccs.append(ccs_code)
        if all_ccs:
            summary_parts.append(f"history: {', '.join(all_ccs[:3])}")
    
    return "ICU patient with " + "; ".join(summary_parts) if summary_parts else "ICU patient"

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

For a patient:
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
    
    input_str += patient_data['clinical_summary'] + """
</patient_data>
"""
    return input_str

def create_dataset_entry(patient_info: Dict, matrix: List[List], 
                        chart_items: List[str], lab_items: List[str],
                        index: int, split: str, ccs_descriptions: Dict = None) -> Dict:
    """
    Create a single dataset entry in S3 format
    """
    # Create clinical summary
    clinical_summary = format_patient_summary(patient_info, matrix, chart_items, lab_items, ccs_descriptions)
    
    # Create patient data dict
    patient_data = {
        'clinical_summary': clinical_summary,
        'patient_id': patient_info['patient_id']
    }
    
    # Create the prompt
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
            "style": "medical_evaluation",
            "ground_truth": {
                "patient_id": patient_info['patient_id'],
                "clinical_summary": clinical_summary,
                "key_values": {
                    "lab_count": len([v for v in patient_info['lab_values'] if v != "None"]),
                    "icu_hours": patient_info['n_hours'],
                    "has_history": len(patient_info.get('past_visit_ccs', [])) > 0
                }
            }
        },
        "extra_info": {
            'split': split,
            'index': index,
            'patient_id': patient_info['patient_id']
        }
    }
    
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/scratch/bcew/ruikez2/intern/s3_med/patient_matrices_optimized')
    parser.add_argument('--output_dir', default='./data/mimic_mortality')
    parser.add_argument('--retriever', default="e5")
    parser.add_argument('--max_patients', type=int, default=None)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading MIMIC data from {args.data_dir}...")
    
    with open(os.path.join(args.data_dir, 'patient_info_enriched.json'), 'r') as f:
        patient_info_list = json.load(f)
    
    with open(os.path.join(args.data_dir, 'patient_matrices.json'), 'r') as f:
        patient_matrices = json.load(f)
    
    with open(os.path.join(args.data_dir, 'chart_items.json'), 'r') as f:
        chart_items = json.load(f)
    
    with open(os.path.join(args.data_dir, 'lab_items.json'), 'r') as f:
        lab_items = json.load(f)
    
    # Load CCS descriptions
    ccs_descriptions = load_ccs_descriptions()
    
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
            entry = create_dataset_entry(
                patient_info_list[i],
                patient_matrices[i],
                chart_items,
                lab_items,
                i - start_idx,
                split_name,
                ccs_descriptions
            )
            dataset_entries.append(entry)
        
        # Save to parquet
        df = pd.DataFrame(dataset_entries)
        output_file = os.path.join(args.output_dir, f'{split_name}_{args.retriever}_ug.parquet')
        df.to_parquet(output_file)
        
        print(f"Saved {len(df)} entries to {output_file}")
        
        # Print sample
        if len(df) > 0:
            print(f"\nSample prompt (first 500 chars):")
            print(df.iloc[0]['prompt'][0]['content'][:500])
            print("...")

if __name__ == '__main__':
    main()