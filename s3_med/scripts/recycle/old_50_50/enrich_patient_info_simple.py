#!/usr/bin/env python3
"""
Enrich patient info with past hospital visits - simplified version
Only adds three lists: past_visit_times, past_visit_icds, past_visit_ccs
"""

import json
import gzip
from datetime import datetime
from collections import defaultdict
import os
import csv
from tqdm import tqdm
import re

def load_icd_to_ccs_mappings():
    """Load both ICD9 and ICD10 to CCS mappings"""
    icd_to_ccs = {}
    
    # Load ICD9 mappings
    icd9_file = '/scratch/bcew/ruikez2/intern/KARE/kg_construct/resources/ICD9CM_to_CCSCM.csv'
    try:
        with open(icd9_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                icd_to_ccs[f"ICD9:{row['ICD9CM']}"] = row['CCSCM']
    except:
        print("Warning: Could not load ICD9 mappings")
    
    # Load ICD10 mappings
    icd10_file = '/scratch/bcew/ruikez2/intern/KARE/ehr_prepare/resources/ICD10CM_to_CCSCM.csv'
    try:
        with open(icd10_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                icd_to_ccs[f"ICD10:{row['ICD10CM']}"] = row['CCSCM']
    except:
        print("Warning: Could not load ICD10 mappings")
    
    return icd_to_ccs

def convert_icd_to_ccs(icd_code, icd_version, icd_to_ccs):
    """Convert ICD to CCS with various format attempts"""
    # Direct lookup
    key = f"ICD{icd_version}:{icd_code}"
    if key in icd_to_ccs:
        return icd_to_ccs[key]
    
    # Try with/without decimal points
    if icd_version == 9 and len(icd_code) > 3 and '.' not in icd_code:
        key_with_dot = f"ICD9:{icd_code[:3]}.{icd_code[3:]}"
        if key_with_dot in icd_to_ccs:
            return icd_to_ccs[key_with_dot]
    elif icd_version == 10:
        # Remove dots
        clean_code = icd_code.replace('.', '')
        if f"ICD10:{clean_code}" in icd_to_ccs:
            return icd_to_ccs[f"ICD10:{clean_code}"]
        # Add dot after 3rd char
        if len(icd_code) > 3 and '.' not in icd_code:
            formatted = f"ICD10:{icd_code[:3]}.{icd_code[3:]}"
            if formatted in icd_to_ccs:
                return icd_to_ccs[formatted]
    
    return None

def main():
    # Load patient info
    input_file = '/scratch/bcew/ruikez2/intern/s3_med/patient_matrices_optimized/patient_info.json'
    output_file = '/scratch/bcew/ruikez2/intern/s3_med/patient_matrices_optimized/patient_info_enriched.json'
    
    print("Loading patient info...")
    with open(input_file, 'r') as f:
        patient_info = json.load(f)
    print(f"Loaded {len(patient_info)} patients")
    
    # Load mappings
    print("Loading ICD to CCS mappings...")
    icd_to_ccs = load_icd_to_ccs_mappings()
    print(f"Loaded {len(icd_to_ccs)} mappings")
    
    # Load data files
    data_dir = '/scratch/bcew/ruikez2/intern/s3_med/data/mimiciv/3.1'
    
    # Build subject to hadm mapping from patient info
    subject_to_current_hadm = {}
    for patient in patient_info:
        subject_id = patient['patient_id'].split('_')[0]
        if 'hadm_id' in patient:
            subject_to_current_hadm[subject_id] = patient['hadm_id']
    
    # Load admissions
    print("\nLoading admissions...")
    admissions = defaultdict(list)  # subject_id -> list of (hadm_id, admit_time)
    
    with gzip.open(os.path.join(data_dir, 'hosp/admissions.csv.gz'), 'rt') as f:
        header = f.readline().strip().split(',')
        subject_idx = header.index('subject_id')
        hadm_idx = header.index('hadm_id')
        admittime_idx = header.index('admittime')
        
        for line in tqdm(f, desc="Reading admissions"):
            parts = line.strip().split(',')
            subject_id = parts[subject_idx]
            if subject_id in subject_to_current_hadm:
                hadm_id = parts[hadm_idx]
                admit_time = parts[admittime_idx]
                admissions[subject_id].append((hadm_id, admit_time))
    
    # Load diagnoses
    print("\nLoading diagnoses...")
    diagnoses = defaultdict(lambda: defaultdict(set))  # subject_id -> hadm_id -> set of icd:version
    
    with gzip.open(os.path.join(data_dir, 'hosp/diagnoses_icd.csv.gz'), 'rt') as f:
        header = f.readline().strip().split(',')
        subject_idx = header.index('subject_id')
        hadm_idx = header.index('hadm_id')
        icd_code_idx = header.index('icd_code')
        icd_version_idx = header.index('icd_version')
        
        for line in tqdm(f, desc="Reading diagnoses"):
            parts = line.strip().split(',')
            subject_id = parts[subject_idx]
            if subject_id in subject_to_current_hadm:
                hadm_id = parts[hadm_idx]
                icd_code = parts[icd_code_idx]
                icd_version = parts[icd_version_idx]
                diagnoses[subject_id][hadm_id].add(f"{icd_code}:{icd_version}")
    
    # Process each patient
    print("\nProcessing patients...")
    enriched_count = 0
    
    for patient in tqdm(patient_info):
        subject_id = patient['patient_id'].split('_')[0]
        current_hadm = patient.get('hadm_id')
        
        if not current_hadm or subject_id not in admissions:
            # Add empty lists
            patient['past_visit_times'] = []
            patient['past_visit_icds'] = []
            patient['past_visit_ccs'] = []
            continue
        
        # Get current admission time
        current_time = None
        for hadm_id, admit_time in admissions[subject_id]:
            if hadm_id == current_hadm:
                try:
                    current_time = datetime.strptime(admit_time, '%Y-%m-%d %H:%M:%S')
                except:
                    pass
                break
        
        if not current_time:
            patient['past_visit_times'] = []
            patient['past_visit_icds'] = []
            patient['past_visit_ccs'] = []
            continue
        
        # Collect past visits
        past_visits = []
        for hadm_id, admit_time in admissions[subject_id]:
            if hadm_id != current_hadm:
                try:
                    visit_time = datetime.strptime(admit_time, '%Y-%m-%d %H:%M:%S')
                    if visit_time < current_time:
                        past_visits.append((hadm_id, admit_time))
                except:
                    continue
        
        # Sort by time
        past_visits.sort(key=lambda x: x[1], reverse=True)
        
        # Collect data
        past_visit_times = []
        past_visit_icds = []
        past_visit_ccs = []
        
        for hadm_id, admit_time in past_visits:
            past_visit_times.append(admit_time)
            
            # Get ICDs for this visit
            visit_icds = []
            visit_ccs = []
            
            if hadm_id in diagnoses[subject_id]:
                for icd_version_pair in diagnoses[subject_id][hadm_id]:
                    icd_code, icd_version = icd_version_pair.split(':')
                    visit_icds.append(f"ICD{icd_version}:{icd_code}")
                    
                    # Convert to CCS
                    ccs = convert_icd_to_ccs(icd_code, int(icd_version), icd_to_ccs)
                    if ccs:
                        visit_ccs.append(f"CCS:{ccs}")
            
            past_visit_icds.append(sorted(visit_icds))
            past_visit_ccs.append(sorted(list(set(visit_ccs))))  # Remove duplicates
        
        # Add to patient
        patient['past_visit_times'] = past_visit_times
        patient['past_visit_icds'] = past_visit_icds
        patient['past_visit_ccs'] = past_visit_ccs
        
        if past_visit_times:
            enriched_count += 1
    
    # Save to new file with compact lab_values format
    print("\nSaving enriched data...")
    
    # First convert to JSON string
    json_str = json.dumps(patient_info, indent=2)
    
    # Fix lab_values formatting to match original compact style
    # Pattern: match arrays with two elements where second is a number
    def compact_lab_value(match):
        # Extract the content and make it compact
        content = match.group(0)
        # Remove extra whitespace and newlines
        compact = re.sub(r'\s*\n\s*', ' ', content)
        compact = re.sub(r'\s+', ' ', compact)
        compact = compact.replace('[ ', '[').replace(' ]', ']')
        return compact
    
    # Match lab value arrays (string followed by number)
    pattern = r'\[\s*"[^"]*"\s*,\s*\d+\s*\]'
    json_str = re.sub(pattern, compact_lab_value, json_str)
    
    with open(output_file, 'w') as f:
        f.write(json_str)
    
    print(f"\nEnriched data saved to: {output_file}")
    print(f"Enriched {enriched_count} patients with past visit data")
    
    # Show sample
    for patient in patient_info:
        if patient.get('past_visit_times'):
            print(f"\nSample patient: {patient['patient_id']}")
            print(f"  Past visits: {len(patient['past_visit_times'])}")
            if patient['past_visit_times']:
                print(f"  Most recent past visit: {patient['past_visit_times'][0]}")
                print(f"  ICDs in that visit: {patient['past_visit_icds'][0][:5]}...")
                print(f"  CCS in that visit: {patient['past_visit_ccs'][0][:5]}...")
            break

if __name__ == "__main__":
    main()