#!/usr/bin/env python3
"""
Fast version: Update patient info with ICU visit list and lab abnormal flags
Single pass through lab events file
"""

import json
import gzip
from datetime import datetime
from collections import defaultdict
import os
from tqdm import tqdm

def main():
    # File paths
    labevents_file = '/scratch/bcew/ruikez2/intern/s3_med/data/mimiciv/3.1/hosp/labevents.csv.gz'
    icustays_file = '/scratch/bcew/ruikez2/intern/s3_med/data/mimiciv/3.1/icu/icustays.csv.gz'
    
    # Load existing patient info
    with open('/scratch/bcew/ruikez2/intern/s3_med/patient_matrices_optimized/patient_info.json', 'r') as f:
        existing_info = json.load(f)
    
    with open('/scratch/bcew/ruikez2/intern/s3_med/lab_items_analysis/top_50_lab_itemids.json', 'r') as f:
        lab_items = json.load(f)
    
    lab_items_set = set(lab_items)
    
    print(f"Loaded {len(existing_info)} patients from existing patient_info.json")
    print(f"Lab items: {len(lab_items)}")
    
    # Create lookup structures
    patient_keys = set()
    info_by_patient = {}
    for info in existing_info:
        patient_key = info['patient_id']
        patient_keys.add(patient_key)
        info_by_patient[patient_key] = info
    
    # Step 1: Build hadm_id mapping and get admission times
    print("\nStep 1: Building ICU stay mappings...")
    patient_to_hadm = {}
    patient_admission_times = {}
    hadm_to_stays = defaultdict(list)
    subject_to_patients = defaultdict(list)
    
    with gzip.open(icustays_file, 'rt') as f:
        header = f.readline().strip().split(',')
        subject_idx = header.index('subject_id')
        stay_idx = header.index('stay_id')
        hadm_idx = header.index('hadm_id')
        intime_idx = header.index('intime')
        
        for line in tqdm(f, desc="Reading ICU stays"):
            parts = line.strip().split(',')
            subject_id = parts[subject_idx]
            stay_id = parts[stay_idx]
            patient_key = f"{subject_id}_{stay_id}"
            hadm_id = parts[hadm_idx]
            
            # Build hadm to stays mapping
            hadm_to_stays[hadm_id].append(stay_id)
            
            if patient_key in patient_keys:
                patient_to_hadm[patient_key] = hadm_id
                patient_admission_times[patient_key] = datetime.strptime(parts[intime_idx], '%Y-%m-%d %H:%M:%S')
                subject_to_patients[subject_id].append(patient_key)
    
    print(f"Found admission times for {len(patient_admission_times)} patients")
    print(f"Unique subjects with valid patients: {len(subject_to_patients)}")
    
    # Step 2: Single pass through lab events
    print("\nStep 2: Reading lab events (single pass)...")
    patient_labs = defaultdict(dict)
    
    # First, let's check file size
    line_count = 0
    with gzip.open(labevents_file, 'rt') as f:
        header = f.readline().strip().split(',')
        subject_idx = header.index('subject_id')
        itemid_idx = header.index('itemid')
        charttime_idx = header.index('charttime')
        value_idx = header.index('value')
        valuenum_idx = header.index('valuenum')
        flag_idx = header.index('flag')
        
        for line in tqdm(f, desc="Processing lab events", unit=" lines"):
            line_count += 1
            if line_count % 1000000 == 0:
                print(f"  Processed {line_count/1000000:.1f}M lines, found {sum(len(v) for v in patient_labs.values())} lab values")
            
            parts = line.strip().split(',')
            if len(parts) <= max(subject_idx, itemid_idx, charttime_idx, value_idx, valuenum_idx, flag_idx):
                continue
            
            subject_id = parts[subject_idx]
            # Skip if this subject has no relevant patients
            if subject_id not in subject_to_patients:
                continue
            
            itemid = parts[itemid_idx]
            # Skip if not a tracked lab item
            if itemid not in lab_items_set:
                continue
            
            try:
                lab_dt = datetime.strptime(parts[charttime_idx], '%Y-%m-%d %H:%M:%S')
                value = parts[valuenum_idx] if parts[valuenum_idx].strip() else parts[value_idx]
                abnormal_flag = 1 if parts[flag_idx].strip().lower() == 'abnormal' else 0
                
                # Check each patient for this subject
                for patient_key in subject_to_patients[subject_id]:
                    if lab_dt >= patient_admission_times[patient_key]:
                        # Only store first occurrence
                        if itemid not in patient_labs[patient_key]:
                            patient_labs[patient_key][itemid] = (value, abnormal_flag)
            except:
                continue
    
    print(f"\nProcessed {line_count} total lines")
    print(f"Found lab values for {len(patient_labs)} patients")
    
    # Step 3: Update patient info
    print("\nStep 3: Updating patient info...")
    updated_info = []
    
    for patient_key, info in tqdm(info_by_patient.items(), desc="Updating patient info"):
        # Add hadm_id and ICU visit list
        hadm_id = patient_to_hadm.get(patient_key, "unknown")
        icu_visit_list = hadm_to_stays.get(hadm_id, []) if hadm_id != "unknown" else []
        
        # Update lab values with abnormal flags
        new_lab_values = []
        
        # Get existing lab values to preserve
        existing_lab_values = info.get('lab_values', [])
        
        for i, itemid in enumerate(lab_items):
            if patient_key in patient_labs and itemid in patient_labs[patient_key]:
                value, flag = patient_labs[patient_key][itemid]
                new_lab_values.append([value, flag])
            else:
                # Keep existing value if available
                if i < len(existing_lab_values):
                    old_value = existing_lab_values[i]
                    # Assume normal if no flag info
                    new_lab_values.append([old_value, 0])
                else:
                    new_lab_values.append(["None", 0])
        
        # Create updated info
        updated_entry = info.copy()
        updated_entry['hadm_id'] = hadm_id
        updated_entry['icu_visit_list'] = icu_visit_list
        updated_entry['lab_values'] = new_lab_values
        
        updated_info.append(updated_entry)
    
    # Print sample
    print("\n" + "="*60)
    print("FIRST PATIENT SAMPLE (UPDATED):")
    print("="*60)
    first_info = updated_info[0]
    print(f"\nPatient ID: {first_info['patient_id']}")
    print(f"Hospital Admission ID: {first_info['hadm_id']}")
    print(f"ICU visits in this hospital admission: {first_info['icu_visit_list']}")
    print(f"Number of hours: {first_info['n_hours']}")
    print(f"\nFirst 10 lab values [value, abnormal_flag]:")
    for i in range(min(10, len(lab_items))):
        value, flag = first_info['lab_values'][i]
        flag_str = "ABNORMAL" if flag == 1 else "normal"
        print(f"  Lab {lab_items[i]}: [{value}, {flag}] - {flag_str}")
    
    # Save updated results
    print("\nSaving updated results...")
    output_dir = 'patient_matrices_optimized_updated'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'patient_info.json'), 'w') as f:
        json.dump(updated_info, f, indent=2)
    
    with open(os.path.join(output_dir, 'lab_items.json'), 'w') as f:
        json.dump(lab_items, f)
    
    # Summary
    summary = {
        'total_patients': len(updated_info),
        'n_lab_items': len(lab_items),
        'updates': 'Added hadm_id, icu_visit_list, and abnormal flags to lab values',
        'note': 'lab_values now contains lists of [value, abnormal_flag]'
    }
    
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nCompleted! Updated {len(updated_info)} patient records")
    print(f"Results saved to {output_dir}/")

if __name__ == "__main__":
    main()