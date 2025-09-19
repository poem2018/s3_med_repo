#!/usr/bin/env python3
"""
Optimized script to create patient matrices for patients with >48h ICU stays
"""

import json
import gzip
from datetime import datetime
from collections import defaultdict
import numpy as np
import os
from tqdm import tqdm

def main():
    # File paths
    chartevents_file = '/scratch/bcew/ruikez2/intern/s3_med/data/mimiciv/3.1/icu/chartevents.csv.gz'
    labevents_file = '/scratch/bcew/ruikez2/intern/s3_med/data/mimiciv/3.1/hosp/labevents.csv.gz'
    icustays_file = '/scratch/bcew/ruikez2/intern/s3_med/data/mimiciv/3.1/icu/icustays.csv.gz'
    
    # Load predefined items
    with open('/scratch/bcew/ruikez2/intern/s3_med/mimic_matrices_count_only/item_order.json', 'r') as f:
        chart_items = json.load(f)
    
    with open('/scratch/bcew/ruikez2/intern/s3_med/lab_items_analysis/top_50_lab_itemids.json', 'r') as f:
        lab_items = json.load(f)
    
    # Convert to sets for O(1) lookup
    chart_items_set = set(chart_items)
    lab_items_set = set(lab_items)
    
    print(f"Chart items: {len(chart_items)}, Lab items: {len(lab_items)}")
    
    # Step 1: Find patients with >48h stays
    print("\nStep 1: Finding patients with >48 hour ICU stays...")
    long_stay_patients = {}  # patient_key -> admission_time
    
    with gzip.open(icustays_file, 'rt') as f:
        header = f.readline().strip().split(',')
        subject_idx = header.index('subject_id')
        stay_idx = header.index('stay_id')
        los_idx = header.index('los')
        intime_idx = header.index('intime')
        
        for line in tqdm(f, desc="Reading ICU stays"):
            parts = line.strip().split(',')
            try:
                los = float(parts[los_idx])
                if los >= 2.0:  # >48 hours
                    patient_key = f"{parts[subject_idx]}_{parts[stay_idx]}"
                    long_stay_patients[patient_key] = datetime.strptime(parts[intime_idx], '%Y-%m-%d %H:%M:%S')
            except:
                continue
    
    print(f"Found {len(long_stay_patients)} patients with >48 hour stays")
    
    # Step 2: Read chart events efficiently
    print("\nStep 2: Reading chart events...")
    patient_data = defaultdict(lambda: defaultdict(dict))  # patient -> hour -> itemid -> value
    patient_hours = defaultdict(set)  # patient -> set of hours
    
    with gzip.open(chartevents_file, 'rt') as f:
        # Skip header
        f.readline()
        
        for line in tqdm(f, desc="Processing chart events"):
            parts = line.strip().split(',')
            if len(parts) < 9:
                continue
            
            patient_key = f"{parts[0]}_{parts[2]}"
            if patient_key not in long_stay_patients:
                continue
            
            itemid = parts[6]
            if itemid not in chart_items_set:
                continue
            
            try:
                chart_dt = datetime.strptime(parts[4], '%Y-%m-%d %H:%M:%S')
                hour_key = chart_dt.strftime('%Y-%m-%d %H:00:00')
                
                value = parts[8] if parts[8].strip() else parts[7]  # valuenum or value
                
                # Only store first value per hour per item
                if itemid not in patient_data[patient_key][hour_key]:
                    patient_data[patient_key][hour_key][itemid] = value
                    patient_hours[patient_key].add(hour_key)
            except:
                continue
    
    # Filter patients with sufficient data
    valid_patients = {p: patient_hours[p] for p in patient_hours if len(patient_hours[p]) >= 24}
    print(f"{len(valid_patients)} patients have >=24 hours of chart data")
    
    if not valid_patients:
        print("No valid patients found!")
        return
    
    # Step 3: Read lab events
    print("\nStep 3: Reading lab events...")
    patient_labs = defaultdict(dict)  # patient -> itemid -> first value
    
    # Create subject_id to patient_key mapping for faster lookup
    subject_to_patients = defaultdict(list)
    for patient_key in valid_patients:
        subject_id = patient_key.split('_')[0]
        subject_to_patients[subject_id].append(patient_key)
    
    with gzip.open(labevents_file, 'rt') as f:
        # Skip header
        f.readline()
        
        for line in tqdm(f, desc="Processing lab events"):
            parts = line.strip().split(',')
            if len(parts) < 9:
                continue
            
            subject_id = parts[1]
            if subject_id not in subject_to_patients:
                continue
            
            itemid = parts[4]
            if itemid not in lab_items_set:
                continue
            
            try:
                lab_dt = datetime.strptime(parts[6], '%Y-%m-%d %H:%M:%S')
                value = parts[9] if parts[9].strip() else parts[8]  # valuenum or value
                
                # Find matching patient and check if after ICU admission
                for patient_key in subject_to_patients[subject_id]:
                    if lab_dt >= long_stay_patients[patient_key]:
                        if itemid not in patient_labs[patient_key]:
                            patient_labs[patient_key][itemid] = value
            except:
                continue
    
    # Step 4: Create matrices
    print("\nStep 4: Creating matrices...")
    
    # Process first patient for demo
    first_patient = None
    first_matrix = None
    first_info = None
    
    output_matrices = []
    output_info = []
    
    for i, (patient_key, hours_set) in enumerate(tqdm(valid_patients.items(), desc="Creating matrices")):
        hours = sorted(hours_set)
        
        # Create matrix: rows=items, cols=hours
        matrix = []
        for itemid in chart_items:
            row = []
            for hour in hours:
                value = patient_data[patient_key][hour].get(itemid, "None")
                row.append(value)
            matrix.append(row)
        
        # Get lab values
        lab_values = []
        for itemid in lab_items:
            value = patient_labs[patient_key].get(itemid, "None")
            lab_values.append(value)
        
        # Create patient info
        info = {
            'patient_id': patient_key,
            'n_hours': len(hours),
            'start_time': hours[0],
            'end_time': hours[-1],
            'lab_values': lab_values
        }
        
        output_matrices.append(matrix)
        output_info.append(info)
        
        # Save first patient for demo
        if i == 0:
            first_patient = patient_key
            first_matrix = matrix
            first_info = info
    
    # Print first sample
    print("\n" + "="*60)
    print("FIRST PATIENT SAMPLE:")
    print("="*60)
    print(f"\nPatient ID: {first_patient}")
    print(f"Matrix shape: {len(first_matrix)} items × {len(first_matrix[0])} hours")
    print(f"\nFirst 5 chart items (first 5 hours):")
    for i in range(min(5, len(first_matrix))):
        print(f"  Item {chart_items[i]}: {first_matrix[i][:5]}")
    print(f"\nPatient info:")
    print(f"  Duration: {first_info['n_hours']} hours")
    print(f"  Start: {first_info['start_time']}")
    print(f"  End: {first_info['end_time']}")
    print(f"\nFirst 10 lab values:")
    for i in range(min(10, len(lab_items))):
        print(f"  Lab {lab_items[i]}: {first_info['lab_values'][i]}")
    
    # Step 5: Save results
    print("\nStep 5: Saving results...")
    output_dir = 'patient_matrices_optimized'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'patient_matrices.json'), 'w') as f:
        json.dump(output_matrices, f)
    
    with open(os.path.join(output_dir, 'patient_info.json'), 'w') as f:
        json.dump(output_info, f, indent=2)
    
    # Save item lists
    with open(os.path.join(output_dir, 'chart_items.json'), 'w') as f:
        json.dump(chart_items, f)
    
    with open(os.path.join(output_dir, 'lab_items.json'), 'w') as f:
        json.dump(lab_items, f)
    
    # Summary
    summary = {
        'total_patients': len(output_matrices),
        'n_chart_items': len(chart_items),
        'n_lab_items': len(lab_items),
        'matrix_shape': f"{len(chart_items)} × variable hours"
    }
    
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nCompleted! Processed {len(output_matrices)} patients")
    print(f"Results saved to {output_dir}/")

if __name__ == "__main__":
    main()