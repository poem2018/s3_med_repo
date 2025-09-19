#!/usr/bin/env python3
"""
Debug version: Find any 10 patients with >48h stays and create matrices
"""

import json
from collections import defaultdict
import gzip
from datetime import datetime
from tqdm import tqdm
import sys
import os

def create_patient_matrices_debug(chartevents_file, icustays_file, d_items_file, n_patients=10, output_dir='output'):
    """
    Create matrices for debug - find ANY n patients with >48h stays
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Step 1: Find ALL patients with stay > 48 hours
    print("Step 1: Finding all patients with >48 hour stays...")
    long_stay_patients = set()
    
    with gzip.open(icustays_file, 'rt') as f:
        header = f.readline().strip().split(',')
        subject_idx = header.index('subject_id')
        stay_idx = header.index('stay_id')
        los_idx = header.index('los')
        
        for line in f:
            parts = line.strip().split(',')
            if len(parts) > los_idx:
                try:
                    los = float(parts[los_idx])
                    if los >= 2.0:  # 2 days = 48 hours
                        subject_id = parts[subject_idx]
                        stay_id = parts[stay_idx]
                        patient_key = f"{subject_id}_{stay_id}"
                        long_stay_patients.add(patient_key)
                except:
                    continue
    
    print(f"Found {len(long_stay_patients)} patients with >48 hour stays")
    
    # Step 2: Scan chartevents and collect data for first n patients found
    print(f"\nStep 2: Finding data for any {n_patients} patients from the list...")
    
    selected_patients = set()
    patient_item_hours = defaultdict(lambda: defaultdict(set))
    
    line_count = 0
    with gzip.open(chartevents_file, 'rt') as f:
        header = f.readline().strip().split(',')
        subject_idx = header.index('subject_id')
        stay_idx = header.index('stay_id')
        charttime_idx = header.index('charttime')
        itemid_idx = header.index('itemid')
        
        for line in tqdm(f, desc="Finding patients"):
            line_count += 1
            
            parts = line.strip().split(',')
            if len(parts) < max(subject_idx, stay_idx, charttime_idx, itemid_idx) + 1:
                continue
                
            subject_id = parts[subject_idx]
            stay_id = parts[stay_idx]
            patient_key = f"{subject_id}_{stay_id}"
            
            # Check if this patient is in our long stay list
            if patient_key in long_stay_patients:
                selected_patients.add(patient_key)
                
                charttime = parts[charttime_idx]
                itemid = parts[itemid_idx]
                
                try:
                    chart_dt = datetime.strptime(charttime, '%Y-%m-%d %H:%M:%S')
                    hour_key = chart_dt.strftime('%Y-%m-%d %H:00:00')
                    patient_item_hours[patient_key][itemid].add(hour_key)
                except:
                    continue
                
                # Stop when we have enough patients
                if len(selected_patients) >= n_patients:
                    print(f"\nFound {n_patients} patients after scanning {line_count} lines")
                    break
    
    print(f"Selected patients: {list(selected_patients)[:5]}...")  # Show first 5
    
    # Step 3: Find top 50 items from these patients
    print(f"\nStep 3: Finding top 50 items from {len(selected_patients)} patients...")
    
    item_patient_counts = defaultdict(set)
    
    # Count items measured >=24 times
    for patient_key, item_hours in patient_item_hours.items():
        for itemid, hours in item_hours.items():
            if len(hours) >= 24:
                item_patient_counts[itemid].add(patient_key)
    
    # Get top 50 items
    item_counts_list = [(itemid, len(patients)) for itemid, patients in item_patient_counts.items()]
    item_counts_list.sort(key=lambda x: x[1], reverse=True)
    top_50_items = item_counts_list[:50]
    top_50_itemids = [item[0] for item in top_50_items]
    
    print(f"\nTop 10 most frequent items:")
    for i, (itemid, count) in enumerate(top_50_items[:10]):
        print(f"  {i+1}. Item {itemid}: used by {count} patients ({count/len(selected_patients)*100:.1f}%)")
    
    # Save top 50 metrics immediately
    print("\nSaving top 50 metrics to JSON file...")
    top_50_metrics = {
        'top_50_items': [
            {
                'itemid': itemid,
                'patient_count': count,
                'percentage': count/len(selected_patients)*100
            }
            for itemid, count in top_50_items
        ],
        'total_patients': len(selected_patients),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(output_dir, 'top_50_metrics.json'), 'w') as f:
        json.dump(top_50_metrics, f, indent=2)
    print(f"Saved top 50 metrics to {os.path.join(output_dir, 'top_50_metrics.json')}")
    
    # Save item order
    with open(os.path.join(output_dir, 'item_order.json'), 'w') as f:
        json.dump(top_50_itemids, f, indent=2)
    
    # Step 4: Get item definitions
    print("\nStep 4: Reading item definitions...")
    item_definitions = {}
    
    with gzip.open(d_items_file, 'rt') as f:
        header = f.readline().strip().split(',')
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 6:
                itemid = parts[0]
                if itemid in top_50_itemids:
                    item_definitions[itemid] = {
                        'itemid': itemid,
                        'label': parts[1],
                        'unit': parts[5] if len(parts) > 5 else '',
                        'category': parts[4] if len(parts) > 4 else ''
                    }
    
    # Save item dictionary
    item_dict = []
    for itemid in top_50_itemids:
        if itemid in item_definitions:
            item_dict.append(item_definitions[itemid])
        else:
            item_dict.append({
                'itemid': itemid,
                'label': 'Unknown',
                'unit': '',
                'category': ''
            })
    
    with open(os.path.join(output_dir, 'item_dictionary.json'), 'w') as f:
        json.dump(item_dict, f, indent=2)
    
    # Step 5: Extract full data for selected patients
    print(f"\nStep 5: Extracting complete data for {len(selected_patients)} patients...")
    
    patient_data = defaultdict(lambda: defaultdict(dict))
    patient_hours = defaultdict(set)
    patients_with_enough_data = set()
    
    line_count = 0
    with gzip.open(chartevents_file, 'rt') as f:
        header = f.readline().strip().split(',')
        subject_idx = header.index('subject_id')
        stay_idx = header.index('stay_id')
        charttime_idx = header.index('charttime')
        itemid_idx = header.index('itemid')
        value_idx = header.index('value')
        valuenum_idx = header.index('valuenum')
        
        for line in tqdm(f, desc="Extracting data"):
            line_count += 1
            
            parts = line.strip().split(',')
            if len(parts) < max(subject_idx, stay_idx, charttime_idx, itemid_idx, value_idx, valuenum_idx) + 1:
                continue
                
            subject_id = parts[subject_idx]
            stay_id = parts[stay_idx]
            patient_key = f"{subject_id}_{stay_id}"
            
            if patient_key not in selected_patients:
                continue
                
            itemid = parts[itemid_idx]
            if itemid not in top_50_itemids:
                continue
            
            charttime = parts[charttime_idx]
            value = parts[value_idx]
            valuenum = parts[valuenum_idx]
            
            try:
                chart_dt = datetime.strptime(charttime, '%Y-%m-%d %H:%M:%S')
                hour_key = chart_dt.strftime('%Y-%m-%d %H:00:00')
                
                final_value = str(valuenum) if valuenum.strip() else str(value)
                
                if itemid not in patient_data[patient_key][hour_key]:
                    patient_data[patient_key][hour_key][itemid] = final_value
                    patient_hours[patient_key].add(hour_key)
                    
                    # Check if this patient has enough data (e.g., >24 hours)
                    if len(patient_hours[patient_key]) >= 24:
                        patients_with_enough_data.add(patient_key)
            except:
                continue
            
            # For debug: stop early if all patients have enough data
            if len(patients_with_enough_data) == len(selected_patients):
                print(f"\n  All patients have sufficient data. Stopping at line {line_count}")
                break
                
            # Also stop if we've scanned too many lines (e.g., 10M for debug)
            if line_count > 10000000:
                print(f"\n  Debug limit reached at {line_count} lines")
                break
    
    print(f"  Scanned {line_count} lines")
    
    # Create matrices - maintain consistent order
    print("\nCreating matrices...")
    all_patient_matrices = []
    patient_info = []
    
    # Sort patients to ensure consistent order
    sorted_patients = sorted(selected_patients)
    
    for i, patient_key in enumerate(sorted_patients):
        if patient_key not in patient_hours or not patient_hours[patient_key]:
            continue
            
        hours = sorted(patient_hours[patient_key])
        
        # Create matrix: each row is an item, each column is an hour
        matrix = []
        for itemid in top_50_itemids:
            row = []
            for hour in hours:
                value = patient_data[patient_key][hour].get(itemid, "None")
                row.append(value)
            matrix.append(row)
        
        all_patient_matrices.append(matrix)
        patient_info.append({
            'patient_id': patient_key,
            'n_hours': len(hours),
            'start_time': hours[0],
            'end_time': hours[-1]
        })
        
        print(f"\nPatient {i+1}: {patient_key}")
        print(f"  Matrix shape: {len(top_50_itemids)} items × {len(hours)} hours")
        print(f"  Time range: {hours[0]} to {hours[-1]}")
        
        if len(matrix) > 0 and len(matrix[0]) > 0:
            non_none = sum(1 for row in matrix for val in row if val != "None")
            total_values = len(matrix) * len(matrix[0])
            print(f"  Data density: {non_none}/{total_values} ({non_none/total_values*100:.1f}%)")
    
    print(f"\nTotal patients with data: {len(all_patient_matrices)}")
    
    # Save results
    with open(os.path.join(output_dir, 'patient_matrices.json'), 'w') as f:
        json.dump(all_patient_matrices, f)
    
    with open(os.path.join(output_dir, 'patient_info.json'), 'w') as f:
        json.dump(patient_info, f, indent=2)
    
    # Save summary
    summary = {
        'n_patients_selected': len(selected_patients),
        'n_patients_with_data': len(all_patient_matrices),
        'n_items': len(top_50_itemids),
        'top_10_items': [(item[0], item[1], item_definitions.get(item[0], {}).get('label', 'Unknown')) 
                         for item in top_50_items[:10]]
    }
    
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nAll files saved successfully!")

def main():
    chartevents_file = '/scratch/bcew/ruikez2/intern/s3_med/data/mimiciv/3.1/icu/chartevents.csv.gz'
    icustays_file = '/scratch/bcew/ruikez2/intern/s3_med/data/mimiciv/3.1/icu/icustays.csv.gz'
    d_items_file = '/scratch/bcew/ruikez2/intern/s3_med/data/mimiciv/3.1/icu/d_items.csv.gz'
    
    # Parse command line arguments
    n_patients = 10  # default
    if len(sys.argv) > 1:
        try:
            n_patients = int(sys.argv[1])
            print(f"Will collect data for {n_patients} patients")
        except ValueError:
            print(f"Invalid number of patients: {sys.argv[1]}, using default 10")
            n_patients = 10
    else:
        print("Usage: python create_patient_matrices_debug.py [number_of_patients]")
        print("Using default: 10 patients")
    
    # Set output directory based on number of patients
    output_dir = f'mimic_matrices_{n_patients}_patients'
    
    create_patient_matrices_debug(chartevents_file, icustays_file, d_items_file, n_patients=n_patients, output_dir=output_dir)

if __name__ == "__main__":
    main()