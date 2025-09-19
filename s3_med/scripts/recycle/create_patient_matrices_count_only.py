#!/usr/bin/env python3
"""
Count-only version: only count item frequencies without storing hour details
"""

import json
from collections import defaultdict, Counter
import gzip
from datetime import datetime
from tqdm import tqdm
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

def process_batch_count_only(batch_info):
    """Only count items per patient, don't store hour details"""
    lines, long_stay_patients, sampled_patients = batch_info
    
    # Only track: patient -> item -> hour_count (not the actual hours)
    local_patient_item_counts = defaultdict(lambda: defaultdict(int))
    
    for line in lines:
        parts = line.strip().split(',')
        if len(parts) < 9:
            continue
        
        try:
            patient_key = f"{parts[0]}_{parts[2]}"
            
            if patient_key not in long_stay_patients or patient_key not in sampled_patients:
                continue
            
            itemid = parts[6]
            chart_dt = datetime.strptime(parts[4], '%Y-%m-%d %H:%M:%S')
            hour_key = chart_dt.strftime('%Y-%m-%d %H:00:00')
            
            # Just count unique hours per item per patient
            key = f"{patient_key}|{itemid}|{hour_key}"
            local_patient_item_counts[patient_key][itemid] = 1  # Mark as seen
            
        except:
            continue
    
    # Convert to counts
    result = {}
    for patient, items in local_patient_item_counts.items():
        result[patient] = list(items.keys())  # Just item IDs
    
    return result

def find_top_items_streaming(chartevents_file, long_stay_patients, sampled_patients, n_workers):
    """Find top items by streaming through file and only counting"""
    
    print("Finding top items by streaming...")
    
    # First pass: count hours per item per patient
    patient_item_hours = defaultdict(set)  # patient -> set of "itemid|hour" strings
    
    batch_size = 100000
    batch = []
    batch_count = 0
    
    with gzip.open(chartevents_file, 'rt') as f:
        f.readline()  # Skip header
        
        for line in tqdm(f, desc="Counting items"):
            parts = line.strip().split(',')
            if len(parts) < 9:
                continue
            
            try:
                patient_key = f"{parts[0]}_{parts[2]}"
                
                if patient_key not in long_stay_patients or patient_key not in sampled_patients:
                    continue
                
                itemid = parts[6]
                chart_dt = datetime.strptime(parts[4], '%Y-%m-%d %H:%M:%S')
                hour_key = chart_dt.strftime('%Y-%m-%d %H')  # Just hour
                
                # Store as string to save memory
                patient_item_hours[patient_key].add(f"{itemid}|{hour_key}")
                
            except:
                continue
            
            # Periodically clean up to save memory
            if len(patient_item_hours) > 5000:
                # Count and aggregate
                item_patient_counts = defaultdict(set)
                
                for patient, item_hours in patient_item_hours.items():
                    item_hour_count = defaultdict(int)
                    for item_hour in item_hours:
                        itemid, _ = item_hour.split('|')
                        item_hour_count[itemid] += 1
                    
                    for itemid, count in item_hour_count.items():
                        if count >= 24:
                            item_patient_counts[itemid].add(patient)
                
                # Clear memory
                patient_item_hours.clear()
                
                print(f"Processed batch, found {len(item_patient_counts)} items so far")
    
    # Final count
    print("Final counting...")
    item_patient_final_counts = defaultdict(set)
    
    for patient, item_hours in patient_item_hours.items():
        item_hour_count = defaultdict(int)
        for item_hour in item_hours:
            itemid, _ = item_hour.split('|')
            item_hour_count[itemid] += 1
        
        for itemid, count in item_hour_count.items():
            if count >= 24:
                item_patient_final_counts[itemid].add(patient)
    
    # Get top 50
    item_counts_list = [(itemid, len(patients)) for itemid, patients in item_patient_final_counts.items()]
    item_counts_list.sort(key=lambda x: x[1], reverse=True)
    
    return item_counts_list[:50]

def process_batch_for_data(batch_info):
    """Process batch to extract data (unchanged)"""
    lines, long_stay_patients, top_50_itemids = batch_info
    
    local_patient_data = defaultdict(lambda: defaultdict(dict))
    local_patient_hours = defaultdict(set)
    
    for line in lines:
        parts = line.strip().split(',')
        if len(parts) < 9:
            continue
        
        try:
            patient_key = f"{parts[0]}_{parts[2]}"
            
            if patient_key not in long_stay_patients:
                continue
            
            itemid = parts[6]
            if itemid not in top_50_itemids:
                continue
            
            chart_dt = datetime.strptime(parts[4], '%Y-%m-%d %H:%M:%S')
            hour_key = chart_dt.strftime('%Y-%m-%d %H:00:00')
            
            value = parts[7]
            valuenum = parts[8]
            final_value = str(valuenum) if valuenum.strip() else str(value)
            
            if itemid not in local_patient_data[patient_key][hour_key]:
                local_patient_data[patient_key][hour_key][itemid] = final_value
                local_patient_hours[patient_key].add(hour_key)
                
        except:
            continue
    
    return {
        'patient_data': dict(local_patient_data),
        'patient_hours': dict(local_patient_hours)
    }

def create_patient_matrices_count_only(chartevents_file, icustays_file, d_items_file, 
                                      output_dir='output', sample_size=20000):
    """
    Memory-efficient version using counting instead of storing all data
    """
    
    n_cores = multiprocessing.cpu_count()
    n_workers = min(n_cores - 2, 24)
    print(f"Using {n_workers} workers")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Find ALL patients with >48h stays
    print("Step 1: Finding all patients with >48 hour stays...")
    long_stay_patients = set()
    
    with gzip.open(icustays_file, 'rt') as f:
        header = f.readline().strip().split(',')
        subject_idx = header.index('subject_id')
        stay_idx = header.index('stay_id')
        los_idx = header.index('los')
        
        for line in tqdm(f, desc="Reading ICU stays"):
            parts = line.strip().split(',')
            if len(parts) > los_idx:
                try:
                    los = float(parts[los_idx])
                    if los >= 2.0:
                        patient_key = f"{parts[subject_idx]}_{parts[stay_idx]}"
                        long_stay_patients.add(patient_key)
                except:
                    continue
    
    print(f"Found {len(long_stay_patients)} patients with >48 hour stays")
    
    # Sample first N patients
    sampled_patients = set(list(long_stay_patients)[:sample_size])
    print(f"Using first {len(sampled_patients)} patients to find top items")
    
    # Step 2: Find top 50 items using streaming approach
    print(f"\nStep 2: Finding top 50 items from {sample_size} sampled patients...")
    
    top_50_items = find_top_items_streaming(chartevents_file, long_stay_patients, 
                                           sampled_patients, n_workers)
    
    top_50_itemids = set([item[0] for item in top_50_items])
    top_50_itemids_list = [item[0] for item in top_50_items]
    
    print(f"\nTop 10 items:")
    for i, (itemid, count) in enumerate(top_50_items[:10]):
        print(f"  {i+1}. Item {itemid}: {count} patients")
    
    # Save top 50 metrics
    with open(os.path.join(output_dir, 'top_50_metrics.json'), 'w') as f:
        json.dump({
            'top_50_items': [{'itemid': itemid, 'patient_count': count} 
                           for itemid, count in top_50_items],
            'sample_size': sample_size
        }, f, indent=2)
    
    with open(os.path.join(output_dir, 'item_order.json'), 'w') as f:
        json.dump(top_50_itemids_list, f)
    
    # Step 3: Get item definitions
    print("\nStep 3: Reading item definitions...")
    item_definitions = {}
    
    with gzip.open(d_items_file, 'rt') as f:
        f.readline()
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
    
    item_dict = []
    for itemid in top_50_itemids_list:
        if itemid in item_definitions:
            item_dict.append(item_definitions[itemid])
        else:
            item_dict.append({'itemid': itemid, 'label': 'Unknown', 'unit': '', 'category': ''})
    
    with open(os.path.join(output_dir, 'item_dictionary.json'), 'w') as f:
        json.dump(item_dict, f, indent=2)
    
    # Step 4: Second pass - extract data for ALL patients
    print(f"\nStep 4: Extracting data for ALL patients using top 50 items...")
    
    batch_size = 100000
    batches = []
    
    with gzip.open(chartevents_file, 'rt') as f:
        f.readline()  # Skip header
        
        batch = []
        for line in tqdm(f, desc="Reading chartevents"):
            batch.append(line)
            
            if len(batch) >= batch_size:
                batches.append(batch)
                batch = []
        
        if batch:
            batches.append(batch)
    
    print(f"Created {len(batches)} batches")
    
    # Process batches in parallel
    patient_data = defaultdict(lambda: defaultdict(dict))
    patient_hours = defaultdict(set)
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for batch in batches:
            future = executor.submit(process_batch_for_data, 
                                   (batch, long_stay_patients, top_50_itemids))
            futures.append(future)
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing batches"):
            result = future.result()
            
            for patient, hours_data in result['patient_data'].items():
                for hour, items in hours_data.items():
                    patient_data[patient][hour].update(items)
            
            for patient, hours in result['patient_hours'].items():
                patient_hours[patient].update(hours)
    
    # Create matrices
    print("\nCreating final matrices...")
    patients_with_enough_data = {p for p in patient_hours if len(patient_hours[p]) >= 24}
    print(f"{len(patients_with_enough_data)} patients have >=24 hours of data")
    
    all_patient_matrices = []
    patient_info = []
    
    for patient_key in tqdm(sorted(patients_with_enough_data), desc="Creating matrices"):
        hours = sorted(patient_hours[patient_key])
        
        matrix = []
        for itemid in top_50_itemids_list:
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
    
    # Save results
    print("\nSaving results...")
    with open(os.path.join(output_dir, 'patient_matrices.json'), 'w') as f:
        json.dump(all_patient_matrices, f)
    
    with open(os.path.join(output_dir, 'patient_info.json'), 'w') as f:
        json.dump(patient_info, f, indent=2)
    
    summary = {
        'n_long_stay_patients': len(long_stay_patients),
        'n_sampled_patients': sample_size,
        'n_patients_with_data': len(all_patient_matrices),
        'n_items': len(top_50_itemids),
        'n_workers': n_workers
    }
    
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nCompleted!")
    print(f"- Used {sample_size} patients to find top 50 items")
    print(f"- Extracted data for {len(all_patient_matrices)} patients total")

def main():
    chartevents_file = '/scratch/bcew/ruikez2/intern/s3_med/data/mimiciv/3.1/icu/chartevents.csv.gz'
    icustays_file = '/scratch/bcew/ruikez2/intern/s3_med/data/mimiciv/3.1/icu/icustays.csv.gz'
    d_items_file = '/scratch/bcew/ruikez2/intern/s3_med/data/mimiciv/3.1/icu/d_items.csv.gz'
    
    output_dir = 'mimic_matrices_count_only'
    
    create_patient_matrices_count_only(chartevents_file, icustays_file, d_items_file, 
                                     output_dir=output_dir, sample_size=20000)

if __name__ == "__main__":
    main()