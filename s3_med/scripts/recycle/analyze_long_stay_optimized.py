#!/usr/bin/env python3
"""
Optimized analysis of patients with ICU stay > 96 hours
"""

import numpy as np
from collections import defaultdict
import gzip
from datetime import datetime
import time
from tqdm import tqdm

def analyze_long_stay_summary(chartevents_file, icustays_file):
    """Analyze patients staying >96 hours with optimized processing"""
    
    start_time = time.time()
    
    # First, get all patients and those with stay > 96 hours
    print("Reading ICU stays data...")
    long_stay_patients = {}  # patient_key -> los (in days)
    all_patients = 0
    
    with gzip.open(icustays_file, 'rt') as f:
        header = f.readline().strip().split(',')
        subject_idx = header.index('subject_id')
        stay_idx = header.index('stay_id')
        los_idx = header.index('los')
        
        # Count total lines for progress bar
        lines = f.readlines()
        
        for line in tqdm(lines, desc="Reading ICU stays"):
            parts = line.strip().split(',')
            if len(parts) > los_idx:
                all_patients += 1
                try:
                    los = float(parts[los_idx])
                    if los >= 4.0:  # 4 days = 96 hours
                        subject_id = parts[subject_idx]
                        stay_id = parts[stay_idx]
                        patient_key = f"{subject_id}_{stay_id}"
                        long_stay_patients[patient_key] = los
                except:
                    continue
    
    print(f"Total patients in ICU: {all_patients}")
    print(f"Patients with >96 hour stays: {len(long_stay_patients)} ({len(long_stay_patients)/all_patients*100:.1f}%)")
    print(f"Average LOS for >96h patients: {np.mean(list(long_stay_patients.values())):.1f} days")
    
    # Sample analysis - process a subset to get estimates
    print("\nProcessing a sample of chartevents to estimate statistics...")
    
    sample_size = min(1000, len(long_stay_patients))
    sampled_patients = set(list(long_stay_patients.keys())[:sample_size])
    
    patient_item_counts = defaultdict(lambda: defaultdict(int))
    patients_with_data = set()
    
    line_count = 0
    max_lines = 10000000  # Process first 10M lines for estimation
    
    with gzip.open(chartevents_file, 'rt') as f:
        header = f.readline().strip().split(',')
        subject_idx = header.index('subject_id')
        stay_idx = header.index('stay_id')
        charttime_idx = header.index('charttime')
        itemid_idx = header.index('itemid')
        
        seen_hours = defaultdict(set)
        
        with tqdm(total=max_lines, desc="Processing chartevents") as pbar:
            for line in f:
                line_count += 1
                if line_count > max_lines:
                    break
                
                pbar.update(1)
                
                parts = line.strip().split(',')
                if len(parts) < max(subject_idx, stay_idx, charttime_idx, itemid_idx) + 1:
                    continue
                    
                subject_id = parts[subject_idx]
                stay_id = parts[stay_idx]
                patient_key = f"{subject_id}_{stay_id}"
                
                # Only process sampled long-stay patients
                if patient_key not in sampled_patients:
                    continue
                
                charttime = parts[charttime_idx]
                itemid = parts[itemid_idx]
                
                try:
                    chart_dt = datetime.strptime(charttime, '%Y-%m-%d %H:%M:%S')
                    hour_key = chart_dt.strftime('%Y-%m-%d %H:00:00')
                    
                    hour_item_key = f"{patient_key}_{itemid}"
                    if hour_key not in seen_hours[hour_item_key]:
                        seen_hours[hour_item_key].add(hour_key)
                        patient_item_counts[patient_key][itemid] += 1
                    
                    patients_with_data.add(patient_key)
                    
                except:
                    continue
    
    print(f"\nSample analysis complete. Processed {line_count} lines.")
    print(f"Found data for {len(patients_with_data)} out of {sample_size} sampled long-stay patients")
    
    # Analyze the sample
    patient_frequent_item_counts = []
    item_frequency = defaultdict(int)
    
    for patient_key in patients_with_data:
        item_counts = patient_item_counts[patient_key]
        
        # Count items measured 24+ times
        frequent_items = [itemid for itemid, count in item_counts.items() if count >= 24]
        patient_frequent_item_counts.append(len(frequent_items))
        
        for itemid in frequent_items:
            item_frequency[itemid] += 1
    
    # Calculate statistics
    if patient_frequent_item_counts:
        avg_items = np.mean(patient_frequent_item_counts)
        std_items = np.std(patient_frequent_item_counts)
        min_items = min(patient_frequent_item_counts)
        max_items = max(patient_frequent_item_counts)
        median_items = np.median(patient_frequent_item_counts)
    else:
        avg_items = std_items = min_items = max_items = median_items = 0
    
    # Extrapolate to full population
    estimated_total_with_data = len(long_stay_patients) * (len(patients_with_data) / sample_size)
    
    print("\n" + "="*80)
    print("SUMMARY FOR PATIENTS WITH >96 HOUR ICU STAYS:")
    print("="*80)
    print(f"Total ICU patients: {all_patients}")
    print(f"Patients with >96h stays: {len(long_stay_patients)} ({len(long_stay_patients)/all_patients*100:.1f}%)")
    print(f"\nBased on sample of {sample_size} patients:")
    print(f"Estimated long-stay patients with chart data: ~{int(estimated_total_with_data)}")
    print("\nItems measured 24+ times per patient:")
    print(f"  Average: {avg_items:.1f} items")
    print(f"  Standard deviation: {std_items:.1f}")
    print(f"  Minimum: {min_items} items")
    print(f"  Maximum: {max_items} items")
    print(f"  Median: {median_items:.1f} items")
    
    # Show top items
    print(f"\nTop 15 most common frequently-measured items (from sample):")
    sorted_items = sorted(item_frequency.items(), key=lambda x: x[1], reverse=True)[:15]
    print(f"{'ItemID':<10} {'% in Sample':<15}")
    print("-" * 25)
    for itemid, count in sorted_items:
        print(f"{itemid:<10} {count/len(patients_with_data)*100:<15.1f}%")
    
    elapsed_time = time.time() - start_time
    print(f"\nAnalysis completed in {elapsed_time:.1f} seconds")

def main():
    chartevents_file = '/scratch/bcew/ruikez2/intern/s3_med/data/mimiciv/3.1/icu/chartevents.csv.gz'
    icustays_file = '/scratch/bcew/ruikez2/intern/s3_med/data/mimiciv/3.1/icu/icustays.csv.gz'
    
    analyze_long_stay_summary(chartevents_file, icustays_file)

if __name__ == "__main__":
    main()