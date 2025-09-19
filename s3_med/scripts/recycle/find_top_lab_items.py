#!/usr/bin/env python3
"""
Find top 50 most frequently used lab items from LABEVENT table
"""

import gzip
from collections import Counter
from tqdm import tqdm
import json
import os

def find_top_lab_items(labevent_file, output_dir='output', top_n=50):
    """
    Find the top N most frequently occurring itemids in LABEVENT table
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Analyzing LABEVENT table for top {top_n} lab items...")
    
    # Count occurrences of each itemid
    itemid_counter = Counter()
    total_rows = 0
    
    with gzip.open(labevent_file, 'rt') as f:
        # Read header to find column indices
        header = f.readline().strip().split(',')
        
        # Find itemid column index
        try:
            itemid_idx = header.index('itemid')
        except ValueError:
            print("Error: 'itemid' column not found in header")
            print(f"Available columns: {header}")
            return
        
        # Count itemids
        for line in tqdm(f, desc="Counting lab items"):
            parts = line.strip().split(',')
            if len(parts) > itemid_idx:
                try:
                    itemid = parts[itemid_idx]
                    if itemid and itemid.strip():  # Skip empty itemids
                        itemid_counter[itemid] += 1
                        total_rows += 1
                except:
                    continue
    
    print(f"\nProcessed {total_rows:,} rows")
    print(f"Found {len(itemid_counter):,} unique itemids")
    
    # Get top N items
    top_items = itemid_counter.most_common(top_n)
    
    # Display top 10
    print(f"\nTop 10 most frequent lab items:")
    for i, (itemid, count) in enumerate(top_items[:10]):
        percentage = (count / total_rows) * 100
        print(f"  {i+1}. ItemID {itemid}: {count:,} occurrences ({percentage:.2f}%)")
    
    # Save results
    results = {
        'top_lab_items': [{'itemid': itemid, 'count': count, 'percentage': (count/total_rows)*100} 
                         for itemid, count in top_items],
        'total_rows': total_rows,
        'unique_itemids': len(itemid_counter)
    }
    
    output_file = os.path.join(output_dir, 'top_50_lab_items.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save just the itemid list
    itemid_list = [item[0] for item in top_items]
    itemid_list_file = os.path.join(output_dir, 'top_50_lab_itemids.json')
    with open(itemid_list_file, 'w') as f:
        json.dump(itemid_list, f, indent=2)
    
    print(f"\nResults saved to:")
    print(f"  - {output_file}")
    print(f"  - {itemid_list_file}")
    
    return top_items

def main():
    # LABEVENT file path
    labevent_file = '/scratch/bcew/ruikez2/intern/s3_med/data/mimiciv/3.1/hosp/labevents.csv.gz'
    
    # Output directory
    output_dir = 'lab_items_analysis'
    
    # Find top 50 lab items
    find_top_lab_items(labevent_file, output_dir=output_dir, top_n=50)

if __name__ == "__main__":
    main()