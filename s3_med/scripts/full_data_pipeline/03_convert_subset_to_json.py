#!/usr/bin/env python3
"""
Convert a subset of test episodes to JSON for testing.
"""

import os
import sys

# Set parameters for subset processing
NUM_PATIENTS = int(sys.argv[1]) if len(sys.argv) > 1 else 100

# Special case: when NUM_PATIENTS >= 5000, process all test data without limits
if NUM_PATIENTS >= 5000:
    os.environ['TEST_MODE'] = 'true'  # Keep TEST_MODE true to use correct paths
    print(f"Processing ALL test data (no limit)...")
    PROCESS_ALL = True
else:
    os.environ['TEST_MODE'] = 'true'
    print(f"Processing {NUM_PATIENTS} patients...")
    PROCESS_ALL = False

# Import the main conversion script
exec_globals = {
    '__name__': '__main__',
    'TEST_MODE': True,  # Always use test mode paths
    'MAX_PATIENTS': NUM_PATIENTS,
    'PROCESS_ALL': PROCESS_ALL
}

# Read and execute the main script with modified globals
with open('03_convert_full_episodes_to_json.py', 'r') as f:
    code = f.read()
    # Replace the MAX_PATIENTS value
    code = code.replace('MAX_PATIENTS = 10', f'MAX_PATIENTS = {NUM_PATIENTS}')
    
    # Replace the limiting logic when processing all
    if PROCESS_ALL:
        # Replace the first check
        code = code.replace(
            '''    if TEST_MODE and len(subject_dirs) > 1000:
        print(f"Processing full test set: {len(subject_dirs)} subjects")
    elif TEST_MODE and MAX_PATIENTS < len(subject_dirs):
        subject_dirs = subject_dirs[:MAX_PATIENTS]
        print(f"TEST MODE: Processing only {len(subject_dirs)} subjects")''',
            '''    # Process all subjects when PROCESS_ALL is True
    if PROCESS_ALL:
        print(f"Processing full test set: {len(subject_dirs)} subjects")
    elif TEST_MODE and MAX_PATIENTS < len(subject_dirs):
        subject_dirs = subject_dirs[:MAX_PATIENTS]
        print(f"TEST MODE: Processing only {len(subject_dirs)} subjects")'''
        )
    
    exec(code, exec_globals)