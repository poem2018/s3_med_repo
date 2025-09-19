#!/usr/bin/env python3
"""
Clean up incorrect RAG output files that were generated with wrong dataset names
"""

import os
import shutil

# Directories to clean
dirs_to_clean = [
    'data/RAG_retrieval/mimic_train',
    'data/RAG_retrieval/mimic_val', 
    'data/RAG_retrieval/mimic_test'
]

# Files to remove (wrong dataset names)
wrong_files = [
    '2wikimultihopqa_output_sequences.json',
    'bamboogle_output_sequences.json',
    'hotpotqa_output_sequences.json',
    'musique_output_sequences.json',
    'nq_output_sequences.json',
    'popqa_output_sequences.json',
    'triviaqa_output_sequences.json'
]

for dir_path in dirs_to_clean:
    if os.path.exists(dir_path):
        print(f"Cleaning directory: {dir_path}")
        for wrong_file in wrong_files:
            file_path = os.path.join(dir_path, wrong_file)
            if os.path.exists(file_path):
                print(f"  Removing: {wrong_file}")
                os.remove(file_path)
        
        # Check if directory is now empty
        if not os.listdir(dir_path):
            print(f"  Directory is empty, keeping for future use")
    else:
        print(f"Directory not found: {dir_path}")

print("\nCleanup complete!")