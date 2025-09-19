#!/bin/bash
# Run pipeline on test subset only to avoid OOM
# This ensures we always use the same test set

set -e  # Exit on error

echo "========================================="
echo "MIMIC-IV Test Subset Pipeline"
echo "========================================="
echo "This pipeline processes only the test set to avoid memory issues."
echo "The test set is fixed and consistent across runs."

# Steps 1-4 are commented out - starting directly from Step 5 (03_convert_subset_to_json.py)
# # Step 1: Create or verify fixed splits
echo -e "\n[Step 1] Creating/verifying fixed data splits..."
python 00_create_fixed_splits.py

# # Step 2: Process ALL test data (4723 patients)
echo -e "\n[Step 2] Processing ALL test data (4723 patients)..."
python 02_process_subset_data.py test 5000  # Set to 5000 to get all 4723

# # Now run the rest of the pipeline on the subset data
SUBSET_DIR="/scratch/bcew/ruikez2/intern/s3_med/data/baseline_text/subset_test"

# # Step 3: Run benchmark extraction on subset
echo -e "\n[Step 3] Extracting subjects from subset..."
cd /scratch/bcew/ruikez2/intern/mimic-iv-benchmarks

python -m mimic4benchmark.scripts.extract_subjects \
     ${SUBSET_DIR} \
     /scratch/bcew/ruikez2/intern/s3_med/data/baseline_text/benchmark_test_subset/

# # Step 4: Extract episodes
echo -e "\n[Step 4] Extracting episodes..."
python -m mimic4benchmark.scripts.extract_episodes_from_subjects \
     /scratch/bcew/ruikez2/intern/s3_med/data/baseline_text/benchmark_test_subset/

# Step 5: Convert to our JSON format (Starting from here)
echo -e "\n[Step 5] Converting to JSON format..."
cd /scratch/bcew/ruikez2/intern/s3_med/scripts/full_data_pipeline

# Run the conversion script for ALL test data (4573 subjects)
echo "Processing ALL test data (4573 subjects)..."
python 03_convert_subset_to_json.py 5000  # Process all test subjects (5000 triggers no limit mode)

# Step 6: Add temporal information
echo -e "\n[Step 6] Adding temporal information..."
export TEST_MODE=true  # Use test data paths
python 04_add_temporal_info_full.py

# Step 7: Convert to text format
echo -e "\n[Step 7] Converting to text format..."
export TEST_MODE=true  # Use test data paths
python 06_convert_json_to_text_simple.py

echo -e "\n========================================="
echo "Test subset pipeline completed!"
echo "========================================="
echo "Output files:"
echo "  - /scratch/bcew/ruikez2/intern/s3_med/data/baseline_text/chartevents_matrices_test.json"
echo "  - /scratch/bcew/ruikez2/intern/s3_med/data/baseline_text/patient_info_test.json"
echo "  - /scratch/bcew/ruikez2/intern/s3_med/data/baseline_text/patient_data_with_temporal_test.json"
echo "  - /scratch/bcew/ruikez2/intern/s3_med/data/baseline_text/*.txt"

# Step 8: Preview the results
echo -e "\n[Step 8] Previewing results..."
python preview_processed_data.py