#!/bin/bash
# Run the complete data processing pipeline for full MIMIC-IV dataset
# Usage: 
#   ./run_full_pipeline.sh         # Run on full dataset
#   ./run_full_pipeline.sh test     # Run on small test subset

set -e  # Exit on error

# Check for test mode
if [ "$1" == "test" ]; then
    export TEST_MODE=true
    echo "🧪 RUNNING IN TEST MODE - Processing small subset only"
    SUFFIX="_test"
    MODE_DISPLAY="TEST"
else
    export TEST_MODE=false
    SUFFIX="_full"
    MODE_DISPLAY="FULL"
fi

echo "========================================="
echo "Starting MIMIC-IV Data Pipeline"
echo "Mode: $MODE_DISPLAY"
echo "========================================="

# Step 1: Extract and prepare MIMIC-IV data
echo -e "\n[Step 1] Extracting MIMIC-IV data..."
python 01_prepare_full_mimiciv.py

# Step 2: Adapt column names
echo -e "\n[Step 2] Adapting column names..."
python 02_adapt_full_mimic4_columns.py

# Step 3: Extract subjects using benchmark tools
echo -e "\n[Step 3] Extracting subjects..."
cd /scratch/bcew/ruikez2/intern/mimic-iv-benchmarks
python -m mimic4benchmark.scripts.extract_subjects \
    /scratch/bcew/ruikez2/intern/s3_med/data/mimiciv_full_adapted \
    /scratch/bcew/ruikez2/intern/s3_med/data/mimic4_benchmark_output_full/

# Step 4: Extract episodes
echo -e "\n[Step 4] Extracting episodes..."
python -m mimic4benchmark.scripts.extract_episodes_from_subjects \
    /scratch/bcew/ruikez2/intern/s3_med/data/mimic4_benchmark_output_full/

# Step 5: Convert to JSON format
echo -e "\n[Step 5] Converting to JSON format..."
cd /scratch/bcew/ruikez2/intern/s3_med/scripts/full_data_pipeline
python 03_convert_full_episodes_to_json.py

# Step 6: Add temporal information
echo -e "\n[Step 6] Adding temporal information..."
python 04_add_temporal_info_full.py

# Step 7: Split dataset
echo -e "\n[Step 7] Splitting dataset (8:1:1)..."
python 05_split_dataset.py

# Step 8: Convert to text format for baselines
echo -e "\n[Step 8] Converting JSON to text format for baselines..."
python 06_convert_json_to_text.py

echo -e "\n========================================="
echo "Pipeline completed successfully!"
echo "========================================="
echo "Output files:"
echo "  JSON format:"
echo "  - /scratch/bcew/ruikez2/intern/s3_med/data/chartevents_matrices${SUFFIX}.json"
echo "  - /scratch/bcew/ruikez2/intern/s3_med/data/patient_info${SUFFIX}.json"
echo "  - /scratch/bcew/ruikez2/intern/s3_med/data/patient_data_with_temporal${SUFFIX}.json"
echo "  - /scratch/bcew/ruikez2/intern/s3_med/data/splits${SUFFIX}/train_data.json"
echo "  - /scratch/bcew/ruikez2/intern/s3_med/data/splits${SUFFIX}/val_data.json"
echo "  - /scratch/bcew/ruikez2/intern/s3_med/data/splits${SUFFIX}/test_data.json"
echo "  Text format:"
echo "  - /scratch/bcew/ruikez2/intern/s3_med/data/text_data${SUFFIX}/train_text.jsonl"
echo "  - /scratch/bcew/ruikez2/intern/s3_med/data/text_data${SUFFIX}/val_text.jsonl"
echo "  - /scratch/bcew/ruikez2/intern/s3_med/data/text_data${SUFFIX}/test_text.jsonl"