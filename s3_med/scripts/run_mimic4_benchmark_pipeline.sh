#!/bin/bash
# Run MIMIC-IV benchmark pipeline on demo data

# Set paths
MIMIC_DATA="/scratch/bcew/ruikez2/intern/s3_med/data/mimiciv_demo_benchmark"
OUTPUT_ROOT="/scratch/bcew/ruikez2/intern/s3_med/data/mimic4_benchmark_output"
BENCHMARK_PATH="/scratch/bcew/ruikez2/intern/mimic-iv-benchmarks"

# Create output directory
mkdir -p $OUTPUT_ROOT

# Change to benchmark directory
cd $BENCHMARK_PATH

echo "Step 1: Extract subjects..."
python -m mimic4benchmark.scripts.extract_subjects $MIMIC_DATA $OUTPUT_ROOT/

echo "Step 2: Validate events..."
python -m mimic4benchmark.scripts.validate_events $OUTPUT_ROOT/

echo "Step 3: Extract episodes from subjects..."
python -m mimic4benchmark.scripts.extract_episodes_from_subjects $OUTPUT_ROOT/

echo "Step 4: Split train and test..."
python -m mimic4benchmark.scripts.split_train_and_test $OUTPUT_ROOT/

echo "Done! Episodes have been extracted to $OUTPUT_ROOT"