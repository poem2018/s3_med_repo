#!/bin/bash

# Prepare MIMIC data for S3 training

DATA_DIR="/scratch/bcew/ruikez2/intern/s3_med/patient_matrices_optimized"
OUTPUT_DIR="./data/mimic_mortality"

echo "Preparing MIMIC mortality prediction data for S3 framework..."

# Create output directory
mkdir -p $OUTPUT_DIR

# Generate training data with patient clinical summaries
python scripts/data_process/train_mimic_ug.py \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --retriever e5 \
    --train_ratio 0.8 \
    --val_ratio 0.1

echo "Data preparation complete!"
echo "Files created in: $OUTPUT_DIR"
ls -la $OUTPUT_DIR/*.parquet