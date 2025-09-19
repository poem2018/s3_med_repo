#!/bin/bash
# Precompute the naïve RAG Cache for MIMIC mortality prediction training

echo "=== IMPORTANT: Before running this script ==="
echo "1. Start the retrieval service: bash scripts/deploy_retriever/retrieval_launch.sh"
echo "2. Start the generator LLM: bash generator_llms/host.sh"
echo "3. Wait for both services to be ready before proceeding"
echo ""
read -p "Press Enter once both services are running..."

echo "=== Step 1: Dataset already constructed with train_mimic_ug_new.py ==="
# The parquet files should already exist at:
# - data/mimic_mortality_ug/train_e5_ug.parquet
# - data/mimic_mortality_ug/val_e5_ug.parquet
# - data/mimic_mortality_ug/test_e5_ug.parquet

echo "=== Step 2: Running E5 Retrieval for MIMIC Training Set ==="
python scripts/baselines/e5_retrieval_mimic.py \
    --input_parquet data/mimic_mortality_ug/train_e5_ug.parquet \
    --output_dir data/RAG_retrieval/mimic_train \
    --endpoint http://127.0.0.1:3000/retrieve

echo "=== Step 3: Running E5 Retrieval for MIMIC Validation Set ==="
python scripts/baselines/e5_retrieval_mimic.py \
    --input_parquet data/mimic_mortality_ug/val_e5_ug.parquet \
    --output_dir data/RAG_retrieval/mimic_val \
    --endpoint http://127.0.0.1:3000/retrieve

echo "=== Step 4: Running E5 Retrieval for MIMIC Test Set ==="
python scripts/baselines/e5_retrieval_mimic.py \
    --input_parquet data/mimic_mortality_ug/test_e5_ug.parquet \
    --output_dir data/RAG_retrieval/mimic_test \
    --endpoint http://127.0.0.1:3000/retrieve

echo "=== Step 5: Generate RAG Cache with Generator LLM ==="
# Make sure the generator LLM is running first!
# bash generator_llms/host.sh

python scripts/evaluation/context_mimic.py \
    --input_file data/mimic_mortality_ug/train_e5_ug.parquet \
    --result_file data/rag_cache/mimic_rag_cache.json \
    --context_dir data/RAG_retrieval/mimic_train \
    --num_workers 16 \
    --topk 3 \
    --model Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4

echo "=== Precomputation Complete ==="
echo "RAG retrieval results saved in: data/RAG_retrieval/mimic_*"
echo "RAG cache saved in: data/rag_cache/mimic_rag_cache.json"