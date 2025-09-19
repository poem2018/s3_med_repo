ran#!/bin/bash
# Run all baseline experiments for ICU mortality prediction

set -e  # Exit on error

echo "========================================="
echo "Running All Baseline Experiments"
echo "========================================="

# Get optional parameter for number of test samples
NUM_TEST_SAMPLES=300 #${1:-"all"}  # Default to "all" if not provided

# Configuration
DATA_DIR="/scratch/bcew/ruikez2/intern/s3_med/data/baseline_text"
RESULTS_DIR="/scratch/bcew/ruikez2/intern/s3_med/results"
SCRIPTS_DIR="/scratch/bcew/ruikez2/intern/s3_med/scripts/baseline_s3_med"

# Add num_samples parameter if specified
if [ "$NUM_TEST_SAMPLES" != "all" ]; then
    echo "Processing only $NUM_TEST_SAMPLES test samples"
    MAX_SAMPLES_ARGS="--max_patients $NUM_TEST_SAMPLES"
else
    echo "Processing all test samples"
    MAX_SAMPLES_ARGS=""
fi

# Retrieval endpoint (update this based on your deployment)
RETRIEVAL_ENDPOINT="http://127.0.0.1:3000/retrieve"

# Create results directory
mkdir -p $RESULTS_DIR

echo ""
echo "Data directory: $DATA_DIR"
echo "Results will be saved to: $RESULTS_DIR"
echo ""

# ============================================
# Baseline 1: Traditional ML Classification
# ============================================
echo "========================================="
echo "[1/4] Running Baseline 1: ML Classification"
echo "========================================="
# python $SCRIPTS_DIR/baseline_1_ml_classification.py \
#     --data_dir $DATA_DIR \
#     --output_dir $RESULTS_DIR/baseline_1 \
#     --test_size 0.2 \
#     --random_seed 42

# echo "✓ Baseline 1 completed"
# echo ""

# ============================================
# Baseline 2: Direct LLM Prediction
# ============================================
echo "========================================="
echo "[2/4] Running Baseline 2: Direct LLM"
echo "========================================="

# Check if LLM server is running
echo "Checking LLM server at localhost:8000..."
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✓ LLM server is running"
else
    echo "⚠ Warning: LLM server may not be running at localhost:8000"
    echo "Please ensure vLLM is running with Qwen2.5-3B-Instruct"
    echo "You can start it with: python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-3B-Instruct --port 8000"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# # Run with direct prompting
# python $SCRIPTS_DIR/baseline_2_direct_llm.py \
#     --data_dir $DATA_DIR \
#     --output_dir $RESULTS_DIR/baseline_2 \
#     --prompt_type direct \
#     --temperature 0.3 \
#     $MAX_SAMPLES_ARGS

# # Run with chain-of-thought prompting
# python $SCRIPTS_DIR/baseline_2_direct_llm.py \
#     --data_dir $DATA_DIR \
#     --output_dir $RESULTS_DIR/baseline_2 \
#     --prompt_type cot \
#     --temperature 0.3 \
#     $MAX_SAMPLES_ARGS 

#  # Run with logprobs-based confidence
# echo "Running Baseline 2 with token probabilities..."
# python $SCRIPTS_DIR/baseline_2_logprobs.py \
#     --data_dir $DATA_DIR \
#     --output_dir $RESULTS_DIR/baseline_2_logprobs \
#     --prompt_type cot \
#     --temperature 0.3 \
#     $MAX_SAMPLES_ARGS

# echo "✓ Baseline 2 completed (direct, CoT, and logprobs)"
# echo ""
# # ============================================
# # Baseline 3: Naive RAG + LLM
# # ============================================
# echo "========================================="
# echo "[3/4] Running Baseline 3: Naive RAG + LLM"
# echo "========================================="

# # Check if retrieval server is running
# echo "Checking retrieval server at $RETRIEVAL_ENDPOINT..."
# if curl -s $RETRIEVAL_ENDPOINT > /dev/null 2>&1; then
#     echo "✓ Retrieval server is running"
# else
#     echo "⚠ Warning: Retrieval server may not be running at $RETRIEVAL_ENDPOINT"
#     echo "Please ensure the medical knowledge retrieval server is running"
#     echo "You can start it with: bash ./scripts/deploy_retriever/retrieval_launch.sh"
#     read -p "Continue anyway? (y/n) " -n 1 -r
#     echo
#     if [[ ! $REPLY =~ ^[Yy]$ ]]; then
#         exit 1
#     fi
# fi

# # python $SCRIPTS_DIR/baseline_3_naive_rag.py \
# #     --data_dir $DATA_DIR \
# #     --output_dir $RESULTS_DIR/baseline_3 \
# #     --retrieval_endpoint $RETRIEVAL_ENDPOINT \
# #     --topk 5 \
# #     --temperature 0.3 \
# #     $MAX_SAMPLES_ARGS

# # echo "✓ Baseline 3 completed"

# # Run with logprobs-based confidence
# echo "Running Baseline 3 with token probabilities..."
# python $SCRIPTS_DIR/baseline_3_logprobs.py \
#     --data_dir $DATA_DIR \
#     --output_dir $RESULTS_DIR/baseline_3_logprobs \
#     --retrieval_endpoint $RETRIEVAL_ENDPOINT \
#     --prompt_type cot \
#     --topk 5 \
#     --temperature 0.3 \
#     --debug \
#     $MAX_SAMPLES_ARGS

# echo "✓ Baseline 3 completed (standard and logprobs)"
# echo ""

# # ============================================
# # Baseline 4: Similar Cases + RAG + LLM
# # ============================================
# echo "========================================="
# echo "[4/5] Running Baseline 4: Similar Cases + RAG + LLM"
# echo "========================================="

# # python $SCRIPTS_DIR/baseline_4_similar_cases_rag.py \
# #     --data_dir $DATA_DIR \
# #     --output_dir $RESULTS_DIR/baseline_4 \
# #     --retrieval_endpoint $RETRIEVAL_ENDPOINT \
# #     --k_similar 3 \
# #     --topk_docs 3 \
# #     --temperature 0.3 \
# #     $MAX_SAMPLES_ARGS

# # echo "✓ Baseline 4 completed"

# # Run with logprobs-based confidence
# echo "Running Baseline 4 with token probabilities..."
# python $SCRIPTS_DIR/baseline_4_logprobs.py \
#     --data_dir $DATA_DIR \
#     --output_dir $RESULTS_DIR/baseline_4_logprobs \
#     --retrieval_endpoint $RETRIEVAL_ENDPOINT \
#     --prompt_type cot \
#     --k_similar 3 \
#     --topk_docs 3 \
#     --temperature 0.3 \
#     $MAX_SAMPLES_ARGS

# echo "✓ Baseline 4 completed (standard and logprobs)"
# echo ""

# ============================================
# Baseline 5: Similar Cases Only (No RAG)
# ============================================
echo "========================================="
echo "[5/5] Running Baseline 5: Similar Cases Only"
echo "========================================="

# Run with logprobs-based confidence
echo "Running Baseline 5 with token probabilities..."
python $SCRIPTS_DIR/baseline_5_logprobs.py \
    --data_dir $DATA_DIR \
    --output_dir $RESULTS_DIR/baseline_5_logprobs \
    --prompt_type cot \
    --k_similar 3 \
    --temperature 0.3 \
    $MAX_SAMPLES_ARGS

echo "✓ Baseline 5 completed"
echo ""

# ============================================
# Summary
# ============================================
echo "========================================="
echo "All Baselines Completed Successfully!"
echo "========================================="
echo ""
echo "Results saved in:"
echo "  - $RESULTS_DIR/baseline_1/"
echo "  - $RESULTS_DIR/baseline_2/"
echo "  - $RESULTS_DIR/baseline_2_logprobs/"
echo "  - $RESULTS_DIR/baseline_3/"
echo "  - $RESULTS_DIR/baseline_3_logprobs/"
echo "  - $RESULTS_DIR/baseline_4/"
echo "  - $RESULTS_DIR/baseline_4_logprobs/"
echo "  - $RESULTS_DIR/baseline_5_logprobs/"
echo ""

# Generate summary report
# echo "Generating summary report..."
# python $SCRIPTS_DIR/summarize_results.py \
#     --results_dir $RESULTS_DIR \
#     --output_file $RESULTS_DIR/baseline_summary.txt

# echo "Summary report saved to: $RESULTS_DIR/baseline_summary.txt"
echo ""
echo "Done!"

