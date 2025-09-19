#!/bin/bash

# Run MIMIC Searcher Experiments
# Three groups: 
# 1. Naive RAG
# 2. Searcher-optimized RAG  
# 3. Searcher-optimized RAG with similar cases

echo "========================================"
echo "MIMIC Searcher Experiments"
echo "========================================"

# Default values
DEMO_FILE="scripts/demo_data_template.json"
OUTPUT_DIR="experiments/searcher_comparison"
RETRIEVAL_ENDPOINT="http://127.0.0.1:3000/retrieve"
TOPK=12

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --demo_file)
            DEMO_FILE="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --endpoint)
            RETRIEVAL_ENDPOINT="$2"
            shift 2
            ;;
        --topk)
            TOPK="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--demo_file FILE] [--output_dir DIR] [--endpoint URL] [--topk K]"
            exit 1
            ;;
    esac
done

# Check if retrieval service is running
echo "Checking retrieval service at $RETRIEVAL_ENDPOINT..."
if curl -s -f -X POST "$RETRIEVAL_ENDPOINT" \
    -H "Content-Type: application/json" \
    -d '{"queries": ["test"], "topk": 1}' > /dev/null 2>&1; then
    echo "✓ Retrieval service is running"
else
    echo "✗ Retrieval service is not running at $RETRIEVAL_ENDPOINT"
    echo "Please start the retrieval service first:"
    echo "  bash ./scripts/deploy_retriever/retrieval_launch.sh"
    exit 1
fi

# Check if demo file exists
if [ ! -f "$DEMO_FILE" ]; then
    echo "✗ Demo file not found: $DEMO_FILE"
    echo "Please create a demo data file with patient cases"
    echo "See scripts/demo_data_template.json for an example"
    exit 1
fi

echo "✓ Demo file found: $DEMO_FILE"

# Create output directory
mkdir -p "$OUTPUT_DIR"
echo "✓ Output directory created: $OUTPUT_DIR"

# Run experiments
echo ""
echo "Running experiments..."
echo "----------------------------------------"
echo "Configuration:"
echo "  Demo file: $DEMO_FILE"
echo "  Output directory: $OUTPUT_DIR"
echo "  Retrieval endpoint: $RETRIEVAL_ENDPOINT"
echo "  Top-k documents: $TOPK"
echo "----------------------------------------"
echo ""

python scripts/run_mimic_searcher_experiments.py \
    --demo_file "$DEMO_FILE" \
    --retrieval_endpoint "$RETRIEVAL_ENDPOINT" \
    --output_dir "$OUTPUT_DIR" \
    --topk "$TOPK"

# Check if experiments completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✓ Experiments completed successfully!"
    echo "========================================"
    echo "Results saved in: $OUTPUT_DIR"
    echo ""
    echo "Files generated:"
    echo "  - Individual case results: $OUTPUT_DIR/*_results.json"
    echo "  - Combined results: $OUTPUT_DIR/all_experiment_results.json"
    echo "  - Comparison report: $OUTPUT_DIR/comparison_report.txt"
    echo ""
    echo "To view the comparison report:"
    echo "  cat $OUTPUT_DIR/comparison_report.txt"
else
    echo ""
    echo "✗ Experiments failed. Please check the error messages above."
    exit 1
fi