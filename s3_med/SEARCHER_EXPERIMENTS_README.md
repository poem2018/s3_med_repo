# MIMIC Searcher Experiments

This directory contains scripts for running three groups of retrieval experiments on MIMIC ICU mortality prediction data:

1. **Naive RAG**: Direct retrieval using question + patient data
2. **Searcher-optimized RAG**: Use S3 searcher to optimize the query before retrieval
3. **Searcher-optimized RAG with similar cases**: Include similar patient cases, optimize with searcher, then retrieve

## Prerequisites

1. **Start the retrieval service**:
   ```bash
   bash ./scripts/deploy_retriever/retrieval_launch.sh
   ```
   The service should be running at `http://127.0.0.1:3000`

2. **Prepare your demo data**:
   - Use `scripts/demo_data_template.json` as a template
   - Each patient case should include:
     - `query`: The question + patient data
     - `similar_case`: (Optional) A similar patient case with outcome
     - `ground_truth`: The actual mortality outcome

## Quick Start

Run all three experiment groups:
```bash
bash scripts/run_searcher_experiments.sh
```

## Custom Configuration

```bash
bash scripts/run_searcher_experiments.sh \
    --demo_file path/to/your/demo.json \
    --output_dir path/to/output \
    --endpoint http://your-retrieval-endpoint:port/retrieve \
    --topk 12
```

## Python Script Usage

You can also run the experiments directly with Python:

```python
python scripts/run_mimic_searcher_experiments.py \
    --demo_file scripts/demo_data_template.json \
    --retrieval_endpoint http://127.0.0.1:3000/retrieve \
    --output_dir experiments/searcher_comparison \
    --topk 12
```

## Demo Data Format

The demo data should be a JSON file with the following structure:

```json
{
  "patient_id": {
    "query": "<question>...</question>\n\n<patient_data>...</patient_data>",
    "similar_case": "Similar Patient Case:\n...\nOutcome: ...",
    "ground_truth": {
      "mortality_label": "low/high mortality risk",
      "mortality": 0 or 1
    }
  }
}
```

## Output Files

The experiments will generate:

1. **Individual case results**: `{case_id}_results.json`
   - Contains results for all three methods for each case

2. **Combined results**: `all_experiment_results.json`
   - All cases combined in a single file

3. **Comparison report**: `comparison_report.txt`
   - Human-readable comparison of the three methods

## Understanding the Results

Each result contains:
- `method`: The experiment group (naive_rag, searcher_optimized, searcher_with_similar)
- `original_query`: The input query
- `final_query`: The query actually used for retrieval (optimized for groups 2 & 3)
- `documents`: Retrieved documents with titles, text, and scores

## Searcher Optimization

The searcher optimization process:
1. Extracts key clinical features (age, gender, conditions)
2. Identifies critical symptoms (hyperglycemia, acidosis, mechanical ventilation)
3. Incorporates similar case outcomes when available
4. Generates a focused search query for better retrieval

## Customizing Searcher Logic

To use a trained S3 searcher model instead of the rule-based optimization:

1. Modify the `_run_searcher_optimization` method in `run_mimic_searcher_experiments.py`
2. Load your trained searcher model
3. Use the model to generate optimized queries

Example:
```python
def _run_searcher_optimization(self, input_text: str) -> str:
    # Load your S3 searcher model
    if self.searcher_model:
        # Use trained model for optimization
        optimized_query = self.searcher_model.generate(input_text)
    else:
        # Use rule-based optimization (current implementation)
        optimized_query = self._extract_clinical_features(input_text)
    return optimized_query
```

## Evaluation

To evaluate the quality of retrieved documents:
1. Check relevance to the patient's condition
2. Compare document rankings across methods
3. Assess whether documents support the mortality prediction

## Troubleshooting

1. **Connection refused error**:
   - Ensure the retrieval service is running
   - Check the endpoint URL and port

2. **Empty retrieval results**:
   - Verify the corpus is properly indexed
   - Check query length (may be too long)

3. **Searcher optimization not working**:
   - Review the clinical feature extraction logic
   - Ensure proper formatting of patient data