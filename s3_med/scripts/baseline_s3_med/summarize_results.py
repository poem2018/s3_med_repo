#!/usr/bin/env python3
"""
Summarize results from all baseline experiments
"""

import json
import argparse
import os
from pathlib import Path
from datetime import datetime

def load_results(results_dir, baseline_name, result_file):
    """Load results from a specific baseline"""
    file_path = os.path.join(results_dir, baseline_name, result_file)
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return None

def format_metrics(metrics, name):
    """Format metrics for display"""
    if not metrics:
        return f"{name}: No results found\n"
    
    result = f"\n{name}:\n"
    result += f"  Accuracy:  {metrics.get('accuracy', 0.0):.3f}\n"
    result += f"  Precision: {metrics.get('precision', 0.0):.3f}\n"
    result += f"  Recall:    {metrics.get('recall', 0.0):.3f}\n"
    result += f"  F1 Score:  {metrics.get('f1', 0.0):.3f}\n"
    result += f"  AUROC:     {metrics.get('auc', 0.0):.3f}\n"
    result += f"  AUPRC:     {metrics.get('auprc', 0.0):.3f}\n"
    return result

def main():
    parser = argparse.ArgumentParser(description="Summarize baseline results")
    parser.add_argument("--results_dir", default="/scratch/bcew/ruikez2/intern/s3_med/results",
                        help="Directory containing results")
    parser.add_argument("--output_file", default="/scratch/bcew/ruikez2/intern/s3_med/results/baseline_summary.txt",
                        help="Output file for summary")
    args = parser.parse_args()
    
    # Define baselines to check
    baselines = [
        # Baseline 1 - ML Classification
        ("baseline_1", "baseline_1_results.json", "Baseline 1: ML Classification"),
        
        # Baseline 2 - Direct LLM
        ("baseline_2", "baseline_2_direct_results.json", "Baseline 2: Direct LLM"),
        ("baseline_2", "baseline_2_cot_results.json", "Baseline 2: Chain-of-Thought"),
        ("baseline_2_logprobs", "baseline_2_logprobs_cot_results.json", "Baseline 2: Logprobs"),

        # Baseline 3 - Naive RAG
        ("baseline_3", "baseline_3_naive_rag_results.json", "Baseline 3: Naive RAG"),
        ("baseline_3_logprobs", "baseline_3_logprobs_cot_results.json", "Baseline 3: RAG + Logprobs"),

        # Baseline 4 - Similar Cases + RAG
        ("baseline_4", "baseline_4_similar_rag_results.json", "Baseline 4: Similar Cases + RAG"),
        ("baseline_4_logprobs", "baseline_4_logprobs_cot_results.json", "Baseline 4: Similar Cases + RAG + Logprobs"),
    ]
    
    summary = []
    summary.append("="*60)
    summary.append("ICU MORTALITY PREDICTION - BASELINE RESULTS SUMMARY")
    summary.append("="*60)
    summary.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary.append("")
    
    # Collect all results
    all_results = []
    for baseline_dir, result_file, name in baselines:
        results = load_results(args.results_dir, baseline_dir, result_file)
        if results and 'metrics' in results:
            metrics = results['metrics']
            all_results.append((name, metrics))
            summary.append(format_metrics(metrics, name))
    
    # Find best performing model for each metric
    if all_results:
        summary.append("\n" + "="*60)
        summary.append("BEST PERFORMING MODELS")
        summary.append("="*60)
        
        metric_names = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'auprc']
        metric_display = {
            'accuracy': 'Accuracy',
            'precision': 'Precision',
            'recall': 'Recall',
            'f1': 'F1 Score',
            'auc': 'AUROC',
            'auprc': 'AUPRC'
        }
        
        for metric in metric_names:
            best_model = max(all_results, key=lambda x: x[1].get(metric, 0.0))
            best_value = best_model[1].get(metric, 0.0)
            summary.append(f"{metric_display[metric]:12s}: {best_model[0]:40s} = {best_value:.3f}")
    
    # Add comparison table
    if all_results:
        summary.append("\n" + "="*60)
        summary.append("COMPARISON TABLE")
        summary.append("="*60)
        
        # Header
        header = f"{'Model':40s} | {'Acc':5s} | {'Prec':5s} | {'Rec':5s} | {'F1':5s} | {'AUC':5s} | {'AUPRC':5s}"
        summary.append(header)
        summary.append("-"*len(header))
        
        # Data rows
        for name, metrics in all_results:
            row = f"{name[:40]:40s} | "
            row += f"{metrics.get('accuracy', 0.0):.3f} | "
            row += f"{metrics.get('precision', 0.0):.3f} | "
            row += f"{metrics.get('recall', 0.0):.3f} | "
            row += f"{metrics.get('f1', 0.0):.3f} | "
            row += f"{metrics.get('auc', 0.0):.3f} | "
            row += f"{metrics.get('auprc', 0.0):.3f}"
            summary.append(row)
    
    # Class imbalance note
    summary.append("\n" + "="*60)
    summary.append("NOTES")
    summary.append("="*60)
    summary.append("- Dataset has class imbalance: ~87% survive, ~13% mortality")
    summary.append("- AUPRC may be more informative than AUROC for this imbalanced dataset")
    summary.append("- Logprobs versions use token probabilities for confidence scores")
    summary.append("- Higher confidence scores should correlate with better AUROC/AUPRC")
    
    # Write summary to file
    summary_text = "\n".join(summary)
    with open(args.output_file, 'w') as f:
        f.write(summary_text)
    
    # Also print to console
    print(summary_text)
    print(f"\nSummary saved to: {args.output_file}")

if __name__ == "__main__":
    main()