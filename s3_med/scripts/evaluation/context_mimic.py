#!/usr/bin/env python3
"""
Generate RAG cache for MIMIC mortality prediction data
Adapted from context.py for MIMIC-specific data format
"""

import os
import pandas as pd
import json
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple

# Import the generate_answer function from the original module
import sys
sys.path.append('/scratch/bcew/ruikez2/intern/s3_med')
from verl.utils.reward_score.rag_2 import generate_answer

def setup_logger(log_file):
    logger = logging.getLogger('mimic_context_processor')
    logger.setLevel(logging.INFO)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def load_context_cache(context_dir: str, data_sources: List[str], logger) -> Dict[str, Dict]:
    """Load all context files into memory at startup"""
    logger.info("Loading context cache...")
    cache = {}
    for source in data_sources:
        context_file = os.path.join(context_dir, f"{source}_output_sequences.json")
        if os.path.exists(context_file):
            logger.info(f"Loading context file: {context_file}")
            with open(context_file, 'r') as f:
                cache[source] = json.load(f)
                logger.info(f"Loaded {len(cache[source])} entries for {source}")
        else:
            logger.warning(f"Context file not found: {context_file}")
    logger.info("Context cache loading complete")
    return cache

def process_questions_batch(questions_batch: List[Dict], context_cache: Dict, topk: int, model: str, logger) -> List[Dict]:
    """Process a batch of questions for MIMIC data"""
    results = []
    
    for row in questions_batch:
        try:
            # Extract information from the row
            ground_truth = row['reward_model']['ground_truth']
            patient_id = ground_truth['patient_id']
            clinical_summary = ground_truth.get('question', ground_truth.get('clinical_summary', ''))
            mortality = ground_truth.get('mortality', 0)
            data_source = row['data_source']
            
            # Get context from cache using patient_id (ensure it's a string)
            patient_context = context_cache.get(data_source, {}).get(str(patient_id), {})
            if not patient_context:
                logger.warning(f"No context found for patient {patient_id}")
                results.append({
                    'patient_id': str(patient_id),
                    'answer': None,
                    'score': 0,
                    'data_source': data_source
                })
                continue
            
            # Get the retrieved context
            context = patient_context.get('context_with_info', '')
            if context:
                # Limit context to topk documents
                context = context.split(f'Doc {topk+1}')[0]
            
            # Create prompt for mortality prediction
            prompt = f"""Based on the following medical literature and patient information, predict if this ICU patient will survive or die during their ICU stay.

Medical Literature:
{context}

Patient Information:
{clinical_summary}

Question: Will this patient die in the ICU? Answer with only "Yes" or "No"."""

            # Generate answer using the model
            answer = generate_answer(prompt, None, model)
            
            # Evaluate answer
            predicted_mortality = 1 if answer.lower().strip() in ["yes", "y", "true", "1"] else 0
            is_correct = predicted_mortality == mortality
            
            results.append({
                'patient_id': str(patient_id),
                'answer': answer,
                'score': 1 if is_correct else 0,
                'data_source': data_source,
                'true_mortality': mortality,
                'predicted_mortality': predicted_mortality
            })
            
        except Exception as e:
            logger.error(f"Error processing patient {row.get('patient_id', 'unknown')}: {str(e)}")
            results.append({
                'patient_id': row.get('patient_id', 'unknown'),
                'answer': None,
                'score': 0,
                'data_source': row.get('data_source', 'unknown')
            })
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Generate RAG cache for MIMIC mortality prediction")
    parser.add_argument("--input_file", required=True, help="Input parquet file with MIMIC data")
    parser.add_argument("--result_file", required=True, help="Output JSON file for RAG cache")
    parser.add_argument("--context_dir", required=True, help="Directory containing retrieval results")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of parallel workers")
    parser.add_argument("--topk", type=int, default=3, help="Number of top documents to use")
    parser.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4", help="Model to use for generation")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for processing")
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = args.result_file.replace('.json', '.log')
    logger = setup_logger(log_file)
    
    logger.info("Starting MIMIC RAG cache generation")
    logger.info(f"Input file: {args.input_file}")
    logger.info(f"Result file: {args.result_file}")
    logger.info(f"Context directory: {args.context_dir}")
    
    # Load data
    df = pd.read_parquet(args.input_file)
    logger.info(f"Loaded {len(df)} entries from parquet file")
    
    # Get unique data sources
    data_sources = df['data_source'].unique()
    logger.info(f"Found data sources: {data_sources}")
    
    # Load context cache
    context_cache = load_context_cache(args.context_dir, data_sources, logger)
    
    # Initialize results
    rag_cache = {}
    
    # Convert DataFrame to list of dicts for processing
    rows = df.to_dict('records')
    
    # Process in batches with parallel workers
    logger.info(f"Processing {len(rows)} patients with {args.num_workers} workers...")
    
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        # Create batches
        batches = [rows[i:i+args.batch_size] for i in range(0, len(rows), args.batch_size)]
        logger.info(f"Created {len(batches)} batches of size {args.batch_size}")
        
        # Submit batches to process pool
        futures = {executor.submit(process_questions_batch, batch, context_cache, args.topk, args.model, logger): i 
                  for i, batch in enumerate(batches)}
        
        # Process results as they complete
        with tqdm(total=len(rows)) as pbar:
            for future in as_completed(futures):
                try:
                    batch_results = future.result()
                    pbar.update(len(batch_results))
                    
                    # Store results in RAG cache
                    for result in batch_results:
                        patient_id = str(result['patient_id'])
                        # Format similar to original cache structure
                        rag_cache[patient_id] = {
                            'answer': result['answer'],
                            'score': result['score']
                        }
                        
                except Exception as e:
                    logger.error(f"Error processing batch: {str(e)}")
    
    # Save RAG cache
    logger.info(f"Saving RAG cache to {args.result_file}")
    os.makedirs(os.path.dirname(args.result_file), exist_ok=True)
    with open(args.result_file, 'w') as f:
        json.dump(rag_cache, f, indent=2)
    
    # Calculate and log statistics
    total = len(rag_cache)
    correct = sum(1 for v in rag_cache.values() if v['score'] == 1)
    accuracy = correct / total if total > 0 else 0
    
    logger.info(f"Processing complete!")
    logger.info(f"Total patients: {total}")
    logger.info(f"Correct predictions: {correct}")
    logger.info(f"Accuracy: {accuracy:.2%}")
    
    # Save statistics
    stats_file = args.result_file.replace('.json', '_stats.json')
    stats = {
        'timestamp': datetime.now().isoformat(),
        'total_patients': total,
        'correct_predictions': correct,
        'accuracy': accuracy,
        'model': args.model,
        'topk': args.topk
    }
    
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Statistics saved to {stats_file}")

if __name__ == "__main__":
    main()