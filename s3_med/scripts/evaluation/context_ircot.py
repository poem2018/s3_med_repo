import os
import random
import pandas as pd
import json
from verl.utils.reward_score.rag_2 import generate_answer, check_answer_correct, em_check, generate_answer_zero_shot
from verl.utils.hdfs_io import copy, makedirs
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
import time
from datetime import datetime
import logging
import numpy as np
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor

# MODEL = "Qwen/Qwen2.5-7B-Instruct"
MODEL = "Claude-Haiku"

# Configure logging
def setup_logger(log_file):
    logger = logging.getLogger('context_processor')
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

def load_previous_results(result_file, logger):
    """Load previous results if they exist"""
    logger.info(f"Checking for previous results at: {result_file}")
    if os.path.exists(result_file):
        logger.info(f"Loading previous results from {result_file}")
        with open(result_file, 'r') as f:
            return json.load(f)
    logger.info("No previous results found")
    return {}

def save_results(answers, result_file, stats_file, total_questions, data_source_stats, logger):
    """Save results and statistics"""
    try:
        # Ensure directories exist
        cache_dir = os.path.dirname(result_file)
        stats_dir = os.path.dirname(stats_file)
        
        logger.info(f"Creating directories if needed:")
        logger.info(f"Cache directory: {cache_dir}")
        logger.info(f"Stats directory: {stats_dir}")
        
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(stats_dir, exist_ok=True)
        
        # Save answers
        logger.info(f"Saving answers to: {result_file}")
        with open(result_file, 'w') as f:
            json.dump(answers, f)
        
        # Save statistics
        stats = {
            'timestamp': datetime.now().isoformat(),
            'total_questions': total_questions,
            'processed_questions': sum(len(answers) for answers in answers.values()),
            'data_source_stats': data_source_stats,
            'remaining_questions': total_questions - sum(len(answers) for answers in answers.values())
        }
        
        logger.info(f"Saving statistics to: {stats_file}")
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
            
        logger.info("Save completed successfully")
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        raise


def process_questions_batch(questions_batch: List[Tuple], context_cache: Dict, logger) -> List[Dict]:
    """Process a batch of questions using batched API calls"""
    results = []
    
    # Prepare prompts for the batch
    prompts = []
    for row in questions_batch:
        question = row['reward_model']['ground_truth']['question']
        data_source = row['data_source']
        
        # Get context from cache
        context = context_cache.get(question, {}).get('all_context', '')
        
        if not context:
            # Skip zero-shot, set score to 0
            results.append({
                'question': question,
                'answer': None,
                'is_correct': False,
                'is_em': False,
                'data_source': data_source
            })
            continue
            
        # Context-based prompt
        prompts.append((question, context, row))
    
    # Process all prompts in parallel using ThreadPoolExecutor
    # Using 16 threads per batch since we have many available cores
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = []
        for question, context, row in prompts:
            future = executor.submit(process_single_question, question, context, row)
            futures.append(future)
        
        # Collect results as they complete
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing question: {str(e)}")
                # Add error result
                results.append({
                    'question': question,
                    'answer': None,
                    'is_correct': False,
                    'is_em': False,
                    'data_source': row['data_source']
                })
    
    # Ensure we return results for all questions in the batch
    if len(results) < len(questions_batch):
        logger.warning(f"Batch processing incomplete: {len(results)}/{len(questions_batch)} questions processed")
        # Add placeholder results for any missing questions
        for row in questions_batch:
            question = row['reward_model']['ground_truth']['question']
            if not any(r['question'] == question for r in results):
                results.append({
                    'question': question,
                    'answer': None,
                    'is_correct': False,
                    'is_em': False,
                    'data_source': row['data_source']
                })
    
    return results

def process_single_question(question: str, context: str, row: Dict) -> Dict:
    """Process a single question with its context"""
    try:
        answer = generate_answer(prompt=question, context=context, model=MODEL)
        golden_answers = row['reward_model']['ground_truth']['target']
        
        # Check if answer is correct
        is_correct = check_answer_correct(answer=answer, golden_answers=golden_answers, model=MODEL)
        is_em = em_check(prediction=answer, golden_answers=golden_answers)
        
        return {
            'question': question,
            'answer': answer,
            'is_correct': is_correct,
            'is_em': is_em,
            'data_source': row['data_source']
        }
    except Exception as e:
        raise Exception(f"Error processing question: {str(e)}")

def process_dataset(input_file: str, result_file: str, context_file: str, num_workers: int = 16, random_seed: int = 42, sampling_enabled: bool = False):
    # Setup logger
    log_file = result_file.replace('.json', '.log')
    logger = setup_logger(log_file)
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Load the dataset
    logger.info(f"Loading dataset from {input_file}")
    df = pd.read_parquet(input_file)
    
    if sampling_enabled:
        # Sample 3000 questions per data source if more than 3000 exist
        sampled_dfs = []
        for data_source in df['data_source'].unique():
            source_df = df[df['data_source'] == data_source]
            if len(source_df) > 3000:
                logger.info(f"Sampling 3000 questions from {data_source} (total: {len(source_df)})")
                sampled_df = source_df.sample(n=3000, random_state=random_seed)
            else:
                logger.info(f"Using all {len(source_df)} questions from {data_source}")
                sampled_df = source_df
            sampled_dfs.append(sampled_df)
        
        # Combine sampled dataframes
        df = pd.concat(sampled_dfs, ignore_index=True)
        logger.info(f"Total questions after sampling: {len(df)}")
    else:
        logger.info("Sampling disabled - using all questions")
    
    # Initialize counters and shared data structures
    total_questions = len(df)
    
    # Load previous results
    answers = load_previous_results(result_file, logger)
    
    # Initialize data source statistics
    data_source_stats = {}
    for data_source in df['data_source'].unique():
        data_source_stats[data_source] = {
            'total': 0,
            'correct': 0,
            'accuracy': 0.0,
            'no_context': 0,
            'em_correct': 0,
            'em_accuracy': 0.0
        }
    
    # Filter out already processed questions
    processed_questions_set = set()
    for data_source, questions in answers.items():
        processed_questions_set.update(questions.keys())
        # Update data source stats from previous results
        correct_count = sum(1 for info in questions.values() if info['score'] == 1)
        em_correct_count = sum(1 for info in questions.values() if info['em_score'] == 1)
        data_source_stats[data_source]['total'] = len(questions)
        data_source_stats[data_source]['correct'] = correct_count
        data_source_stats[data_source]['em_correct'] = em_correct_count
        data_source_stats[data_source]['accuracy'] = correct_count / len(questions) if questions else 0
    
    remaining_df = df[~df['reward_model'].apply(lambda x: x['ground_truth']['question']).isin(processed_questions_set)]
    
    logger.info(f"Found {sum(len(answers) for answers in answers.values())} previously processed questions")
    logger.info(f"Remaining questions to process: {len(remaining_df)}")
    
    # Create stats file path
    stats_file = result_file.replace('.json', '_stats.json')
    logger.info(f"Stats file will be saved to: {stats_file}")
    
    # Load context cache
    context_cache = json.load(open(context_file, 'r'))
    
    # Process remaining questions in parallel using process pool
    logger.info(f"Processing remaining questions with {num_workers} workers...")
    
    # Convert DataFrame to list of rows for processing
    remaining_rows = remaining_df.to_dict('records')
    
    # Process in batches
    batch_size = 8
    results_buffer = []
    processed_count = 0  # Counter for showing statistics
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Create batches
        batches = [remaining_rows[i:i+batch_size] for i in range(0, len(remaining_rows), batch_size)]
        logger.info(f"Created {len(batches)} batches of size {batch_size}")
        
        # Submit batches to process pool
        futures = {executor.submit(process_questions_batch, batch, context_cache, logger): i for i, batch in enumerate(batches)}
        
        # Process results as they complete
        with tqdm(total=len(remaining_rows), desc="Processing questions") as pbar:
            for future in as_completed(futures):
                try:
                    batch_results = future.result()
                    if not batch_results:
                        logger.warning(f"Empty batch results received for batch {futures[future]}")
                        continue
                        
                    results_buffer.extend(batch_results)
                    processed_count += len(batch_results)
                    
                    # Update progress bar
                    pbar.update(len(batch_results))
                    pbar.set_postfix({'processed': len(results_buffer)})
                    
                    # Process results and update statistics
                    for result in batch_results:
                        if result['question'] is not None:
                            data_source = result['data_source']
                            
                            # Initialize data source if not exists
                            if data_source not in answers:
                                answers[data_source] = {}
                            
                            # Store results
                            answers[data_source][result['question']] = {
                                'answer': result['answer'],
                                'score': 1 if result['is_correct'] else 0,
                                'em_score': 1 if result['is_em'] else 0
                            }
                            
                            # Update data source statistics
                            data_source_stats[data_source]['total'] += 1
                            if result['is_correct']:
                                data_source_stats[data_source]['correct'] += 1
                            if result['is_em']:
                                data_source_stats[data_source]['em_correct'] += 1
                            data_source_stats[data_source]['accuracy'] = (
                                data_source_stats[data_source]['correct'] / 
                                data_source_stats[data_source]['total']
                            )
                            data_source_stats[data_source]['em_accuracy'] = (
                                data_source_stats[data_source]['em_correct'] / 
                                data_source_stats[data_source]['total']
                            )
                    
                    # Show statistics every 100 steps
                    if processed_count % 100 == 0:
                        logger.info(f"\nStatistics after {processed_count} questions:")
                        for source, stats in data_source_stats.items():
                            if stats['total'] > 0:  # Only show sources that have been processed
                                logger.info(f"{source}: {stats['correct']}/{stats['total']} correct ({stats['accuracy']:.2%}), {stats['em_correct']}/{stats['total']} em correct ({stats['em_accuracy']:.2%})")
                    
                    # Save results periodically
                    if len(results_buffer) >= 10000:
                        save_results(answers, result_file, stats_file, total_questions, data_source_stats, logger)
                        results_buffer = []
                
                except Exception as e:
                    logger.error(f"Error processing batch {futures[future]}: {str(e)}")
                    # Update progress bar even on error to keep it moving
                    pbar.update(batch_size)
    
    # Save final results
    logger.info("\nSaving final results")
    save_results(answers, result_file, stats_file, total_questions, data_source_stats, logger)
    
    # Print final statistics
    logger.info("\nFinal Statistics per Data Source:")
    for source, stats in data_source_stats.items():
        logger.info(f"{source}: {stats['correct']}/{stats['total']} correct ({stats['accuracy']:.2%}), {stats['em_correct']}/{stats['total']} em correct ({stats['em_accuracy']:.2%})")
    logger.info(f"\nResults saved to: {result_file}")
    logger.info(f"Statistics saved to: {stats_file}")
    logger.info(f"Log file saved to: {log_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default="data/nq_hotpotqa_train/test_e5_ug.parquet", help='Path to input parquet file')
    parser.add_argument('--result_file', default="results/ircot_14b_gen_haiku.json", help='Path to save answers JSON file')
    parser.add_argument('--context_file', default="/shared/eng/pj20/ircot/results_14b.json", help='Directory containing context files')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of worker processes to use')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducible sampling')
    parser.add_argument('--sampling_enabled', action='store_true', help='Enable sampling of questions')
    
    args = parser.parse_args()
    process_dataset(args.input_file, args.result_file, args.context_file, args.num_workers, args.random_seed, True) 