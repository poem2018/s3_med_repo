import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import requests
import math
from transformers import AutoTokenizer
from generator_llms.test_cases import test_cases
import torch
from seper.calculate import gen_answers_batch, calculate_uncertainty_soft_batch, create_collate_fn, process_item_for_seper
from seper.uncertainty_measures.semantic_entropy import EntailmentDeberta
# Load matching tokenizer locally
MODEL = "Qwen/Qwen3-14B" 
# MODEL = "meta-llama/Meta-Llama-3-8B"
 
tokenizer = AutoTokenizer.from_pretrained(MODEL)

def compute_alternative_perplexity(prompt: str, answer: str, num_runs: int = 10) -> float:
    """
    Calculate perplexity using the sliding window approach.
    This makes multiple API calls for each token in the answer.
    Runs multiple times and returns the average perplexity after filtering outliers.
    Uses GPT-2 Small model for more stable perplexity computation.
    """
    # print(f"Computing perplexity for prompt: '{prompt}'")
    # print(f"Will run {num_runs} times and take average after filtering outliers")
    # print(f"Using model: {MODEL}")
    
    all_perplexities = []
    
    for run in range(num_runs):
        # print(f"\nRun {run + 1}/{num_runs}")
        headers = {"Content-Type": "application/json"}
        
        # Prepare the base prompt
        base_prompt = prompt.strip() + "\nThe answer is "
        
        # Tokenize the answer to get its length
        answer_tokens = tokenizer.encode(answer.strip(), add_special_tokens=False)
        answer_token_strings = tokenizer.convert_ids_to_tokens(answer_tokens)
        # print(f"Answer has {len(answer_tokens)} tokens: {answer_token_strings}")
        
        # Go through the answer token by token
        cumulative_answer = ""
        total_logprob = 0
        token_logprobs = []
        
        # For efficiency, we'll process in small batches
        batch_size = 1  # Process one token at a time for accuracy
        
        for i in range(0, len(answer_tokens), batch_size):
            # Get the next batch of tokens
            batch_end = min(i + batch_size, len(answer_tokens))
            next_tokens = answer_tokens[i:batch_end]
            next_text = tokenizer.decode(next_tokens)
            
            # Current context is the base prompt + answer so far
            current_prompt = base_prompt + cumulative_answer
            
            # Make API call to get probability of the next token(s)
            payload = {
                "model": MODEL,
                "prompt": current_prompt,
                "max_tokens": 1,
                "logprobs": True,
                "temperature": 0,  # Use deterministic sampling
                "top_p": 1.0,  # No top-p sampling
                "frequency_penalty": 0.0,  # No frequency penalty
                "presence_penalty": 0.0  # No presence penalty
            }
            
            try:
                response = requests.post("http://localhost:8000/v1/completions", headers=headers, json=payload)
                response.raise_for_status()
                res = response.json()
                
                if "choices" not in res or not res["choices"] or "logprobs" not in res["choices"][0]:
                    # print(f"Warning: Invalid response at position {i}: {res}")
                    continue
                    
                # Get the logprob for the most likely next token
                top_logprobs = res["choices"][0]["logprobs"].get("top_logprobs", [{}])
                
                if not top_logprobs or not isinstance(top_logprobs[0], dict):
                    # print(f"Warning: No top logprobs at position {i}")
                    continue
                    
                # Find the most likely next token and its logprob
                next_token_prob = None
                token_matched = False
                
                for token, logprob in top_logprobs[0].items():
                    if token.strip() in next_text or next_text.strip() in token:
                        next_token_prob = logprob
                        token_matched = True
                        # print(f"Token {i+1}/{len(answer_tokens)}: '{next_text}' - logprob: {logprob:.4f}")
                        break
                        
                if not token_matched and top_logprobs[0]:
                    # If no match, use the most likely token's probability
                    most_likely_token = max(top_logprobs[0].items(), key=lambda x: x[1])
                    next_token_prob = most_likely_token[1]
                    # print(f"Token {i+1}/{len(answer_tokens)}: '{next_text}' - No exact match, using most likely token '{most_likely_token[0]}' with logprob: {next_token_prob:.4f}")
                    
                if next_token_prob is not None:
                    total_logprob += next_token_prob
                    token_logprobs.append(next_token_prob)
                    
                # Update the cumulative answer
                cumulative_answer += next_text
                
            except Exception as e:
                print(f"Error at position {i}: {e}")
                # Continue with the next token
        
        # Calculate perplexity for this run
        if total_logprob == 0 or len(answer_tokens) == 0:
            # print("Warning: Could not calculate logprobs for this run, skipping")
            continue
            
        avg_logprob = total_logprob / len(answer_tokens)
        perplexity = math.exp(-avg_logprob)
        all_perplexities.append(perplexity)
        
        # Show token-by-token breakdown for this run
        # print("\nToken-by-token breakdown:")
        # for i, (token, logprob) in enumerate(zip(answer_token_strings, token_logprobs)):
            # print(f"{i+1}. '{token}': logprob = {logprob:.4f}, perplexity = {math.exp(-logprob):.4f}")
        
        # print(f"\nRun {run + 1} results:")
        # print(f"Total logprob: {total_logprob:.4f}")
        # print(f"Average logprob: {avg_logprob:.4f}")
        # print(f"Perplexity: {perplexity:.4f}")
    
    if not all_perplexities:
        # print("Warning: No valid perplexity calculations, returning default value")
        return 1.0
    
    def filter_outliers(values, max_iterations=5):
        """Recursively filter outliers until values stabilize or max iterations reached"""
        if len(values) < 3 or max_iterations == 0:
            return values
            
        sorted_values = sorted(values)
        q1 = sorted_values[len(sorted_values)//4]
        q3 = sorted_values[3*len(sorted_values)//4]
        iqr = q3 - q1
        
        # Use a more aggressive threshold for multiple outliers
        lower_bound = q1 - 2.0 * iqr
        upper_bound = q3 + 2.0 * iqr
        
        filtered = [v for v in values if lower_bound <= v <= upper_bound]
        
        # If no values were filtered or we've reached the minimum allowed values, return
        if len(filtered) == len(values) or len(filtered) < 3:
            return filtered
            
        # Recursively filter remaining values
        return filter_outliers(filtered, max_iterations - 1)
    
    # Apply recursive outlier filtering
    filtered_perplexities = filter_outliers(all_perplexities)
    
    if not filtered_perplexities:
        # print("Warning: All values were filtered as outliers, using original values")
        filtered_perplexities = all_perplexities
    
    # Calculate and return the average perplexity after filtering outliers
    avg_perplexity = sum(filtered_perplexities) / len(filtered_perplexities)
    
    return avg_perplexity


def compare_hotpotqa_style_examples():
    """
    Compare perplexity of short, factual answers with and without context,
    similar to HotpotQA examples.
    """
    # Define several test cases more similar to HotpotQA
    
    results = []
    
    for case in test_cases:
        print("\n" + "=" * 80)
        print(f"TEST CASE: {case['name']}")
        print("=" * 80)
        
        # Case 1: Question only
        print("\nCASE 1: QUESTION ONLY")
        print("-" * 40)
        question_only_ppl = compute_alternative_perplexity(case['question'], case['answer'])
        
        # Case 2: Context + Question
        print("\nCASE 2: CONTEXT + QUESTION")
        print("-" * 40)
        context_question_ppl = compute_alternative_perplexity(case['context'] + "\n\n" + case['question'], case['answer'])
        
        # Store results
        result = {
            "name": case['name'],
            "question": case['question'],
            "answer": case['answer'],
            "q_only_ppl": question_only_ppl,
            "context_q_ppl": context_question_ppl,
            "ratio": question_only_ppl / context_question_ppl if context_question_ppl > 0 else float('inf')
        }
        results.append(result)
    
    # Summary of all results
    print("\n" + "=" * 80)
    print("SUMMARY OF ALL RESULTS")
    print("=" * 80)
    print(f"{'Test Case':<15} {'Q-Only PPL':<15} {'Context+Q PPL':<15} {'Ratio (Q/C+Q)':<15} {'Effect':<20}")
    print("-" * 80)
    
    for result in results:
        effect = "HELPS ✓" if result['q_only_ppl'] > result['context_q_ppl'] else "HURTS ✗"
        print(f"{result['name']:<15} {result['q_only_ppl']:<15.4f} {result['context_q_ppl']:<15.4f} {result['ratio']:<15.4f} {effect:<20}")


def generate_answers_vllm(example: dict, num_generations: int, max_new_tokens: int) -> list:
    """
    Generate multiple answers using vLLM for a given example.
    
    Args:
        example: Dictionary containing question, context, and answers
        num_generations: Number of answers to generate
        max_new_tokens: Maximum number of tokens to generate
        
    Returns:
        list: List of (answer_text, log_probs) tuples
    """
    headers = {"Content-Type": "application/json"}
    responses = []
    
    # Format the prompt
    context, prompt = format_qa_prompt(example['question'], example['context'])
    full_prompt = context + "\n\n" + prompt
    
    for _ in range(num_generations):
        payload = {
            "model": MODEL,
            "prompt": full_prompt,
            "max_tokens": max_new_tokens,
            "temperature": 1.0,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "logprobs": True
        }
        
        try:
            response = requests.post("http://localhost:8000/v1/completions", headers=headers, json=payload)
            response.raise_for_status()
            res = response.json()
            
            if "choices" not in res or not res["choices"]:
                continue
                
            # Extract text and log probabilities
            choice = res["choices"][0]
            text = choice["text"].strip()
            logprobs = choice.get("logprobs", {}).get("token_logprobs", [])
            
            responses.append((text, logprobs))
            
        except Exception as e:
            print(f"Error generating answer: {e}")
            continue
            
    return responses

def compute_retrieval_utility_score_seper(prompt: str, answer: str, context: str, num_generations: int = 10) -> float:
    """
    Calculate retrieval utility score using SEPER method.
    This makes multiple API calls to generate answers and uses an entailment model
    to compute semantic entropy and belief shift.
    
    Args:
        prompt: The question to be answered
        answer: The expected answer
        context: The context to be added to the question
        num_generations: Number of times to generate answers
        
    Returns:
        float: The ΔSePer score (positive means context helps)
    """
    # Setup parameters for generation
    max_new_tokens = 128
    computation_chunk_size = 8
    device = 'cuda'  # Adjust based on your setup
    
    # Create example dictionaries for both scenarios
    example_with_context = {
        'question': prompt,
        'context': context,
        'answers': [answer]
    }
    
    example_baseline = {
        'question': prompt,
        'context': '',
        'answers': [answer]
    }
    
    # Generate answers for both scenarios using vLLM
    responses_with_context = generate_answers_vllm(example_with_context, num_generations, max_new_tokens)
    responses_baseline = generate_answers_vllm(example_baseline, num_generations, max_new_tokens)
    
    # Format results in SEPER's expected format
    result_with_context = {
        'question': prompt,
        'context': context,
        'answers': [answer],
        'responses': responses_with_context
    }
    
    result_baseline = {
        'question': prompt,
        'context': '',
        'answers': [answer],
        'responses': responses_baseline
    }
    
    # Setup entailment model
    entailment_model = EntailmentDeberta(device=device)
    entailment_model.model.eval()
    
    # Prepare data for SEPER calculation
    keys = ['question', 'response_text', 'answers', 'likelihood', 'context_label', 'log_liks_agg', 'context']
    seper_collate_fn = create_collate_fn(keys)
    
    # Calculate SEPER scores
    with torch.no_grad():
        r_with_context = process_item_for_seper(result_with_context)
        r_baseline = process_item_for_seper(result_baseline)
        seper_input = seper_collate_fn([r_with_context, r_baseline])
        seper_with_context, seper_baseline = calculate_uncertainty_soft_batch(
            seper_input, entailment_model, computation_chunk_size
        )
        
        # Calculate ΔSePer
        d_seper = seper_with_context - seper_baseline
        
    return d_seper

def compute_retrieval_utility_score(prompt: str, answer: str, context: str, num_runs: int = 10) -> float:
    """
    Calculate retrieval utility score using SEPER method.
    This is a wrapper around compute_retrieval_utility_score_seper for backward compatibility.
    """
    return compute_retrieval_utility_score_seper(prompt, answer, context, num_runs)


def generate_answer(context: str, prompt: str) -> str:
    """
    Generate an answer using the local LLM given context and prompt.
    
    Args:
        context: The context information
        prompt: The question or prompt to answer
        
    Returns:
        str: The generated answer
    """
    headers = {"Content-Type": "application/json"}
    
    context, prompt = format_qa_prompt(prompt, context)
    
    # Combine context and prompt
    full_prompt = context + "\n\n" + prompt
    
    # Prepare the payload for the API call
    payload = {
        "model": MODEL,
        "prompt": full_prompt,
        "max_tokens": 20,  # Adjust based on expected answer length
        "temperature": 0.0,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0    
        }
    
    try:
        response = requests.post("http://localhost:8000/v1/completions", headers=headers, json=payload)
        response.raise_for_status()
        res = response.json()
        
        if "choices" not in res or not res["choices"]:
            raise ValueError("Invalid response from LLM")
            
        # Extract and return the generated text
        return res["choices"][0]["text"].strip()
        
    except Exception as e:
        print(f"Error generating answer: {e}")
        return ""


def format_qa_prompt(question: str, context: str) -> tuple[str, str]:
    """
    Format a question and context into the specified prompt format.
    
    Args:
        question: The question to be answered
        context: The context information
        
    Returns:
        tuple: (formatted_context, formatted_prompt)
    """
    formatted_context = f"""Use the following contexts (some might be irrelevant) on demand:\nContexts:
{context}"""
    
    formatted_prompt = f"Question: {question}\n\nImportant: You MUST directly answer the question without any other text and thinking. You should END your response after the answer.\n\nAnswer: "
    
    return formatted_context, formatted_prompt


def display_results_summary(results: list) -> None:
    """
    Display a summary table of retrieval utility scores for all test cases.
    
    Args:
        results: List of dictionaries containing test case results
    """
    print("\n" + "=" * 100)
    print("RETRIEVAL UTILITY SCORES SUMMARY")
    print("=" * 100)
    
    # Print header
    print(f"{'Test Case':<30} {'ΔSePer':<15} {'Context Type':<20} {'Effect':<20}")
    print("-" * 100)
    
    # Group results by topic
    topic_groups = {}
    for result in results:
        topic = result['name'].split(' - ')[0]
        if topic not in topic_groups:
            topic_groups[topic] = []
        topic_groups[topic].append(result)
    
    # Print results by topic
    for topic, cases in topic_groups.items():
        print(f"\n{topic}:")
        for case in cases:
            d_seper = case['d_seper']
            context_type = case['name'].split(' - ')[1]
            effect = "HELPS ✓" if d_seper > 0 else "HURTS ✗"
            print(f"  {context_type:<27} {d_seper:<15.4f} {'':<20} {effect:<20}")
    
    # Print overall statistics
    print("\n" + "=" * 100)
    print("OVERALL STATISTICS")
    print("=" * 100)
    
    total_cases = len(results)
    helpful_cases = sum(1 for r in results if r['d_seper'] > 0)
    harmful_cases = sum(1 for r in results if r['d_seper'] < 0)
    neutral_cases = sum(1 for r in results if r['d_seper'] == 0)
    
    print(f"Total Cases: {total_cases}")
    print(f"Helpful Context: {helpful_cases} ({helpful_cases/total_cases*100:.1f}%)")
    print(f"Harmful Context: {harmful_cases} ({harmful_cases/total_cases*100:.1f}%)")
    print(f"Neutral Context: {neutral_cases} ({neutral_cases/total_cases*100:.1f}%)")
    
    # Calculate average ΔSePer by context type
    context_types = {}
    for result in results:
        context_type = result['name'].split(' - ')[1]
        if context_type not in context_types:
            context_types[context_type] = []
        context_types[context_type].append(result['d_seper'])
    
    print("\nAverage ΔSePer by Context Type:")
    for context_type, scores in context_types.items():
        avg_score = sum(scores) / len(scores)
        print(f"{context_type:<20}: {avg_score:.4f}")

if __name__ == "__main__":
    # Run all test cases and collect results
    results = []
    
    for test_case in test_cases:
        print("\n" + "=" * 80)
        print(f"TEST CASE: {test_case['name']}")
        print("=" * 80)
        
        # Calculate retrieval utility score
        retrieval_score = compute_retrieval_utility_score(
            prompt=test_case['question'],
            answer=test_case['answer'],
            context=test_case['context'],
            num_runs=3  # Increased for more stable results
        )
        
        # Store results
        results.append({
            'name': test_case['name'],
            'question': test_case['question'],
            'answer': test_case['answer'],
            'context': test_case['context'],
            'd_seper': retrieval_score
        })
    
    # Display summary table
    display_results_summary(results)