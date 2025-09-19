#!/usr/bin/env python3
"""
Baseline 4 with Token Probabilities: Similar Cases + RAG + LLM using logprobs
Uses similar patient cases and medical knowledge with token probabilities for confidence
"""

import json
import requests
import argparse
import os
import re
import math
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import time

MODEL = "Qwen/Qwen2.5-3B-Instruct"
RETRIEVAL_ENDPOINT = "http://127.0.0.1:3000/retrieve"

# Indicator weights for similarity calculation - all equal weights (same as baseline_4_similar_cases_rag.py)
INDICATOR_WEIGHTS = {
    0: 1.0,   # Capillary refill rate
    1: 1.0,   # Diastolic blood pressure
    2: 1.0,   # FiO2
    3: 1.0,   # GCS eye
    4: 1.0,   # GCS motor
    5: 1.0,   # GCS total
    6: 1.0,   # GCS verbal
    7: 1.0,   # Glucose
    8: 1.0,   # Heart Rate
    9: 1.0,   # Height
    10: 1.0,  # MAP
    11: 1.0,  # SpO2
    12: 1.0,  # pH
    13: 1.0,  # Respiratory rate
    14: 1.0,  # Systolic BP
    15: 1.0,  # Temperature
    16: 1.0   # Weight
}

def call_llm_with_logprobs(prompt: str, temperature: float = 0.1, max_tokens: int = 20, max_retries: int = 3) -> tuple:
    """
    Call the LLM and get response with token probabilities
    Returns: (response_text, confidence_score)
    """
    headers = {"Content-Type": "application/json"}
    
    # Format as chat messages
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "top_p": 0.95,
        "max_tokens": max_tokens,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "logprobs": True,  # Enable logprobs
        "top_logprobs": 5   # Get top 5 alternative tokens
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "http://localhost:8000/v1/chat/completions", 
                headers=headers, 
                json=payload,
                timeout=30
            )
            
            if response.status_code == 500:
                wait_time = 2 ** attempt
                print(f"Server error (500), retrying in {wait_time}s... (attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
                continue
            
            response.raise_for_status()
            res = response.json()
            
            if "choices" not in res or not res["choices"]:
                raise ValueError("Invalid response from LLM")
            
            choice = res["choices"][0]
            text = choice["message"]["content"].strip()
            
            # Extract confidence from logprobs (pass full prompt to identify if CoT)
            confidence = extract_confidence_from_logprobs(choice, text, prompt)

            return text, confidence
            
        except requests.exceptions.Timeout:
            print(f"Request timeout, retrying... (attempt {attempt+1}/{max_retries})")
            time.sleep(2)
        except Exception as e:
            print(f"Error generating answer: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return "", 0.5
    
    return "", 0.5

def extract_confidence_from_logprobs(choice, response_text, prompt=""):
    """
    Extract confidence score from token log probabilities
    For CoT prompts, only look at tokens after "Final Answer:"
    """
    # Default confidence
    confidence = 0.5
    found_key_token = False
    
    try:
        # Check if logprobs are available
        if "logprobs" in choice and choice["logprobs"]:
            logprobs_data = choice["logprobs"]
            
            # Get the content tokens and their probabilities
            if "content" in logprobs_data and logprobs_data["content"]:
                # For CoT prompts, find where "Final Answer:" appears and only look after that
                start_idx = 0
                if "Final Answer:" in prompt or "Final Answer:" in response_text:
                    # Build the full text from tokens to find where Final Answer appears
                    full_text = ""
                    for i, token_data in enumerate(logprobs_data["content"]):
                        token = token_data.get("token", "")
                        full_text += token
                        if "Final Answer" in full_text or "final answer" in full_text.lower():
                            # Start looking from the next few tokens
                            start_idx = max(0, i - 2)  # Look a bit before in case Final Answer is split
                            break

                # Method 1: Look for the key prediction tokens (starting from start_idx)
                for i, token_data in enumerate(logprobs_data["content"]):
                    if i < start_idx:
                        continue

                    token = token_data.get("token", "").strip().upper()
                    logprob = token_data.get("logprob", -10)

                    # Check if this is a prediction token
                    if token in ["HIGH", "LOW"]:
                        # Convert logprob to probability
                        prob = math.exp(logprob)

                        # For HIGH RISK
                        if token == "HIGH":
                            confidence = prob  # Direct probability of mortality
                        # For LOW RISK
                        elif token == "LOW":
                            confidence = 1 - prob  # Invert for mortality probability

                        found_key_token = True

                        # Look at alternatives for this specific token
                        if "top_logprobs" in token_data:
                            # Calculate proper confidence from alternatives
                            high_prob = prob if token == "HIGH" else 0.0
                            low_prob = prob if token == "LOW" else 0.0

                            for alt in token_data["top_logprobs"]:
                                alt_token = alt.get("token", "").strip().upper()
                                alt_prob = math.exp(alt.get("logprob", -10))
                                if alt_token == "HIGH":
                                    high_prob = alt_prob
                                elif alt_token == "LOW":
                                    low_prob = alt_prob

                            # Recalculate confidence based on HIGH vs LOW probability
                            if high_prob > 0 or low_prob > 0:
                                total = high_prob + low_prob
                                confidence = high_prob / total if total > 0 else 0.5

                        break  # Use the first HIGH/LOW token found after Final Answer
                
                # Method 2: If no key token found, look at first token's alternatives
                if not found_key_token and logprobs_data["content"]:
                    first_token_data = logprobs_data["content"][0]
                    
                    # Check if top alternatives contain HIGH/LOW
                    if "top_logprobs" in first_token_data:
                        high_prob = 0.0
                        low_prob = 0.0
                        
                        # Sum probabilities of all death-related vs survival-related tokens
                        for alt in first_token_data["top_logprobs"]:
                            alt_token = alt.get("token", "").strip().upper()
                            alt_prob = math.exp(alt.get("logprob", -10))
                            
                            if "HIGH" in alt_token or "DIED" in alt_token or "DEATH" in alt_token:
                                high_prob += alt_prob
                            elif "LOW" in alt_token or "SURVIV" in alt_token or "STABLE" in alt_token:
                                low_prob += alt_prob
                        
                        # If we found relevant alternatives, use them
                        if high_prob > 0 or low_prob > 0:
                            total_prob = high_prob + low_prob
                            if total_prob > 0:
                                confidence = high_prob / total_prob  # Probability of HIGH RISK
                                found_key_token = True
                
                # Method 3: Use perplexity/certainty of the response
                if not found_key_token and "content" in logprobs_data:
                    total_logprob = 0
                    token_count = 0
                    
                    for token_data in logprobs_data["content"]:
                        logprob = token_data.get("logprob", -10)
                        total_logprob += logprob
                        token_count += 1
                    
                    if token_count > 0:
                        avg_logprob = total_logprob / token_count
                        # Perplexity = exp(-avg_logprob)
                        perplexity = math.exp(-avg_logprob)
                        
                        # Map perplexity to confidence
                        if perplexity < 2:
                            certainty = 0.9
                        elif perplexity < 5:
                            certainty = 0.7
                        elif perplexity < 10:
                            certainty = 0.5
                        else:
                            certainty = 0.3
                        
                        # Determine direction based on response text
                        response_upper = response_text.upper()
                        if any(word in response_upper for word in ["HIGH", "RISK", "DIED", "DEATH", "CRITICAL"]):
                            confidence = certainty
                        elif any(word in response_upper for word in ["LOW", "STABLE", "SURVIV", "GOOD"]):
                            confidence = 1 - certainty
                        else:
                            confidence = 0.5  # Can't determine direction
                    
    except Exception as e:
        print(f"Error extracting confidence from logprobs: {e}")
    
    return confidence

def retrieve_medical_knowledge(query, retrieval_endpoint="http://127.0.0.1:3000/retrieve", topk=5):
    """Retrieve relevant medical knowledge from the retrieval service"""
    try:
        payload = {
            "queries": [query],  # API expects array of queries
            "topk": topk,
            "return_scores": True  # Required by server to avoid unpacking error
        }

        response = requests.post(retrieval_endpoint, json=payload, timeout=30)
        response.raise_for_status()

        if response.status_code == 200:
            data = response.json()
            # Use 'result' key as in baseline_4_similar_cases_rag.py
            if 'result' in data:
                results = data['result']

                formatted_docs = ""
                for idx, doc_item in enumerate(results[0][:topk]):
                    if 'document' in doc_item:
                        if 'contents' in doc_item['document']:
                            content = f"{doc_item['document'].get('title', '')} {doc_item['document'].get('text', '')}"
                        elif 'content' in doc_item['document']:
                            content = doc_item['document']['content']
                        else:
                            content = str(doc_item['document'].get('text', doc_item['document']))
                    else:
                        content = str(doc_item.get('content', doc_item))

                    # Limit content length
                    content = content[:400]
                    formatted_docs += f"\n[Doc {idx+1}] {content}\n"

                return formatted_docs
            else:
                # No results found
                return "No relevant medical knowledge could be retrieved."
        elif response.status_code == 500:
            # Server error - RAG service has a bug
            print(f"[WARNING] RAG service error 500 - continuing without RAG")
            return "No relevant medical knowledge could be retrieved (service error)."
        else:
            print(f"[WARNING] RAG service returned {response.status_code}")
            return "No relevant medical knowledge could be retrieved."
    except Exception as e:
        print(f"Error retrieving medical knowledge: {e}")
    
    return "No relevant medical knowledge could be retrieved."

def load_data(data_dir):
    """Load all necessary data files (same as baseline_4_similar_cases_rag.py)"""
    # Load text data
    patient_file = os.path.join(data_dir, 'all_patients.json')
    with open(patient_file, 'r') as f:
        patients_text = json.load(f)
    
    # Load matrices for similarity calculation
    matrices_file = os.path.join(data_dir, 'chartevents_matrices_test.json')
    with open(matrices_file, 'r') as f:
        matrices = json.load(f)
    
    # Load patient info
    patient_info_file = os.path.join(data_dir, 'patient_info_test.json')
    with open(patient_info_file, 'r') as f:
        patient_info = json.load(f)
    
    return patients_text, matrices, patient_info

def calculate_matrix_similarity(matrix1, matrix2, max_hours=48):
    """Calculate weighted similarity between two patient matrices (same as baseline_4_similar_cases_rag.py)"""
    similarities = []
    weights = []
    
    # Process each indicator
    for idx in range(min(len(matrix1), len(matrix2), 17)):
        weight = INDICATOR_WEIGHTS.get(idx, 0.1)
        
        # Skip indicators with very low weight
        if weight < 0.05:
            continue
        
        # Get series for both patients (limit to 48 hours)
        series1 = matrix1[idx][:min(len(matrix1[idx]), max_hours)]
        series2 = matrix2[idx][:min(len(matrix2[idx]), max_hours)]
        
        # Clean series (remove None/NaN)
        s1_clean = [x for x in series1 if x is not None and not np.isnan(x)]
        s2_clean = [x for x in series2 if x is not None and not np.isnan(x)]
        
        if len(s1_clean) > 0 and len(s2_clean) > 0:
            # Calculate simple similarity (1 - normalized difference)
            # For simplicity, using mean and std comparison
            mean_diff = abs(np.mean(s1_clean) - np.mean(s2_clean))
            std_diff = abs(np.std(s1_clean) - np.std(s2_clean))
            
            # Normalize differences (simple approach)
            max_val = max(max(s1_clean), max(s2_clean))
            min_val = min(min(s1_clean), min(s2_clean))
            range_val = max_val - min_val if max_val > min_val else 1
            
            norm_mean_diff = mean_diff / range_val if range_val > 0 else 0
            norm_std_diff = std_diff / range_val if range_val > 0 else 0
            
            # Similarity score (1 - normalized difference)
            similarity = 1 - (norm_mean_diff + norm_std_diff) / 2
            similarity = max(0, min(1, similarity))  # Clamp to [0, 1]
            
            similarities.append(similarity)
            weights.append(weight)
    
    # Calculate weighted average similarity
    if weights:
        total_weight = sum(weights)
        weighted_sim = sum(s * w for s, w in zip(similarities, weights)) / total_weight
        return weighted_sim
    else:
        return 0.0

def find_similar_patients(target_idx, all_matrices, all_info, k=3):
    """Find k most similar patients to the target patient (same as baseline_4_similar_cases_rag.py)"""
    similarities = []
    target_matrix = all_matrices[target_idx]
    
    for i, matrix in enumerate(all_matrices):
        if i == target_idx:
            continue  # Skip self
        
        sim = calculate_matrix_similarity(target_matrix, matrix)
        similarities.append((i, sim))
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Get top k similar patients
    similar_patients = []
    for idx, sim in similarities[:k]:
        similar_patients.append({
            'patient_id': all_info[idx]['patient_id'],
            'similarity': sim,
            'mortality': all_info[idx]['mortality'],
            'age': all_info[idx].get('age', 0),
            'gender': all_info[idx].get('gender', 'U')
        })
    
    return similar_patients

def format_similar_cases(similar_patients, all_patients):
    """Format similar patient cases for the prompt"""
    cases = []
    for sp in similar_patients:
        # Find the full patient data
        patient_data = next((p for p in all_patients if p['patient_id'] == sp['patient_id']), None)
        if patient_data:
            # Extract key info without the outcome
            text = patient_data['text']
            if "=== OUTCOME ===" in text:
                text = text.split("=== OUTCOME ===")[0].strip()
            
            # Shorten to key information
            if len(text) > 500:
                text = text[:500] + "..."
            
            case = f"[Similar Case - Similarity: {sp['similarity']:.2f}]\n"
            case += f"Patient ID: {sp['patient_id']}\n"
            case += f"Outcome: {'DIED' if sp['mortality'] == 1 else 'SURVIVED'}\n"
            case += f"Summary: {text}\n"
            cases.append(case)
    
    return "\n".join(cases) if cases else "No similar cases found."

def create_simple_enhanced_prompt(patient_text, similar_cases, retrieved_docs):
    """Create simple prompt with similar cases and medical knowledge for single-token response"""
    # Limit patient text
    if len(patient_text) > 2000:
        if "=== CLINICAL DATA ===" in patient_text:
            parts = patient_text.split("=== CLINICAL DATA ===")
            patient_text = parts[0] + "=== CLINICAL DATA ===" + parts[1][:1500]
        else:
            patient_text = patient_text[:2000]
    
    prompt = f"""You are an expert ICU physician. Based on similar patient cases and medical literature, predict ICU mortality risk.

Instructions:
- Consider the outcomes of similar patients
- Apply medical knowledge to assess risk
- Remember that LOW RISK is more common (87% of ICU patients survive)
- Respond with ONLY ONE WORD: HIGH or LOW

=== CURRENT PATIENT ===
{patient_text}

=== SIMILAR PATIENT CASES ===
{similar_cases}

=== RELEVANT MEDICAL KNOWLEDGE ===
{retrieved_docs}

Answer:"""
    return prompt

def create_cot_enhanced_prompt(patient_text, similar_cases, retrieved_docs):
    """Create COT prompt integrating similar cases and medical knowledge"""
    # Limit patient text for COT to leave room for analysis
    if len(patient_text) > 1500:
        if "=== CLINICAL DATA ===" in patient_text:
            parts = patient_text.split("=== CLINICAL DATA ===")
            patient_text = parts[0] + "=== CLINICAL DATA ===" + parts[1][:1000]
        else:
            patient_text = patient_text[:1500]
    
    prompt = f"""You are an expert ICU physician. Analyze this patient using similar historical cases and medical literature to predict ICU mortality step by step.

=== CURRENT PATIENT ===
{patient_text}

=== SIMILAR PATIENT CASES ===
{similar_cases}

=== RELEVANT MEDICAL KNOWLEDGE ===
{retrieved_docs}

Analyze systematically:
1. Vital signs (BP, HR, O2, temp) - identify critical values
2. Neurological (GCS scores) - assess consciousness level
3. Labs (glucose, pH) - check for severe abnormalities
4. Risk factors - count severe indicators
5: Integration with Medical Evidence and similar patient case
- The outcomes/label of similar patient case
- Synthesize patient data with literature-based risk models
- Apply evidence-based mortality predictors

Important:
- Remember that most ICU patients (87%) survive
- Base assessment on similar patient case and medical literature evidence
- Only predict HIGH if multiple severe indicators align with literature

Respond with EXACTLY these 3 words only: "Final Answer: HIGH" or "Final Answer: LOW"
No analysis, no explanation, no additional text."""
    return prompt

def parse_prediction_with_confidence(response_text, confidence):
    """Parse the response text to get binary prediction, using provided confidence"""
    if not response_text:
        return 0, confidence  # Default to survival
    
    response_upper = response_text.upper()
    
    # Look for HIGH/LOW in the response
    if "HIGH" in response_upper:
        return 1, confidence  # HIGH RISK = mortality
    elif "LOW" in response_upper:
        return 0, confidence  # LOW RISK = survival, confidence already represents mortality probability
    
    # Fallback: look for other keywords
    death_keywords = ["DIE", "DEATH", "MORTALITY", "FATAL"]
    survival_keywords = ["SURVIVE", "SURVIVAL", "RECOVER", "STABLE"]
    
    for keyword in death_keywords:
        if keyword in response_upper:
            return 1, confidence
    
    for keyword in survival_keywords:
        if keyword in response_upper:
            return 0, confidence  # confidence already represents mortality probability
    
    # Default to survival with original confidence
    return 0, confidence

def main():
    parser = argparse.ArgumentParser(description="Baseline 4 with Logprobs: Similar Cases + RAG + LLM with Token Probabilities")
    parser.add_argument("--data_dir", default="/scratch/bcew/ruikez2/intern/s3_med/data/baseline_text_500", 
                        help="Directory containing patient data")
    parser.add_argument("--output_dir", default="/scratch/bcew/ruikez2/intern/s3_med/results/baseline_4_logprobs", 
                        help="Directory to save results")
    parser.add_argument("--retrieval_endpoint", default="http://127.0.0.1:3000/retrieve",
                        help="Endpoint for medical knowledge retrieval")
    parser.add_argument("--k_similar", type=int, default=3,
                        help="Number of similar cases to retrieve")
    parser.add_argument("--topk_docs", type=int, default=3,
                        help="Number of medical documents to retrieve")
    parser.add_argument("--max_patients", type=int, default=None,
                        help="Maximum number of patients to process")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="LLM temperature for generation")
    parser.add_argument("--batch_delay", type=float, default=0.1,
                        help="Delay between requests in seconds")
    parser.add_argument("--prompt_type", choices=["simple", "cot"], default="simple",
                        help="Type of prompt to use: simple (single-token) or cot (chain-of-thought)")
    args = parser.parse_args()
    
    global RETRIEVAL_ENDPOINT
    RETRIEVAL_ENDPOINT = args.retrieval_endpoint
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading patient data from {args.data_dir}...")
    patients_text, all_matrices, all_info = load_data(args.data_dir)
    
    if args.max_patients:
        patients_text = patients_text[:args.max_patients]
        all_matrices = all_matrices[:args.max_patients]
        all_info = all_info[:args.max_patients]
    
    print(f"Processing {len(patients_text)} patients with similar cases, RAG, and logprobs...")
    
    predictions = []
    true_labels = []
    confidences = []
    responses = []  # Store response texts
    failed_patients = []
    
    for i, patient in enumerate(tqdm(patients_text, desc="Processing patients")):
        patient_id = patient['patient_id']
        # Remove the OUTCOME section
        patient_text_full = patient['text']
        if "=== OUTCOME ===" in patient_text_full:
            patient_text = patient_text_full.split("=== OUTCOME ===")[0].strip()
        else:
            patient_text = patient_text_full
        true_mortality = patient['mortality']
        
        # Find similar patients using matrix similarity
        similar_patients = find_similar_patients(i, all_matrices, all_info, k=args.k_similar)
        similar_cases_text = format_similar_cases(similar_patients, patients_text)
        
        # Create query for medical knowledge retrieval
        similar_outcomes = [sp['mortality'] for sp in similar_patients]
        if sum(similar_outcomes) > len(similar_outcomes) / 2:
            query_context = "high mortality risk ICU patients similar cases died"
        else:
            query_context = "ICU patients recovery patterns similar cases survived"
        
        patient_summary = patient_text.split("=== CLINICAL DATA ===")[0] if "=== CLINICAL DATA ===" in patient_text else patient_text[:300]
        retrieval_query = f"{query_context} {patient_summary}"
        
        # Retrieve medical knowledge
        retrieved_docs = retrieve_medical_knowledge(retrieval_query, 
                                                     retrieval_endpoint=args.retrieval_endpoint,
                                                     topk=args.topk_docs)
        
        # Create enhanced prompt based on selected type
        if args.prompt_type == "cot":
            prompt = create_cot_enhanced_prompt(patient_text, similar_cases_text, retrieved_docs)
        else:
            prompt = create_simple_enhanced_prompt(patient_text, similar_cases_text, retrieved_docs)
        
        # Add delay to avoid overwhelming the server
        time.sleep(args.batch_delay)
        
        # Get LLM prediction with logprobs
        response_text, confidence = call_llm_with_logprobs(prompt, temperature=args.temperature)
        
        if not response_text:
            failed_patients.append(patient_id)
            prediction = 0
            confidence = 0.5
        else:
            # Parse prediction with confidence
            prediction, adjusted_confidence = parse_prediction_with_confidence(response_text, confidence)
            confidence = adjusted_confidence
        
        # Store results
        predictions.append(prediction)
        true_labels.append(true_mortality)
        confidences.append(confidence)
        responses.append(response_text if response_text else "NO_RESPONSE")
        
        # Print example for first patient
        if len(predictions) == 1:
            print("\nExample prediction:")
            print(f"Patient ID: {patient_id}")
            print(f"Similar patients: {[sp['patient_id'] for sp in similar_patients]}")
            print(f"Similar outcomes: {similar_outcomes}")
            print(f"True mortality: {true_mortality}")
            print(f"Response: {response_text}")
            print(f"Prediction: {prediction}")
            print(f"Confidence: {confidence:.3f}")
    
    # Print failure statistics
    if failed_patients:
        print(f"\nWarning: {len(failed_patients)} patients failed to get predictions")
        print(f"Failed rate: {len(failed_patients)/len(patients_text)*100:.1f}%")
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(true_labels, predictions),
        'precision': precision_score(true_labels, predictions, zero_division=0),
        'recall': recall_score(true_labels, predictions, zero_division=0),
        'f1': f1_score(true_labels, predictions, zero_division=0)
    }
    
    # Calculate AUC using confidence scores
    try:
        metrics['auc'] = roc_auc_score(true_labels, confidences)
        metrics['auprc'] = average_precision_score(true_labels, confidences)
    except:
        metrics['auc'] = 0.0
        metrics['auprc'] = 0.0
    
    # Save results
    results = {
        'model': MODEL,
        'method': 'similar_cases_rag_logprobs',
        'prompt_type': args.prompt_type,
        'k_similar': args.k_similar,
        'topk_docs': args.topk_docs,
        'num_patients': len(patients_text),
        'num_failed': len(failed_patients),
        'metrics': metrics,
        'predictions': [
            {
                'patient_id': patients_text[i]['patient_id'],
                'true_label': true_labels[i],
                'prediction': predictions[i],
                'confidence': confidences[i],
                'response': responses[i]  # Include response text
            }
            for i in range(len(predictions))
        ],
        'failed_patients': failed_patients
    }
    
    output_file = os.path.join(args.output_dir, f'baseline_4_logprobs_{args.prompt_type}_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    # Print summary
    print("\n" + "="*50)
    print("BASELINE 4: SIMILAR CASES + RAG + LOGPROBS")
    print("="*50)
    print(f"Model: {MODEL}")
    print(f"Similar cases: {args.k_similar}")
    print(f"Medical docs: {args.topk_docs}")
    print(f"Number of patients: {len(patients_text)}")
    print(f"Failed predictions: {len(failed_patients)}")
    print(f"\nMetrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall:    {metrics['recall']:.3f}")
    print(f"  F1 Score:  {metrics['f1']:.3f}")
    print(f"  AUROC:     {metrics['auc']:.3f}")
    print(f"  AUPRC:     {metrics['auprc']:.3f}")
    
    # Confusion matrix
    tp = sum(1 for p, t in zip(predictions, true_labels) if p == 1 and t == 1)
    tn = sum(1 for p, t in zip(predictions, true_labels) if p == 0 and t == 0)
    fp = sum(1 for p, t in zip(predictions, true_labels) if p == 1 and t == 0)
    fn = sum(1 for p, t in zip(predictions, true_labels) if p == 0 and t == 1)
    
    print(f"\nConfusion Matrix:")
    print(f"  True Positive:  {tp}")
    print(f"  True Negative:  {tn}")
    print(f"  False Positive: {fp}")
    print(f"  False Negative: {fn}")

if __name__ == "__main__":
    main()