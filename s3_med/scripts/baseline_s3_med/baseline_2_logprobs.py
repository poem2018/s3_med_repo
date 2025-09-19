#!/usr/bin/env python3
"""
Baseline 2 with Token Probabilities: Direct LLM Prediction using logprobs
Uses token probabilities from the model to calculate confidence scores
"""

import json
import requests
import argparse
import os
import math
import time
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

MODEL = "Qwen/Qwen2.5-3B-Instruct"

def call_llm_with_logprobs(prompt: str, temperature: float = 0.1, max_tokens: int = 20, max_retries: int = 3) -> tuple:
    """Call the LLM and get response with token probabilities"""
    headers = {"Content-Type": "application/json"}
    
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": 0.95,
        "max_tokens": max_tokens,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "logprobs": True,
        "top_logprobs": 5
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post("http://localhost:8000/v1/chat/completions", 
                                   headers=headers, json=payload, timeout=30)
            
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
            confidence = extract_confidence_from_logprobs(choice, text, prompt)
            return text, confidence
            
        except requests.exceptions.HTTPError as e:
            print(f"\nHTTP Error {response.status_code}: {e}")
            try:
                error_detail = response.json()
                print(f"Error details: {error_detail}")
            except:
                print(f"Response text: {response.text}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return "", 0.5
        except requests.exceptions.Timeout:
            print(f"Request timeout, retrying... (attempt {attempt+1}/{max_retries})")
            time.sleep(2)
        except Exception as e:
            print(f"\nError generating answer: {e}")
            import json as json_debug
            print(f"Payload sent: {json_debug.dumps(payload, indent=2)[:500]}...")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return "", 0.5
    
    return "", 0.5

def extract_confidence_from_logprobs(choice, response_text, prompt=""):
    """Extract confidence score from token log probabilities"""
    confidence = 0.5
    
    try:
        if not (choice.get("logprobs") and choice["logprobs"].get("content")):
            return confidence
            
        logprobs_data = choice["logprobs"]["content"]
        
        # Find start index for CoT prompts (look after "Final Answer:")
        start_idx = 0
        if "Final Answer:" in prompt or "Final Answer:" in response_text:
            full_text = ""
            for i, token_data in enumerate(logprobs_data):
                full_text += token_data.get("token", "")
                if "Final Answer" in full_text or "final answer" in full_text.lower():
                    start_idx = max(0, i - 2)
                    break

        # Method 1: Look for tokens containing HIGH/LOW (starting from start_idx)
        for i, token_data in enumerate(logprobs_data):
            if i < start_idx:
                continue

            token = token_data.get("token", "").strip().upper()

            # Match tokens containing HIGH or LOW (not just exact match)
            if "HIGH" in token:
                prob = math.exp(token_data.get("logprob", -10))
                confidence = prob  # HIGH = death probability
                return confidence
            elif "LOW" in token:
                prob = math.exp(token_data.get("logprob", -10))
                confidence = 1 - prob  # LOW = survival, invert for death probability
                return confidence

        # Method 2: Look through alternatives in all tokens
        for token_data in logprobs_data:
            if not token_data.get("top_logprobs"):
                continue
                
            high_prob = low_prob = 0.0
            for alt in token_data["top_logprobs"]:
                alt_token = alt.get("token", "").strip().upper()
                alt_prob = math.exp(alt.get("logprob", -10))
                
                if "HIGH" in alt_token or "DIED" in alt_token or "DEATH" in alt_token:
                    high_prob += alt_prob
                elif "LOW" in alt_token or "SURVIV" in alt_token or "STABLE" in alt_token:
                    low_prob += alt_prob
            
            if high_prob > 0 or low_prob > 0:
                total_prob = high_prob + low_prob
                confidence = high_prob / total_prob if total_prob > 0 else 0.5
                return confidence

        # Method 3: Use perplexity as fallback
        total_logprob = sum(td.get("logprob", -10) for td in logprobs_data)
        token_count = len(logprobs_data)
        
        if token_count > 0:
            avg_logprob = total_logprob / token_count
            perplexity = math.exp(-avg_logprob)
            
            # Map perplexity to certainty
            if perplexity < 2:
                certainty = 0.9
            elif perplexity < 5:
                certainty = 0.7
            elif perplexity < 10:
                certainty = 0.5
            else:
                certainty = 0.3
            
            # Determine direction from response text
            response_upper = response_text.upper()
            if any(word in response_upper for word in ["HIGH", "RISK", "DIED", "DEATH", "CRITICAL"]):
                confidence = certainty
            elif any(word in response_upper for word in ["LOW", "STABLE", "SURVIV", "GOOD"]):
                confidence = 1 - certainty
                
    except Exception as e:
        print(f"Error extracting confidence from logprobs: {e}")
    
    return confidence

def create_simple_prompt(patient_text):
    """Create a simple prompt that encourages single-token response"""
    return f"""You are an ICU physician. Based on this patient data, predict mortality risk.

{patient_text}

Remember: 87% survive. Only predict HIGH if severe indicators present.

Respond with exactly one word: HIGH or LOW"""

def create_cot_prompt(patient_text):
    """Create a chain-of-thought prompt with clear output instruction"""
    return f"""You are an ICU physician. Analyze this patient's mortality risk step by step.

=== PATIENT DATA ===
{patient_text}

Analyze systematically:
1. Vital signs (BP, HR, O2, temp) - identify critical values
2. Neurological (GCS scores) - assess consciousness level
3. Labs (glucose, pH) - check for severe abnormalities
4. Risk factors - count severe indicators

Remember: Remember that most ICU patients (87%) survive. Only predict HIGH if multiple severe indicators present.

Respond with EXACTLY these 3 words only: "Final Answer: HIGH" or "Final Answer: LOW"
No analysis, no explanation, no additional text."""

def parse_prediction_with_confidence(response_text, confidence):
    """Parse the response text to get binary prediction"""
    if not response_text:
        return 0, confidence
    
    response_upper = response_text.upper()
    
    if "HIGH" in response_upper:
        return 1, confidence
    elif "LOW" in response_upper:
        return 0, confidence
    
    # Fallback keywords
    death_keywords = ["DIE", "DEATH", "MORTALITY", "FATAL"]
    survival_keywords = ["SURVIVE", "SURVIVAL", "RECOVER", "STABLE"]
    
    for keyword in death_keywords:
        if keyword in response_upper:
            return 1, confidence
    
    for keyword in survival_keywords:
        if keyword in response_upper:
            return 0, confidence
    
    return 0, confidence

def load_patient_data(data_dir):
    """Load patient text data"""
    patient_file = os.path.join(data_dir, 'all_patients.json')
    with open(patient_file, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="Baseline 2 with Logprobs: Token Probability-based Confidence")
    parser.add_argument("--data_dir", default="/scratch/bcew/ruikez2/intern/s3_med/data/baseline_text_500", 
                        help="Directory containing patient data")
    parser.add_argument("--output_dir", default="/scratch/bcew/ruikez2/intern/s3_med/results/baseline_2_logprobs", 
                        help="Directory to save results")
    parser.add_argument("--max_patients", type=int, default=None, help="Maximum number of patients to process")
    parser.add_argument("--temperature", type=float, default=0.3, help="LLM temperature for generation")
    parser.add_argument("--batch_delay", type=float, default=0.1, help="Delay between requests in seconds")
    parser.add_argument("--prompt_type", choices=["simple", "cot"], default="simple",
                        help="Type of prompt to use: simple (single-token) or cot (chain-of-thought)")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading patient data from {args.data_dir}...")
    patients = load_patient_data(args.data_dir)
    
    if args.max_patients:
        patients = patients[:args.max_patients]
    
    print(f"Processing {len(patients)} patients with logprobs-based confidence...")
    
    predictions, true_labels, confidences, responses = [], [], [], []
    failed_patients = []
    
    for patient in tqdm(patients, desc="Processing patients"):
        patient_id = patient['patient_id']
        
        # Remove outcome section to avoid data leakage
        patient_text = patient['text']
        if "=== OUTCOME ===" in patient_text:
            patient_text = patient_text.split("=== OUTCOME ===")[0].strip()
        
        true_mortality = patient['mortality']
        
        # Create prompt based on selected type
        prompt = create_cot_prompt(patient_text) if args.prompt_type == "cot" else create_simple_prompt(patient_text)
        
        time.sleep(args.batch_delay)
        
        # Get LLM prediction with logprobs
        response_text, confidence = call_llm_with_logprobs(prompt, temperature=args.temperature)
        
        if not response_text:
            failed_patients.append(patient_id)
            prediction, confidence = 0, 0.5
        else:
            prediction, confidence = parse_prediction_with_confidence(response_text, confidence)
        
        predictions.append(prediction)
        true_labels.append(true_mortality)
        confidences.append(confidence)
        responses.append(response_text or "NO_RESPONSE")
        
        # Print example for first patient
        if len(predictions) == 1:
            print(f"\nExample prediction:")
            print(f"Patient ID: {patient_id}")
            print(f"True mortality: {true_mortality}")
            print(f"Response: {response_text}")
            print(f"Prediction: {prediction}")
            print(f"Confidence: {confidence:.3f}")
    
    if failed_patients:
        print(f"\nWarning: {len(failed_patients)} patients failed ({len(failed_patients)/len(patients)*100:.1f}%)")
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(true_labels, predictions),
        'precision': precision_score(true_labels, predictions, zero_division=0),
        'recall': recall_score(true_labels, predictions, zero_division=0),
        'f1': f1_score(true_labels, predictions, zero_division=0)
    }
    
    try:
        metrics['auc'] = roc_auc_score(true_labels, confidences)
        metrics['auprc'] = average_precision_score(true_labels, confidences)
    except:
        metrics['auc'] = metrics['auprc'] = 0.0
    
    # Save results
    results = {
        'model': MODEL,
        'method': 'logprobs',
        'prompt_type': args.prompt_type,
        'num_patients': len(patients),
        'num_failed': len(failed_patients),
        'metrics': metrics,
        'predictions': [
            {
                'patient_id': patients[i]['patient_id'],
                'true_label': true_labels[i],
                'prediction': predictions[i],
                'confidence': confidences[i],
                'response': responses[i]  # Store full response
            }
            for i in range(len(predictions))
        ],
        'failed_patients': failed_patients
    }
    
    output_file = os.path.join(args.output_dir, f'baseline_2_logprobs_{args.prompt_type}_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    # Print summary
    print(f"\n{'='*50}")
    print("BASELINE 2: LOGPROBS-BASED CONFIDENCE")
    print(f"{'='*50}")
    print(f"Model: {MODEL}")
    print(f"Patients: {len(patients)} (Failed: {len(failed_patients)})")
    print(f"\nMetrics:")
    for metric, value in metrics.items():
        print(f"  {metric.upper():10}: {value:.3f}")
    
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