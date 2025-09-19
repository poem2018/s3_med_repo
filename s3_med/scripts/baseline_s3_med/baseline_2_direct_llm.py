#!/usr/bin/env python3
"""
Baseline 2: Direct LLM Prediction
Directly feed patient data text into LLM (Qwen2.5-3B-Instruct) for mortality prediction
"""

import json
import requests
import argparse
import os
from pathlib import Path
from tqdm import tqdm
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

MODEL = "Qwen/Qwen2.5-3B-Instruct"

def call_llm(prompt: str, temperature: float = 0.3, max_tokens: int = 20, max_retries: int = 3) -> str:
    """
    Call the LLM with a prompt and get response with retry logic
    Lower temperature for more deterministic predictions
    """
    import time
    headers = {"Content-Type": "application/json"}
    
    # Truncate prompt if too long (keep first and last parts)
    if len(prompt) > 8000:
        print(f"Warning: Truncating prompt from {len(prompt)} to 8000 chars")
        prompt = prompt[:4000] + "\n\n[... clinical data truncated ...]\n\n" + prompt[-3500:]
    
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
        "presence_penalty": 0.0    
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "http://localhost:8000/v1/chat/completions", 
                headers=headers, 
                json=payload,
                timeout=30  # Add timeout
            )
            
            if response.status_code == 500:
                # Server error - wait and retry
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Server error (500), retrying in {wait_time}s... (attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
                continue
            elif response.status_code == 400:
                # Bad request - likely prompt too long, truncate more aggressively
                print(f"Bad request (400), truncating prompt further...")
                if len(prompt) > 4000:
                    prompt = prompt[:2000] + "\n\n[... truncated ...]\n\n" + prompt[-1500:]
                    messages[0]["content"] = prompt  # Update user message
                    payload["messages"] = messages
                    continue
                else:
                    print(f"Error 400 even with short prompt: {response.text[:200]}")
                    return ""
            
            response.raise_for_status()
            res = response.json()
            
            if "choices" not in res or not res["choices"]:
                raise ValueError("Invalid response from LLM")
                
            return res["choices"][0]["message"]["content"].strip()
            
        except requests.exceptions.Timeout:
            print(f"Request timeout, retrying... (attempt {attempt+1}/{max_retries})")
            time.sleep(2)
        except requests.exceptions.ConnectionError:
            print(f"Connection error, retrying... (attempt {attempt+1}/{max_retries})")
            time.sleep(2)
        except Exception as e:
            print(f"Error generating answer: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return ""
    
    return ""

def load_patient_data(data_dir):
    """Load patient text data"""
    patient_file = os.path.join(data_dir, 'all_patients.json')
    
    with open(patient_file, 'r') as f:
        patients = json.load(f)
    
    return patients

def create_direct_prompt(patient_text):
    """Create a simple prompt that encourages single-token response"""
    prompt = f"""You are an expert ICU physician. Based on the following patient data, predict if the patient has HIGH or LOW risk of ICU mortality.

{patient_text}

Important: 
- Remember that most ICU patients (87%) survive
- Only predict HIGH risk if there are clear, severe indicators
- Respond with ONLY ONE WORD: HIGH or LOW

Answer:"""
    return prompt

def create_cot_prompt(patient_text):
    """Create a chain-of-thought prompt for better reasoning"""
    prompt = f"""You are an expert ICU physician. Analyze the following patient's clinical data step by step to predict ICU mortality.

{patient_text}

Please analyze this patient systematically:

Step 1: Vital Signs Analysis
- Blood pressure trends (systolic, diastolic, mean arterial pressure)
- Heart rate patterns
- Respiratory rate and oxygen saturation
- Temperature abnormalities

Step 2: Neurological Status
- Glasgow Coma Scale scores (eye, motor, verbal responses)
- Trends in consciousness level

Step 3: Laboratory Values
- Glucose levels
- pH (if available)
- Other abnormalities

Step 4: Medical History
- Relevant past medical conditions
- Risk factors for poor outcome

Step 5: Overall Assessment
Based on your analysis, predict whether this patient has:
- HIGH RISK of ICU mortality (likely to die)
- LOW RISK of ICU mortality (likely to survive)

Final Answer:
PREDICTION: [HIGH RISK or LOW RISK]
CONFIDENCE: [High/Medium/Low]
KEY FACTORS: [List 2-3 most important factors]
"""
    return prompt

def parse_prediction(response):
    """Parse the LLM response to extract prediction"""
    if not response:
        # If no response, use a random prediction based on base rate
        import random
        return random.choice([0, 0, 0, 0, 0, 0, 0, 1])  # ~12.5% mortality rate
    
    response_upper = response.upper()
    
    # Look for explicit prediction
    if "HIGH RISK" in response_upper or "HIGH-RISK" in response_upper:
        return 1
    elif "LOW RISK" in response_upper or "LOW-RISK" in response_upper:
        return 0
    
    # Fallback: look for mortality/death keywords
    death_keywords = ["WILL DIE", "LIKELY TO DIE", "MORTALITY LIKELY", "POOR PROGNOSIS", "FATAL"]
    survival_keywords = ["WILL SURVIVE", "LIKELY TO SURVIVE", "GOOD PROGNOSIS", "FAVORABLE"]
    
    for keyword in death_keywords:
        if keyword in response_upper:
            return 1
    
    for keyword in survival_keywords:
        if keyword in response_upper:
            return 0
    
    # Default to survival if uncertain
    return 0

def parse_all_predictions(predictions):
    """Parse all text predictions to binary labels"""
    parsed_predictions = []
    
    for pred in predictions:
        response = pred.get('raw_response', '')
        
        # Parse the response
        if not response:
            # No response, use default
            predicted_label = 0
            confidence = 0.0
            prediction_text = "NO_RESPONSE"
        else:
            response_upper = response.upper()
            
            # Extract prediction text - look for HIGH or LOW (with or without RISK)
            if "HIGH RISK" in response_upper or "HIGH-RISK" in response_upper or response_upper.strip() == "HIGH" or response_upper.startswith("HIGH"):
                predicted_label = 1
                prediction_text = "HIGH_RISK"
            elif "LOW RISK" in response_upper or "LOW-RISK" in response_upper or response_upper.strip() == "LOW" or response_upper.startswith("LOW"):
                predicted_label = 0
                prediction_text = "LOW_RISK"
            else:
                # Fallback: look for mortality/death keywords
                death_keywords = ["WILL DIE", "LIKELY TO DIE", "MORTALITY LIKELY", "POOR PROGNOSIS", "FATAL"]
                survival_keywords = ["WILL SURVIVE", "LIKELY TO SURVIVE", "GOOD PROGNOSIS", "FAVORABLE"]
                
                predicted_label = 0  # Default
                prediction_text = "UNCLEAR"
                
                for keyword in death_keywords:
                    if keyword in response_upper:
                        predicted_label = 1
                        prediction_text = f"DEATH_KEYWORD:{keyword}"
                        break
                
                if predicted_label == 0:  # Only check if not already found death keyword
                    for keyword in survival_keywords:
                        if keyword in response_upper:
                            predicted_label = 0
                            prediction_text = f"SURVIVAL_KEYWORD:{keyword}"
                            break
            
            # Since we don't have explicit confidence in the simple prompt,
            # assign a fixed confidence based on the prediction
            if prediction_text == "HIGH_RISK":
                confidence = 0.9  # High confidence for mortality
            elif prediction_text == "LOW_RISK":
                confidence = 0.1  # Low confidence for mortality (high confidence for survival)
            else:
                confidence = 0.5  # Uncertain
        
        # Create parsed prediction
        parsed_pred = {
            'patient_id': pred['patient_id'],
            'true_label': pred['true_label'],
            'prediction': predicted_label,
            'prediction_text': prediction_text,
            'confidence': confidence
        }
        parsed_predictions.append(parsed_pred)
    
    return parsed_predictions

def evaluate_predictions(predictions, true_labels):
    """Calculate evaluation metrics"""
    # Parse predictions first
    parsed_predictions = parse_all_predictions(predictions)
    
    # Filter out NO_RESPONSE predictions for metrics calculation
    valid_predictions = []
    valid_true_labels = []
    no_response_count = 0
    
    for i, p in enumerate(parsed_predictions):
        if p['prediction_text'] != 'NO_RESPONSE':
            valid_predictions.append(p['prediction'])
            valid_true_labels.append(true_labels[i])
        else:
            no_response_count += 1
    
    # Calculate metrics only on valid predictions
    if len(valid_predictions) > 0:
        metrics = {
            'accuracy': accuracy_score(valid_true_labels, valid_predictions),
            'precision': precision_score(valid_true_labels, valid_predictions, zero_division=0),
            'recall': recall_score(valid_true_labels, valid_predictions, zero_division=0),
            'f1': f1_score(valid_true_labels, valid_predictions, zero_division=0),
            'valid_predictions': len(valid_predictions),
            'no_response_count': no_response_count,
            'response_rate': len(valid_predictions) / len(parsed_predictions)
        }
    else:
        metrics = {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'valid_predictions': 0,
            'no_response_count': no_response_count,
            'response_rate': 0.0
        }
    
    # Calculate confidence scores for AUC if available (only for valid predictions)
    if len(valid_predictions) > 0:
        valid_confidences = []
        for i, p in enumerate(parsed_predictions):
            if p['prediction_text'] != 'NO_RESPONSE' and 'confidence' in p:
                valid_confidences.append(p['confidence'])
        
        if len(valid_confidences) == len(valid_predictions):
            try:
                metrics['auc'] = roc_auc_score(valid_true_labels, valid_confidences)
                metrics['auprc'] = average_precision_score(valid_true_labels, valid_confidences)
            except:
                metrics['auc'] = 0.0
                metrics['auprc'] = 0.0
        else:
            metrics['auc'] = 0.0
            metrics['auprc'] = 0.0
    else:
        metrics['auc'] = 0.0
        metrics['auprc'] = 0.0
    
    return metrics, parsed_predictions

def main():
    parser = argparse.ArgumentParser(description="Baseline 2: Direct LLM Prediction for ICU Mortality")
    parser.add_argument("--data_dir", default="/scratch/bcew/ruikez2/intern/s3_med/data/baseline_text_500", 
                        help="Directory containing patient data")
    parser.add_argument("--output_dir", default="/scratch/bcew/ruikez2/intern/s3_med/results/baseline_2", 
                        help="Directory to save results")
    parser.add_argument("--prompt_type", default="direct", choices=["direct", "cot"],
                        help="Type of prompt to use (direct or chain-of-thought)")
    parser.add_argument("--max_patients", type=int, default=None,
                        help="Maximum number of patients to process")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="LLM temperature for generation")
    parser.add_argument("--batch_delay", type=float, default=0.1,
                        help="Delay between requests in seconds")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading patient data from {args.data_dir}...")
    patients = load_patient_data(args.data_dir)
    
    if args.max_patients:
        patients = patients[:args.max_patients]
    
    print(f"Processing {len(patients)} patients with {args.prompt_type} prompting...")
    
    predictions = []
    true_labels = []
    failed_patients = []
    
    for patient in tqdm(patients, desc="Processing patients"):
        patient_id = patient['patient_id']
        # Remove the OUTCOME section to avoid data leakage
        patient_text_full = patient['text']
        if "=== OUTCOME ===" in patient_text_full:
            patient_text = patient_text_full.split("=== OUTCOME ===")[0].strip()
        else:
            patient_text = patient_text_full
        true_mortality = patient['mortality']
        
        # Create prompt based on type
        if args.prompt_type == "cot":
            prompt = create_cot_prompt(patient_text)
        else:
            prompt = create_direct_prompt(patient_text)
        
        # Add delay to avoid overwhelming the server
        import time
        time.sleep(args.batch_delay)
        
        # Get LLM prediction with retries
        response = call_llm(prompt, temperature=args.temperature)
        
        if not response:
            failed_patients.append(patient_id)
        
        # Store raw response first, parse later
        prediction_result = {
            'patient_id': patient_id,
            'true_label': true_mortality,
            'raw_response': response  # Store full response
        }
        
        predictions.append(prediction_result)
        true_labels.append(true_mortality)
        
        # Print example for first patient
        if len(predictions) == 1:
            print("\nExample prediction:")
            print(f"Patient ID: {patient_id}")
            print(f"True mortality: {true_mortality}")
            print(f"Response preview: {response[:200] if response else 'No response'}...")
    
    # Print failure statistics
    if failed_patients:
        print(f"\nWarning: {len(failed_patients)} patients failed to get predictions")
        print(f"Failed rate: {len(failed_patients)/len(patients)*100:.1f}%")
    
    # Evaluate results
    print("\nEvaluating predictions...")
    metrics, parsed_predictions = evaluate_predictions(predictions, true_labels)
    
    # Save results
    results = {
        'model': MODEL,
        'prompt_type': args.prompt_type,
        'num_patients': len(patients),
        'num_failed': len(failed_patients),
        'metrics': metrics,
        'raw_predictions': predictions,  # Save raw responses
        'parsed_predictions': parsed_predictions,  # Save parsed predictions
        'failed_patients': failed_patients
    }
    
    output_file = os.path.join(args.output_dir, f'baseline_2_{args.prompt_type}_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    # Print summary
    print("\n" + "="*50)
    print(f"BASELINE 2: DIRECT LLM PREDICTION ({args.prompt_type.upper()})")
    print("="*50)
    print(f"Model: {MODEL}")
    print(f"Number of patients: {len(patients)}")
    print(f"Failed predictions: {len(failed_patients)}")
    print(f"\nMetrics (on {metrics.get('valid_predictions', 0)} valid predictions):")
    print(f"  Response Rate: {metrics.get('response_rate', 0):.1%}")
    print(f"  Accuracy:  {metrics['accuracy']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall:    {metrics['recall']:.3f}")
    print(f"  F1 Score:  {metrics['f1']:.3f}")
    print(f"  AUROC:     {metrics['auc']:.3f}")
    print(f"  AUPRC:     {metrics.get('auprc', 0.0):.3f}")
    
    # Confusion matrix (use parsed predictions)
    tp = sum(1 for p, t in zip(parsed_predictions, true_labels) if p['prediction'] == 1 and t == 1)
    tn = sum(1 for p, t in zip(parsed_predictions, true_labels) if p['prediction'] == 0 and t == 0)
    fp = sum(1 for p, t in zip(parsed_predictions, true_labels) if p['prediction'] == 1 and t == 0)
    fn = sum(1 for p, t in zip(parsed_predictions, true_labels) if p['prediction'] == 0 and t == 1)
    
    print(f"\nConfusion Matrix:")
    print(f"  True Positive:  {tp}")
    print(f"  True Negative:  {tn}")
    print(f"  False Positive: {fp}")
    print(f"  False Negative: {fn}")

if __name__ == "__main__":
    main()