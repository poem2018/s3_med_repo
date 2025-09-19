#!/usr/bin/env python3
"""
Baseline 3 with Token Probabilities: Naive RAG + LLM using logprobs
Retrieves relevant medical knowledge and uses it with token probabilities for confidence
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

                    # Check if token contains HIGH or LOW (not just exact match)
                    if "HIGH" in token:
                        # Convert logprob to probability
                        prob = math.exp(logprob)
                        confidence = prob  # HIGH = death probability
                        found_key_token = True
                    elif "LOW" in token:
                        # Convert logprob to probability
                        prob = math.exp(logprob)
                        confidence = 1 - prob  # LOW = survival, invert for death probability
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

def retrieve_medical_knowledge(query, retrieval_endpoint="http://127.0.0.1:3000/retrieve", topk=5, debug=False):
    """Retrieve relevant medical knowledge from the retrieval service"""
    if debug:
        print(f"\n[DEBUG] RAG Retrieval:")
        print(f"  Endpoint: {retrieval_endpoint}")
        print(f"  Query preview: {query[:100]}...")

    try:
        payload = {
            "queries": [query],  # API expects array of queries
            "topk": topk,
            "return_scores": True  # Add this to match what server expects
        }

        response = requests.post(retrieval_endpoint, json=payload, timeout=30)  # Increase timeout

        if debug:
            print(f"  Response status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            # Try 'result' key (as in baseline_3_naive_rag.py)
            if 'result' in data:
                results = data['result']
                if isinstance(results, list) and len(results) > 0 and isinstance(results[0], list):
                    # Format retrieved documents similar to baseline_3_naive_rag.py
                    formatted_docs = ""
                    for idx, doc_item in enumerate(results[0][:topk]):
                        # Extract document content
                        if 'document' in doc_item:
                            if 'contents' in doc_item['document']:
                                content = f"{doc_item['document'].get('title', '')} {doc_item['document'].get('text', '')}"
                            elif 'content' in doc_item['document']:
                                content = doc_item['document']['content']
                            else:
                                content = str(doc_item['document'].get('text', doc_item['document']))
                        elif 'contents' in doc_item:
                            content = f"{doc_item.get('title', '')} {doc_item.get('text', '')}"
                        elif 'content' in doc_item:
                            content = doc_item['content']
                        else:
                            content = str(doc_item)

                        # Format for output
                        if isinstance(content, str) and "\n" in content:
                            title = content.split("\n")[0]
                            text = "\n".join(content.split("\n")[1:])[:500]
                        else:
                            title = f"Document {idx+1}"
                            text = str(content)[:500]

                        formatted_docs += f"\n[Doc {idx+1}] {title}\n{text}\n"

                        # Limit total docs length
                        if len(formatted_docs) > 2000:
                            formatted_docs += "\n[Additional documents truncated...]\n"
                            break

                    if debug:
                        print(f"  Retrieved documents successfully")
                    return formatted_docs
            # Fallback to 'results' key if 'result' not found
            elif data.get("results"):
                # Original handling code...
                results = data.get("results")
                if isinstance(results, list) and len(results) > 0:
                    if isinstance(results[0], list):
                        results = results[0]
                    docs = []
                    for i, result in enumerate(results[:topk], 1):
                        doc = f"[Document {i}]\n"
                        doc += f"Title: {result.get('title', 'N/A')}\n"
                        doc += f"Content: {result.get('text', '')[:500]}...\n"
                        docs.append(doc)
                    if debug:
                        print(f"  Retrieved {len(docs)} documents successfully")
                    return "\n".join(docs)
            else:
                if debug:
                    print(f"  No results in response or success=False")
        elif response.status_code == 500:
            if debug:
                print(f"  Server error 500 - RAG service has internal error")
                print(f"  Continuing without RAG results...")
        else:
            if debug:
                print(f"  Error response: {response.text[:200]}")
    except requests.exceptions.ConnectionError as e:
        print(f"[ERROR] Cannot connect to RAG service at {retrieval_endpoint}")
        print(f"  Make sure the RAG service is running!")
        if debug:
            print(f"  Error details: {e}")
    except Exception as e:
        print(f"[ERROR] Retrieving medical knowledge: {e}")

    return "No relevant medical knowledge could be retrieved."

def extract_key_features(patient_text):
    """Extract key features from patient text to form retrieval query"""
    features = []
    
    # Look for vital signs patterns
    if "Blood Pressure" in patient_text:
        bp_values = re.findall(r'Blood Pressure \(systolic/diastolic\): (\d+)/(\d+)', patient_text)
        if bp_values:
            # Check for hypotension
            for sys, dia in bp_values[-3:]:  # Check last 3 readings
                if int(sys) < 90 or int(dia) < 60:
                    features.append("severe hypotension ICU mortality")
                    break
    
    # Look for oxygen patterns
    if "Oxygen saturation" in patient_text:
        o2_values = re.findall(r'Oxygen saturation \(SaO2\): ([\d.]+)%', patient_text)
        if o2_values:
            for val in o2_values[-3:]:
                if float(val) < 90:
                    features.append("hypoxemia respiratory failure ICU")
                    break
    
    # Look for glucose abnormalities
    if "Glucose" in patient_text:
        glucose_values = re.findall(r'Glucose: (\d+)', patient_text)
        if glucose_values:
            for val in glucose_values:
                if int(val) > 200:
                    features.append("hyperglycemia critical illness mortality")
                    break
    
    # Look for temperature abnormalities
    if "Temperature" in patient_text:
        temp_values = re.findall(r'Temperature \(Celsius\): ([\d.]+)', patient_text)
        if temp_values:
            for val in temp_values:
                if float(val) > 38.5 or float(val) < 35:
                    features.append("sepsis temperature abnormality ICU")
                    break
    
    # Look for heart failure indicators
    if "heart failure" in patient_text.lower() or "CHF" in patient_text:
        features.append("congestive heart failure ICU prognosis")
    
    # Age as risk factor
    age_match = re.search(r'Age: (\d+)', patient_text)
    if age_match:
        age = int(age_match.group(1))
        if age > 70:
            features.append("elderly ICU mortality risk factors")
    
    return " ".join(features) if features else "ICU patient mortality prediction vital signs"

def create_simple_rag_prompt(patient_text, retrieved_docs):
    """Create a simple RAG-enhanced prompt for single-token response"""
    # Limit patient text to focus on most important parts
    if len(patient_text) > 3000:
        # Keep patient info and early clinical data
        if "=== CLINICAL DATA ===" in patient_text:
            parts = patient_text.split("=== CLINICAL DATA ===")
            patient_text = parts[0] + "=== CLINICAL DATA ===" + parts[1][:2000]
        else:
            patient_text = patient_text[:3000]
    
    prompt = f"""You are an expert ICU physician. Based on the patient's clinical data and relevant medical literature, predict ICU mortality risk.

Instructions:
- Remember that LOW RISK is more common (87% of ICU patients survive)
- Only predict HIGH if there are clear, severe indicators
- Respond with ONLY ONE WORD: HIGH or LOW

=== PATIENT CLINICAL DATA ===
{patient_text}

=== RELEVANT MEDICAL KNOWLEDGE ===
{retrieved_docs}

Answer:"""
    return prompt

def create_cot_rag_prompt(patient_text, retrieved_docs):
    """Create a COT RAG-enhanced prompt with medical literature integration"""
    # Limit patient text to focus on most important parts
    if len(patient_text) > 2500:
        # Keep patient info and early clinical data
        if "=== CLINICAL DATA ===" in patient_text:
            parts = patient_text.split("=== CLINICAL DATA ===")
            patient_text = parts[0] + "=== CLINICAL DATA ===" + parts[1][:1500]
        else:
            patient_text = patient_text[:2500]
    
    prompt = f"""You are an expert ICU physician. Analyze the patient's clinical data using medical literature to predict ICU mortality step by step.

=== PATIENT CLINICAL DATA ===
{patient_text}

=== RELEVANT MEDICAL KNOWLEDGE ===
{retrieved_docs}

Analyze systematically:
1. Vital signs (BP, HR, O2, temp) - identify critical values
2. Neurological (GCS scores) - assess consciousness level
3. Labs (glucose, pH) - check for severe abnormalities
4. Risk factors - count severe indicators
5: Integration with Medical Evidence
- Synthesize patient data with literature-based risk models
- Apply evidence-based mortality predictors

Important:
- Remember that most ICU patients (87%) survive
- Base assessment on medical literature evidence
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

def load_patient_data(data_dir):
    """Load patient text data"""
    patient_file = os.path.join(data_dir, 'all_patients.json')
    
    with open(patient_file, 'r') as f:
        patients = json.load(f)
    
    return patients

def main():
    parser = argparse.ArgumentParser(description="Baseline 3 with Logprobs: RAG + Token Probability-based Confidence")
    parser.add_argument("--data_dir", default="/scratch/bcew/ruikez2/intern/s3_med/data/baseline_text_500", 
                        help="Directory containing patient data")
    parser.add_argument("--output_dir", default="/scratch/bcew/ruikez2/intern/s3_med/results/baseline_3_logprobs", 
                        help="Directory to save results")
    parser.add_argument("--retrieval_endpoint", default="http://127.0.0.1:3000/retrieve",
                        help="Endpoint for medical knowledge retrieval")
    parser.add_argument("--topk", type=int, default=5,
                        help="Number of documents to retrieve")
    parser.add_argument("--max_patients", type=int, default=None,
                        help="Maximum number of patients to process")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="LLM temperature for generation")
    parser.add_argument("--batch_delay", type=float, default=0.1,
                        help="Delay between requests in seconds")
    parser.add_argument("--prompt_type", choices=["simple", "cot"], default="simple",
                        help="Type of prompt to use: simple (single-token) or cot (chain-of-thought)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug output for RAG retrieval")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading patient data from {args.data_dir}...")
    patients = load_patient_data(args.data_dir)
    
    if args.max_patients:
        patients = patients[:args.max_patients]
    
    print(f"Processing {len(patients)} patients with RAG and logprobs...")
    
    predictions = []
    true_labels = []
    confidences = []
    responses = []  # Store response texts
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
        
        # Extract query features
        query_features = extract_key_features(patient_text)
        
        # Retrieve medical knowledge
        retrieved_docs = retrieve_medical_knowledge(query_features,
                                                     retrieval_endpoint=args.retrieval_endpoint,
                                                     topk=args.topk,
                                                     debug=args.debug)
        
        # Create RAG-enhanced prompt based on selected type
        if args.prompt_type == "cot":
            prompt = create_cot_rag_prompt(patient_text, retrieved_docs)
        else:
            prompt = create_simple_rag_prompt(patient_text, retrieved_docs)
        
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
            print(f"Query: {query_features[:100]}...")
            print(f"True mortality: {true_mortality}")
            print(f"Response: {response_text}")
            print(f"Prediction: {prediction}")
            print(f"Confidence: {confidence:.3f}")
    
    # Print failure statistics
    if failed_patients:
        print(f"\nWarning: {len(failed_patients)} patients failed to get predictions")
        print(f"Failed rate: {len(failed_patients)/len(patients)*100:.1f}%")
    
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
        'method': 'rag_logprobs',
        'prompt_type': args.prompt_type,
        'topk_docs': args.topk,
        'num_patients': len(patients),
        'num_failed': len(failed_patients),
        'metrics': metrics,
        'predictions': [
            {
                'patient_id': patients[i]['patient_id'],
                'true_label': true_labels[i],
                'prediction': predictions[i],
                'confidence': confidences[i],
                'response': responses[i]  # Include response text
            }
            for i in range(len(predictions))
        ],
        'failed_patients': failed_patients
    }
    
    output_file = os.path.join(args.output_dir, f'baseline_3_logprobs_{args.prompt_type}_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    # Print summary
    print("\n" + "="*50)
    print("BASELINE 3: RAG + LOGPROBS-BASED CONFIDENCE")
    print("="*50)
    print(f"Model: {MODEL}")
    print(f"Number of patients: {len(patients)}")
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