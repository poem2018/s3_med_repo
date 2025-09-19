#!/usr/bin/env python3
"""
Baseline 4: Similar Cases + RAG + LLM
Find similar patient cases, use them with RAG to retrieve medical knowledge, 
then feed everything to LLM for prediction
"""

import json
import numpy as np
import requests
import argparse
import os
from pathlib import Path
from tqdm import tqdm
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import warnings
warnings.filterwarnings('ignore')

MODEL = "Qwen/Qwen2.5-3B-Instruct"
RETRIEVAL_ENDPOINT = "http://127.0.0.1:3000/retrieve"

# Indicator weights for similarity calculation - all equal weights
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

def call_llm(prompt: str, temperature: float = 0.3, max_tokens: int = 2000) -> str:
    """Call the LLM with a prompt and get response"""
    headers = {"Content-Type": "application/json"}
    
    # Truncate prompt if too long (keep first part)
    if len(prompt) > 8000:
        print(f"Warning: Truncating prompt from {len(prompt)} to 8000 chars")
        prompt = prompt[:8000]
    
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
    
    try:
        response = requests.post("http://localhost:8000/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        res = response.json()
        
        if "choices" not in res or not res["choices"]:
            raise ValueError("Invalid response from LLM")
            
        return res["choices"][0]["message"]["content"].strip()
        
    except Exception as e:
        print(f"Error generating answer: {e}")
        return ""

def retrieve_medical_knowledge(query: str, topk: int = 3) -> str:
    """Retrieve relevant medical knowledge from the knowledge base"""
    payload = {
        "queries": [query],
        "topk": topk,
        "return_scores": True
    }
    
    try:
        response = requests.post(RETRIEVAL_ENDPOINT, json=payload)
        response.raise_for_status()
        results = response.json()['result']
        
        formatted_docs = ""
        for idx, doc_item in enumerate(results[0]):
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
    
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        return ""

def load_data(data_dir):
    """Load all necessary data files"""
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
    """Calculate weighted similarity between two patient matrices"""
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
    """Find k most similar patients to the target patient"""
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

def format_similar_cases(similar_patients, patients_text):
    """Format similar patient cases for the prompt"""
    formatted = ""
    for i, similar in enumerate(similar_patients, 1):
        # Find the full text for this patient
        patient_text = None
        for p in patients_text:
            if p['patient_id'] == similar['patient_id']:
                patient_text = p['text']
                break
        
        if patient_text:
            # Extract summary (first part before clinical data)
            summary = patient_text.split("=== CLINICAL DATA ===")[0] if "=== CLINICAL DATA ===" in patient_text else patient_text[:300]
            outcome = "DIED" if similar['mortality'] == 1 else "SURVIVED"
            
            formatted += f"\n--- Similar Case {i} (Similarity: {similar['similarity']:.2f}) ---\n"
            formatted += f"{summary}\n"
            formatted += f"OUTCOME: {outcome}\n"
    
    return formatted

def create_enhanced_prompt(patient_text, similar_cases, retrieved_docs):
    """Create a prompt enhanced with similar cases and medical knowledge"""
    prompt = f"""You are an expert ICU physician. Analyze the patient's data using similar historical cases and medical literature to predict ICU mortality.

Instructions:
- Remember that LOW RISK is more common (87% of ICU patients survive)
- Only predict HIGH RISK if there are clear, severe indicators of imminent death
- Weight the similar cases based on their similarity scores
- Use medical literature to validate patterns observed in similar cases
- Provide a clear prediction: HIGH RISK (likely to die) or LOW RISK (likely to survive)

=== CURRENT PATIENT ===
{patient_text}

=== SIMILAR HISTORICAL CASES ===
{similar_cases}

=== RELEVANT MEDICAL LITERATURE ===
{retrieved_docs}

=== TASK ===
Based on the comprehensive information above:

1. Compare the current patient with similar historical cases
2. Identify patterns and risk factors from both cases and literature
3. Consider the outcomes of similar patients
4. Make an evidence-based prediction

Format your response as:
PREDICTION: [HIGH RISK or LOW RISK]
SIMILAR CASE PATTERN: [What pattern do you see in similar cases?]
KEY RISK FACTORS: [List 3 main factors]
EVIDENCE: [Cite supporting evidence from cases and literature]
CONFIDENCE: [High/Medium/Low]
"""
    return prompt

def parse_prediction(response):
    """Parse the LLM response to extract prediction and confidence"""
    response_upper = response.upper()
    
    # Extract prediction
    if "HIGH RISK" in response_upper or "HIGH-RISK" in response_upper:
        prediction = 1
    elif "LOW RISK" in response_upper or "LOW-RISK" in response_upper:
        prediction = 0
    else:
        # Fallback
        death_keywords = ["WILL DIE", "LIKELY TO DIE", "MORTALITY LIKELY", "POOR PROGNOSIS"]
        prediction = 1 if any(kw in response_upper for kw in death_keywords) else 0
    
    # Extract confidence
    confidence = 0.5
    if "CONFIDENCE:" in response_upper:
        conf_match = re.search(r'CONFIDENCE:\s*(\w+)', response, re.IGNORECASE)
        if conf_match:
            conf_text = conf_match.group(1).upper()
            if "HIGH" in conf_text:
                confidence = 0.9 if prediction == 1 else 0.1
            elif "MEDIUM" in conf_text:
                confidence = 0.6 if prediction == 1 else 0.4
            elif "LOW" in conf_text:
                confidence = 0.3 if prediction == 1 else 0.7
    
    return prediction, confidence

def evaluate_predictions(predictions, true_labels):
    """Calculate evaluation metrics"""
    pred_binary = [p['prediction'] for p in predictions]
    true_binary = true_labels
    
    metrics = {
        'accuracy': accuracy_score(true_binary, pred_binary),
        'precision': precision_score(true_binary, pred_binary, zero_division=0),
        'recall': recall_score(true_binary, pred_binary, zero_division=0),
        'f1': f1_score(true_binary, pred_binary, zero_division=0)
    }
    
    confidence_scores = [p['confidence'] for p in predictions]
    try:
        metrics['auc'] = roc_auc_score(true_binary, confidence_scores)
        metrics['auprc'] = average_precision_score(true_binary, confidence_scores)
    except:
        metrics['auc'] = 0.0
        metrics['auprc'] = 0.0
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Baseline 4: Similar Cases + RAG + LLM for ICU Mortality")
    parser.add_argument("--data_dir", default="/scratch/bcew/ruikez2/intern/s3_med/data/baseline_text_500", 
                        help="Directory containing patient data")
    parser.add_argument("--output_dir", default="/scratch/bcew/ruikez2/intern/s3_med/results/baseline_4", 
                        help="Directory to save results")
    parser.add_argument("--retrieval_endpoint", default="http://127.0.0.1:3000/retrieve",
                        help="Medical knowledge retrieval endpoint")
    parser.add_argument("--k_similar", type=int, default=3,
                        help="Number of similar cases to retrieve")
    parser.add_argument("--topk_docs", type=int, default=3,
                        help="Number of medical documents to retrieve")
    parser.add_argument("--max_patients", type=int, default=None,
                        help="Maximum number of patients to process")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="LLM temperature for generation")
    args = parser.parse_args()
    
    global RETRIEVAL_ENDPOINT
    RETRIEVAL_ENDPOINT = args.retrieval_endpoint
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading data from {args.data_dir}...")
    patients_text, matrices, patient_info = load_data(args.data_dir)
    
    if args.max_patients:
        patients_text = patients_text[:args.max_patients]
        matrices = matrices[:args.max_patients]
        patient_info = patient_info[:args.max_patients]
    
    print(f"Processing {len(patients_text)} patients with Similar Cases + RAG...")
    print(f"K similar cases: {args.k_similar}")
    print(f"Top-k documents: {args.topk_docs}")
    
    predictions = []
    true_labels = []
    
    for i, patient in enumerate(tqdm(patients_text, desc="Processing patients")):
        patient_id = patient['patient_id']
        # Remove the OUTCOME section to avoid data leakage
        patient_text_full = patient['text']
        if "=== OUTCOME ===" in patient_text_full:
            patient_text = patient_text_full.split("=== OUTCOME ===")[0].strip()
        else:
            patient_text = patient_text_full
        true_mortality = patient['mortality']
        
        # Find similar patients
        similar_patients = find_similar_patients(i, matrices, patient_info, k=args.k_similar)
        similar_cases_text = format_similar_cases(similar_patients, patients_text)
        
        # Create query for medical knowledge retrieval
        # Combine patient info with similar case patterns
        similar_outcomes = [sp['mortality'] for sp in similar_patients]
        if sum(similar_outcomes) > len(similar_outcomes) / 2:
            query_context = "high mortality risk ICU patients similar cases died"
        else:
            query_context = "ICU patients recovery patterns similar cases survived"
        
        patient_summary = patient_text.split("=== CLINICAL DATA ===")[0] if "=== CLINICAL DATA ===" in patient_text else patient_text[:300]
        retrieval_query = f"{query_context} {patient_summary}"
        
        # Retrieve medical knowledge
        retrieved_docs = retrieve_medical_knowledge(retrieval_query, topk=args.topk_docs)
        
        # Create enhanced prompt
        prompt = create_enhanced_prompt(patient_text, similar_cases_text, retrieved_docs)
        
        # Get LLM prediction
        response = call_llm(prompt, temperature=args.temperature)
        
        # Parse prediction
        predicted_mortality, confidence = parse_prediction(response)
        
        # Store results
        prediction_result = {
            'patient_id': patient_id,
            'prediction': predicted_mortality,
            'true_label': true_mortality,
            'confidence': confidence,
            'similar_patients': similar_patients,
            'response': response  # Store full response
        }
        
        predictions.append(prediction_result)
        true_labels.append(true_mortality)
        
        # Print example for first patient
        if len(predictions) == 1:
            print("\nExample prediction:")
            print(f"Patient ID: {patient_id}")
            print(f"Similar patients: {[sp['patient_id'] for sp in similar_patients]}")
            print(f"Similar outcomes: {similar_outcomes}")
            print(f"True mortality: {true_mortality}")
            print(f"Predicted: {predicted_mortality}")
            print(f"Confidence: {confidence}")
            print(f"Response preview: {response[:200]}...")
    
    # Evaluate results
    print("\nEvaluating predictions...")
    metrics = evaluate_predictions(predictions, true_labels)
    
    # Save results
    results = {
        'model': MODEL,
        'k_similar': args.k_similar,
        'topk_docs': args.topk_docs,
        'num_patients': len(patients_text),
        'metrics': metrics,
        'predictions': predictions
    }
    
    output_file = os.path.join(args.output_dir, 'baseline_4_similar_rag_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    # Print summary
    print("\n" + "="*50)
    print("BASELINE 4: SIMILAR CASES + RAG + LLM")
    print("="*50)
    print(f"Model: {MODEL}")
    print(f"Similar cases: {args.k_similar}")
    print(f"Medical docs: {args.topk_docs}")
    print(f"Number of patients: {len(patients_text)}")
    print(f"\nMetrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall:    {metrics['recall']:.3f}")
    print(f"  F1 Score:  {metrics['f1']:.3f}")
    print(f"  AUROC:     {metrics['auc']:.3f}")
    print(f"  AUPRC:     {metrics.get('auprc', 0.0):.3f}")
    
    # Confusion matrix
    tp = sum(1 for p, t in zip(predictions, true_labels) if p['prediction'] == 1 and t == 1)
    tn = sum(1 for p, t in zip(predictions, true_labels) if p['prediction'] == 0 and t == 0)
    fp = sum(1 for p, t in zip(predictions, true_labels) if p['prediction'] == 1 and t == 0)
    fn = sum(1 for p, t in zip(predictions, true_labels) if p['prediction'] == 0 and t == 1)
    
    print(f"\nConfusion Matrix:")
    print(f"  True Positive:  {tp}")
    print(f"  True Negative:  {tn}")
    print(f"  False Positive: {fp}")
    print(f"  False Negative: {fn}")

if __name__ == "__main__":
    main()