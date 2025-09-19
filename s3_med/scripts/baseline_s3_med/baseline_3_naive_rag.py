#!/usr/bin/env python3
"""
Baseline 3: Naive RAG + LLM
Use patient data text to retrieve relevant medical knowledge, then feed both to LLM for prediction
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
RETRIEVAL_ENDPOINT = "http://127.0.0.1:3000/retrieve"  # Medical knowledge retriever endpoint

def call_llm(prompt: str, temperature: float = 0.3, max_tokens: int = 2000, max_retries: int = 3) -> str:
    """Call the LLM with a prompt and get response"""
    headers = {"Content-Type": "application/json"}
    
    # Smart truncation to avoid breaking prompt structure
    MAX_PROMPT_LENGTH = 6000  # Reduced to be safer with token limits
    
    if len(prompt) > MAX_PROMPT_LENGTH:
        print(f"Warning: Truncating prompt from {len(prompt)} to {MAX_PROMPT_LENGTH} chars")
        
        # Try to preserve structure by finding key sections
        patient_start = prompt.find("=== PATIENT CLINICAL DATA ===")
        knowledge_start = prompt.find("=== RELEVANT MEDICAL KNOWLEDGE ===")
        task_start = prompt.find("=== TASK ===")
        
        if patient_start != -1 and knowledge_start != -1 and task_start != -1:
            # Smart truncation: keep intro, truncate middle sections, keep instructions
            intro = prompt[:patient_start]
            patient_section = prompt[patient_start:knowledge_start]
            knowledge_section = prompt[knowledge_start:task_start]
            task_section = prompt[task_start:]
            
            # Calculate space for each section
            fixed_len = len(intro) + len(task_section) + 200  # buffer
            available = MAX_PROMPT_LENGTH - fixed_len
            
            # Allocate 60% to patient data, 40% to knowledge
            patient_max = int(available * 0.6)
            knowledge_max = int(available * 0.4)
            
            if len(patient_section) > patient_max:
                patient_section = patient_section[:patient_max] + "\n[...truncated...]\n"
            if len(knowledge_section) > knowledge_max:
                knowledge_section = knowledge_section[:knowledge_max] + "\n[...truncated...]\n"
            
            prompt = intro + patient_section + knowledge_section + task_section
        else:
            # Fallback: simple truncation but keep the task instructions at the end
            prompt = prompt[:MAX_PROMPT_LENGTH-500] + "\n[...truncated...]\n" + prompt[-500:]
    
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
            response = requests.post("http://localhost:8000/v1/chat/completions", headers=headers, json=payload, timeout=30)
            
            if response.status_code == 400:
                print(f"400 Bad Request - likely prompt too long or format issue")
                # Try with shorter prompt on 400 error
                if len(prompt) > 4000 and attempt < max_retries - 1:
                    prompt = prompt[:4000] + "\n\nPredict: HIGH RISK or LOW RISK?"
                    messages = [{"role": "user", "content": prompt}]
                    payload["messages"] = messages
                    continue
            
            response.raise_for_status()
            res = response.json()
            
            if "choices" not in res or not res["choices"]:
                raise ValueError("Invalid response from LLM")
                
            content = res["choices"][0]["message"]["content"].strip()
            if content:  # Only return if we got actual content
                return content
            
        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries} - Error: {str(e)[:100]}")
            if attempt < max_retries - 1:
                import time
                time.sleep(1)  # Brief pause before retry
    
    # If all retries failed, return a default prediction
    print("All retries failed, using fallback response")
    return "PREDICTION: LOW RISK\nKEY RISK FACTORS: Unable to analyze due to technical error\nCONFIDENCE: Low"

def retrieve_medical_knowledge(query: str, topk: int = 5) -> str:
    """Retrieve relevant medical knowledge from the knowledge base"""
    # Limit query length to avoid issues
    if len(query) > 1000:
        query = query[:1000]
    
    payload = {
        "queries": [query],
        "topk": topk,
        "return_scores": True
    }
    
    try:
        response = requests.post(RETRIEVAL_ENDPOINT, json=payload)
        response.raise_for_status()
        results = response.json()['result']
        
        # Format retrieved documents
        formatted_docs = ""
        for idx, doc_item in enumerate(results[0]):
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
                text = "\n".join(content.split("\n")[1:])[:500]  # Limit text length
            else:
                title = f"Document {idx+1}"
                text = str(content)[:500]
            
            formatted_docs += f"\n[Doc {idx+1}] {title}\n{text}\n"
            
            # Limit total docs length to save space
            if len(formatted_docs) > 2000:
                formatted_docs += "\n[Additional documents truncated...]\n"
                break
        
        return formatted_docs
    
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        return ""

def load_patient_data(data_dir):
    """Load patient text data"""
    patient_file = os.path.join(data_dir, 'all_patients.json')
    
    with open(patient_file, 'r') as f:
        patients = json.load(f)
    
    return patients

def extract_key_features(patient_text):
    """Extract key features from patient text for better retrieval"""
    # Extract vital signs patterns
    features = []
    
    # Look for abnormal values
    if "blood pressure" in patient_text.lower():
        # Check for hypotension or hypertension
        bp_values = re.findall(r'blood pressure: (\d+)', patient_text)
        if bp_values:
            bp_nums = [int(x) for x in bp_values[:5]]  # Check first few values
            if any(bp < 60 for bp in bp_nums):
                features.append("severe hypotension")
            elif any(bp > 140 for bp in bp_nums):
                features.append("hypertension")
    
    # Check Glasgow Coma Scale
    if "glascow coma scale" in patient_text.lower():
        gcs_motor = re.findall(r'motor response: (\d+)', patient_text.lower())
        if gcs_motor and any(int(x) < 4 for x in gcs_motor[:3]):
            features.append("altered consciousness low GCS score")
    
    # Check for medical history
    if "septicemia" in patient_text.lower():
        features.append("sepsis ICU mortality")
    if "renal failure" in patient_text.lower():
        features.append("acute kidney injury mortality")
    if "heart failure" in patient_text.lower():
        features.append("congestive heart failure ICU prognosis")
    
    # Age as risk factor
    age_match = re.search(r'Age: (\d+)', patient_text)
    if age_match:
        age = int(age_match.group(1))
        if age > 70:
            features.append("elderly ICU mortality risk factors")
    
    return " ".join(features) if features else "ICU patient mortality prediction vital signs"

def create_rag_prompt(patient_text, retrieved_docs):
    """Create a RAG-enhanced prompt for the LLM"""
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
- Only predict HIGH RISK if there are clear, severe indicators of imminent death
- Use the medical literature to support your assessment
- Focus on evidence-based risk factors
- Provide a clear prediction: HIGH RISK (likely to die) or LOW RISK (likely to survive)

=== PATIENT CLINICAL DATA ===
{patient_text}

=== RELEVANT MEDICAL KNOWLEDGE ===
{retrieved_docs}

=== TASK ===
Based on the patient data and medical literature above:
1. Identify key risk factors from the patient's data that align with the medical literature
2. Assess the severity of the patient's condition
3. Predict the ICU mortality outcome



Format your response as:
PREDICTION: [HIGH RISK or LOW RISK]
KEY RISK FACTORS: [List 3 main factors based on data and literature]
EVIDENCE: [Cite relevant information from the medical knowledge]
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
    confidence = 0.5  # Default
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
    
    # Calculate AUC and AUPRC if confidence scores available
    confidence_scores = [p['confidence'] for p in predictions]
    try:
        metrics['auc'] = roc_auc_score(true_binary, confidence_scores)
        metrics['auprc'] = average_precision_score(true_binary, confidence_scores)
    except:
        metrics['auc'] = 0.0
        metrics['auprc'] = 0.0
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Baseline 3: Naive RAG + LLM for ICU Mortality")
    parser.add_argument("--data_dir", default="/scratch/bcew/ruikez2/intern/s3_med/data/baseline_text_500", 
                        help="Directory containing patient data")
    parser.add_argument("--output_dir", default="/scratch/bcew/ruikez2/intern/s3_med/results/baseline_3", 
                        help="Directory to save results")
    parser.add_argument("--retrieval_endpoint", default="http://127.0.0.1:3000/retrieve",
                        help="Medical knowledge retrieval endpoint")
    parser.add_argument("--topk", type=int, default=5,
                        help="Number of documents to retrieve")
    parser.add_argument("--max_patients", type=int, default=None,
                        help="Maximum number of patients to process")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="LLM temperature for generation")
    parser.add_argument("--max_retries", type=int, default=3,
                        help="Maximum retry attempts for failed LLM calls")
    args = parser.parse_args()
    
    # Update global endpoint
    global RETRIEVAL_ENDPOINT
    RETRIEVAL_ENDPOINT = args.retrieval_endpoint
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading patient data from {args.data_dir}...")
    patients = load_patient_data(args.data_dir)
    
    if args.max_patients:
        patients = patients[:args.max_patients]
    
    print(f"Processing {len(patients)} patients with Naive RAG...")
    print(f"Retrieval endpoint: {RETRIEVAL_ENDPOINT}")
    print(f"Top-k documents: {args.topk}")
    
    predictions = []
    true_labels = []
    
    for patient in tqdm(patients, desc="Processing patients"):
        patient_id = patient['patient_id']
        # Remove the OUTCOME section to avoid data leakage
        patient_text_full = patient['text']
        if "=== OUTCOME ===" in patient_text_full:
            patient_text = patient_text_full.split("=== OUTCOME ===")[0].strip()
        else:
            patient_text = patient_text_full
        true_mortality = patient['mortality']
        
        # Extract key features for retrieval
        query_features = extract_key_features(patient_text)
        
        # Create retrieval query
        # Combine patient summary with key features
        patient_summary = patient_text.split("=== CLINICAL DATA ===")[0] if "=== CLINICAL DATA ===" in patient_text else patient_text[:500]
        retrieval_query = f"ICU mortality prediction {query_features} {patient_summary}"
        
        # Retrieve relevant medical knowledge
        retrieved_docs = retrieve_medical_knowledge(retrieval_query, topk=args.topk)
        
        # Create RAG-enhanced prompt
        prompt = create_rag_prompt(patient_text, retrieved_docs)
        
        # Get LLM prediction
        max_retries = args.max_retries if hasattr(args, 'max_retries') else 3
        response = call_llm(prompt, temperature=args.temperature, max_retries=max_retries)
        
        # Parse prediction and confidence
        predicted_mortality, confidence = parse_prediction(response)
        
        # Store results
        prediction_result = {
            'patient_id': patient_id,
            'prediction': predicted_mortality,
            'true_label': true_mortality,
            'confidence': confidence,
            'query_features': query_features,
            'retrieved_docs': retrieved_docs[:1000],  # Store first 1000 chars
            'response': response  # Store full response
        }
        
        predictions.append(prediction_result)
        true_labels.append(true_mortality)
        
        # Print example for first patient
        if len(predictions) == 1:
            print("\nExample prediction:")
            print(f"Patient ID: {patient_id}")
            print(f"Query features: {query_features}")
            print(f"Retrieved docs preview: {retrieved_docs[:200]}...")
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
        'retrieval_endpoint': RETRIEVAL_ENDPOINT,
        'topk': args.topk,
        'num_patients': len(patients),
        'metrics': metrics,
        'predictions': predictions
    }
    
    output_file = os.path.join(args.output_dir, 'baseline_3_naive_rag_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    # Print summary
    print("\n" + "="*50)
    print("BASELINE 3: NAIVE RAG + LLM PREDICTION")
    print("="*50)
    print(f"Model: {MODEL}")
    print(f"Retrieval: {args.topk} documents")
    print(f"Number of patients: {len(patients)}")
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