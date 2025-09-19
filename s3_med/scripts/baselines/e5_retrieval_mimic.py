#!/usr/bin/env python3
"""
Run E5 retrieval for MIMIC ICU mortality data
"""

import pandas as pd
import requests
import json
import argparse
import os
from tqdm import tqdm
import numpy as np

def search(query: str, endpoint: str):
    """Search using the retrieval API"""
    payload = {
        "queries": [query],
        "topk": 5,   #todo
        "return_scores": True
    }
    try:
        response = requests.post(endpoint, json=payload)
        response.raise_for_status()
        results = response.json()['result']
    except Exception as e:
        return ""

    def _passages2string(retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            # Handle different possible data structures
            if 'document' in doc_item:
                if 'contents' in doc_item['document']:
                    ##content revise
                    content = f"{doc_item['document']['title']} {doc_item['document']['text']}"
                elif 'content' in doc_item['document']:
                    content = doc_item['document']['content']
                else:
                    # Try to get any text field
                    content = str(doc_item['document'].get('text', doc_item['document']))
            elif 'contents' in doc_item:
                ##content revise
                content = f"{doc_item['title']} {doc_item['text']}"
            elif 'content' in doc_item:
                content = doc_item['content']
            else:
                content = str(doc_item)
            
            # Extract title and text
            if isinstance(content, str) and "\n" in content:
                title = content.split("\n")[0]
                text = "\n".join(content.split("\n")[1:])
            else:
                title = f"Document {idx+1}"
                text = str(content)
            
            format_reference += f"Doc {idx+1} (Title: {title}) {text}\n"
        return format_reference

    formatted_docs = _passages2string(results[0])
    return formatted_docs

def extract_question_from_prompt(prompt_content: str) -> str:
    """Extract the question and patient data from the prompt"""
    import re
    
    # Extract question from the prompt
    question_match = re.search(r'<question>(.*?)</question>', prompt_content, re.DOTALL)
    question = question_match.group(1).strip() if question_match else ""
    
    # Find ALL patient_data sections
    patient_matches = re.findall(r'<patient_data>(.*?)</patient_data>', prompt_content, re.DOTALL)
    
    # The last one should be the actual patient data (not the placeholder)
    patient_data = ""
    if patient_matches:
        for match in reversed(patient_matches):
            cleaned_match = match.strip()
            # Skip placeholders like "[patient clinical information]"
            if not cleaned_match.startswith('[') and not cleaned_match.endswith(']'):
                patient_data = cleaned_match
                break
        # If all are placeholders, use the last one
        if not patient_data and patient_matches:
            patient_data = patient_matches[-1].strip()
    
    # Combine question and patient data for the query
    if question and patient_data:
        return f"{question} {patient_data}"
    elif patient_data:
        return patient_data
    else:
        return "No patient data found"

def main():
    parser = argparse.ArgumentParser(description="Run retrieval for MIMIC data and save JSON outputs.")
    parser.add_argument("--input_parquet", required=True, help="Input .parquet file with MIMIC data.")
    parser.add_argument("--output_dir", required=True, help="Directory to store output JSON files.")
    parser.add_argument("--endpoint", required=True, help="Retrieval API endpoint URL (e.g., http://127.0.0.1:3000/retrieve)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Read the parquet file
    df = pd.read_parquet(args.input_parquet)
    print(f"[INFO] Loaded {len(df)} entries from {args.input_parquet}")

    # Get unique data sources in the file
    data_sources = df['data_source'].unique()
    print(f"[INFO] Found data sources: {data_sources}")

    for data_source in data_sources:
        print(f"[INFO] Processing: {data_source}")
        retrieval_info = {}
        qa_data = df[df['data_source'] == data_source]
        
        for index, row in tqdm(qa_data.iterrows(), total=len(qa_data), desc=f"Processing {data_source}"):
            # Extract question and patient data from prompt
            prompt_content = row['prompt'][0]['content']
            question_and_patient_data = extract_question_from_prompt(prompt_content)
            
            # Get patient ID and other info from ground truth
            ground_truth = row['reward_model']['ground_truth']
            patient_id = ground_truth.get('patient_id', f'patient_{index}')
            mortality = ground_truth.get('mortality', 0)
            
            # Create search query - use original question + patient data directly
            # Using raw question + patient data without any modifications
            # import pdb; pdb.set_trace()
            search_query = question_and_patient_data
            
            # Perform retrieval
            retrieval_result = search(search_query, args.endpoint)
            # import pdb; pdb.set_trace()
            
            # Convert any numpy types to Python native types
            golden_answers = ground_truth.get('answers', ["high mortality risk" if mortality else "low mortality risk"])
            if isinstance(golden_answers, np.ndarray):
                golden_answers = golden_answers.tolist()
            
            # Store retrieval info with patient ID as key
            question_info = {
                'patient_id': str(patient_id),
                'query_used': question_and_patient_data,  # Store the actual query used for retrieval
                'mortality': int(mortality) if not isinstance(mortality, np.integer) else int(mortality.item()),
                'golden_answers': golden_answers,
                'context_with_info': retrieval_result
            }
            
            # Use patient_id as key for easier lookup
            retrieval_info[str(patient_id)] = question_info

        # Save output
        out_path = os.path.join(args.output_dir, f"{data_source}_output_sequences.json")
        with open(out_path, 'w') as f:
            json.dump(retrieval_info, f, indent=4)
        print(f"[INFO] Saved {len(retrieval_info)} entries to: {out_path}")

if __name__ == "__main__":
    main()