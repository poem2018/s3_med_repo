#!/usr/bin/env python3
"""
Run three groups of experiments for MIMIC ICU mortality prediction:
1. Naive RAG: Direct retrieval with question + patient data
2. Searcher-optimized RAG: Use S3 searcher to optimize query, then retrieve
3. Searcher-optimized RAG with similar cases: Add similar patient cases, optimize with searcher, then retrieve
"""

import json
import argparse
import os
import requests
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import re


class SearcherExperiment:
    """Run searcher-based experiments for MIMIC data"""
    
    def __init__(self, retrieval_endpoint: str, searcher_model: str = None, topk: int = 12):
        self.retrieval_endpoint = retrieval_endpoint
        self.searcher_model = searcher_model
        self.topk = topk
        
    def naive_rag(self, query: str) -> Dict[str, Any]:
        """
        Group 1: Naive RAG - Direct retrieval without searcher optimization
        """
        payload = {
            "queries": [query],
            "topk": self.topk,
            "return_scores": True
        }
        
        try:
            response = requests.post(self.retrieval_endpoint, json=payload, timeout=30)
            response.raise_for_status()
            results = response.json()['result']
            return {
                'method': 'naive_rag',
                'original_query': query,
                'final_query': query,  # Same as original for naive RAG
                'documents': self._format_documents(results[0] if results else [])
            }
        except Exception as e:
            print(f"[ERROR] Naive RAG failed: {e}")
            return {
                'method': 'naive_rag',
                'original_query': query,
                'final_query': query,
                'documents': [],
                'error': str(e)
            }
    
    def searcher_optimized_rag(self, query: str) -> Dict[str, Any]:
        """
        Group 2: Searcher-optimized RAG - Use searcher to generate optimized query
        """
        # Generate searcher-optimized query
        optimized_query = self._run_searcher_optimization(query)
        
        # Retrieve with optimized query
        payload = {
            "queries": [optimized_query],
            "topk": self.topk,
            "return_scores": True
        }
        
        try:
            response = requests.post(self.retrieval_endpoint, json=payload, timeout=30)
            response.raise_for_status()
            results = response.json()['result']
            return {
                'method': 'searcher_optimized',
                'original_query': query,
                'final_query': optimized_query,
                'documents': self._format_documents(results[0] if results else [])
            }
        except Exception as e:
            print(f"[ERROR] Searcher-optimized RAG failed: {e}")
            return {
                'method': 'searcher_optimized',
                'original_query': query,
                'final_query': optimized_query,
                'documents': [],
                'error': str(e)
            }
    
    def searcher_with_similar_cases_rag(self, query: str, similar_case: str) -> Dict[str, Any]:
        """
        Group 3: Searcher-optimized RAG with similar patient cases
        """
        # Combine original query with similar case
        combined_input = f"{query}\n\n<similar_case>\n{similar_case}\n</similar_case>"
        
        # Generate searcher-optimized query
        optimized_query = self._run_searcher_optimization(combined_input)
        
        # Retrieve with optimized query
        payload = {
            "queries": [optimized_query],
            "topk": self.topk,
            "return_scores": True
        }
        
        try:
            response = requests.post(self.retrieval_endpoint, json=payload, timeout=30)
            response.raise_for_status()
            results = response.json()['result']
            return {
                'method': 'searcher_with_similar_cases',
                'original_query': query,
                'similar_case': similar_case,
                'final_query': optimized_query,
                'documents': self._format_documents(results[0] if results else [])
            }
        except Exception as e:
            print(f"[ERROR] Searcher with similar cases RAG failed: {e}")
            return {
                'method': 'searcher_with_similar_cases',
                'original_query': query,
                'similar_case': similar_case,
                'final_query': optimized_query,
                'documents': [],
                'error': str(e)
            }
    
    def _run_searcher_optimization(self, input_text: str) -> str:
        """
        Simulate S3 searcher optimization process
        This should ideally call the actual S3 searcher model
        For now, we'll create a simplified version that extracts key clinical features
        """
        # Extract key clinical features for search optimization
        optimized_parts = []
        
        # Extract age and gender
        age_match = re.search(r'(\d+)\s*year\s*old', input_text, re.IGNORECASE)
        gender_match = re.search(r'(male|female)', input_text, re.IGNORECASE)
        
        if age_match and gender_match:
            optimized_parts.append(f"{age_match.group(1)} year old {gender_match.group(1)} ICU patient")
        
        # Extract clinical conditions
        conditions = []
        if 'mechanical ventilation' in input_text.lower():
            conditions.append('mechanical ventilation')
        if 'hyperglycemia' in input_text.lower() or re.search(r'glucose[>=\s]+(\d+)', input_text):
            conditions.append('hyperglycemia')
        if 'acidosis' in input_text.lower() or re.search(r'pH[<\s]+([\d.]+)', input_text):
            conditions.append('acidosis')
        if 'alkalosis' in input_text.lower() or re.search(r'pH[>\s]+([\d.]+)', input_text):
            conditions.append('alkalosis')
        if 'tachycardia' in input_text.lower() or re.search(r'HR[>=\s]+(\d+)', input_text):
            conditions.append('tachycardia')
        
        if conditions:
            optimized_parts.append(' '.join(conditions))
        
        # Add mortality prediction context
        optimized_parts.append('ICU mortality prediction risk factors prognosis')
        
        # If similar case is present, extract outcome
        if '<similar_case>' in input_text:
            if 'mortality: 1' in input_text or 'died' in input_text.lower():
                optimized_parts.append('high mortality risk factors')
            elif 'mortality: 0' in input_text or 'survived' in input_text.lower():
                optimized_parts.append('survival predictors protective factors')
        
        # Combine optimized parts
        optimized_query = ' '.join(optimized_parts)
        
        # Limit query length
        if len(optimized_query) > 500:
            optimized_query = optimized_query[:500]
        
        return optimized_query
    
    def _format_documents(self, retrieval_results: List[Dict]) -> List[Dict]:
        """Format retrieval results for output"""
        formatted_docs = []
        for idx, doc in enumerate(retrieval_results[:self.topk]):
            # Handle different document formats
            if isinstance(doc, dict):
                if 'document' in doc:
                    ##content revise
                    content = f"{doc['document'].get('title', '')} {doc['document'].get('text', '')}"
                    score = doc.get('score', 0.0)
                elif 'contents' in doc:
                    ##content revise
                    content = f"{doc.get('title', '')} {doc.get('text', '')}"
                    score = doc.get('score', 0.0)
                elif 'content' in doc:
                    content = doc['content']
                    score = doc.get('score', 0.0)
                else:
                    content = str(doc)
                    score = 0.0
            else:
                content = str(doc)
                score = 0.0
            
            # Extract title if present
            if isinstance(content, str) and "\n" in content:
                lines = content.split("\n")
                title = lines[0]
                text = "\n".join(lines[1:])
            else:
                title = f"Document {idx+1}"
                text = str(content)
            
            formatted_docs.append({
                'rank': idx + 1,
                'title': title,
                'text': text[:1000],  # Truncate for readability
                'score': score
            })
        
        return formatted_docs


def load_demo_data(demo_file: str) -> Dict[str, Any]:
    """Load demo patient data and similar cases"""
    with open(demo_file, 'r') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Run MIMIC searcher experiments")
    parser.add_argument("--demo_file", required=True, help="Path to demo data JSON file")
    parser.add_argument("--retrieval_endpoint", default="http://127.0.0.1:3000/retrieve",
                       help="Retrieval service endpoint")
    parser.add_argument("--output_dir", required=True, help="Directory to save experiment results")
    parser.add_argument("--topk", type=int, default=12, help="Number of documents to retrieve")
    parser.add_argument("--searcher_model", help="Path to searcher model (optional)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load demo data
    print(f"Loading demo data from {args.demo_file}...")
    demo_data = load_demo_data(args.demo_file)
    
    # Initialize experiment runner
    experiment = SearcherExperiment(
        retrieval_endpoint=args.retrieval_endpoint,
        searcher_model=args.searcher_model,
        topk=args.topk
    )
    
    # Run experiments for each demo case
    all_results = []
    
    for case_id, case_data in tqdm(demo_data.items(), desc="Running experiments"):
        print(f"\n{'='*60}")
        print(f"Processing case: {case_id}")
        print(f"{'='*60}")
        
        # Prepare query (question + patient data)
        query = case_data.get('query', '')
        if not query and 'question' in case_data and 'patient_data' in case_data:
            query = f"{case_data['question']}\n\n{case_data['patient_data']}"
        
        similar_case = case_data.get('similar_case', '')
        
        # Group 1: Naive RAG
        print("\n[Group 1] Running Naive RAG...")
        naive_result = experiment.naive_rag(query)
        
        # Group 2: Searcher-optimized RAG
        print("\n[Group 2] Running Searcher-optimized RAG...")
        searcher_result = experiment.searcher_optimized_rag(query)
        
        # Group 3: Searcher with similar cases RAG
        print("\n[Group 3] Running Searcher with similar cases RAG...")
        if similar_case:
            similar_case_result = experiment.searcher_with_similar_cases_rag(query, similar_case)
        else:
            print("  No similar case provided, skipping Group 3")
            similar_case_result = None
        
        # Combine results
        case_results = {
            'case_id': case_id,
            'ground_truth': case_data.get('ground_truth', {}),
            'naive_rag': naive_result,
            'searcher_optimized': searcher_result,
            'searcher_with_similar': similar_case_result
        }
        
        all_results.append(case_results)
        
        # Print summary
        print(f"\nResults summary for {case_id}:")
        print(f"  Naive RAG: Retrieved {len(naive_result['documents'])} documents")
        print(f"  Searcher-optimized: Retrieved {len(searcher_result['documents'])} documents")
        if similar_case_result:
            print(f"  Searcher with similar: Retrieved {len(similar_case_result['documents'])} documents")
        
        # Save intermediate results
        intermediate_file = os.path.join(args.output_dir, f"{case_id}_results.json")
        with open(intermediate_file, 'w') as f:
            json.dump(case_results, f, indent=2)
    
    # Save all results
    final_output_file = os.path.join(args.output_dir, "all_experiment_results.json")
    with open(final_output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Experiments completed!")
    print(f"Results saved to: {args.output_dir}")
    print(f"Final results file: {final_output_file}")
    
    # Generate comparison report
    generate_comparison_report(all_results, args.output_dir)


def generate_comparison_report(results: List[Dict], output_dir: str):
    """Generate a comparison report of the three methods"""
    report_file = os.path.join(output_dir, "comparison_report.txt")
    
    with open(report_file, 'w') as f:
        f.write("MIMIC Searcher Experiments Comparison Report\n")
        f.write("=" * 60 + "\n\n")
        
        for case_result in results:
            case_id = case_result['case_id']
            f.write(f"Case ID: {case_id}\n")
            f.write("-" * 40 + "\n")
            
            # Ground truth
            gt = case_result.get('ground_truth', {})
            f.write(f"Ground Truth: {gt.get('mortality_label', 'Unknown')}\n\n")
            
            # Naive RAG
            naive = case_result['naive_rag']
            f.write("1. Naive RAG:\n")
            f.write(f"   Query length: {len(naive['final_query'])} chars\n")
            f.write(f"   Documents retrieved: {len(naive['documents'])}\n")
            if naive['documents']:
                f.write(f"   Top document: {naive['documents'][0]['title'][:100]}...\n")
            f.write("\n")
            
            # Searcher-optimized
            searcher = case_result['searcher_optimized']
            f.write("2. Searcher-optimized RAG:\n")
            f.write(f"   Original query length: {len(searcher['original_query'])} chars\n")
            f.write(f"   Optimized query length: {len(searcher['final_query'])} chars\n")
            f.write(f"   Documents retrieved: {len(searcher['documents'])}\n")
            if searcher['documents']:
                f.write(f"   Top document: {searcher['documents'][0]['title'][:100]}...\n")
            f.write("\n")
            
            # Searcher with similar cases
            if case_result['searcher_with_similar']:
                similar = case_result['searcher_with_similar']
                f.write("3. Searcher with similar cases RAG:\n")
                f.write(f"   Original query length: {len(similar['original_query'])} chars\n")
                f.write(f"   Optimized query length: {len(similar['final_query'])} chars\n")
                f.write(f"   Documents retrieved: {len(similar['documents'])}\n")
                if similar['documents']:
                    f.write(f"   Top document: {similar['documents'][0]['title'][:100]}...\n")
            else:
                f.write("3. Searcher with similar cases RAG: Not performed (no similar case)\n")
            
            f.write("\n" + "=" * 60 + "\n\n")
    
    print(f"Comparison report saved to: {report_file}")


if __name__ == "__main__":
    main()