# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import string
import random
from pyserini.eval.evaluate_dpr_retrieval import has_answers, SimpleTokenizer
from generator_llms.local_inst import *

_tokenizer = SimpleTokenizer()


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def answer_span_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    normalized_golden_answers = [normalize_answer(golden_answer) for golden_answer in golden_answers]
    if has_answers(normalized_prediction, normalized_golden_answers, _tokenizer, regex=False):
        score = 1
    return score


def extract_titles_and_texts(solution_str):
    """
    Extract titles and texts from information blocks, handling important documents
    and preserving their order. Properly handles duplicated documents.
    
    <important_info> tags only apply to the most recent <information> block before them,
    even if there are other tags or content in between them.
    
    If an <information> block doesn't have a corresponding <important_info> tag,
    ALL documents from that block are included.
    
    Returns a list of (title, text) tuples for each document.
    """
    try:
        # Extract all information blocks and important_info tags with their positions
        info_blocks = []
        important_infos = []
        
        # Find all information blocks with their positions
        info_pattern = re.compile(r'<information>(.*?)</information>', re.DOTALL)
        for match in info_pattern.finditer(solution_str):
            info_blocks.append({
                'position': match.start(),
                'content': match.group(1),
                'important_ids': None,  # Will be filled if there's a matching important_info
                'processed': False  # Track if this block has been processed
            })
        
        # Find all important_info tags with their positions
        important_pattern = re.compile(r'<important_info>(.*?)</important_info>', re.DOTALL)
        for match in important_pattern.finditer(solution_str):
            # Parse important doc IDs
            important_ids = []
            try:
                # Extract the content and clean it
                content = match.group(1).strip()
                
                # Extract all numbers from the content
                numbers = re.findall(r'\d+', content)
                # Only keep IDs 1, 2, and 3
                important_ids = [int(num) for num in numbers if int(num) in [1, 2, 3]]
                
                # Remove duplicates while preserving order
                important_ids = list(dict.fromkeys(important_ids))
                
            except Exception as e:
                print(f"Warning: Error parsing important document IDs: {str(e)}")
                important_ids = []
            
            important_infos.append({
                'position': match.start(),
                'important_ids': important_ids
            })
        
        # Process each important_info tag and associate it with the closest unprocessed information block
        all_docs = []
        seen_docs = set()  # To track unique documents
        
        # First process all information blocks that have important_info tags
        for imp_info in important_infos:
            # Find the closest unprocessed information block that appears before this important_info
            closest_info = None
            min_distance = float('inf')
            
            for info_block in info_blocks:
                if not info_block['processed'] and info_block['position'] < imp_info['position']:
                    # Only consider information blocks that are before this important_info
                    # and don't have any other information blocks between them
                    has_info_between = False
                    for other_info in info_blocks:
                        if (other_info['position'] > info_block['position'] and 
                            other_info['position'] < imp_info['position']):
                            has_info_between = True
                            break
                    
                    if not has_info_between:
                        distance = imp_info['position'] - info_block['position']
                        if distance < min_distance:
                            min_distance = distance
                            closest_info = info_block
            
            # If we found a matching information block, process it
            if closest_info:
                closest_info['processed'] = True  # Mark as processed
                info_content = closest_info['content']
                important_ids = imp_info['important_ids']
                
                docs_in_block = []
                try:
                    # Extract individual documents from the info block
                    doc_pattern = re.compile(r'Doc\s*(\d+)\s*\(Title:\s*"?([^")]+)"?\)\s*(.*?)(?=Doc\s*\d+\s*\(Title:|$)', re.DOTALL)
                    
                    for match in doc_pattern.finditer(info_content):
                        try:
                            doc_id = int(match.group(1))
                            title = match.group(2).strip()
                            text = match.group(3).strip()
                            docs_in_block.append((doc_id, title, text))
                        except (IndexError, ValueError) as e:
                            print(f"Warning: Error parsing document: {str(e)}")
                            continue
                except Exception as e:
                    print(f"Warning: Error extracting documents from info block: {str(e)}")
                    continue
                
                # Filter by important_ids
                try:
                    filtered_docs = [(title, text) for doc_id, title, text in docs_in_block 
                                    if doc_id in important_ids]
                except Exception as e:
                    print(f"Warning: Error filtering documents: {str(e)}")
                    filtered_docs = []
                
                # Add unique documents to the result
                for title, text in filtered_docs:
                    if text not in seen_docs:
                        seen_docs.add(text)
                        all_docs.append((title, text))
        
        # Then process all remaining unprocessed information blocks (those without important_info tags)
        for info_block in info_blocks:
            if not info_block['processed']:
                info_content = info_block['content']
                try:
                    # Extract individual documents from the info block
                    doc_pattern = re.compile(r'Doc\s*(\d+)\s*\(Title:\s*"?([^")]+)"?\)\s*(.*?)(?=Doc\s*\d+\s*\(Title:|$)', re.DOTALL)
                    
                    for match in doc_pattern.finditer(info_content):
                        try:
                            doc_id = int(match.group(1))
                            title = match.group(2).strip()
                            text = match.group(3).strip()
                            
                            # Add all documents from unprocessed blocks
                            if text not in seen_docs:
                                seen_docs.add(text)
                                all_docs.append((title, text))
                        except (IndexError, ValueError) as e:
                            print(f"Warning: Error parsing document: {str(e)}")
                            continue
                except Exception as e:
                    print(f"Warning: Error extracting documents from info block: {str(e)}")
                    continue
        
        return all_docs
        
    except Exception as e:
        print(f"Warning: Error in extract_titles_and_texts: {str(e)}")
        return []  # Return empty list if any unexpected error occurs


def check_answer_correct(answer, golden_answers):
    answer_context_score = answer_span_check(
        prediction=answer,
        golden_answers=golden_answers
    )
    if answer_context_score == 0:
        answer_context_score = 1 if check_if_response_is_correct_llm(
            response=answer,
            gold_answers=golden_answers
        ) else 0
    return answer_context_score


def check_answer_span_in_context(context, golden_answers):
    answer_context_score = answer_span_check(
        prediction=context,
        golden_answers=golden_answers
    )
    if answer_context_score == 0:
        answer_context_score = 1 if check_if_context_contains_golden_answers(
            context=context,
            gold_answers=golden_answers
        ) else 0
    return answer_context_score


def compute_score_rag(solution_str, ground_truth, zeroshot_answers, data_source, use_utility_score=True, use_generation_score=True):
    """
    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        zeroshot_answers: dictionary containing cached zeroshot answers
    """
    
    
    question = ground_truth['question']
    golden_answers = ground_truth['target'].tolist()
    
    # Get documents with titles, handling important documents
    response_str = solution_str.split("Now, start the loop with the following question and initial searched results:")[1]
    docs = extract_titles_and_texts(solution_str=response_str)
    
    # Build context with unique documents
    seen_docs = set()
    doc_id = 1
    context_with_info = ""
    for title, text in docs:
        doc_key = (title, text)
        if doc_key not in seen_docs:
            seen_docs.add(doc_key)
            context_with_info += f"Doc {doc_id} (Title: {title})\n{text}\n\n"
            doc_id += 1
    
    
    retrieval_score = check_answer_span_in_context(
        context=context_with_info,
        golden_answers=golden_answers
    )
    
    do_print = random.randint(1, 16) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Question: {question}")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted doc_info: {context_with_info}")
        print(f"Retrieval score: {retrieval_score}")
        print(f"Solution string: {solution_str}")
        
    return retrieval_score, None, None



def output_sequence(solution_str, ground_truth):
    """
    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        zeroshot_answers: dictionary containing cached zeroshot answers
    """
    
    
    question = ground_truth['question']
    golden_answers = ground_truth['target'].tolist()
    
    # Get documents with titles, handling important documents
    response_str = solution_str.split("Now, start the loop with the following question:")[1]
    docs = extract_titles_and_texts(solution_str=response_str)
    
    # Build context with unique documents
    seen_docs = set()
    doc_id = 1
    context_with_info = ""
    for title, text in docs:
        doc_key = (title, text)
        if doc_key not in seen_docs:
            seen_docs.add(doc_key)
            context_with_info += f"Doc {doc_id} (Title: {title})\n{text}\n\n"
            doc_id += 1
        
    do_print = random.randint(1, 16) == 1
        
    if do_print:
        print(f"--------------------------------")
        print(f"Question: {question}")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted doc_info: {context_with_info}")
        print(f"Solution string: {solution_str}")
    
    return question, golden_answers, context_with_info, response_str
    