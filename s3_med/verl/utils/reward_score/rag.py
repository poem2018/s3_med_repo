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

_tokenizer = SimpleTokenizer()

def validate_response_structure(processed_str: str, do_print: bool) -> bool:
    """Performs comprehensive validation of response structure.
    
    Args:
        processed_str: Processed response string from the model
        
    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    if do_print:
        print("\n[Structure Validation]")
    validation_passed = True

    # Check required tags
    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1),
        'answer_start': ('<answer>', 1),
        'answer_end': ('</answer>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        positions[tag_name] = processed_str.find(tag_str)

    # Verify tag order
    if (positions['think_start'] > positions['think_end'] or
        positions['think_end'] > positions['answer_start'] or
        positions['answer_start'] > positions['answer_end']):
        if do_print:
            print("  [Error] Incorrect tag order: Expected <think>...</think><answer>...</answer>")
        validation_passed = False
    else:
        if do_print:
            print("  Tag sequence validation passed")
    
    return validation_passed

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

def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score

def extract_titles(text):
    # Find all sections between <information> and </information>
    info_sections = re.findall(r'<information>(.*?)</information>', text, re.DOTALL)

    titles = []
    for section in info_sections:
        # In each section, find all titles using the pattern
        found_titles = re.findall(r'Doc\s+\d+\(Title:\s*(.*?)\)', section)
        titles.extend(found_titles)
    
    return titles

def extract_texts(solution_str):
    # Find all sections between <information> and </information>
    info_sections = re.findall(r'<information>(.*?)</information>', solution_str, re.DOTALL)

    texts = []
    for section in info_sections:
        # Extract the {text} part after each Doc (Title: xxx)
        found_texts = re.findall(r'Doc\s+\d+\(Title:\s*.*?\)\s*(.*)', section)
        texts.extend(found_texts)
    
    return texts

def extract_answer(solution_str):
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    # If there are 0 match, return None
    if len(matches) < 1:
        return None
    # If there are 1 or more matches, return the last one
    return matches[-1].group(1).strip()

def extract_zeroshot_answer(solution_str):
    answer_pattern = r'<zeroshot_answer>(.*?)</zeroshot_answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    return matches[-1].group(1).strip() if len(matches) > 0 else None


def compute_score_rag(solution_str, ground_truth, format_score=0.):
    """
    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    retrieval_score = 0
    answer_score = 0
    zeroshot_answer_score = 0
    
    # retrieval_check = "recall" if ground_truth["gt_docs"] != [] else "span"
    retrieval_check = "span"
    
    answer = extract_answer(solution_str=solution_str)
    zeroshot_answer = extract_zeroshot_answer(solution_str=solution_str)
    
    if answer is not None:
        answer_score = 2 if answer_span_check(answer, ground_truth['target']) else 0
    
    if zeroshot_answer is not None:
        zeroshot_answer_score = 1 if answer_span_check(zeroshot_answer, ground_truth['target']) else 0
    
    label = ground_truth['target'].tolist()
    
    if retrieval_check == "recall":
        # Extract titles from solution
        searched_titles = extract_titles(solution_str=solution_str)
        doc_info = searched_titles
        gt_titles = ground_truth["gt_docs"]["title"]

        # Normalize both predicted docs and ground truth docs
        normalized_searched_titles = {normalize_answer(title) for title in searched_titles}
        normalized_gt_titles = {normalize_answer(title) for title in gt_titles}

        # Compute relaxed recall
        hit_titles = normalized_searched_titles & normalized_gt_titles
        recall = len(hit_titles) / len(normalized_gt_titles)
        retrieval_score = recall
        
    elif retrieval_check == "span":
        searched_texts = extract_texts(solution_str=solution_str)
        doc_info = searched_texts
        for i in range(len(searched_texts)):
            assert isinstance(label, list)
            if has_answers(searched_texts[i], label, _tokenizer, regex=False):
                retrieval_score = 1
                break
        
    do_print = random.randint(1, 16) == 1
        
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Extracted zeroshot answer: {zeroshot_answer}")
        print(f"Extracted doc_info: {doc_info}")
        print(f"Solution string: {solution_str}")
    
    
    # # if both answer and zeroshot answer are correct, the correctness is not coming from retrieval, but it is good, thus we return weak combined score
    # if answer_score > 0 and zeroshot_answer_score > 0:
    #     return retrieval_score + answer_score / 2
    
    # # if answer is correct but zeroshot answer is incorrect, the correctness is coming from retrieval, thus we return combined score
    # if answer_score > 0 and zeroshot_answer_score == 0:
    #     return retrieval_score + answer_score
    
    # # if answer is incorrect but zeroshot answer is correct, the wrongness is coming from retrieval, thus we return negative score
    # if answer_score == 0 and zeroshot_answer_score > 0:
    #     return -1
    
    # # if both are incorrect, we return 0, we don't know how retrieval contributes to the wrongness, thus we return retrieval score
    # if answer_score == 0 and zeroshot_answer_score == 0:
    #     return retrieval_score
    
    return answer_score
    