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
from generator_llms.local import *

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


def answer_span_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    normalized_golden_answers = [normalize_answer(golden_answer) for golden_answer in golden_answers]
    if has_answers(normalized_prediction, normalized_golden_answers, _tokenizer, regex=False):
        score = 1
    return score


def extract_titles(solution_str):
    # Find all sections between <information> and </information>
    info_sections = re.findall(r'<information>(.*?)</information>', solution_str, re.DOTALL)

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


def compute_score_rag(solution_str, ground_truth, zeroshot_answers):
    """
    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        zeroshot_answers: dictionary containing cached zeroshot answers
    """
    
    utility_score = 0
    generation_score = 0
    
    question = ground_truth['question']
    golden_answers = ground_truth['target'].tolist()
    
    searched_texts = extract_texts(solution_str=solution_str)
    titles = extract_titles(solution_str=solution_str)
        
    context_with_info = ""
    for i, text in enumerate(searched_texts):
        context_with_info += f"Doc {i+1} (Title: {titles[i]})\n{text}\n\n"
        
    answer_context = generate_answer(prompt=question, context=context_with_info)
    answer_context_score = check_answer_correct(answer=answer_context, golden_answers=golden_answers)
        
    if question in zeroshot_answers:
        answer_zeroshot = zeroshot_answers[question]['answer']
        answer_zeroshot_score = zeroshot_answers[question]['score']
    else:
        answer_zeroshot = generate_answer_zero_shot(prompt=question)
        answer_zeroshot_score = check_answer_correct(answer=answer_zeroshot, golden_answers=golden_answers)
        
    utility_score = answer_context_score - answer_zeroshot_score
    generation_score = answer_context_score
    
    score = utility_score + generation_score
    
    do_print = random.randint(1, 16) == 1
        
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer_context}")
        print(f"Extracted zeroshot answer: {answer_zeroshot}")
        print(f"Answer context score: {answer_context_score}")
        print(f"Answer zeroshot score: {answer_zeroshot_score}")
        print(f"Utility score: {utility_score}")
        print(f"Generation score: {generation_score}")
        print(f"Extracted doc_info: {context_with_info}")
        print(f"Solution string: {solution_str}")
    
    return score, answer_zeroshot, answer_zeroshot_score
    