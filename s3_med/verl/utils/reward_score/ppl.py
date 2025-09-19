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
from generator_llms.local import compute_retrieval_utility_score, generate_answer
from pyserini.eval.evaluate_dpr_retrieval import has_answers, SimpleTokenizer

_tokenizer = SimpleTokenizer()

def normalize_answer(s):
    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


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


def compute_score_ppl(solution_str, ground_truth, retrieval=True, generation=True):
    """
    compute perplexity score, and return as the reward
    """
    highest_ppl_score = 0
    
    question = ground_truth['question']
    golden_answers = ground_truth['target'].tolist()

    searched_texts = extract_texts(solution_str=solution_str)
    titles = extract_titles(solution_str=solution_str)
    
    context_with_info = ""
    for i, text in enumerate(searched_texts):
        context_with_info += f"Doc {i+1} (Title: {titles[i]})\n{text}\n\n"
    
    for answer in golden_answers:
        retrieval_score = compute_retrieval_utility_score(
            prompt=question,
            answer=answer,
            context=context_with_info,
            num_runs=6
        )
        if retrieval_score > highest_ppl_score:
            highest_ppl_score = retrieval_score
            
        do_print = random.randint(1, 16) == 1
        
    generation_score = 0
    generated_answer = None
    if generation:
        generated_answer = generate_answer(
            prompt=question,
            context=context_with_info
        )
        
        if generated_answer is not None:
            generation_score = 2.1 if answer_span_check(generated_answer, ground_truth['target']) else 0
        
    if do_print:
        print(f"--------------------------------")
        print(f"Question: {question}")
        if generated_answer is not None:
            print(f"Generated answer: {generated_answer}")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Solution string: {solution_str}")
        
    retrieval_score = 0
    if highest_ppl_score >= 0.9:
        retrieval_score = 3
    if highest_ppl_score >= 0.8:
        retrieval_score = 2
    elif highest_ppl_score >= 0.6:
        retrieval_score = 1
    elif highest_ppl_score >= 0.5:
        retrieval_score = 0
    elif highest_ppl_score >= 0.3:
        retrieval_score = -1
    else:
        retrieval_score = -3
        
        
    return retrieval_score + generation_score
            
    # if highest_ppl_score >= 0.9:
    #     return 5
    # if highest_ppl_score >= 0.8:
    #     return 4
    # elif highest_ppl_score >= 0.6:
    #     return 3
    # elif highest_ppl_score >= 0.5:
    #     return 1
    # elif highest_ppl_score >= 0.4:
    #     return 0
    # elif highest_ppl_score >= 0.3:
    #     return -1
    # else:
    #     return -3
    
    