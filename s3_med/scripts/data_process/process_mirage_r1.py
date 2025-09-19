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
"""
Process the Mirage dataset using the QA search test merge prompt format
"""

import json
import os
import datasets
import argparse
from verl.utils.hdfs_io import copy, makedirs


def make_prefix(dp, template_type):
    question = dp['question']

    if template_type == 'base':
        """This works for any base model"""
        prefix = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""
    else:
        raise NotImplementedError
    return prefix


def process_mirage_data(file_path: str, template_type: str) -> list:
    """Process Mirage benchmark.json into the required format."""
    with open(file_path, 'r') as f:
        benchmark_data = json.load(f)
    
    processed_data = []
    
    for data_source, questions in benchmark_data.items():
        for q_id, q_data in questions.items():
            # Get the question and options
            question = q_data['question']
            options = q_data['options']
            
            # Create the full question with options
            full_question = f"{question}\nOptions:\n"
            for opt_key, opt_text in options.items():
                full_question += f"{opt_key}: {opt_text}\n"
            
            # Get the correct answer text based on the answer key
            answer_key = q_data['answer']
            golden_answer = [f"{answer_key}: {options[answer_key]}"]
            
            # Create the data point in the required format
            data_point = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": make_prefix({"question": full_question}, template_type),
                }],
                "ability": "fact-reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {
                        "question": full_question,
                        "target": golden_answer,
                        "gt_docs": []  # Empty list as we don't have supporting facts
                    }
                },
                "extra_info": {
                    'split': 'test',
                    'index': q_id,
                }
            }
            
            processed_data.append(data_point)
    
    return processed_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/mirage')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, default='base')
    parser.add_argument('--input_file', default='data/mirage/benchmark.json')

    args = parser.parse_args()

    # Process the data
    processed_data = process_mirage_data(args.input_file, args.template_type)
    
    # Convert to dataset and save as parquet
    dataset = datasets.Dataset.from_list(processed_data)
    
    # Create local directory if it doesn't exist
    os.makedirs(args.local_dir, exist_ok=True)
    
    # Save to parquet
    dataset.to_parquet(os.path.join(args.local_dir, 'test_mirage_r1.parquet'))
    
    # Print statistics
    print(f"Total number of questions: {len(dataset)}")
    for data_source in dataset.unique('data_source'):
        count = len(dataset.filter(lambda x: x['data_source'] == data_source))
        print(f"{data_source}: {count}")

    # Copy to HDFS if specified
    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=args.local_dir, dst=args.hdfs_dir) 