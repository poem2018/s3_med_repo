import json

file_path = '/scratch/bcew/ruikez2/intern/s3_med/data/demo_exp/demo_corpus.jsonl'

with open(file_path, 'r', encoding='utf-8') as f:
    first_line = f.readline()
    first_item = json.loads(first_line)
    print(json.dumps(first_item, indent=2, ensure_ascii=False))