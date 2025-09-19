import json
import tqdm

input_file = '/scratch/bcew/ruikez2/intern/s3_med/data/medrag/pubmed/pubmed_all_formatted.jsonl'
output_file = '/scratch/bcew/ruikez2/intern/s3_med/data/medrag/pubmed/pubmed_all_formatted_with_contents.jsonl'

with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
    for line in tqdm.tqdm(f_in, desc="Processing lines"):
        item = json.loads(line)
        item['contents'] = f"{item['title']} {item['text']}"
        f_out.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"Conversion complete. Output saved to: {output_file}")