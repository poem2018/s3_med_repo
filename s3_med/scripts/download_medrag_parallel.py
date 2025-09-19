import argparse
from huggingface_hub import HfApi, hf_hub_download
import os
import json
import gzip
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

parser = argparse.ArgumentParser(description="Download MedRAG PubMed corpus with parallel downloads.")
parser.add_argument("--save_path", type=str, required=True, help="Local directory to save files")
parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel download workers")
parser.add_argument("--chunk_start", type=int, default=0, help="Starting chunk index")
parser.add_argument("--chunk_end", type=int, default=-1, help="Ending chunk index (-1 for all)")
    
args = parser.parse_args()

# Create save directory
corpus_dir = os.path.join(args.save_path, "pubmed")
os.makedirs(corpus_dir, exist_ok=True)

api = HfApi()
repo_id = "MedRAG/pubmed"

print(f"Fetching file list from {repo_id}...")

# List all files
files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
chunk_files = sorted([f for f in files if f.startswith("chunk/") and f.endswith(".jsonl")])

print(f"Found {len(chunk_files)} chunk files total")

# Determine range to download
start_idx = args.chunk_start
end_idx = args.chunk_end if args.chunk_end > 0 else len(chunk_files)
chunk_files_to_download = chunk_files[start_idx:end_idx]

print(f"Will download chunks {start_idx} to {end_idx-1} ({len(chunk_files_to_download)} files)")

def download_and_process_chunk(chunk_file, chunk_idx):
    """Download and process a single chunk file."""
    try:
        # Download chunk
        local_file = hf_hub_download(
            repo_id=repo_id,
            filename=chunk_file,
            repo_type="dataset",
            local_dir=corpus_dir,
        )
        
        # Process and convert to s3 format
        output_file = os.path.join(corpus_dir, f"pubmed_chunk_{chunk_idx:04d}.jsonl.gz")
        
        doc_count = 0
        with gzip.open(output_file, 'wt', encoding='utf-8') as out_f:
            with open(local_file, 'r', encoding='utf-8') as in_f:
                for line in in_f:
                    try:
                        data = json.loads(line.strip())
                        
                        # Extract content
                        title = data.get('title', '')
                        contents = data.get('contents', '')
                        text = f'"{title}"\n{contents}' if title else contents
                        
                        if not text:
                            continue
                        
                        # Global ID includes chunk number
                        entry = {
                            "id": f"{chunk_idx}_{doc_count}",
                            "contents": text
                        }
                        
                        out_f.write(json.dumps(entry) + '\n')
                        doc_count += 1
                        
                    except json.JSONDecodeError:
                        continue
        
        # Remove original chunk file to save space
        os.remove(local_file)
        
        return chunk_idx, doc_count, None
        
    except Exception as e:
        return chunk_idx, 0, str(e)

# Download with parallel workers
print(f"\nStarting parallel download with {args.num_workers} workers...")

successful_chunks = 0
total_docs = 0
failed_chunks = []

with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
    # Submit all download tasks
    future_to_chunk = {
        executor.submit(download_and_process_chunk, chunk_file, start_idx + i): (chunk_file, start_idx + i)
        for i, chunk_file in enumerate(chunk_files_to_download)
    }
    
    # Process completed downloads
    with tqdm(total=len(chunk_files_to_download), desc="Downloading chunks") as pbar:
        for future in as_completed(future_to_chunk):
            chunk_file, chunk_idx = future_to_chunk[future]
            
            try:
                idx, doc_count, error = future.result()
                
                if error:
                    print(f"\nError processing chunk {idx}: {error}")
                    failed_chunks.append(idx)
                else:
                    successful_chunks += 1
                    total_docs += doc_count
                    
            except Exception as e:
                print(f"\nUnexpected error with chunk {chunk_idx}: {e}")
                failed_chunks.append(chunk_idx)
            
            pbar.update(1)

print(f"\n{'='*50}")
print(f"Download completed!")
print(f"Successfully downloaded: {successful_chunks}/{len(chunk_files_to_download)} chunks")
print(f"Total documents: {total_docs}")
if failed_chunks:
    print(f"Failed chunks: {failed_chunks}")

print(f"\nNext steps:")
print(f"1. Combine all chunks into single file:")
print(f"   zcat {corpus_dir}/pubmed_chunk_*.jsonl.gz > {corpus_dir}/pubmed_all.jsonl")
print(f"2. Compress combined file:")
print(f"   gzip {corpus_dir}/pubmed_all.jsonl")
print(f"3. Build E5 index:")
print(f"   python s3/search/index_builder.py --retrieval_method e5 --model_path intfloat/e5-base-v2 --corpus_path {corpus_dir}/pubmed_all.jsonl --save_dir {corpus_dir}/indexes")