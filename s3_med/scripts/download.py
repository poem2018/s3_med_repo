import argparse
from huggingface_hub import hf_hub_download

parser = argparse.ArgumentParser(description="Download files from a Hugging Face dataset repository.")
parser.add_argument("--repo_id", type=str, default="PeterJinGo/wiki-18-e5-index", help="Hugging Face repository ID")
parser.add_argument("--save_path", type=str, required=True, help="Local directory to save files")
    
args = parser.parse_args()

repo_id = "PeterJinGo/wiki-18-e5-index"
for file in ["part_aa", "part_ab"]:
    hf_hub_download(
        repo_id=repo_id,
        filename=file,  # e.g., "e5_Flat.index"
        repo_type="dataset",
        local_dir=args.save_path,
    )

repo_id = "PeterJinGo/wiki-18-corpus"
hf_hub_download(
        repo_id=repo_id,
        filename="wiki-18.jsonl.gz",
        repo_type="dataset",
        local_dir=args.save_path,
)


'''
import argparse
from huggingface_hub import hf_hub_download

parser = argparse.ArgumentParser(description="Download medical corpus files from Hugging Face dataset repository.")
parser.add_argument("--repo_id", type=str, default="medical-corpus/pubmed-e5-index", help="Hugging Face repository ID for medical index")
parser.add_argument("--corpus_repo_id", type=str, default="medical-corpus/pubmed-biomedical", help="Hugging Face repository ID for medical corpus")
parser.add_argument("--save_path", type=str, required=True, help="Local directory to save files")
parser.add_argument("--corpus_type", type=str, choices=["pubmed", "medical_books", "clinical_guidelines"], 
                   default="pubmed", help="Type of medical corpus to download")
    
args = parser.parse_args()

# Download medical retrieval index (E5 embeddings for medical corpus)
print(f"Downloading medical retrieval index from {args.repo_id}...")
index_repo_id = args.repo_id
for file in ["medical_part_aa", "medical_part_ab"]:  # Changed from wiki parts
    hf_hub_download(
        repo_id=index_repo_id,
        filename=file,
        repo_type="dataset",
        local_dir=args.save_path,
    )

# Download medical corpus based on type
print(f"Downloading {args.corpus_type} corpus from {args.corpus_repo_id}...")
corpus_repo_id = args.corpus_repo_id

if args.corpus_type == "pubmed":
    # PubMed abstracts and papers
    corpus_filename = "pubmed-medical.jsonl.gz"
elif args.corpus_type == "medical_books": 
    # Medical textbooks and reference materials
    corpus_filename = "medical-books.jsonl.gz"
elif args.corpus_type == "clinical_guidelines":
    # Clinical practice guidelines and protocols
    corpus_filename = "clinical-guidelines.jsonl.gz"
else:
    corpus_filename = "pubmed-medical.jsonl.gz"  # default fallback

hf_hub_download(
    repo_id=corpus_repo_id,
    filename=corpus_filename,
    repo_type="dataset",
    local_dir=args.save_path,
)

print("Medical corpus download completed!")
print(f"Files saved to: {args.save_path}")
print(f"Index files: medical_part_aa, medical_part_ab")
print(f"Corpus file: {corpus_filename}")
'''