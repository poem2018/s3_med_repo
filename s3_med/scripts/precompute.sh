# precompute the naïve RAG Cache for training

# preconstruct dataset without RAG Retrieval
bash scripts/dataset_construct/data_process_s3_pre.sh 

# prepare Naïve RAG Retrieval for Training and Test Set
bash scripts/baselines/run_retrieval.sh 

# run Generator with RAG Retrieval and save the RAG Cache
bash scripts/evaluation/run_rag_cache.sh

# construct dataset with RAG Retrieval
bash scripts/dataset_construct/data_process_s3.sh