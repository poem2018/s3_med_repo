
corpus_file=/scratch/bcew/ruikez2/intern/s3_med/data/medrag/pubmed/pubmed_all_formatted_with_contents.jsonl # jsonl
save_dir=/scratch/bcew/ruikez2/intern/s3_med/data/medrag/pubmed/4gpu_indexes
retriever_name=e5 # this is for indexing naming
retriever_model=intfloat/e5-base-v2

# change faiss_type to HNSW32/64/128 for ANN indexing
# change retriever_name to bm25 for BM25 indexing
CUDA_VISIBLE_DEVICES=0,1,2,3 python /scratch/bcew/ruikez2/intern/s3_med/s3/search/index_builder.py \
    --retrieval_method $retriever_name \
    --model_path $retriever_model \
    --corpus_path $corpus_file \
    --save_dir $save_dir \
    --use_fp16 \
    --max_length 256 \
    --batch_size 128 \
    --pooling_method mean \
    --faiss_type Flat \
    --save_embedding

# corpus_file=/shared/eng/pj20/s3_medcorp/medcorpus.jsonl # jsonl
# save_dir=/shared/eng/pj20/s3_medcorp/bm25
# retriever_name=bm25 # this is for indexing naming
# retriever_model=bm25

# # change faiss_type to HNSW32/64/128 for ANN indexing
# # change retriever_name to bm25 for BM25 indexing
# CUDA_VISIBLE_DEVICES=0,1 python search_c1/search/index_builder.py \
#     --retrieval_method $retriever_name \
#     --model_path $retriever_model \
#     --corpus_path $corpus_file \
#     --save_dir $save_dir \
#     --use_fp16 \
#     --max_length 256 \
#     --batch_size 1024 \
#     --pooling_method mean \
#     --faiss_type Flat \
#     --save_embedding
