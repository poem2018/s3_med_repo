export CUDA_VISIBLE_DEVICES=0,1,2,3
# Updated to use PubMed corpus instead of Wikipedia
# index_file=/scratch/bcew/ruikez2/intern/s3_med/data/medrag/pubmed/indexes/e5_Flat.index
# corpus_file=/scratch/bcew/ruikez2/intern/s3_med/data/medrag/pubmed/pubmed_all_formatted.jsonl

index_file=/scratch/bcew/ruikez2/intern/s3_med/data/demo_exp/indexes/e5_Flat.index
corpus_file=/scratch/bcew/ruikez2/intern/s3_med/data/demo_exp/demo_corpus.jsonl
retriever_name=e5
retriever_path=intfloat/e5-base-v2

python s3/search/retrieval_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 4 \
                                            --retriever_name $retriever_name \
                                            --retriever_model $retriever_path \
                                            --faiss_gpu  \
                                            --port 3000
                                            # --port 7000
