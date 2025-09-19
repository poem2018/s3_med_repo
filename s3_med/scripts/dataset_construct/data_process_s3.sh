LOCAL_DIR=data/nq_hotpotqa_train

pwd

## process multiple dataset search format train file
DATA=nq,hotpotqa
python scripts/data_process/train_s3.py --local_dir $LOCAL_DIR --data_sources $DATA --retriever e5

## process multiple dataset search format test file
DATA=nq,triviaqa,popqa,hotpotqa,2wikimultihopqa,musique,bamboogle
python scripts/data_process/test_s3.py --local_dir $LOCAL_DIR --data_sources $DATA --retriever e5

## For a more efficient evaluation, we sample a subset (max 3000 samples per data_source) of the test set
DATA=nq,triviaqa,popqa,hotpotqa,2wikimultihopqa,musique,bamboogle
python scripts/data_process/test_s3_sampled.py --local_dir $LOCAL_DIR --data_sources $DATA --retriever e5
