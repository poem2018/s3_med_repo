#!/bin/bash
# Compare retrieval quality for three MIMIC mortality experiments
# Only focusing on retrieval results, not final answer generation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DISABLE_FLASH_ATTN=1
export TRANSFORMERS_NO_FLASH_ATTN=1
export RAY_DISABLE_IMPORT_WARNING=1
# Settings
DATA_DIR="/scratch/bcew/ruikez2/intern/s3_med/data/demo_exp"
OUTPUT_DIR="/scratch/bcew/ruikez2/intern/s3_med/output/demo_exp/retrieval"
RETRIEVAL_ENDPOINT="http://127.0.0.1:3000/retrieve"

RANDOM_SEED=${1:-42}

# Create output directory
mkdir -p $OUTPUT_DIR

echo "================================"
echo "Retrieval Quality Comparison"
echo "================================"

# ===============================
# Experiment 1: Naive RAG (no similar cases)
# Using e5_retrieval_mimic.py
# ===============================
echo ""
echo "Experiment 1: Naive E5 Retrieval (no similar cases)"
echo "----------------------------------------"

python /scratch/bcew/ruikez2/intern/s3_med/scripts/baselines/e5_retrieval_mimic.py \
    --input_parquet $DATA_DIR/demo_mimic_mortality_simple.parquet \
    --output_dir $OUTPUT_DIR/exp1_naive_nosimilar \
    --endpoint $RETRIEVAL_ENDPOINT

echo "Saved to: $OUTPUT_DIR/exp1_naive_nosimilar/"

# ===============================
# Experiment 1b: Naive RAG (WITH similar cases)
# Using e5_retrieval_mimic.py
# ===============================
echo ""
echo "Experiment 1b: Naive E5 Retrieval (WITH similar cases)"
echo "----------------------------------------"

python /scratch/bcew/ruikez2/intern/s3_med/scripts/baselines/e5_retrieval_mimic.py \
    --input_parquet $DATA_DIR/demo_mimic_mortality_simple_with_similar.parquet \
    --output_dir $OUTPUT_DIR/exp1_naive_similar \
    --endpoint $RETRIEVAL_ENDPOINT

echo "Saved to: $OUTPUT_DIR/exp1_naive_similar/"

# ===============================
# Experiment 2: S3 One-turn Searcher (no similar cases)
# Need to run S3 inference with max_turns=1
# ===============================
echo ""
echo "Experiment 2: S3 Searcher 1-turn (no similar cases)"
echo "----------------------------------------"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_ATTENTION_BACKEND=XFORMERS

# Use Qwen model directly like in evaluate-8-3-3.sh
S3_MODEL_PATH="Qwen/Qwen2.5-3B-Instruct"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/demo_mimic_mortality_no_similar.parquet \
    data.val_files=$DATA_DIR/demo_mimic_mortality_no_similar.parquet \
    data.train_data_num=null \
    data.val_data_num=4 \
    data.train_batch_size=4 \
    data.val_batch_size=4 \
    data.max_prompt_length=20000 \
    data.max_response_length=500 \
    data.max_start_length=8000 \
    data.max_obs_length=800 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=gae \
    actor_rollout_ref.model.path=$S3_MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.95 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size=4 \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=4 \
    actor_rollout_ref.rollout.n_agent=1 \
    actor_rollout_ref.actor.state_masking=true \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.optim.lr_warmup_steps_ratio=0.05 \
    critic.model.path=$S3_MODEL_PATH \
    critic.model.enable_gradient_checkpointing=true \
    critic.ppo_micro_batch_size=4 \
    critic.ppo_mini_batch_size=4 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.no_think_rl=false \
    trainer.critic_warmup=0 \
    trainer.logger=[] \
    +trainer.val_only=true \
    +trainer.val_before_train=true \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    max_turns=1 \
    +data.random_seed=$RANDOM_SEED \
    +generator_llm="Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4" \
    +output_context_dir=$OUTPUT_DIR/exp2_searcher_nosimilar_${RANDOM_SEED} \
    retriever.url=$RETRIEVAL_ENDPOINT \
    retriever.topk=5 \
    2>&1 | tee $OUTPUT_DIR/exp2_searcher_nosimilar.log

# ===============================
# Experiment 3: S3 One-turn Searcher (with similar cases)
# ===============================
echo ""
echo "Experiment 3: S3 Searcher 1-turn (with similar cases)"
echo "----------------------------------------"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/demo_mimic_mortality.parquet \
    data.val_files=$DATA_DIR/demo_mimic_mortality.parquet \
    data.train_data_num=null \
    data.val_data_num=4 \
    data.train_batch_size=4 \
    data.val_batch_size=4 \
    data.max_prompt_length=20000 \
    data.max_response_length=700 \
    data.max_start_length=8000 \
    data.max_obs_length=800 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=gae \
    actor_rollout_ref.model.path=$S3_MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.95 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size=4 \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=4 \
    actor_rollout_ref.rollout.n_agent=1 \
    actor_rollout_ref.actor.state_masking=true \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.optim.lr_warmup_steps_ratio=0.05 \
    critic.model.path=$S3_MODEL_PATH \
    critic.model.enable_gradient_checkpointing=true \
    critic.ppo_micro_batch_size=4 \
    critic.ppo_mini_batch_size=4 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.no_think_rl=false \
    trainer.critic_warmup=0 \
    trainer.logger=[] \
    +trainer.val_only=true \
    +trainer.val_before_train=true \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    max_turns=1 \
    +data.random_seed=$RANDOM_SEED \
    +generator_llm="Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4" \
    +output_context_dir=$OUTPUT_DIR/exp3_searcher_similar_${RANDOM_SEED} \
    retriever.url=$RETRIEVAL_ENDPOINT \
    retriever.topk=5 \
    2>&1 | tee $OUTPUT_DIR/exp3_searcher_similar.log

# ===============================
# Analyze retrieval results
# ===============================
# echo ""
# echo "Analyzing retrieval quality..."
# python /scratch/bcew/ruikez2/intern/s3_med/scripts/analyze_retrieval_quality.py \
#     --output_dir $OUTPUT_DIR \
#     --data_dir $DATA_DIR

# echo ""
# echo "================================"
# echo "Retrieval comparison complete!"
# echo "Results saved to $OUTPUT_DIR"
# echo "================================"