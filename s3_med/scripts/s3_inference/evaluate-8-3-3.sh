export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DISABLE_FLASH_ATTN=1
export TRANSFORMERS_NO_FLASH_ATTN=1
export RAY_DISABLE_IMPORT_WARNING=1

data_name=RAG_retrieval/test

RANDOM_SEED=${1:-42}

export CUDA_VISIBLE_DEVICES=0,1,2,3
export DATA_DIR=data/${data_name}

export BASE_MODEL=""
export VLLM_ATTENTION_BACKEND=XFORMERS

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train_e5_s3.parquet  \
    data.val_files=$DATA_DIR/test_e5_s3.parquet \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=2 \
    data.val_batch_size=8 \
    data.max_prompt_length=2000 \
    data.max_response_length=150 \
    data.max_start_length=600 \
    data.max_obs_length=400 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=gae \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.95 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size=2 \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=4 \
    actor_rollout_ref.rollout.n_agent=1 \
    actor_rollout_ref.actor.state_masking=true \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.optim.lr_warmup_steps_ratio=0.05 \
    critic.model.path=Qwen/Qwen2.5-3B-Instruct \
    critic.model.enable_gradient_checkpointing=true \
    critic.ppo_micro_batch_size=2 \
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
    max_turns=3 \
    +data.random_seed=$RANDOM_SEED \
    +generator_llm="Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4" \
    +output_context_dir=data/output_sequences_s3_8_3_3_step_20_${RANDOM_SEED} \
    retriever.url="http://127.0.0.1:3000/retrieve" \
    retriever.topk=8 \
    2>&1 | tee evaluation_s3_8_3_3_step_20_${RANDOM_SEED}.log