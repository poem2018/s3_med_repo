#!/bin/bash
# Host script for Qwen2.5-3B-Instruct for baseline experiments

# Adjust GPU usage based on your available GPUs
export CUDA_VISIBLE_DEVICES=0

python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-3B-Instruct \
    --port 8000 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9