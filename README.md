# virtual_path_scalling

# Medical Reasoning Results - Qwen2.5-3B-Instruct (logprobs score)


| Model | Acc | Prec | Rec | F1 | AUC | AUPRC |
|-------|-----|------|-----|----|----- |-------|
| Baseline 1: Random Forest | 0.907 | 0.545 | 0.316 | 0.400 | 0.907 | 0.456 |
| Baseline 1: Logistic Regression | 0.825 | 0.317 | 0.684 | 0.433 | 0.874 | 0.514 |
| Baseline 2: Direct Input + CoT + Logprobs | 0.180 | 0.100 | 0.931 | 0.180 | 0.519 | 0.100 |
| Baseline 3: RAG + CoT + Logprobs | 0.183 | 0.103 | 0.966 | 0.186 | 0.534 | 0.103 |
| Baseline 4: Similar Cases + RAG + CoT | 0.180 | 0.097 | 0.897 | 0.175 | 0.571 | 0.114 |
| Baseline 5: Similar Cases + CoT | 0.627 | 0.140 | 0.552 | 0.223 | 0.608 | 0.139 |

# running command
bash scripts/baseline_s3_med/host_llm_3b.sh

bash scripts/deploy_retriever/retrieval_launch.sh

bash scripts/baseline_s3_med/run_all_baselines.sh
