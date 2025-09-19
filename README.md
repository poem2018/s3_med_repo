# virtual_path_scalling

# Medical Reasoning Results - Qwen2.5-3B-Instruct (logprobs score)

| Method | Prompt Type | Additional Parameters | Num Patients | Num Failed | Accuracy | Precision | Recall | F1 | AUC | AUPRC |
|--------|-------------|----------------------|--------------|------------|----------|-----------|--------|----|----- |-------|
| direct input | cot | - | 300 | 6 | 0.18 | 0.0996 | 0.9310 | 0.18 | 0.5187 | 0.1001 |
| with rag | cot | topk_docs: 5 | 300 | 0 | 0.1833 | 0.1029 | 0.9655 | 0.1860 | 0.5340 | 0.1034 |
| similar_cases + rag | cot | k_similar: 3, topk_docs: 3 | 300 | 0 | 0.18 | 0.0967 | 0.8966 | 0.1745 | 0.5709 | 0.1135 |
| similar_cases | cot | k_similar: 3 | 300 | 0 | 0.63 | 0.1404 | 0.5517 | 0.2238 | 0.6288 | 0.1591 |

# running command
bash scripts/baseline_s3_med/host_llm_3b.sh

bash scripts/deploy_retriever/retrieval_launch.sh

bash scripts/baseline_s3_med/run_all_baselines.sh
