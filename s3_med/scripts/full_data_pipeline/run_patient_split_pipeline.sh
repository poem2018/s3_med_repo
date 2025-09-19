#!/bin/bash
# 运行基于患者分割的数据处理pipeline
# 用于生成baseline实验所需的数据

set -e  # Exit on error

echo "========================================="
echo "Patient-based Split Data Pipeline"
echo "========================================="
echo "使用基于患者ID的数据分割（同一患者的所有ICU stays在同一个split中）"
echo ""

# 配置
NUM_TEST_SAMPLES=4500  # 处理前4500个测试样本
OUTPUT_DIR="/scratch/bcew/ruikez2/intern/s3_med/data/baseline_text_patient_split"

# Step 1: 创建或验证基于患者的固定分割
echo "[Step 1] 创建/验证基于患者的数据分割..."
python 00_create_fixed_splits.py
echo "✓ Patient-based splits ready"
echo ""

# Step 2: 处理测试数据子集
echo "[Step 2] 处理前${NUM_TEST_SAMPLES}个测试样本..."
python 02_process_subset_data.py test ${NUM_TEST_SAMPLES}
echo "✓ Test subset processed"
echo ""

# Step 3: 运行benchmark工具提取数据
echo "[Step 3] 使用benchmark工具提取数据..."
SUBSET_DIR="/scratch/bcew/ruikez2/intern/s3_med/data/baseline_text/subset_test"
BENCHMARK_DIR="/scratch/bcew/ruikez2/intern/s3_med/data/baseline_text/benchmark_test_subset"

cd /scratch/bcew/ruikez2/intern/mimic-iv-benchmarks
python -m mimic4benchmark.scripts.extract_subjects \
     ${SUBSET_DIR} \
     ${BENCHMARK_DIR}/

python -m mimic4benchmark.scripts.extract_episodes_from_subjects \
     ${BENCHMARK_DIR}/
echo "✓ Benchmark extraction complete"
echo ""

# Step 4: 转换为JSON格式
cd /scratch/bcew/ruikez2/intern/s3_med/scripts/full_data_pipeline
echo "[Step 4] 转换为JSON格式..."
python 03_convert_subset_to_json.py ${NUM_TEST_SAMPLES}
echo "✓ JSON conversion complete"
echo ""

# Step 5: 添加时间信息
echo "[Step 5] 添加时间信息..."
export TEST_MODE=true
python 04_add_temporal_info_full.py
echo "✓ Temporal information added"
echo ""

# Step 6: 转换为文本格式（用于baseline）
echo "[Step 6] 转换为文本格式..."
python 06_convert_json_to_text_simple.py
echo "✓ Text conversion complete"
echo ""

# 创建输出目录并复制结果
echo "[Step 7] 整理输出文件..."
mkdir -p ${OUTPUT_DIR}
cp /scratch/bcew/ruikez2/intern/s3_med/data/baseline_text/*.json ${OUTPUT_DIR}/ 2>/dev/null || true
cp -r /scratch/bcew/ruikez2/intern/s3_med/data/baseline_text/patient_text_files ${OUTPUT_DIR}/ 2>/dev/null || true
echo "✓ Output files organized in ${OUTPUT_DIR}"
echo ""

echo "========================================="
echo "Pipeline完成！"
echo "========================================="
echo "输出位置: ${OUTPUT_DIR}"
echo ""
echo "现在可以运行baseline实验："
echo "  cd /scratch/bcew/ruikez2/intern/s3_med/scripts/baseline_s3_med"
echo "  ./run_all_baselines.sh"
echo ""