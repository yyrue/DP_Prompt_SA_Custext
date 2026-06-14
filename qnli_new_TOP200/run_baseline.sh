#!/bin/bash

# Baseline 实验：用原始数据微调 BERT（不做隐私化替换）
# 用法：bash run_baseline.sh [GPU_ID]

set -o pipefail

# ============ 环境配置 ============
source activate custext 2>/dev/null || conda activate custext

DEVICE=${1:-0}
if [ "${DEVICE}" = "cpu" ]; then
    export CUDA_VISIBLE_DEVICES=""
    USE_CUDA_FLAG=""
    echo "使用 CPU 运行"
else
    export CUDA_VISIBLE_DEVICES=${DEVICE}
    USE_CUDA_FLAG="--use_cuda"
    echo "使用 GPU: ${CUDA_VISIBLE_DEVICES}"
fi

# ============ 实验参数 ============
SEEDS=(42 43 44 45 46)
DATASET="qnli"
MODEL_TYPE="/data/youyaru/SanText-main/bert-base-uncased"

export TRANSFORMERS_OFFLINE=1

RESULT_DIR="./experiment_results"
mkdir -p ${RESULT_DIR}

echo "========================================" | tee -a ${RESULT_DIR}/baseline_log.txt
echo "Baseline 实验开始: $(date)" | tee -a ${RESULT_DIR}/baseline_log.txt
echo "========================================" | tee -a ${RESULT_DIR}/baseline_log.txt

for seed in "${SEEDS[@]}"
do
    echo "  运行 baseline, seed=${seed}..." | tee -a ${RESULT_DIR}/baseline_log.txt

    python main_baseline.py \
        --dataset ${DATASET} \
        --model_type ${MODEL_TYPE} \
        ${USE_CUDA_FLAG} \
        --seed ${seed} \
        2>&1 | tee ${RESULT_DIR}/baseline_seed_${seed}.log

    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "  ✓ baseline, seed=${seed} 完成" | tee -a ${RESULT_DIR}/baseline_log.txt
    else
        echo "  ✗ baseline, seed=${seed} 失败" | tee -a ${RESULT_DIR}/baseline_log.txt
    fi
done

echo "========================================" | tee -a ${RESULT_DIR}/baseline_log.txt
echo "Baseline 实验完成: $(date)" | tee -a ${RESULT_DIR}/baseline_log.txt
echo "========================================" | tee -a ${RESULT_DIR}/baseline_log.txt

# 汇总结果
echo ""
echo "===== Baseline 结果汇总 ====="
for seed in "${SEEDS[@]}"
do
    echo "  seed=${seed}: $(grep 'Baseline test acc' ${RESULT_DIR}/baseline_seed_${seed}.log)"
done