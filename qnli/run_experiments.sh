#!/bin/bash

# 批量运行 QNLI 实验脚本
# 功能：对不同 eps 值，每个配置运行 5 次（随机种子 42-46）
# 结果自动记录到各自的日志文件中，最后汇总
#
# 用法：
#   bash run_experiments.sh [GPU_ID]
#   例如：bash run_experiments.sh 0       # 使用 GPU 0
#         bash run_experiments.sh 5       # 使用 GPU 5
#         bash run_experiments.sh cpu     # 使用 CPU 运行

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
EPS_VALUES=(1)
SEEDS=(42 43 44 45 46)
DATASET="qnli"
TOP_K=20
EMBEDDING_TYPE="glove_840B-300d"
MAPPING_STRATEGY="conservative"

PRIVATIZATION_STRATEGY="s1"
MODEL_TYPE="/data/youyaru/SanText-main/bert-base-uncased"

export TRANSFORMERS_OFFLINE=1

# 创建结果目录
RESULT_DIR="./experiment_results"
mkdir -p ${RESULT_DIR}

# 记录实验开始时间
echo "========================================" | tee -a ${RESULT_DIR}/experiment_log.txt
echo "实验开始时间: $(date)" | tee -a ${RESULT_DIR}/experiment_log.txt
echo "设备: $(if [ -n "${USE_CUDA_FLAG}" ]; then echo "GPU ${CUDA_VISIBLE_DEVICES}"; else echo "CPU"; fi)" | tee -a ${RESULT_DIR}/experiment_log.txt
echo "eps 值: ${EPS_VALUES[*]}" | tee -a ${RESULT_DIR}/experiment_log.txt
echo "seeds: ${SEEDS[*]}" | tee -a ${RESULT_DIR}/experiment_log.txt
echo "========================================" | tee -a ${RESULT_DIR}/experiment_log.txt

# 遍历所有 eps 值
for eps in "${EPS_VALUES[@]}"
do
    echo "" | tee -a ${RESULT_DIR}/experiment_log.txt
    echo "开始运行 eps=${eps} 的实验..." | tee -a ${RESULT_DIR}/experiment_log.txt

    # 对每个 eps 值运行 5 次（不同随机种子）
    for seed in "${SEEDS[@]}"
    do
        echo "  运行 eps=${eps}, seed=${seed}..." | tee -a ${RESULT_DIR}/experiment_log.txt

        # 运行训练
        python main.py \
            --dataset ${DATASET} \
            --eps ${eps} \
            --top_k ${TOP_K} \
            --embedding_type ${EMBEDDING_TYPE} \
            --mapping_strategy ${MAPPING_STRATEGY} \
            --privatization_strategy ${PRIVATIZATION_STRATEGY} \
            --model_type ${MODEL_TYPE} \
            ${USE_CUDA_FLAG} \
            --seed ${seed} \
            2>&1 | tee ${RESULT_DIR}/eps_${eps}_seed_${seed}.log

        # 检查是否成功
        if [ $? -eq 0 ]; then
            echo "  ✓ eps=${eps}, seed=${seed} 完成" | tee -a ${RESULT_DIR}/experiment_log.txt
        else
            echo "  ✗ eps=${eps}, seed=${seed} 失败" | tee -a ${RESULT_DIR}/experiment_log.txt
        fi
    done

    echo "eps=${eps} 的所有实验完成" | tee -a ${RESULT_DIR}/experiment_log.txt
done

echo "" | tee -a ${RESULT_DIR}/experiment_log.txt
echo "========================================" | tee -a ${RESULT_DIR}/experiment_log.txt
echo "所有实验完成时间: $(date)" | tee -a ${RESULT_DIR}/experiment_log.txt
echo "========================================" | tee -a ${RESULT_DIR}/experiment_log.txt

# 汇总结果
echo "" | tee -a ${RESULT_DIR}/experiment_log.txt
echo "开始收集结果..." | tee -a ${RESULT_DIR}/experiment_log.txt
python collect_results.py

echo "实验结果已保存到 ${RESULT_DIR}/"