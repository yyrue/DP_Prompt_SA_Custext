#!/bin/bash

# Sample Amplification 批量实验脚本
# 功能：对混合数据（mix eps_low 和 eps_high）的不同目标 eps'，每个配置运行 5 次
#
# 用法：
#   bash run_sample_amplification.sh [GPU_ID] [EPS_LOW] [EPS_HIGH]
#   例如：bash run_sample_amplification.sh 0 0.0 32.0
#         bash run_sample_amplification.sh 0 0.0 18.0
#         bash run_sample_amplification.sh cpu 0.0 32.0

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
EPS_LOW=${2:-0.0}
EPS_HIGH=${3:-32.0}
SEEDS=(42 43 44 45 46)

DATASET="sst2"
TOP_K=20
EMBEDDING_TYPE="glove_840B-300d"
MAPPING_STRATEGY="conservative"
PRIVATIZATION_STRATEGY="s1"
MODEL_TYPE="/data/youyaru/SanText-main/bert-base-uncased"

export TRANSFORMERS_OFFLINE=1

# 自动推算 eps_targets：(eps_low, eps_high) 之间的整数，即 eps_low+1 到 eps_high-1
EPS_LOW_INT=$(printf "%.0f" ${EPS_LOW})
EPS_HIGH_INT=$(printf "%.0f" ${EPS_HIGH})
EPS_TARGETS=()
for ((target=EPS_LOW_INT+1; target<=EPS_HIGH_INT-1; target++)); do
    EPS_TARGETS+=($target)
done

# 结果保存到区分 mix 来源的子目录
RESULT_DIR="./experiment_results_sa/mix_${EPS_LOW}_${EPS_HIGH}"
mkdir -p ${RESULT_DIR}

# 记录实验开始
echo "========================================" | tee -a ${RESULT_DIR}/experiment_log.txt
echo "Sample Amplification 实验开始: $(date)" | tee -a ${RESULT_DIR}/experiment_log.txt
echo "混合源: eps_low=${EPS_LOW}, eps_high=${EPS_HIGH}" | tee -a ${RESULT_DIR}/experiment_log.txt
echo "目标 eps': ${EPS_TARGETS[*]}" | tee -a ${RESULT_DIR}/experiment_log.txt
echo "设备: $(if [ -n "${USE_CUDA_FLAG}" ]; then echo "GPU ${CUDA_VISIBLE_DEVICES}"; else echo "CPU"; fi)" | tee -a ${RESULT_DIR}/experiment_log.txt
echo "========================================" | tee -a ${RESULT_DIR}/experiment_log.txt

for eps in "${EPS_TARGETS[@]}"
do
    echo "" | tee -a ${RESULT_DIR}/experiment_log.txt
    echo "开始运行 eps'=${eps} (mix ${EPS_LOW} & ${EPS_HIGH})..." | tee -a ${RESULT_DIR}/experiment_log.txt

    for seed in "${SEEDS[@]}"
    do
        echo "  运行 eps'=${eps}, seed=${seed}..." | tee -a ${RESULT_DIR}/experiment_log.txt

        python main_sample_amplification.py \
            --dataset ${DATASET} \
            --eps ${eps} \
            --eps_low ${EPS_LOW} \
            --eps_high ${EPS_HIGH} \
            --top_k ${TOP_K} \
            --embedding_type ${EMBEDDING_TYPE} \
            --mapping_strategy ${MAPPING_STRATEGY} \
            --privatization_strategy ${PRIVATIZATION_STRATEGY} \
            --model_type ${MODEL_TYPE} \
            ${USE_CUDA_FLAG} \
            --seed ${seed} \
            2>&1 | tee ${RESULT_DIR}/sa_eps_${eps}_seed_${seed}.log

        if [ $? -eq 0 ]; then
            echo "  ✓ eps'=${eps}, seed=${seed} 完成" | tee -a ${RESULT_DIR}/experiment_log.txt
        else
            echo "  ✗ eps'=${eps}, seed=${seed} 失败" | tee -a ${RESULT_DIR}/experiment_log.txt
        fi
    done

    echo "eps'=${eps} 的所有实验完成" | tee -a ${RESULT_DIR}/experiment_log.txt
done

echo "" | tee -a ${RESULT_DIR}/experiment_log.txt
echo "========================================" | tee -a ${RESULT_DIR}/experiment_log.txt
echo "所有 SA 实验完成: $(date)" | tee -a ${RESULT_DIR}/experiment_log.txt
echo "========================================" | tee -a ${RESULT_DIR}/experiment_log.txt

# 收集结果（传入 eps_low/eps_high 参数）
echo "开始收集 SA 实验结果..." | tee -a ${RESULT_DIR}/experiment_log.txt
python collect_sa_results.py --eps_low ${EPS_LOW} --eps_high ${EPS_HIGH}
echo "结果已保存到 ${RESULT_DIR}/"
