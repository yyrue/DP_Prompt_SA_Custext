#!/bin/bash

# Sample Amplification 批量训练脚本
# 前提：混合数据已通过 generate_sample_amplification.py 生成完毕
# 本脚本直接从已有的混合数据训练评估
#
# 用法：
#   # 按步长自动推算 eps'（默认步长 2：0,2,4,...,eps_high）
#   bash run_sample_amplification.sh [GPU_ID] [EPS_LOW] [EPS_HIGH]
#   bash run_sample_amplification.sh 0 0.0 20.0
#
#   # 指定若干目标 eps'（第 4 个参数起，空格分隔）
#   bash run_sample_amplification.sh 0 0.0 18.0 0 2 4 6 8 10 12 14 16 18
#   bash run_sample_amplification.sh 0 0.0 14.0 0 7 10 14
#
#   # 或用逗号写在一个参数里
#   bash run_sample_amplification.sh 0 0.0 18.0 "0,2,4,6,8,10,12,14,16,18"
#
#   # 或用环境变量（会覆盖命令行第 4 参数及以后）
#   EPS_TARGETS="0 7 10 14" bash run_sample_amplification.sh 0 0.0 18.0
#
#   bash run_sample_amplification.sh cpu 0.0 32.0

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
EPS_STEP=2
SEEDS=(42 43 44 45 46)

DATASET="sst2"
TOP_K=200
EMBEDDING_TYPE="glove_840B-300d"
MAPPING_STRATEGY="paper"
PRIVATIZATION_STRATEGY="s1"
SAVE_STOP_WORDS="False"
if [ "${SAVE_STOP_WORDS}" = "True" ]; then
    SAVE_STOP_FLAG="--save_stop_words"
else
    SAVE_STOP_FLAG=""
fi
MODEL_TYPE="/data/youyaru/SanText-main/bert-base-uncased"

# 混合数据的根目录
MIXED_DATA_DIR="./privatized_dataset_mixed/${EMBEDDING_TYPE}/${MAPPING_STRATEGY}"

export TRANSFORMERS_OFFLINE=1

# ============ 目标 eps' 列表 ============
# 优先级：环境变量 EPS_TARGETS > 命令行第 4 参数起 > 按 EPS_STEP 在 [EPS_LOW, EPS_HIGH] 上扫描
_EPS_TARGETS_ENV="${EPS_TARGETS:-}"
EPS_TARGETS=()
if [ -n "${_EPS_TARGETS_ENV}" ]; then
    _targets="${_EPS_TARGETS_ENV//,/ }"
    read -ra EPS_TARGETS <<< "${_targets}"
    echo "使用环境变量 EPS_TARGETS: ${EPS_TARGETS[*]}"
elif [ $# -ge 4 ]; then
  if [[ "$4" == *","* ]]; then
    IFS=',' read -ra EPS_TARGETS <<< "$4"
  else
    EPS_TARGETS=("${@:4}")
  fi
  echo "使用命令行指定的 eps': ${EPS_TARGETS[*]}"
else
  EPS_LOW_INT=$(printf "%.0f" "${EPS_LOW}")
  EPS_HIGH_INT=$(printf "%.0f" "${EPS_HIGH}")
  for ((target=EPS_LOW_INT; target<=EPS_HIGH_INT; target+=EPS_STEP)); do
    EPS_TARGETS+=("${target}")
  done
  echo "按步长 ${EPS_STEP} 自动推算 eps': ${EPS_TARGETS[*]}"
fi

# 结果保存到区分 mix 来源和 top_k 的子目录
RESULT_DIR="./experiment_results_sa/mix_${EPS_LOW}_${EPS_HIGH}_topk_${TOP_K}"
mkdir -p "${RESULT_DIR}"

# 记录实验开始
echo "========================================" | tee -a "${RESULT_DIR}/experiment_log.txt"
echo "Sample Amplification 实验开始: $(date)" | tee -a "${RESULT_DIR}/experiment_log.txt"
echo "混合源: eps_low=${EPS_LOW}, eps_high=${EPS_HIGH}" | tee -a "${RESULT_DIR}/experiment_log.txt"
echo "目标 eps': ${EPS_TARGETS[*]}" | tee -a "${RESULT_DIR}/experiment_log.txt"
echo "seeds: ${SEEDS[*]}" | tee -a "${RESULT_DIR}/experiment_log.txt"
echo "设备: $(if [ -n "${USE_CUDA_FLAG}" ]; then echo "GPU ${CUDA_VISIBLE_DEVICES}"; else echo "CPU"; fi)" | tee -a "${RESULT_DIR}/experiment_log.txt"
echo "========================================" | tee -a "${RESULT_DIR}/experiment_log.txt"

# ============ 训练 + 评估 ============
echo "" | tee -a "${RESULT_DIR}/experiment_log.txt"
echo "开始训练..." | tee -a "${RESULT_DIR}/experiment_log.txt"

for eps in "${EPS_TARGETS[@]}"
do
    echo "" | tee -a "${RESULT_DIR}/experiment_log.txt"
    echo "开始运行 eps'=${eps} (mix ${EPS_LOW} & ${EPS_HIGH})..." | tee -a "${RESULT_DIR}/experiment_log.txt"

    for seed in "${SEEDS[@]}"
    do
        echo "  运行 eps'=${eps}, seed=${seed}..." | tee -a "${RESULT_DIR}/experiment_log.txt"

        python main_sample_amplification.py \
            --dataset "${DATASET}" \
            --eps "${eps}" \
            --eps_low "${EPS_LOW}" \
            --eps_high "${EPS_HIGH}" \
            --top_k "${TOP_K}" \
            --embedding_type "${EMBEDDING_TYPE}" \
            --mapping_strategy "${MAPPING_STRATEGY}" \
            --privatization_strategy "${PRIVATIZATION_STRATEGY}" \
            --model_type "${MODEL_TYPE}" \
            ${USE_CUDA_FLAG} \
            --seed "${seed}" \
            --mixed_data_dir "${MIXED_DATA_DIR}" \
            ${SAVE_STOP_FLAG} \
            2>&1 | tee "${RESULT_DIR}/sa_eps_${eps}_topk_${TOP_K}_seed_${seed}.log"

        if [ $? -eq 0 ]; then
            echo "  ✓ eps'=${eps}, seed=${seed} 完成" | tee -a "${RESULT_DIR}/experiment_log.txt"
        else
            echo "  ✗ eps'=${eps}, seed=${seed} 失败" | tee -a "${RESULT_DIR}/experiment_log.txt"
        fi
    done

    echo "eps'=${eps} 的所有实验完成" | tee -a "${RESULT_DIR}/experiment_log.txt"
done

echo "" | tee -a "${RESULT_DIR}/experiment_log.txt"
echo "========================================" | tee -a "${RESULT_DIR}/experiment_log.txt"
echo "所有 SA 实验完成: $(date)" | tee -a "${RESULT_DIR}/experiment_log.txt"
echo "========================================" | tee -a "${RESULT_DIR}/experiment_log.txt"

# 收集结果
SEEDS_FOR_COLLECT=$(IFS=,; echo "${SEEDS[*]}")
EPS_TARGETS_STR=$(IFS=,; echo "${EPS_TARGETS[*]}")

echo "开始收集 SA 实验结果..." | tee -a "${RESULT_DIR}/experiment_log.txt"
python collect_sa_results.py \
    --eps_low "${EPS_LOW}" \
    --eps_high "${EPS_HIGH}" \
    --top_k "${TOP_K}" \
    --seeds "${SEEDS_FOR_COLLECT}" \
    --eps_values "${EPS_TARGETS_STR}"
echo "结果已保存到 ${RESULT_DIR}/"
