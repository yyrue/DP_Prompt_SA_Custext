#!/bin/bash

# QNLI Sample Amplification 批量训练脚本
# 前提：mixed 数据已通过 generate_sample_amplification.py 生成完毕
#
# 用法：
#   # 按步长生成 eps' 列表（默认步长 2）
#   bash run_sample_amplification.sh [GPU_ID] [EPS_LOW] [EPS_HIGH]
#   bash run_sample_amplification.sh 0 0.0 18.0
#
#   # 显式指定目标 eps'（第 4 个参数及以后）
#   bash run_sample_amplification.sh 0 0.0 18.0 0 2 4 6 8 10 12 14 16 18
#
#   # 或一个参数里逗号分隔
#   bash run_sample_amplification.sh 0 0.0 18.0 "0,2,4,6,8,10,12,14,16,18"
#
#   # 或使用环境变量（优先级最高）
#   EPS_TARGETS="0 4 8 12 16 18" bash run_sample_amplification.sh 0 0.0 18.0

set -o pipefail

if [ "${CONDA_DEFAULT_ENV:-}" != "custext" ]; then
    if [ -n "${CONDA_EXE:-}" ] && [ -f "$(dirname "${CONDA_EXE}")/../etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1091
        source "$(dirname "${CONDA_EXE}")/../etc/profile.d/conda.sh"
        conda activate custext
    else
        echo "请先执行: conda activate custext"
        exit 1
    fi
fi

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

EPS_LOW=${2:-0.0}
EPS_HIGH=${3:-20.0}
EPS_STEP=2
SEEDS_STR=${SEEDS_STR:-"46"}
read -r -a SEEDS <<< "${SEEDS_STR}"

DATASET="qnli"
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
MIXED_DATA_DIR="./privatized_dataset_mixed/${EMBEDDING_TYPE}/${MAPPING_STRATEGY}"

export TRANSFORMERS_OFFLINE=1

_EPS_TARGETS_ENV="${EPS_TARGETS:-}"
EPS_TARGETS=()
if [ -n "${_EPS_TARGETS_ENV}" ]; then
    _targets="${_EPS_TARGETS_ENV//,/ }"
    read -r -a EPS_TARGETS <<< "${_targets}"
    echo "使用环境变量 EPS_TARGETS: ${EPS_TARGETS[*]}"
elif [ $# -ge 4 ]; then
    if [[ "$4" == *","* ]]; then
        IFS=',' read -r -a EPS_TARGETS <<< "$4"
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

RESULT_DIR="./experiment_results_sa/mix_${EPS_LOW}_${EPS_HIGH}_topk_${TOP_K}"
mkdir -p "${RESULT_DIR}"

echo "========================================" | tee -a "${RESULT_DIR}/experiment_log.txt"
echo "QNLI SA 实验开始: $(date)" | tee -a "${RESULT_DIR}/experiment_log.txt"
echo "混合源: eps_low=${EPS_LOW}, eps_high=${EPS_HIGH}" | tee -a "${RESULT_DIR}/experiment_log.txt"
echo "目标 eps': ${EPS_TARGETS[*]}" | tee -a "${RESULT_DIR}/experiment_log.txt"
echo "seeds: ${SEEDS[*]}" | tee -a "${RESULT_DIR}/experiment_log.txt"
echo "========================================" | tee -a "${RESULT_DIR}/experiment_log.txt"

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

        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            echo "  ✓ eps'=${eps}, seed=${seed} 完成" | tee -a "${RESULT_DIR}/experiment_log.txt"
        else
            echo "  ✗ eps'=${eps}, seed=${seed} 失败" | tee -a "${RESULT_DIR}/experiment_log.txt"
        fi
    done
done

echo "" | tee -a "${RESULT_DIR}/experiment_log.txt"
echo "所有 QNLI SA 实验完成: $(date)" | tee -a "${RESULT_DIR}/experiment_log.txt"

SEEDS_FOR_COLLECT=$(IFS=,; echo "${SEEDS[*]}")
EPS_TARGETS_STR=$(IFS=,; echo "${EPS_TARGETS[*]}")
python collect_sa_results.py \
    --eps_low "${EPS_LOW}" \
    --eps_high "${EPS_HIGH}" \
    --top_k "${TOP_K}" \
    --seeds "${SEEDS_FOR_COLLECT}" \
    --eps_values "${EPS_TARGETS_STR}" | tee -a "${RESULT_DIR}/experiment_log.txt"

echo "结果目录: ${RESULT_DIR}/"
