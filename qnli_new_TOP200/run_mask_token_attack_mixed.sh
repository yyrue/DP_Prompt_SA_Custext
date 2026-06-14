#!/bin/bash

# QNLI mixed 数据 Mask Token Attack 批量脚本
# 用法：
#   bash run_mask_token_attack_mixed.sh [GPU_ID] [SKIP_BASELINE]

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
SKIP_BASELINE=${2:-0}
if [ "${DEVICE}" = "cpu" ]; then
    export CUDA_VISIBLE_DEVICES=""
    echo "使用 CPU 运行"
else
    export CUDA_VISIBLE_DEVICES=${DEVICE}
    echo "使用 GPU: ${CUDA_VISIBLE_DEVICES}"
fi

EPS_LOW=${EPS_LOW:-0.0}
EPS_HIGH=${EPS_HIGH:-24.0}
EPS_PRIME_VALUES_STR=${EPS_PRIME_VALUES_STR:-"24"}
read -r -a EPS_PRIME_LIST <<< "${EPS_PRIME_VALUES_STR}"
SEEDS_STR=${SEEDS_STR:-"42 43 44 45 46"}
read -r -a SEEDS <<< "${SEEDS_STR}"

DATASET="qnli"
TOP_K=200
EMBEDDING_TYPE="glove_840B-300d"
MAPPING_STRATEGY="paper"
PRIVATIZATION_STRATEGY="s1"
PRIVATIZED_ROOT="./privatized_dataset_mixed"
MODEL_PATH="/data/youyaru/SanText-main/bert-base-uncased"
BATCH_SIZE=256
MAX_SEQ_LENGTH=128
SAVE_STOP_WORDS="False"

if [ "${SAVE_STOP_WORDS}" = "True" ]; then
    SAVE_STOP_FLAG="--save_stop_words"
else
    SAVE_STOP_FLAG=""
fi

RESULT_DIR_ROOT="./attack_results_mixed"
MIX_SUBDIR="mix_${EPS_LOW}_${EPS_HIGH}"
RESULT_DIR="${RESULT_DIR_ROOT}/${MIX_SUBDIR}"
mkdir -p "${RESULT_DIR}"

export TRANSFORMERS_OFFLINE=1

if [ "${SKIP_BASELINE}" != "1" ]; then
    for seed in "${SEEDS[@]}"; do
        python mask_token_attack_mixed.py \
            --dataset "${DATASET}" \
            --model_path "${MODEL_PATH}" \
            --output_dir "${RESULT_DIR}" \
            --privatized_root "${PRIVATIZED_ROOT}" \
            --eps_low "${EPS_LOW}" \
            --eps_high "${EPS_HIGH}" \
            --max_seq_length "${MAX_SEQ_LENGTH}" \
            --batch_size "${BATCH_SIZE}" \
            --seed "${seed}" \
            --top_k "${TOP_K}" \
            --embedding_type "${EMBEDDING_TYPE}" \
            --mapping_strategy "${MAPPING_STRATEGY}" \
            --privatization_strategy "${PRIVATIZATION_STRATEGY}" \
            ${SAVE_STOP_FLAG} \
            --attack_original \
            2>&1 | tee -a "${RESULT_DIR}/attack_mixed_baseline_seed_${seed}.log"
    done
fi

for seed in "${SEEDS[@]}"; do
    python mask_token_attack_mixed.py \
        --dataset "${DATASET}" \
        --model_path "${MODEL_PATH}" \
        --output_dir "${RESULT_DIR}" \
        --privatized_root "${PRIVATIZED_ROOT}" \
        --eps_low "${EPS_LOW}" \
        --eps_high "${EPS_HIGH}" \
        --eps_list "${EPS_PRIME_LIST[@]}" \
        --top_k "${TOP_K}" \
        --embedding_type "${EMBEDDING_TYPE}" \
        --mapping_strategy "${MAPPING_STRATEGY}" \
        --privatization_strategy "${PRIVATIZATION_STRATEGY}" \
        --seed "${seed}" \
        --max_seq_length "${MAX_SEQ_LENGTH}" \
        --batch_size "${BATCH_SIZE}" \
        ${SAVE_STOP_FLAG} \
        2>&1 | tee -a "${RESULT_DIR}/attack_mixed_mix_${EPS_LOW}_${EPS_HIGH}_seed_${seed}.log"

    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✓ seed=${seed} 完成"
    else
        echo "✗ seed=${seed} 失败"
    fi
done

echo "结果目录: ${RESULT_DIR}/"
