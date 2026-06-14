#!/bin/bash

# QNLI Mask Token Attack 批量运行脚本
# 用法：
#   bash run_mask_token_attack.sh [GPU_ID] [SKIP_BASELINE]

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

EPS_VALUES_STR=${EPS_VALUES_STR:-"18"}
read -r -a EPS_VALUES <<< "${EPS_VALUES_STR}"
SEEDS_STR=${SEEDS_STR:-"42 43 44"}
read -r -a SEEDS <<< "${SEEDS_STR}"

DATASET="qnli"
TOP_K=20
EMBEDDING_TYPE="glove_840B-300d"
MAPPING_STRATEGY="paper"
PRIVATIZATION_STRATEGY="s1"
SAVE_STOP_WORDS="False"
MODEL_PATH="/data/youyaru/SanText-main/bert-base-uncased"
BATCH_SIZE=256
MAX_SEQ_LENGTH=128

if [ "${SAVE_STOP_WORDS}" = "True" ]; then
    SAVE_STOP_FLAG="--save_stop_words"
else
    SAVE_STOP_FLAG=""
fi

export TRANSFORMERS_OFFLINE=1

RESULT_DIR="./attack_results"
mkdir -p "${RESULT_DIR}"

echo "========================================" | tee -a "${RESULT_DIR}/attack_log.txt"
echo "QNLI Mask Token Attack 开始: $(date)" | tee -a "${RESULT_DIR}/attack_log.txt"
echo "eps_values: ${EPS_VALUES[*]}" | tee -a "${RESULT_DIR}/attack_log.txt"
echo "seeds: ${SEEDS[*]}" | tee -a "${RESULT_DIR}/attack_log.txt"
echo "========================================" | tee -a "${RESULT_DIR}/attack_log.txt"

if [ "${SKIP_BASELINE}" != "1" ]; then
    for seed in "${SEEDS[@]}"; do
        echo "[Baseline] seed=${seed}" | tee -a "${RESULT_DIR}/attack_log.txt"
        python mask_token_attack.py \
            --dataset "${DATASET}" \
            --model_path "${MODEL_PATH}" \
            --output_dir "${RESULT_DIR}" \
            --max_seq_length "${MAX_SEQ_LENGTH}" \
            --batch_size "${BATCH_SIZE}" \
            --seed "${seed}" \
            --top_k "${TOP_K}" \
            --embedding_type "${EMBEDDING_TYPE}" \
            --mapping_strategy "${MAPPING_STRATEGY}" \
            --attack_original \
            2>&1 | tee -a "${RESULT_DIR}/attack_original_seed_${seed}.log"
    done
fi

for eps in "${EPS_VALUES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "eps=${eps}, seed=${seed}" | tee -a "${RESULT_DIR}/attack_log.txt"
        python mask_token_attack.py \
            --dataset "${DATASET}" \
            --model_path "${MODEL_PATH}" \
            --output_dir "${RESULT_DIR}" \
            --max_seq_length "${MAX_SEQ_LENGTH}" \
            --batch_size "${BATCH_SIZE}" \
            --eps "${eps}" \
            --top_k "${TOP_K}" \
            --embedding_type "${EMBEDDING_TYPE}" \
            --mapping_strategy "${MAPPING_STRATEGY}" \
            --privatization_strategy "${PRIVATIZATION_STRATEGY}" \
            --seed "${seed}" \
            ${SAVE_STOP_FLAG} \
            2>&1 | tee -a "${RESULT_DIR}/attack_eps_${eps}_seed_${seed}.log"

        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            echo "  ✓ eps=${eps}, seed=${seed} 完成" | tee -a "${RESULT_DIR}/attack_log.txt"
        else
            echo "  ✗ eps=${eps}, seed=${seed} 失败" | tee -a "${RESULT_DIR}/attack_log.txt"
        fi
    done
done

python collect_attack_results.py \
    --result_dir "${RESULT_DIR}" \
    --eps_values "$(IFS=,; echo "${EPS_VALUES[*]}")" \
    --seeds "$(IFS=,; echo "${SEEDS[*]}")" \
    --embedding_type "${EMBEDDING_TYPE}" \
    --mapping_strategy "${MAPPING_STRATEGY}" \
    --top_k "${TOP_K}" \
    --save_stop_words "${SAVE_STOP_WORDS}"

echo "结果目录: ${RESULT_DIR}/"
