#!/bin/bash
# KNN Attack 批量实验（CusText privatized_dataset + Mix privatized_dataset_mixed）
#
# 用法:
#   bash run_knn_attack.sh [GPU_ID]     # 本攻击不用 GPU，参数可忽略
#   DATA_SOURCE=custext bash run_knn_attack.sh
#   DATA_SOURCE=mixed EPS_LOW=0 EPS_HIGH=20 bash run_knn_attack.sh

set -e
cd "$(dirname "$0")"

if [ "${CONDA_DEFAULT_ENV:-}" != "custext" ]; then
    if [ -n "${CONDA_EXE:-}" ] && [ -f "$(dirname "${CONDA_EXE}")/../etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1091
        source "$(dirname "${CONDA_EXE}")/../etc/profile.d/conda.sh"
        conda activate custext
    fi
fi

DATASET="sst2"
SPLIT="${SPLIT:-test}"
ATTACK_K="${ATTACK_K:-10}"
TOP_K=20
EMBEDDING_TYPE="glove_840B-300d"
MAPPING_STRATEGY="paper"
DATA_SOURCE="${DATA_SOURCE:-custext}"   # custext | mixed
EPS_LOW="${EPS_LOW:-0.0}"
EPS_HIGH="${EPS_HIGH:-20.0}"
# 与常见实验一致的 eps；可按需改
EPS_VALUES="${EPS_VALUES:-0 2 4 6 8 10 12 14 16 18 20}"
SEEDS="${SEEDS:-42 43 44 45 46 47 48 49 50 51}"
RESULT_DIR="./knn_attack_results/${DATA_SOURCE}"
mkdir -p "${RESULT_DIR}"

echo "KNN Attack: source=${DATA_SOURCE}, split=${SPLIT}, k=${ATTACK_K}" | tee "${RESULT_DIR}/knn_attack_log.txt"
echo "eps: ${EPS_VALUES}" | tee -a "${RESULT_DIR}/knn_attack_log.txt"
echo "seeds: ${SEEDS}" | tee -a "${RESULT_DIR}/knn_attack_log.txt"

# 原始数据自检（可选，取消注释）
# python knn_attack.py --attack_original --split "${SPLIT}" --seed 42 \
#     --output_dir "${RESULT_DIR}" --attack_k "${ATTACK_K}" 2>&1 | tee -a "${RESULT_DIR}/knn_attack_log.txt"

for eps in ${EPS_VALUES}; do
    for seed in ${SEEDS}; do
        echo "  eps=${eps}, seed=${seed} ..." | tee -a "${RESULT_DIR}/knn_attack_log.txt"
        EXTRA=()
        if [ "${DATA_SOURCE}" = "mixed" ]; then
            EXTRA=(--data_source mixed --eps_low "${EPS_LOW}" --eps_high "${EPS_HIGH}")
        else
            EXTRA=(--data_source custext)
        fi
        python knn_attack.py \
            --dataset "${DATASET}" \
            --split "${SPLIT}" \
            --eps "${eps}" \
            --seed "${seed}" \
            --top_k "${TOP_K}" \
            --embedding_type "${EMBEDDING_TYPE}" \
            --mapping_strategy "${MAPPING_STRATEGY}" \
            --attack_k "${ATTACK_K}" \
            --output_dir "${RESULT_DIR}" \
            "${EXTRA[@]}" \
            2>&1 | tee -a "${RESULT_DIR}/knn_eps_${eps}_seed_${seed}.log" || true
    done
done

echo "收集汇总..." | tee -a "${RESULT_DIR}/knn_attack_log.txt"
python collect_knn_attack_results.py --result_dir "${RESULT_DIR}"
echo "完成: ${RESULT_DIR}"
