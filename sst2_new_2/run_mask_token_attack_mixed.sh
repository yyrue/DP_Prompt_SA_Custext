#!/bin/bash

# 对 privatized_dataset_mixed（Sample Amplification 混合数据）批量运行 Mask Token Attack
#
# 流程说明：
#   - 每个 seed：调用一次 mask_token_attack_mixed.py，并传入整段 --eps_list。
#   - 这一次运行会按顺序跑完列表里每个 ε'，每个 ε' 写一个 txt；若 ε' 多于 1 个，
#     同一 seed 还会生成一个汇总 CSV（attack_summary_mixed_*_seed_{seed}.csv），
#     即「一个 seed 对应一张表（多行 eps_prime）」。
#   - 不同 (EPS_LOW, EPS_HIGH) 的结果写到 attack_results_mixed 下的独立子目录，
#     避免多种 mixed 区间混在同一层目录。
#
# 用法：
#   bash run_mask_token_attack_mixed.sh [GPU_ID] [SKIP_BASELINE]
#   例如：bash run_mask_token_attack_mixed.sh 0
#         bash run_mask_token_attack_mixed.sh 0 1   # 跳过 baseline
#         bash run_mask_token_attack_mixed.sh cpu

source activate custext 2>/dev/null || conda activate custext

DEVICE=${1:-0}
SKIP_BASELINE=${2:-0}
if [ "${DEVICE}" = "cpu" ]; then
    export CUDA_VISIBLE_DEVICES=""
    echo "使用 CPU 运行"
else
    export CUDA_VISIBLE_DEVICES=${DEVICE}
    echo "使用 GPU: ${CUDA_VISIBLE_DEVICES}"
fi
if [ "${SKIP_BASELINE}" = "1" ]; then
    echo "跳过 Baseline（SKIP_BASELINE=1）"
else
    echo "将运行 Baseline（SKIP_BASELINE=0）"
fi

# ============ 与 generate_sample_amplification 生成目录一致 ============
EPS_LOW=0.0
EPS_HIGH=14.0
# 混合数据里各子目录的 ε' 步长与生成脚本默认一致：0,2,4,...,eps_high
EPS_PRIME_LIST=(0 2 4 6 8 10 12 14)

SEEDS=(42 43 44 45 46)
DATASET="sst2"
TOP_K=200
EMBEDDING_TYPE="glove_840B-300d"
MAPPING_STRATEGY="paper"
PRIVATIZATION_STRATEGY="s1"
PRIVATIZED_ROOT="./privatized_dataset_mixed"
MODEL_PATH="/data/youyaru/SanText-main/bert-base-uncased"
BATCH_SIZE=256
MAX_SEQ_LENGTH=128

# 按 mixed 区间分子目录：换一组 EPS_LOW/EPS_HIGH 再跑，结果会进新的文件夹
RESULT_DIR_ROOT="./attack_results_mixed"
MIX_SUBDIR="mix_${EPS_LOW}_${EPS_HIGH}"
RESULT_DIR="${RESULT_DIR_ROOT}/${MIX_SUBDIR}"
mkdir -p "${RESULT_DIR}"

SAVE_STOP_FLAG=""
# 若混合数据为 save_stop_words_True 生成，改为: SAVE_STOP_FLAG="--save_stop_words"

export TRANSFORMERS_OFFLINE=1

echo "========================================" | tee -a "${RESULT_DIR}/attack_mixed_log.txt"
echo "输出目录: ${RESULT_DIR}" | tee -a "${RESULT_DIR}/attack_mixed_log.txt"
echo "Mask Token Attack (mixed) 开始: $(date)" | tee -a "${RESULT_DIR}/attack_mixed_log.txt"
echo "mix: mix_${EPS_LOW}_${EPS_HIGH}, eps_prime: ${EPS_PRIME_LIST[*]}" | tee -a "${RESULT_DIR}/attack_mixed_log.txt"
echo "seeds: ${SEEDS[*]}" | tee -a "${RESULT_DIR}/attack_mixed_log.txt"
echo "========================================" | tee -a "${RESULT_DIR}/attack_mixed_log.txt"

if [ "${SKIP_BASELINE}" != "1" ]; then
    for seed in "${SEEDS[@]}"; do
        echo "[Baseline] seed=${seed}..." | tee -a "${RESULT_DIR}/attack_mixed_log.txt"
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
else
    echo "[Baseline] 已跳过" | tee -a "${RESULT_DIR}/attack_mixed_log.txt"
fi

for seed in "${SEEDS[@]}"; do
    echo "" | tee -a "${RESULT_DIR}/attack_mixed_log.txt"
    echo "seed=${seed}: 跑全部 ε'（一条命令内汇总 CSV）..." | tee -a "${RESULT_DIR}/attack_mixed_log.txt"

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

    if [ $? -eq 0 ]; then
        echo "  ✓ seed=${seed} 完成" | tee -a "${RESULT_DIR}/attack_mixed_log.txt"
    else
        echo "  ✗ seed=${seed} 失败" | tee -a "${RESULT_DIR}/attack_mixed_log.txt"
    fi
done

echo "" | tee -a "${RESULT_DIR}/attack_mixed_log.txt"
echo "全部完成: $(date)" | tee -a "${RESULT_DIR}/attack_mixed_log.txt"
echo "单条 txt 与按 seed 的汇总 CSV 见: ${RESULT_DIR}/"
