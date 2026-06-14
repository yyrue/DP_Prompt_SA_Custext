#!/bin/bash

# 批量运行 Mask Token Inference Attack 实验
# 对不同 eps 值的脱敏数据执行 BERT MLM 攻击，评估隐私保护强度
#
# 用法：
#   bash run_mask_token_attack.sh [GPU_ID] [SKIP_BASELINE]
#   例如：bash run_mask_token_attack.sh 0       # 使用 GPU 0，正常跑 baseline + eps
#         bash run_mask_token_attack.sh 0 1     # 使用 GPU 0，跳过 baseline，只跑 eps
#         bash run_mask_token_attack.sh cpu     # 使用 CPU 运行

# ============ 环境配置 ============
source activate custext 2>/dev/null || conda activate custext

# 设置 GPU / CPU（从命令行参数读取，默认为 0）
DEVICE=${1:-0}
# 是否跳过 baseline（第二个参数，默认 0=不跳过，1=跳过）
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

# ============ 实验参数 ============
EPS_VALUES=(0 2 4 6 8 10 12 14 16 18 20)   # 要测试的 eps 列表
SEEDS=(42 43 44)              # 随机种子列表
DATASET="sst2"
TOP_K=20
EMBEDDING_TYPE="glove_840B-300d"
MAPPING_STRATEGY="paper"
PRIVATIZATION_STRATEGY="s1"
SAVE_STOP_WORDS="False"
MODEL_PATH="/data/youyaru/SanText-main/bert-base-uncased"  # MLM 模型路径
BATCH_SIZE=256
MAX_SEQ_LENGTH=128

if [ "${SAVE_STOP_WORDS}" = "True" ]; then
    SAVE_STOP_FLAG="--save_stop_words"
else
    SAVE_STOP_FLAG=""
fi

# 禁止 transformers 联网下载
export TRANSFORMERS_OFFLINE=1

# 结果目录
RESULT_DIR="./attack_results"
mkdir -p ${RESULT_DIR}

echo "========================================" | tee -a ${RESULT_DIR}/attack_log.txt
echo "Mask Token Attack 实验开始: $(date)" | tee -a ${RESULT_DIR}/attack_log.txt
echo "设备: $(if [ -n "${CUDA_VISIBLE_DEVICES}" ]; then echo "GPU ${CUDA_VISIBLE_DEVICES}"; else echo "CPU"; fi)" | tee -a ${RESULT_DIR}/attack_log.txt
echo "eps_values: ${EPS_VALUES[@]}" | tee -a ${RESULT_DIR}/attack_log.txt
echo "seeds: ${SEEDS[@]}" | tee -a ${RESULT_DIR}/attack_log.txt
echo "========================================" | tee -a ${RESULT_DIR}/attack_log.txt

# ---------- 1. Baseline: 对原始数据执行 attack ----------
if [ "${SKIP_BASELINE}" != "1" ]; then
    echo "" | tee -a ${RESULT_DIR}/attack_log.txt
    echo "[Baseline] 对原始数据执行 Mask Token Attack..." | tee -a ${RESULT_DIR}/attack_log.txt

    for seed in "${SEEDS[@]}"
    do
        echo "  seed=${seed}..." | tee -a ${RESULT_DIR}/attack_log.txt
        python mask_token_attack.py \
            --dataset ${DATASET} \
            --model_path ${MODEL_PATH} \
            --output_dir ${RESULT_DIR} \
            --max_seq_length ${MAX_SEQ_LENGTH} \
            --batch_size ${BATCH_SIZE} \
            --seed ${seed} \
            --top_k ${TOP_K} \
            --embedding_type ${EMBEDDING_TYPE} \
            --mapping_strategy ${MAPPING_STRATEGY} \
            --attack_original \
            2>&1 | tee -a ${RESULT_DIR}/attack_original_seed_${seed}.log
    done
else
    echo "" | tee -a ${RESULT_DIR}/attack_log.txt
    echo "[Baseline] 已跳过（SKIP_BASELINE=1）" | tee -a ${RESULT_DIR}/attack_log.txt
fi

# ---------- 2. 对不同 eps 的脱敏数据执行 attack ----------
for eps in "${EPS_VALUES[@]}"
do
    echo "" | tee -a ${RESULT_DIR}/attack_log.txt
    echo "eps=${eps}: 开始..." | tee -a ${RESULT_DIR}/attack_log.txt

    for seed in "${SEEDS[@]}"
    do
        echo "  eps=${eps}, seed=${seed}..." | tee -a ${RESULT_DIR}/attack_log.txt

        python mask_token_attack.py \
            --dataset ${DATASET} \
            --model_path ${MODEL_PATH} \
            --output_dir ${RESULT_DIR} \
            --max_seq_length ${MAX_SEQ_LENGTH} \
            --batch_size ${BATCH_SIZE} \
            --eps ${eps} \
            --top_k ${TOP_K} \
            --embedding_type ${EMBEDDING_TYPE} \
            --mapping_strategy ${MAPPING_STRATEGY} \
            --privatization_strategy ${PRIVATIZATION_STRATEGY} \
            --seed ${seed} \
            ${SAVE_STOP_FLAG} \
            2>&1 | tee -a ${RESULT_DIR}/attack_eps_${eps}_seed_${seed}.log

        if [ $? -eq 0 ]; then
            echo "  ✓ eps=${eps}, seed=${seed} 完成" | tee -a ${RESULT_DIR}/attack_log.txt
        else
            echo "  ✗ eps=${eps}, seed=${seed} 失败（可能脱敏数据不存在）" | tee -a ${RESULT_DIR}/attack_log.txt
        fi
    done

    echo "eps=${eps}: 完成" | tee -a ${RESULT_DIR}/attack_log.txt
done

echo "" | tee -a ${RESULT_DIR}/attack_log.txt
echo "========================================" | tee -a ${RESULT_DIR}/attack_log.txt
echo "所有 Mask Token Attack 实验完成: $(date)" | tee -a ${RESULT_DIR}/attack_log.txt
echo "========================================" | tee -a ${RESULT_DIR}/attack_log.txt

# ---------- 3. 汇总结果 ----------
echo "" | tee -a ${RESULT_DIR}/attack_log.txt
echo "开始汇总结果..." | tee -a ${RESULT_DIR}/attack_log.txt

python collect_attack_results.py \
    --result_dir ${RESULT_DIR} \
    --eps_values "$(IFS=,; echo "${EPS_VALUES[*]}")" \
    --seeds "$(IFS=,; echo "${SEEDS[*]}")" \
    --embedding_type ${EMBEDDING_TYPE} \
    --mapping_strategy ${MAPPING_STRATEGY} \
    --top_k ${TOP_K} \
    --save_stop_words ${SAVE_STOP_WORDS}

echo "结果汇总完成！查看 ${RESULT_DIR}/attack_summary.csv"