#!/bin/bash

# 批量运行 QNLI 实验（与 sst2_new 流程一致：先脱敏缓存 + batch 生成，再 --use_saved_private_data 训练）
#
# 用法：
#   bash run_build_mapping_cache.sh          # 首次：构建 sim_word_dict / p_dict
#   bash batch 或 python batch_generate_private_data.py ...
#   bash run_experiments.sh [GPU_ID]

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

EPS_VALUES_STR=${EPS_VALUES_STR:-"18"}
read -r -a EPS_VALUES <<< "${EPS_VALUES_STR}"
NUM_RUNS=3
NUM_EPOCHS=3
DATASET="qnli"
TOP_K=20
EMBEDDING_TYPE="glove_840B-300d"
MAPPING_STRATEGY="paper"
PRIVATIZATION_STRATEGY="s1"
SAVE_STOP_WORDS="False"
if [ "${SAVE_STOP_WORDS}" = "True" ]; then
    SAVE_STOP_FLAG="--save_stop_words"
    SAVE_STOP_LOG_TAG="_savestopword"
else
    SAVE_STOP_FLAG=""
    SAVE_STOP_LOG_TAG=""
fi
MODEL_TYPE="/data/youyaru/SanText-main/bert-base-uncased"

export TRANSFORMERS_OFFLINE=1

RESULT_DIR="./experiment_results"
mkdir -p ${RESULT_DIR}

echo "========================================" | tee -a ${RESULT_DIR}/experiment_log.txt
echo "QNLI 实验开始: $(date)" | tee -a ${RESULT_DIR}/experiment_log.txt
echo "mapping: ${MAPPING_STRATEGY}, top_k: ${TOP_K}" | tee -a ${RESULT_DIR}/experiment_log.txt
echo "eps_values: ${EPS_VALUES[*]}" | tee -a ${RESULT_DIR}/experiment_log.txt
echo "num_runs: ${NUM_RUNS}, num_epochs: ${NUM_EPOCHS}" | tee -a ${RESULT_DIR}/experiment_log.txt
echo "========================================" | tee -a ${RESULT_DIR}/experiment_log.txt

for eps in "${EPS_VALUES[@]}"
do
    echo "" | tee -a ${RESULT_DIR}/experiment_log.txt
    echo "开始 eps=${eps} ..." | tee -a ${RESULT_DIR}/experiment_log.txt

    for (( i=0; i<NUM_RUNS; i++ ))
    do
        seed=$((42 + i))
        echo "  eps=${eps}, seed=${seed} (${i+1}/${NUM_RUNS})..." | tee -a ${RESULT_DIR}/experiment_log.txt

        python main.py \
            --dataset ${DATASET} \
            --eps ${eps} \
            --top_k ${TOP_K} \
            --embedding_type ${EMBEDDING_TYPE} \
            --mapping_strategy ${MAPPING_STRATEGY} \
            --privatization_strategy ${PRIVATIZATION_STRATEGY} \
            --model_type ${MODEL_TYPE} \
            --epochs ${NUM_EPOCHS} \
            ${USE_CUDA_FLAG} \
            --seed ${seed} \
            ${SAVE_STOP_FLAG} \
            --use_saved_private_data \
            2>&1 | tee ${RESULT_DIR}/eps_${eps}_topk_${TOP_K}_seed_${seed}${SAVE_STOP_LOG_TAG}.log

        if [ $? -eq 0 ]; then
            echo "  ✓ 完成" | tee -a ${RESULT_DIR}/experiment_log.txt
        else
            echo "  ✗ 失败" | tee -a ${RESULT_DIR}/experiment_log.txt
        fi
    done
done

echo "" | tee -a ${RESULT_DIR}/experiment_log.txt
echo "所有实验完成: $(date)" | tee -a ${RESULT_DIR}/experiment_log.txt
echo "开始收集结果..." | tee -a ${RESULT_DIR}/experiment_log.txt
python collect_results.py

echo "结果目录: ${RESULT_DIR}/"
