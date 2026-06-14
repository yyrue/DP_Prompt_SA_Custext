#!/bin/bash

# 批量运行 SST-2 实验脚本
# 功能：对不同 eps 值，每个配置运行 NUM_RUNS 次（随机种子从42开始递增）
# 结果自动记录到各自的日志文件中
#
# 用法：
#   bash run_experiments.sh [GPU_ID]
#   例如：bash run_experiments.sh 0       # 使用 GPU 0
#         bash run_experiments.sh 1       # 使用 GPU 1
#         bash run_experiments.sh 0,1     # 使用 GPU 0 和 1
#         bash run_experiments.sh cpu     # 使用 CPU 运行
#         bash run_experiments.sh         # 默认使用 GPU 0

# ============ 环境配置 ============
# 激活 conda 虚拟环境
source activate custext 2>/dev/null || conda activate custext

# 设置 GPU / CPU（从命令行参数读取，默认为 0）
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
EPS_VALUES=(14 16 18 20 22)
NUM_RUNS=5            # 每个配置运行的次数（随机种子从42开始递增，即42,43,...,42+NUM_RUNS-1）
NUM_EPOCHS=3          # 训练轮数
DATASET="sst2"
TOP_K=100  
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

# 禁止 transformers 联网下载，强制使用本地模型
export TRANSFORMERS_OFFLINE=1

# 创建结果目录（根据是否使用 save_stop_words 加标识）
if [ "${SAVE_STOP_WORDS}" = "True" ]; then
    RESULT_DIR="./experiment_results_savestopword"
else
    RESULT_DIR="./experiment_results"
fi
mkdir -p ${RESULT_DIR}

# 记录实验开始时间
echo "========================================" | tee -a ${RESULT_DIR}/experiment_log.txt
echo "实验开始时间: $(date)" | tee -a ${RESULT_DIR}/experiment_log.txt
echo "设备: $(if [ -n "${USE_CUDA_FLAG}" ]; then echo "GPU ${CUDA_VISIBLE_DEVICES}"; else echo "CPU"; fi)" | tee -a ${RESULT_DIR}/experiment_log.txt
echo "save_stop_words: ${SAVE_STOP_WORDS}" | tee -a ${RESULT_DIR}/experiment_log.txt
echo "num_runs: ${NUM_RUNS}, num_epochs: ${NUM_EPOCHS}" | tee -a ${RESULT_DIR}/experiment_log.txt
echo "========================================" | tee -a ${RESULT_DIR}/experiment_log.txt

# 遍历所有 eps 值
for eps in "${EPS_VALUES[@]}"
do
    echo "" | tee -a ${RESULT_DIR}/experiment_log.txt
    echo "开始运行 eps=${eps}, top_k=${TOP_K} 的实验..." | tee -a ${RESULT_DIR}/experiment_log.txt
    
    # 对每个 eps 值运行 NUM_RUNS 次（随机种子从42开始递增）
    for (( i=0; i<NUM_RUNS; i++ ))
    do
        seed=$((42 + i))
        echo "  运行 eps=${eps}, top_k=${TOP_K}, seed=${seed} (${i+1}/${NUM_RUNS})..." | tee -a ${RESULT_DIR}/experiment_log.txt
        
        # 运行训练（使用已保存的脱敏数据集）
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
        
        # 检查是否成功
        if [ $? -eq 0 ]; then
            echo "  ✓ eps=${eps}, top_k=${TOP_K}, seed=${seed} 完成" | tee -a ${RESULT_DIR}/experiment_log.txt
        else
            echo "  ✗ eps=${eps}, top_k=${TOP_K}, seed=${seed} 失败" | tee -a ${RESULT_DIR}/experiment_log.txt
        fi
    done
    
    echo "eps=${eps}, top_k=${TOP_K} 的所有实验完成" | tee -a ${RESULT_DIR}/experiment_log.txt
done

echo "" | tee -a ${RESULT_DIR}/experiment_log.txt
echo "========================================" | tee -a ${RESULT_DIR}/experiment_log.txt
echo "所有实验完成时间: $(date)" | tee -a ${RESULT_DIR}/experiment_log.txt
echo "========================================" | tee -a ${RESULT_DIR}/experiment_log.txt

# 构造实验参数列表，传给 collect_results.py
SEEDS_STR=$(seq -s, 42 $((42 + NUM_RUNS - 1)))
EPS_STR=$(IFS=,; echo "${EPS_VALUES[*]}")

# 调用结果收集脚本
echo "" | tee -a ${RESULT_DIR}/experiment_log.txt
echo "开始收集结果..." | tee -a ${RESULT_DIR}/experiment_log.txt
python collect_results.py \
    --result_dir ${RESULT_DIR} \
    --eps_values "${EPS_STR}" \
    --top_k_values "${TOP_K}" \
    --seeds "${SEEDS_STR}" \
    --save_stop_words ${SAVE_STOP_WORDS}

echo "实验结果已保存到 ${RESULT_DIR}/"