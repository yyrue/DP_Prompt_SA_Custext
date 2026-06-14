#!/bin/bash
# 在 qnli_new 目录下构建 sim_word_dict / sim_dist_dict / p_dict 缓存

set -e
cd "$(dirname "$0")"

# 已在 custext 时勿再 conda activate（非交互 bash 未 conda init 会报错）
if [ "${CONDA_DEFAULT_ENV:-}" != "custext" ]; then
    if [ -n "${CONDA_EXE:-}" ] && [ -f "$(dirname "${CONDA_EXE}")/../etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1091
        source "$(dirname "${CONDA_EXE}")/../etc/profile.d/conda.sh"
        conda activate custext
    else
        echo "请先执行: conda activate custext"
        echo "（当前 CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-未设置}）"
        exit 1
    fi
fi

TOP_K=20
EMBEDDING_TYPE="glove_840B-300d"
MAPPING_STRATEGY="paper"
EPS_LIST="0 1 2 4 6 8 10 12 14 16 18 20 22"

echo "构建映射缓存: embedding=${EMBEDDING_TYPE}, mapping=${MAPPING_STRATEGY}, top_k=${TOP_K}"
python build_mapping_cache.py \
    --top_k "${TOP_K}" \
    --embedding_type "${EMBEDDING_TYPE}" \
    --mapping_strategy "${MAPPING_STRATEGY}" \
    --eps_list ${EPS_LIST}

echo "缓存已写入 ./sim_word_dict ./sim_dist_dict ./p_dict"
