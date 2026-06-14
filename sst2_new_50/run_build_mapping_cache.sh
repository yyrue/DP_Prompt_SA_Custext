#!/bin/bash
# 构建 sim_word_dict / sim_dist_dict / p_dict（默认 top_k=50，可按需改 TOP_K）

set -e
cd "$(dirname "$0")"

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

TOP_K=${TOP_K:-500}
EMBEDDING_TYPE="${EMBEDDING_TYPE:-glove_840B-300d}"
MAPPING_STRATEGY="${MAPPING_STRATEGY:-paper}"
EPS_LIST="${EPS_LIST:-0 1 2 4 6 8 10 12 14 16 18 20 22 24}"

echo "构建映射缓存: top_k=${TOP_K}, embedding=${EMBEDDING_TYPE}, mapping=${MAPPING_STRATEGY}"
python build_mapping_cache.py \
    --top_k "${TOP_K}" \
    --embedding_type "${EMBEDDING_TYPE}" \
    --mapping_strategy "${MAPPING_STRATEGY}" \
    --eps_list ${EPS_LIST}

echo "完成 → ./sim_word_dict ./sim_dist_dict ./p_dict (top_${TOP_K})"
