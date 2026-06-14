#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仅在 qnli_new 目录下构建 CusText 映射缓存（不生成脱敏 TSV、不训练）：

  - sim_word_dict/{embedding}/{mapping}/top_{K}.txt
  - sim_dist_dict/{embedding}/{mapping}/top_{K}.txt
  - p_dict/{embedding}/{mapping}/eps_{ε}_top_{K}.txt

与 sst2_new 相同：Algorithm 1、data-independent，只依赖 embedding 与 top_k。

用法（在 qnli_new 目录下）:
  python build_mapping_cache.py
  python build_mapping_cache.py --top_k 20 --embedding_type glove_840B-300d --mapping_strategy paper
  python build_mapping_cache.py --eps_list 0 1 2 4 6 8 10 12 14 16 18 20
  python build_mapping_cache.py --force  # 即使已有缓存也重新计算 sim_word_dict
"""

import json
import os
import sys

from args import get_parser

_parser = get_parser()
_parser.add_argument(
    "--eps_list",
    type=float,
    nargs="+",
    default=None,
    help="要预计算的 p_dict 的 eps 列表；默认仅 --eps",
)
_parser.add_argument(
    "--force",
    action="store_true",
    help="强制重新构建 sim_word_dict / sim_dist_dict",
)
_args, _ = _parser.parse_known_args()
sys.argv = [sys.argv[0]]

import utils as _utils_mod
from utils import build_sim_word_dict, compute_p_dict


def main():
    args = _args
    _utils_mod.args = args

    eps_list = args.eps_list if args.eps_list else [args.eps]

    sim_word_path = (
        f"./sim_word_dict/{args.embedding_type}/{args.mapping_strategy}/top_{args.top_k}.txt"
    )
    sim_dist_path = (
        f"./sim_dist_dict/{args.embedding_type}/{args.mapping_strategy}/top_{args.top_k}.txt"
    )

    print("=" * 60)
    print("构建 CusText 映射缓存 (qnli_new)")
    print("=" * 60)
    print(f"  embedding:  {args.embedding_type}")
    print(f"  mapping:    {args.mapping_strategy}")
    print(f"  top_k:      {args.top_k}")
    print(f"  eps_list:   {eps_list}")
    print("=" * 60)

    if (
        not args.force
        and os.path.isfile(sim_word_path)
        and os.path.isfile(sim_dist_path)
    ):
        print(f"\n[1/2] 已存在，跳过构建:\n  {sim_word_path}\n  {sim_dist_path}")
        with open(sim_dist_path, "r") as f:
            sim_dist_dict = json.load(f)
    else:
        print("\n[1/2] 构建 sim_word_dict / sim_dist_dict（可能较久）...")
        _, sim_dist_dict = build_sim_word_dict(top_k=args.top_k)
        print(f"  已写入: {sim_word_path}")
        print(f"  已写入: {sim_dist_path}")

    print("\n[2/2] 构建 p_dict...")
    for eps in eps_list:
        p_path = (
            f"./p_dict/{args.embedding_type}/{args.mapping_strategy}/"
            f"eps_{eps}_top_{args.top_k}.txt"
        )
        if os.path.isfile(p_path) and not args.force:
            print(f"  eps={eps}: 已存在，跳过 → {p_path}")
            continue
        print(f"  eps={eps}: 计算中...")
        compute_p_dict(sim_dist_dict, eps=eps)
        print(f"  eps={eps}: 已写入 → {p_path}")

    print("\n完成。缓存目录: ./sim_word_dict  ./sim_dist_dict  ./p_dict")


if __name__ == "__main__":
    main()
