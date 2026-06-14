#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量为 QNLI 生成脱敏数据集（与 sst2_new/batch_generate_private_data.py 流程一致）。

  - sim_word_dict / sim_dist_dict / p_dict 只构建或加载一次
  - 对 train / test 的 sentence、question 两列脱敏
  - 每个 (eps, seed) 单独目录

用法:
  cd qnli_new
  python build_mapping_cache.py --eps_list 0 1 2 4 6 8 10 12 14 16 18 20
  python batch_generate_private_data.py --seeds 42 43 44 45 46 47 48 49 50 51 --eps_list 0 18
"""

import copy
import json
import os
import random
import sys

import numpy as np
import torch

from args import get_parser

_parser = get_parser()
_parser.add_argument(
    "--seeds",
    type=int,
    nargs="+",
    default=[42, 43, 44, 45, 46, 47, 48, 49, 50, 51],
    help="脱敏随机种子列表",
)
_parser.add_argument(
    "--eps_list",
    type=float,
    nargs="+",
    default=None,
    help="eps 列表；不指定则用 --eps",
)
_args, _ = _parser.parse_known_args()
sys.argv = [sys.argv[0]]

import utils as _utils_mod
from utils import load_data, build_sim_word_dict, compute_p_dict, generate_new_sents_s1


def main():
    args = _args
    _utils_mod.args = args
    eps_list = args.eps_list if args.eps_list else [args.eps]

    print("=" * 60)
    print("批量生成 QNLI 脱敏数据集")
    print("=" * 60)
    print(f"  seeds:     {args.seeds}")
    print(f"  eps_list:  {eps_list}")
    print(f"  top_k:     {args.top_k}")
    print(f"  embedding: {args.embedding_type}")
    print(f"  mapping:   {args.mapping_strategy}")
    print("=" * 60)

    print("\n[1/4] 加载原始 QNLI 数据...")
    train_orig, test_orig = load_data(args.dataset)
    print(f"  train: {len(train_orig)}, test: {len(test_orig)}")

    sim_word_path = (
        f"./sim_word_dict/{args.embedding_type}/{args.mapping_strategy}/top_{args.top_k}.txt"
    )
    sim_dist_path = (
        f"./sim_dist_dict/{args.embedding_type}/{args.mapping_strategy}/top_{args.top_k}.txt"
    )

    if os.path.isfile(sim_word_path) and os.path.isfile(sim_dist_path):
        print("\n[2/4] 加载已缓存的 sim_word_dict / sim_dist_dict...")
        with open(sim_word_path, "r") as f:
            sim_word_dict = json.load(f)
        with open(sim_dist_path, "r") as f:
            sim_dist_dict = json.load(f)
    else:
        print("\n[2/4] 构建 sim_word_dict / sim_dist_dict...")
        sim_word_dict, sim_dist_dict = build_sim_word_dict(top_k=args.top_k)

    p_dict_cache = {}
    for eps in eps_list:
        p_path = (
            f"./p_dict/{args.embedding_type}/{args.mapping_strategy}/"
            f"eps_{eps}_top_{args.top_k}.txt"
        )
        if os.path.isfile(p_path):
            print(f"\n[3/4] 加载 p_dict (eps={eps})...")
            with open(p_path, "r") as f:
                p_dict_cache[eps] = json.load(f)
        else:
            print(f"\n[3/4] 计算 p_dict (eps={eps})...")
            p_dict_cache[eps] = compute_p_dict(sim_dist_dict, eps=eps)

    total = len(eps_list) * len(args.seeds)
    done = 0
    skipped = 0

    print(f"\n[4/4] 生成脱敏 TSV，共 {total} 个 (eps, seed) 组合...\n")

    for eps in eps_list:
        p_dict = p_dict_cache[eps]
        for seed in args.seeds:
            done += 1
            priv_dir = (
                f"./privatized_dataset/{args.embedding_type}/{args.mapping_strategy}"
                f"/eps_{eps}_top_{args.top_k}_{args.privatization_strategy}"
                f"_save_stop_words_{args.save_stop_words}_seed_{seed}"
            )
            train_path = os.path.join(priv_dir, "train.tsv")
            test_path = os.path.join(priv_dir, "test.tsv")

            if os.path.isfile(train_path) and os.path.isfile(test_path):
                print(f"  [{done}/{total}] 已存在，跳过: eps={eps}, seed={seed}")
                skipped += 1
                continue

            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)

            train_data = train_orig.copy()
            test_data = test_orig.copy()
            args.seed = seed
            args.eps = eps

            print(f"  [{done}/{total}] 生成: eps={eps}, seed={seed} ...", flush=True, end="")

            if args.privatization_strategy == "s1":
                generate_new_sents_s1(
                    df=train_data,
                    sim_word_dict=sim_word_dict,
                    p_dict=p_dict,
                    save_stop_words=args.save_stop_words,
                    type="train",
                )
                generate_new_sents_s1(
                    df=test_data,
                    sim_word_dict=sim_word_dict,
                    p_dict=p_dict,
                    save_stop_words=args.save_stop_words,
                    type="test",
                )

            print(" 完成")

    print(f"\n{'=' * 60}")
    print(f"完成：共 {total} 组，新增 {total - skipped}，跳过 {skipped}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
