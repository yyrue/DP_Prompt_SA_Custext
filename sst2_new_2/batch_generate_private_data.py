"""
批量生成脱敏数据集脚本

相比反复调用 generate_private_data.py，此脚本的优势：
  - sim_word_dict / sim_dist_dict / p_dict 只加载一次（与 seed 无关）
  - 原始数据只加载一次
  - 避免了多进程重复初始化的开销

用法示例：
  # 默认参数：5个seed，eps=1.0
  python batch_generate_private_data.py

  # 自定义 seed 列表和 eps
  python batch_generate_private_data.py --seeds 42 123 456 789 2024 --eps 1.0

  # 多组 eps
  python batch_generate_private_data.py --seeds 42 123 456 --eps_list 0.5 1.0 2.0 4.0

  # 完整参数
  python batch_generate_private_data.py \
    --seeds 52 53 54 55 56 57 58 59 60 61 \
    --eps_list 0 18 \
    --top_k 20 \
    --embedding_type glove_840B-300d \
    --mapping_strategy paper \
    --privatization_strategy s1 \
    --dataset sst2
"""

import os
import sys
import json
import copy
import random
import numpy as np
import pandas as pd
import torch

from args import get_parser

# 在 import utils 之前先解析参数，以免 utils.py 模块级的 parse_args() 报错
# 先注册批量脚本的专属参数到 parser，再 import utils
_parser = get_parser()
_parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 2024],
                     help="要生成的 seed 列表，默认 [42, 123, 456, 789, 2024]")
_parser.add_argument("--eps_list", type=float, nargs="+", default=None,
                     help="要生成的 eps 列表，不指定则使用 --eps 的值")
_args, _unknown = _parser.parse_known_args()
# 把解析结果放到全局，这样 utils.py 里的 parse_args() 执行时也不会因为新参数报错
sys.argv = [sys.argv[0]]  # 清空 argv，让 utils.py 的 parse_args() 只拿到默认值

import utils as _utils_mod
from utils import load_data, build_sim_word_dict, compute_p_dict, generate_new_sents_s1


def main():
    # 使用之前已解析的参数
    args = _args

    # 同步更新 utils 模块中的全局 args，因为 generate_new_sents_s1 等函数
    # 内部通过 utils.args 读取 embedding_type / seed / eps 等来构建保存路径
    _utils_mod.args = args

    # 如果指定了 eps_list，用它；否则只生成 --eps 对应的单个值
    eps_list = args.eps_list if args.eps_list else [args.eps]

    print("=" * 60)
    print("批量生成脱敏数据集")
    print("=" * 60)
    print(f"  seeds:        {args.seeds}")
    print(f"  eps_list:     {eps_list}")
    print(f"  top_k:        {args.top_k}")
    print(f"  embedding:    {args.embedding_type}")
    print(f"  mapping:      {args.mapping_strategy}")
    print(f"  strategy:     {args.privatization_strategy}")
    print(f"  save_stop_words: {args.save_stop_words}")
    print(f"  dataset:      {args.dataset}")
    print("=" * 60)

    # 1. 加载原始数据（只需一次）
    print("\n[1/4] 加载原始数据...")
    train_data_orig, dev_data_orig, test_data_orig = load_data(args.dataset)
    print(f"  train: {len(train_data_orig)}, dev: {len(dev_data_orig)}, test: {len(test_data_orig)}")

    # 2. 加载 / 构建 sim_word_dict（只依赖 embedding_type + mapping_strategy + top_k，与 eps/seed 无关）
    sim_word_dict_path = f"./sim_word_dict/{args.embedding_type}/{args.mapping_strategy}/top_{args.top_k}.txt"
    sim_dist_dict_path = f"./sim_dist_dict/{args.embedding_type}/{args.mapping_strategy}/top_{args.top_k}.txt"

    if os.path.exists(sim_word_dict_path) and os.path.exists(sim_dist_dict_path):
        print("\n[2/4] 加载已缓存的 sim_word_dict 和 sim_dist_dict...")
        with open(sim_word_dict_path, 'r') as f:
            sim_word_dict = json.load(f)
        with open(sim_dist_dict_path, 'r') as f:
            sim_dist_dict = json.load(f)
    else:
        print("\n[2/4] 构建 sim_word_dict 和 sim_dist_dict（首次运行，耗时较长）...")
        sim_word_dict, sim_dist_dict = build_sim_word_dict(top_k=args.top_k)

    # 3. 对每个 eps 加载 / 计算 p_dict（只依赖 eps + top_k，与 seed 无关）
    p_dict_cache = {}
    for eps in eps_list:
        p_dict_path = f"./p_dict/{args.embedding_type}/{args.mapping_strategy}/eps_{eps}_top_{args.top_k}.txt"
        if os.path.exists(p_dict_path):
            print(f"\n[3/4] 加载已缓存的 p_dict (eps={eps})...")
            with open(p_dict_path, 'r') as f:
                p_dict_cache[eps] = json.load(f)
        else:
            print(f"\n[3/4] 计算 p_dict (eps={eps})...")
            p_dict_cache[eps] = compute_p_dict(sim_dist_dict, eps=eps)

    # 4. 遍历所有 (eps, seed) 组合生成脱敏数据
    total = len(eps_list) * len(args.seeds)
    done = 0
    skipped = 0

    print(f"\n[4/4] 开始生成脱敏数据，共 {total} 个组合...\n")

    for eps in eps_list:
        p_dict = p_dict_cache[eps]
        for seed in args.seeds:
            done += 1

            # 构建保存路径
            priv_dir = (
                f"./privatized_dataset/{args.embedding_type}/{args.mapping_strategy}"
                f"/eps_{eps}_top_{args.top_k}_{args.privatization_strategy}"
                f"_save_stop_words_{args.save_stop_words}_seed_{seed}"
            )
            train_path = os.path.join(priv_dir, "train.tsv")
            test_path = os.path.join(priv_dir, "test.tsv")

            # 检查是否已存在
            if os.path.exists(train_path) and os.path.exists(test_path):
                print(f"  [{done}/{total}] 已存在，跳过: eps={eps}, seed={seed}")
                skipped += 1
                continue

            # 设置随机种子（每次生成脱敏数据前都要重置，保证可复现）
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)

            # 深拷贝原始数据，因为 generate_new_sents_s1 会原地修改 DataFrame
            train_data = train_data_orig.copy()
            test_data = test_data_orig.copy()

            # 更新 args 中的 seed 和 eps（generate_new_sents_s1 内部通过 args 构建保存路径）
            args.seed = seed
            args.eps = eps

            print(f"  [{done}/{total}] 生成中: eps={eps}, seed={seed} ...", end="", flush=True)

            if args.privatization_strategy == "s1":
                generate_new_sents_s1(
                    df=train_data,
                    sim_word_dict=sim_word_dict,
                    p_dict=p_dict,
                    save_stop_words=args.save_stop_words,
                    type="train"
                )
                generate_new_sents_s1(
                    df=test_data,
                    sim_word_dict=sim_word_dict,
                    p_dict=p_dict,
                    save_stop_words=args.save_stop_words,
                    type="test"
                )

            print(" 完成")

    print(f"\n{'=' * 60}")
    print(f"批量生成完成！共 {total} 个组合，新增 {total - skipped} 个，跳过 {skipped} 个（已存在）")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()