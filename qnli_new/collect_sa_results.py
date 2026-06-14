#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QNLI Sample Amplification 实验结果收集脚本。

从 experiment_results_sa/mix_{eps_low}_{eps_high}_topk_{K}/ 下的
sa_eps_*_topk_*_seed_*.log 提取 test acc，并输出明细与统计。
"""

import argparse
import os
import re
from pathlib import Path

import pandas as pd


LOG_NAME_RE = re.compile(r"^sa_eps_(?P<eps>[\d.]+)_topk_(?P<topk>\d+)_seed_(?P<seed>\d+)\.log$")
TEST_ACC_RE = re.compile(r"test acc = (\d+\.\d+)")


def extract_test_acc(log_file: Path):
    try:
        content = log_file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    matches = TEST_ACC_RE.findall(content)
    if not matches:
        return None
    return float(matches[-1])


def main():
    root = Path(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description="收集 QNLI SA 实验结果")
    parser.add_argument("--eps_low", type=float, required=True)
    parser.add_argument("--eps_high", type=float, required=True)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--seeds", type=str, required=True, help="逗号分隔")
    parser.add_argument("--eps_values", type=str, required=True, help="逗号分隔")
    parser.add_argument(
        "--base_result_dir",
        type=str,
        default=str(root / "experiment_results_sa"),
        help="结果根目录",
    )
    args = parser.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    eps_values = [float(x.strip()) for x in args.eps_values.split(",") if x.strip()]
    mix_dir = Path(args.base_result_dir) / f"mix_{args.eps_low}_{args.eps_high}_topk_{args.top_k}"

    if not mix_dir.is_dir():
        raise SystemExit(f"目录不存在: {mix_dir}")

    rows = []
    for eps in eps_values:
        eps_str = str(int(eps)) if abs(eps - round(eps)) < 1e-9 else str(eps)
        for seed in seeds:
            log_file = mix_dir / f"sa_eps_{eps_str}_topk_{args.top_k}_seed_{seed}.log"
            if not log_file.is_file():
                print(f"✗ 缺失: {log_file.name}")
                continue
            acc = extract_test_acc(log_file)
            if acc is None:
                print(f"✗ 无 test acc: {log_file.name}")
                continue
            rows.append(
                {
                    "eps": float(eps),
                    "top_k": args.top_k,
                    "seed": seed,
                    "test_acc": acc,
                    "log_file": log_file.name,
                }
            )
            print(f"✓ eps={eps}, seed={seed}: test_acc={acc:.4f}")

    if not rows:
        print("未收集到任何结果。")
        return

    df = pd.DataFrame(rows).sort_values(["eps", "seed"]).reset_index(drop=True)
    summary_file = mix_dir / f"results_summary_topk_{args.top_k}.csv"
    df.to_csv(summary_file, index=False)
    print(f"明细已保存: {summary_file}")

    stats = df.groupby("eps")["test_acc"].agg(["mean", "std", "min", "max", "count"]).reset_index()
    stats.columns = ["eps", "平均值", "标准差", "最小值", "最大值", "运行次数"]
    stats["标准差"] = stats["标准差"].fillna(0.0)
    stats_file = mix_dir / f"results_statistics_topk_{args.top_k}.csv"
    stats.to_csv(stats_file, index=False)
    print(f"统计已保存: {stats_file}")
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
