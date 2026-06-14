#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sample Amplification 实验结果收集脚本
从日志文件中提取测试准确率，汇总到 CSV 文件。

支持通过 --eps_low / --eps_high 区分不同 mix 来源：
- 日志目录: experiment_results_sa/mix_{eps_low}_{eps_high}/
- target eps' 范围自动推算为 (eps_low+1) 到 (eps_high-1) 的整数
"""

import os
import re
import argparse
import pandas as pd


def extract_test_acc(log_file):
    """从日志文件中提取 test acc"""
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()
            match = re.search(r"test acc = (\d+\.\d+)", content)
            return float(match.group(1)) if match else None
    except Exception as e:
        print(f"读取文件 {log_file} 时出错: {e}")
        return None


def collect_results(eps_low, eps_high, seeds, base_result_dir="./experiment_results_sa"):
    """收集指定 mix 来源的所有 SA 实验结果"""

    # 自动推算 target 范围：eps_low+1 到 eps_high-1 的整数
    eps_targets = list(range(int(eps_low) + 1, int(eps_high)))

    # 日志所在子目录
    result_dir = os.path.join(base_result_dir, f"mix_{eps_low}_{eps_high}")

    print(f"收集 mix_{eps_low}_{eps_high} 的实验结果")
    print(f"目标 eps' 范围: {eps_targets}")
    print(f"日志目录: {result_dir}")
    print("=" * 60)

    results = []
    for eps in eps_targets:
        for seed in seeds:
            log_file = os.path.join(result_dir, f"sa_eps_{eps}_seed_{seed}.log")
            if not os.path.exists(log_file):
                print(f"✗ eps'={eps}, seed={seed}: 日志文件不存在")
                continue

            test_acc = extract_test_acc(log_file)
            if test_acc is not None:
                results.append({
                    "eps_low": eps_low,
                    "eps_high": eps_high,
                    "eps_target": eps,
                    "seed": seed,
                    "test_acc": test_acc,
                })
                print(f"✓ eps'={eps}, seed={seed}: test_acc={test_acc:.4f}")
            else:
                print(f"✗ eps'={eps}, seed={seed}: 未找到测试准确率")

    print("=" * 60)

    if not results:
        print("\n未找到任何实验结果！")
        return None

    df = pd.DataFrame(results).sort_values(["eps_target", "seed"])

    # 保存明细
    summary_file = os.path.join(result_dir, "sa_results_summary.csv")
    df.to_csv(summary_file, index=False)
    print(f"\n明细已保存到: {summary_file}")

    # 统计信息
    stats = df.groupby("eps_target")["test_acc"].agg(["mean", "std", "min", "max", "count"])
    stats.columns = ["mean", "std", "min", "max", "count"]
    print("\n" + "=" * 60)
    print("统计摘要:")
    print("=" * 60)
    print(stats.to_string())

    stats_file = os.path.join(result_dir, "sa_results_statistics.csv")
    stats.to_csv(stats_file)
    print(f"\n统计信息已保存到: {stats_file}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="收集 Sample Amplification 实验结果")
    parser.add_argument("--eps_low", type=float, default=0.0,
                        help="混合的低隐私预算源")
    parser.add_argument("--eps_high", type=float, default=32.0,
                        help="混合的高隐私预算源")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--base_result_dir", type=str, default="./experiment_results_sa")
    args = parser.parse_args()

    os.makedirs(args.base_result_dir, exist_ok=True)
    collect_results(
        eps_low=args.eps_low,
        eps_high=args.eps_high,
        seeds=args.seeds,
        base_result_dir=args.base_result_dir,
    )
