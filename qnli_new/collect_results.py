#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
QNLI 实验结果收集脚本
功能：从日志文件中提取测试准确率，汇总到 CSV 文件，并计算每个 eps 的均值
"""

import os
import re
import pandas as pd
from pathlib import Path


def extract_test_acc_from_log(log_file):
    """
    从日志文件中提取测试准确率
    匹配模式: "test acc = 0.xxxx"
    """
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # 匹配 "test acc = 0.xxxx"，只匹配数字（含一个小数点）
            match = re.search(r'test acc = (\d+\.\d+)', content)
            if match:
                return float(match.group(1))
            else:
                return None
    except Exception as e:
        print(f"读取文件 {log_file} 时出错: {e}")
        return None


def collect_results(result_dir='./experiment_results'):
    """
    收集所有实验结果
    """
    results = []

    # 实验参数（与 run_experiments.sh 保持一致）
    eps_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 20, 32]
    seeds = [42, 43, 44, 45, 46]

    print("开始收集 QNLI 实验结果...")
    print("=" * 60)

    # 遍历所有 eps 和 seed 组合
    for eps in eps_values:
        for seed in seeds:
            log_file = os.path.join(result_dir, f'eps_{eps}_seed_{seed}.log')

            if os.path.exists(log_file):
                test_acc = extract_test_acc_from_log(log_file)

                if test_acc is not None:
                    results.append({
                        'eps': eps,
                        'seed': seed,
                        'test_acc': test_acc
                    })
                    print(f"  eps={eps}, seed={seed}: test_acc={test_acc:.4f}")
                else:
                    print(f"  eps={eps}, seed={seed}: 日志存在但未找到 test acc")
            else:
                print(f"  eps={eps}, seed={seed}: 日志文件不存在")

    print("=" * 60)

    if len(results) == 0:
        print("\n未找到任何实验结果！")
        return None

    # 转换为 DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values(['eps', 'seed'])

    # 保存每次运行的详细结果
    detail_file = os.path.join(result_dir, 'results_detail.csv')
    df.to_csv(detail_file, index=False)
    print(f"\n详细结果已保存到: {detail_file}")

    # 计算每个 eps 的统计信息（均值、标准差、最小值、最大值、运行次数）
    stats = df.groupby('eps')['test_acc'].agg(['mean', 'std', 'min', 'max', 'count'])
    stats.columns = ['mean_acc', 'std_acc', 'min_acc', 'max_acc', 'count']

    # 打印统计摘要
    print("\n" + "=" * 60)
    print("统计摘要 (每个 eps 的 5 次运行结果):")
    print("=" * 60)
    print(stats.to_string())

    # 保存统计信息
    stats_file = os.path.join(result_dir, 'results_summary.csv')
    stats.to_csv(stats_file)
    print(f"\n统计结果已保存到: {stats_file}")

    return df


if __name__ == "__main__":
    result_dir = './experiment_results'
    os.makedirs(result_dir, exist_ok=True)
    collect_results(result_dir=result_dir)