#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
QNLI 实验结果收集脚本
功能：从日志文件中提取测试准确率，汇总到 CSV 文件

当 eps_values/top_k_values/seeds 未指定时，自动扫描 result_dir 中的日志文件名来推断参数。
输出文件：
  - results_summary_topk_{K}.csv      明细：eps, top_k, seed, test_acc
  - results_statistics_topk_{K}.csv   统计：eps, 平均值, 标准差, 最小值, 最大值, 运行次数
  - results_summary_all.csv           所有 top_k 汇总明细
  - results_statistics_all.csv        所有 top_k 汇总统计
"""

import os
import re
import pandas as pd
from pathlib import Path

def _num_name_variants(x):
    """
    为数字参数生成文件名匹配候选：
    例如 0.0 -> ["0.0", "0"]，3 -> ["3"]。
    """
    vals = [str(x)]
    try:
        xf = float(x)
        if abs(xf - int(round(xf))) < 1e-9:
            vals.append(str(int(round(xf))))
    except Exception:
        pass
    out = []
    seen = set()
    for v in vals:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out

def extract_test_acc_from_log(log_file):
    """
    从日志文件中提取测试准确率
    匹配模式: "test acc = 0.xxxx" 或 "best accuracy ... -> 0.xxxx"
    """
    try:
        with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()

            content = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', content)
            content = content.replace('\r', '\n')

            matches = re.findall(r'test\s*acc\s*=\s*(\d+\.\d+)', content, flags=re.IGNORECASE)
            if matches:
                return float(matches[-1])

            matches = re.findall(
                r'best\s+accuracy(?:\s+performance)?\s+(?:has\s+been\s+)?updated:\s*\d+\.\d+\s*->\s*(\d+\.\d+)',
                content,
                flags=re.IGNORECASE,
            )
            if matches:
                return float(matches[-1])

            return None
    except Exception as e:
        print(f"读取文件 {log_file} 时出错: {e}")
        return None

def collect_results(result_dir='./experiment_results', log_dir='./log',
                    eps_values=None, top_k_values=None, seeds=None,
                    save_stop_words=None):
    """
    收集所有实验结果

    当 eps_values/top_k_values/seeds 未指定时，自动扫描 result_dir 中的日志文件名来推断参数。
    """
    results = []

    if eps_values is None or top_k_values is None or seeds is None:
        inferred_eps, inferred_topk, inferred_seeds = set(), set(), set()
        if os.path.exists(result_dir):
            for fname in os.listdir(result_dir):
                eps_match = re.search(r'eps_([\d.]+)', fname)
                topk_match = re.search(r'topk_(\d+)', fname)
                seed_match = re.search(r'seed_(\d+)', fname)
                if eps_match:
                    inferred_eps.add(float(eps_match.group(1)))
                if topk_match:
                    inferred_topk.add(int(topk_match.group(1)))
                if seed_match:
                    inferred_seeds.add(int(seed_match.group(1)))

        if eps_values is None:
            eps_values = sorted(inferred_eps) if inferred_eps else [0,1,2,4,6,8,10,12,14,16,18,20,22,24,26]
        if top_k_values is None:
            top_k_values = sorted(inferred_topk) if inferred_topk else [200]
        if seeds is None:
            seeds = sorted(inferred_seeds) if inferred_seeds else list(range(42, 47))

    log_suffix = ""
    if save_stop_words == "True" or save_stop_words is True:
        log_suffix = "_savestopword"

    print("开始收集 QNLI 实验结果...")
    print("=" * 60)
    print(f"  eps 值:     {eps_values}")
    print(f"  top_k 值:   {top_k_values}")
    print(f"  seeds:      {seeds}")
    if log_suffix:
        print(f"  日志后缀:   {log_suffix}")
    print("=" * 60)

    for eps in eps_values:
        for top_k in top_k_values:
            for seed in seeds:
                log_file = None
                eps_name_candidates = _num_name_variants(eps)
                topk_name_candidates = _num_name_variants(top_k)
                for eps_name in eps_name_candidates:
                    for topk_name in topk_name_candidates:
                        candidates = [
                            os.path.join(result_dir, f'eps_{eps_name}_topk_{topk_name}_seed_{seed}{log_suffix}.log'),
                            os.path.join(result_dir, f'eps_{eps_name}_topk_{topk_name}_seed_{seed}.log'),
                            os.path.join(result_dir, f'eps_{eps_name}_seed_{seed}{log_suffix}.log'),
                            os.path.join(result_dir, f'eps_{eps_name}_seed_{seed}.log'),
                        ]
                        for c in candidates:
                            if os.path.exists(c):
                                log_file = c
                                break
                        if log_file is not None:
                            break
                    if log_file is not None:
                        break

                if log_file is not None:
                    test_acc = extract_test_acc_from_log(log_file)

                    if test_acc is not None:
                        results.append({
                            'eps': eps,
                            'top_k': top_k,
                            'seed': seed,
                            'test_acc': test_acc
                        })
                        print(f"  eps={eps}, top_k={top_k}, seed={seed}: test_acc={test_acc:.4f}")
                    else:
                        print(f"  eps={eps}, top_k={top_k}, seed={seed}: 未找到测试准确率")
                else:
                    print(f"  eps={eps}, top_k={top_k}, seed={seed}: 日志文件不存在")

    print("=" * 60)

    if len(results) == 0:
        print("\n未在 experiment_results 目录找到结果，尝试从 log 目录读取...")
        print("=" * 60)

        if os.path.exists(log_dir):
            log_files = list(Path(log_dir).glob('*.txt'))
            for log_file in log_files:
                filename = log_file.name
                eps_match = re.search(r'eps_([\d.]+)', filename)
                topk_match = re.search(r'top_(\d+)', filename)
                seed_match = re.search(r'seed_(\d+)', filename)

                if eps_match:
                    eps = float(eps_match.group(1))
                    top_k = int(topk_match.group(1)) if topk_match else 'unknown'
                    seed = int(seed_match.group(1)) if seed_match else 'unknown'
                    test_acc = extract_test_acc_from_log(log_file)

                    if test_acc is not None:
                        results.append({
                            'eps': eps,
                            'top_k': top_k,
                            'seed': seed,
                            'test_acc': test_acc,
                            'log_file': filename
                        })
                        print(f"  {filename}: eps={eps}, top_k={top_k}, seed={seed}, test_acc={test_acc:.4f}")

    if len(results) > 0:
        df = pd.DataFrame(results)
        df = df.sort_values(['eps', 'top_k', 'seed'])

        top_k_values_in_df = df['top_k'].unique()

        for tk in sorted(top_k_values_in_df):
            df_tk = df[df['top_k'] == tk]

            output_file = os.path.join(result_dir, f'results_summary_topk_{tk}.csv')
            df_tk.to_csv(output_file, index=False)
            print(f"\ntop_k={tk} 结果已保存到: {output_file}")

            print(f"\ntop_k={tk} 统计摘要:")
            print("-" * 60)

            if 'seed' in df_tk.columns and df_tk['seed'].dtype != 'object':
                stats = df_tk.groupby('eps')['test_acc'].agg(['mean', 'std', 'min', 'max', 'count'])
                stats.columns = ['平均值', '标准差', '最小值', '最大值', '运行次数']
                print(stats.to_string())

                stats_file = os.path.join(result_dir, f'results_statistics_topk_{tk}.csv')
                stats.to_csv(stats_file)
                print(f"统计信息已保存到: {stats_file}")
            else:
                print(df_tk.to_string(index=False))

        all_output_file = os.path.join(result_dir, 'results_summary_all.csv')
        df.to_csv(all_output_file, index=False)
        print(f"\n所有 top_k 汇总结果已保存到: {all_output_file}")

        if 'seed' in df.columns and df['seed'].dtype != 'object':
            all_stats = df.groupby(['eps', 'top_k'])['test_acc'].agg(['mean', 'std', 'min', 'max', 'count'])
            all_stats.columns = ['平均值', '标准差', '最小值', '最大值', '运行次数']
            all_stats_file = os.path.join(result_dir, 'results_statistics_all.csv')
            all_stats.to_csv(all_stats_file)
            print(f"所有 top_k 汇总统计已保存到: {all_stats_file}")

        return df
    else:
        print("\n未找到任何实验结果！")
        return None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="收集 QNLI 实验结果")
    parser.add_argument("--result_dir", type=str, default='./experiment_results',
                        help="实验结果目录（默认 ./experiment_results）")
    parser.add_argument("--log_dir", type=str, default='./log',
                        help="logger 日志目录（默认 ./log）")
    parser.add_argument("--eps_values", type=str, default=None,
                        help="eps 值列表，逗号分隔（不指定则自动扫描）")
    parser.add_argument("--top_k_values", type=str, default=None,
                        help="top_k 值列表，逗号分隔（不指定则自动扫描）")
    parser.add_argument("--seeds", type=str, default=None,
                        help="seed 值列表，逗号分隔（不指定则自动扫描）")
    parser.add_argument("--save_stop_words", type=str, default=None,
                        help="是否带 save_stop_words 日志后缀（True/False）")
    cli_args = parser.parse_args()

    result_dir = cli_args.result_dir
    log_dir = cli_args.log_dir
    os.makedirs(result_dir, exist_ok=True)

    eps_values = [float(x.strip()) for x in cli_args.eps_values.split(',')] if cli_args.eps_values else None
    top_k_values = [int(x.strip()) for x in cli_args.top_k_values.split(',')] if cli_args.top_k_values else None
    seeds = [int(x.strip()) for x in cli_args.seeds.split(',')] if cli_args.seeds else None
    save_stop_words = cli_args.save_stop_words

    df = collect_results(result_dir=result_dir, log_dir=log_dir,
                         eps_values=eps_values, top_k_values=top_k_values, seeds=seeds,
                         save_stop_words=save_stop_words)