"""
收集 Mask Token Attack 实验结果并汇总

分别输出 baseline accuracy 和 各 eps 下的 privatized accuracy

用法：
  python collect_attack_results.py --result_dir ./attack_results --eps_values 0,1,2,4,8 --seeds 42,49,123

  # 目录里已有各 eps 的 txt，但不想手写列表时：自动扫描文件名中的 eps / seed
  python collect_attack_results.py --result_dir ./attack_results --auto_scan
"""

import os
import re
import argparse
import pandas as pd


def discover_eps_seeds_from_dir(result_dir, embedding_type, mapping_strategy, top_k, save_stop):
    """
    扫描 result_dir 下与 mask_token_attack.save_results 一致的脱敏结果文件名，
    返回 (eps, seed) 列表。
    """
    want_stop = "True" if save_stop == "True" else "False"
    esc_emb = re.escape(embedding_type)
    esc_map = re.escape(mapping_strategy)
    pattern = re.compile(
        rf"^attack_{esc_emb}_{esc_map}_eps_([\d.]+)_top_{top_k}_seed_(\d+)"
        rf"_save_stop_words_{want_stop}\.txt$"
    )
    pairs = []
    if not os.path.isdir(result_dir):
        return pairs
    for name in os.listdir(result_dir):
        m = pattern.match(name)
        if m:
            pairs.append((float(m.group(1)), int(m.group(2))))
    return pairs


def discover_baseline_seeds(result_dir):
    seeds = []
    if not os.path.isdir(result_dir):
        return seeds
    for name in os.listdir(result_dir):
        m = re.match(r"attack_baseline_seed_(\d+)\.txt$", name)
        if m:
            seeds.append(int(m.group(1)))
    return sorted(set(seeds))


def get_attack_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, default="./attack_results",
                        help="攻击结果目录")
    parser.add_argument(
        "--auto_scan",
        action="store_true",
        help="从目录中自动发现 eps 与 seed（忽略 --eps_values / --seeds 的枚举范围）",
    )
    parser.add_argument("--eps_values", type=str, default="0,0.5,1,2,4,8,12,16,18",
                        help="逗号分隔的 eps 值列表（未使用 --auto_scan 时生效）")
    parser.add_argument("--seeds", type=str, default="42,49,123,456,789",
                        help="逗号分隔的 seed 列表（未使用 --auto_scan 时生效）")
    parser.add_argument("--embedding_type", type=str, default="glove_840B-300d")
    parser.add_argument("--mapping_strategy", type=str, default="paper")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--save_stop_words", type=str, default="False")
    return parser


def parse_result_file(filepath):
    """从攻击结果文件中解析各项指标"""
    results = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            m = re.match(r'\s*(\w+):\s*(.+)', line)
            if m:
                key = m.group(1)
                value = m.group(2).strip()
                try:
                    if '.' in value:
                        results[key] = float(value)
                    else:
                        results[key] = int(value)
                except ValueError:
                    results[key] = value
    return results


def main():
    parser = get_attack_parser()
    args = parser.parse_args()

    save_stop = args.save_stop_words
    if args.auto_scan:
        pairs = discover_eps_seeds_from_dir(
            args.result_dir,
            args.embedding_type,
            args.mapping_strategy,
            args.top_k,
            save_stop,
        )
        seeds_baseline = discover_baseline_seeds(args.result_dir)
        if not pairs and not seeds_baseline:
            print(
                "自动扫描未找到任何 attack 结果文件（脱敏或 baseline）；请检查 "
                "--result_dir / --embedding_type / --mapping_strategy / --top_k / --save_stop_words。"
            )
            return
        eps_values = sorted({p[0] for p in pairs})
        seeds_priv = sorted({p[1] for p in pairs})
        seeds = sorted(set(seeds_priv) | set(seeds_baseline))
        print(
            f"自动扫描: eps 共 {len(eps_values)} 个 {eps_values}, "
            f"seed 共 {len(seeds)} 个 {seeds}"
        )
    else:
        eps_values = [float(x.strip()) for x in args.eps_values.split(',')]
        seeds = [int(x.strip()) for x in args.seeds.split(',')]

    all_rows = []

    # 1. 收集 baseline（原始数据）的结果
    for seed in seeds:
        filename = f"attack_baseline_seed_{seed}.txt"
        filepath = os.path.join(args.result_dir, filename)
        if os.path.exists(filepath):
            results = parse_result_file(filepath)
            results['eps'] = 'baseline'
            results['seed'] = seed
            all_rows.append(results)
            print(f"  已收集 baseline seed={seed}: attack_accuracy={results.get('attack_accuracy', 'N/A')}")

    # 2. 收集不同 eps 的脱敏数据结果
    for eps in eps_values:
        for seed in seeds:
            if save_stop == "True":
                filename = (f"attack_{args.embedding_type}_{args.mapping_strategy}"
                           f"_eps_{eps}_top_{args.top_k}_seed_{seed}_save_stop_words_True.txt")
            else:
                filename = (f"attack_{args.embedding_type}_{args.mapping_strategy}"
                           f"_eps_{eps}_top_{args.top_k}_seed_{seed}"
                           f"_save_stop_words_False.txt")
            filepath = os.path.join(args.result_dir, filename)
            if os.path.exists(filepath):
                results = parse_result_file(filepath)
                results['eps'] = eps
                results['seed'] = seed
                all_rows.append(results)
                print(f"  已收集 eps={eps}, seed={seed}: attack_accuracy={results.get('attack_accuracy', 'N/A')}")

    if not all_rows:
        print("未找到任何结果文件！")
        return

    df = pd.DataFrame(all_rows)
    df = df.sort_values(by=['eps', 'seed']).reset_index(drop=True)

    # 保存汇总 CSV
    summary_path = os.path.join(args.result_dir, "attack_summary.csv")
    df.to_csv(summary_path, index=False)
    print(f"\n汇总结果已保存到: {summary_path}")

    # === 打印汇总表 ===
    print("\n" + "=" * 70)
    print("  Mask Token Attack 结果汇总")
    print("=" * 70)

    # Baseline
    baseline_df = df[df['eps'] == 'baseline']
    if len(baseline_df) > 0:
        baseline_acc_mean = baseline_df['attack_accuracy'].mean()
        baseline_acc_std = baseline_df['attack_accuracy'].std()
        print(f"\n  Baseline (原始数据):")
        print(f"    attack_accuracy = {baseline_acc_mean:.4f} ± {baseline_acc_std:.4f}")
        print(f"    (BERT MLM 对原始文本自身的预测准确率)")
    else:
        print(f"\n  [警告] 未找到 baseline 结果")
        print(f"  请先运行: python mask_token_attack.py --attack_original --seed 42")

    # 各 eps
    priv_df = df[df['eps'] != 'baseline']
    if len(priv_df) > 0:
        grouped = priv_df.groupby('eps').agg({
            'attack_accuracy': ['mean', 'std'],
        })

        print(f"\n  Privatized (脱敏数据):")
        print(f"  {'eps':<10} {'attack_accuracy (mean±std)':<30}")
        print(f"  {'-'*10} {'-'*30}")
        for eps_val in sorted(priv_df['eps'].unique()):
            row = grouped.loc[eps_val]
            acc_mean = row[('attack_accuracy', 'mean')]
            acc_std = row[('attack_accuracy', 'std')]
            # std 只有一个 seed 时为 NaN
            if len(priv_df[priv_df['eps'] == eps_val]) == 1:
                print(f"  {eps_val:<10} {acc_mean:<30.4f}")
            else:
                print(f"  {eps_val:<10} {acc_mean:.4f} ± {acc_std:.4f}")

    print("\n" + "=" * 70)
    print("  对比说明:")
    print("    baseline_accuracy   = BERT MLM 对原始文本 mask 预测的准确率")
    print("    privatized_accuracy = BERT MLM 对脱敏文本 mask 预测原词的准确率")
    print("    privatized_accuracy 越低，说明脱敏越有效阻止了攻击")
    print("=" * 70)


if __name__ == "__main__":
    main()