"""
收集 QNLI Mask Token Attack 实验结果并汇总。
"""

import argparse
import os
import re

import pandas as pd


def parse_result_file(filepath):
    results = {}
    with open(filepath, "r") as f:
        for line in f:
            m = re.match(r"\s*(\w+):\s*(.+)", line.strip())
            if not m:
                continue
            key, value = m.group(1), m.group(2)
            try:
                if "." in value:
                    results[key] = float(value)
                else:
                    results[key] = int(value)
            except ValueError:
                results[key] = value
    return results


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, default="./attack_results")
    parser.add_argument("--eps_values", type=str, default="0,2,4,6,8,10,12,14,16,18")
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46,47,48,49,50,51")
    parser.add_argument("--embedding_type", type=str, default="glove_840B-300d")
    parser.add_argument("--mapping_strategy", type=str, default="paper")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--save_stop_words", type=str, default="False")
    return parser


def main():
    args = get_parser().parse_args()
    eps_values = [float(x.strip()) for x in args.eps_values.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    rows = []

    for seed in seeds:
        fp = os.path.join(args.result_dir, f"attack_baseline_seed_{seed}.txt")
        if os.path.exists(fp):
            r = parse_result_file(fp)
            r["eps"] = "baseline"
            r["seed"] = seed
            rows.append(r)

    for eps in eps_values:
        for seed in seeds:
            fp = os.path.join(
                args.result_dir,
                f"attack_{args.embedding_type}_{args.mapping_strategy}_eps_{eps}_top_{args.top_k}"
                f"_seed_{seed}_save_stop_words_{args.save_stop_words}.txt",
            )
            if os.path.exists(fp):
                r = parse_result_file(fp)
                r["eps"] = eps
                r["seed"] = seed
                rows.append(r)

    if not rows:
        print("未找到任何结果文件。")
        return

    df = pd.DataFrame(rows).sort_values(["eps", "seed"]).reset_index(drop=True)
    out = os.path.join(args.result_dir, "attack_summary.csv")
    df.to_csv(out, index=False)
    print(f"汇总结果已保存到: {out}")


if __name__ == "__main__":
    main()
