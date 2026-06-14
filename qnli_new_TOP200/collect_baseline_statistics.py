#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 baseline 训练日志（无隐私，等价 ε→∞）中提取最终 test acc，汇总为与
results_statistics_topk_*.csv 相同列名的统计 CSV，便于画图作水平参考线。

默认扫描: experiment_results/baseline_seed_*.log
匹配日志中最后一处「Baseline test acc = …」或「test acc = …」。
"""

import argparse
import os
import re
from pathlib import Path

import pandas as pd


def extract_baseline_test_acc(log_path: str):
    """返回最终 test acc；无法解析则返回 None。"""
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except OSError as exc:
        print(f"读取失败 {log_path}: {exc}")
        return None

    m = re.findall(r"Baseline test acc = (\d+\.\d+)", content, flags=re.IGNORECASE)
    if m:
        return float(m[-1])
    m = re.findall(r"test acc = (\d+\.\d+)", content, flags=re.IGNORECASE)
    if m:
        return float(m[-1])
    return None


def main():
    parser = argparse.ArgumentParser(description="汇总 baseline（ε=∞）test acc 到 CSV")
    parser.add_argument(
        "--result_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "experiment_results"),
        help="含 baseline_seed_*.log 的目录",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="baseline_seed_*.log",
        help="glob 相对 result_dir",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="输出 CSV（默认 result_dir/results_baseline_statistics.csv）",
    )
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    paths = sorted(result_dir.glob(args.pattern))
    if not paths:
        print(f"未找到匹配文件: {result_dir / args.pattern}")
        return

    rows = []
    for p in paths:
        sm = re.search(r"seed_(\d+)", p.name)
        seed = int(sm.group(1)) if sm else -1
        acc = extract_baseline_test_acc(str(p))
        if acc is not None:
            rows.append({"seed": seed, "test_acc": acc, "log_file": p.name})
            print(f"  {p.name}: seed={seed}, test_acc={acc:.4f}")
        else:
            print(f"  {p.name}: 未解析到 test acc")

    if not rows:
        print("没有可用结果，未写入 CSV。")
        return

    df = pd.DataFrame(rows).sort_values("seed")
    out_summary = args.out or str(result_dir / "results_baseline_summary.csv")
    df.to_csv(out_summary, index=False)
    print(f"\n逐 seed 明细已保存: {out_summary}")

    s = df["test_acc"]
    stats = pd.DataFrame(
        [
            {
                "eps": "inf",
                "平均值": s.mean(),
                "标准差": s.std(ddof=1) if len(s) > 1 else 0.0,
                "最小值": s.min(),
                "最大值": s.max(),
                "运行次数": len(s),
            }
        ]
    )
    out_stats = str(Path(out_summary).parent / "results_baseline_statistics.csv")
    stats.to_csv(out_stats, index=False)
    print(f"统计汇总（eps=inf）已保存: {out_stats}")


if __name__ == "__main__":
    main()
