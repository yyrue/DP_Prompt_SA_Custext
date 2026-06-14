#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
按 eps 汇总 attack_summary.csv：defense_rate = 1 - attack_accuracy，
输出每个 eps 上的均值、标准差、最小值、最大值与运行次数。

无脱敏 baseline（attack_summary 中 eps=baseline）单独汇总为一行，eps 记为 inf（ε→∞）。

用法:
  python3 aggregate_attack_summary_by_eps.py
  python3 aggregate_attack_summary_by_eps.py --in ./attack_results/attack_summary.csv --out ./attack_results/attack_summary_by_eps.csv
"""

import argparse
import math
import os

import numpy as np
import pandas as pd


def _stats_from_defense(series: pd.Series) -> dict:
    s = series.astype(float)
    std = float(s.std(ddof=1)) if len(s) > 1 else 0.0
    if math.isnan(std):
        std = 0.0
    return {
        "平均值": float(s.mean()),
        "标准差": std,
        "最小值": float(s.min()),
        "最大值": float(s.max()),
        "运行次数": int(len(s)),
    }


def main():
    parser = argparse.ArgumentParser(description="按 eps 汇总 attack_summary 中的 defense rate")
    root = os.path.dirname(os.path.abspath(__file__))
    default_in = os.path.join(root, "attack_results", "attack_summary.csv")
    default_out = os.path.join(root, "attack_results", "attack_summary_by_eps.csv")
    parser.add_argument("--in", dest="inp", type=str, default=default_in, help="attack_summary.csv 路径")
    parser.add_argument("--out", type=str, default=default_out, help="输出统计 CSV 路径")
    args = parser.parse_args()

    df = pd.read_csv(args.inp)
    if "attack_accuracy" not in df.columns or "eps" not in df.columns:
        raise SystemExit("输入 CSV 需包含列: eps, attack_accuracy")

    is_baseline = df["eps"].astype(str).str.lower() == "baseline"
    base = df[is_baseline].copy()
    priv = df[~is_baseline].copy()

    parts = []

    if len(priv) > 0:
        priv["eps"] = pd.to_numeric(priv["eps"], errors="coerce")
        priv = priv.dropna(subset=["eps", "attack_accuracy"])
    if len(priv) > 0:
        priv["defense_rate"] = 1.0 - priv["attack_accuracy"].astype(float)
        stats_priv = (
            priv.groupby("eps", as_index=False)["defense_rate"]
            .agg(平均值="mean", 标准差="std", 最小值="min", 最大值="max", 运行次数="count")
            .sort_values("eps")
            .reset_index(drop=True)
        )
        stats_priv["标准差"] = stats_priv["标准差"].fillna(0.0)
        parts.append(stats_priv)

    if len(base) > 0:
        base["defense_rate"] = 1.0 - base["attack_accuracy"].astype(float)
        st = _stats_from_defense(base["defense_rate"])
        parts.append(
            pd.DataFrame(
                [
                    {
                        "eps": np.inf,
                        **st,
                    }
                ]
            )
        )

    if not parts:
        raise SystemExit("没有可用行：既无脱敏 eps 也无 baseline。")

    stats = pd.concat(parts, ignore_index=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    stats.to_csv(args.out, index=False)
    n_priv_rows = int(len(df[~is_baseline]))
    n_base_rows = int(len(base))
    print(
        f"已写入: {args.out}（attack_summary 脱敏行: {n_priv_rows}, baseline 行: {n_base_rows}，汇总表行数: {len(stats)}）"
    )
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
