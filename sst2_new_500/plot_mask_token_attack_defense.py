#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
绘制 Mask token attack：横轴 eps，纵轴 defense_rate（1 - attack_accuracy）。
默认读取 aggregate_attack_summary_by_eps.py 生成的按 eps 统计 CSV。

含 baseline（eps=inf / 无脱敏）时：曲线仅连接有限 ε；在横轴 ∞ 刻度处单独画一点。

用法:
  python3 aggregate_attack_summary_by_eps.py
  python3 plot_mask_token_attack_defense.py
"""

import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _is_inf_eps(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    mask_str = s.isin(["inf", "infinity"])
    num = pd.to_numeric(series, errors="coerce")
    arr = num.to_numpy(dtype=float, na_value=np.nan)
    mask_num = np.isinf(arr)
    return pd.Series(mask_str.to_numpy() | mask_num, index=series.index)


def main():
    parser = argparse.ArgumentParser(description="Mask token attack: defense rate vs eps")
    root = os.path.dirname(os.path.abspath(__file__))
    default_stats = os.path.join(root, "attack_results", "attack_summary_by_eps.csv")
    parser.add_argument(
        "--stats",
        type=str,
        default=default_stats,
        help="按 eps 汇总后的 CSV（含 平均值/标准差 等列）",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="输出图片路径（默认与 stats 同目录 plot_mask_token_attack_defense.png）",
    )
    parser.add_argument("--no-errorbar", action="store_true", help="不画标准差误差棒")
    args = parser.parse_args()

    df = pd.read_csv(args.stats)
    required = {"eps", "平均值", "标准差"}
    if not required.issubset(df.columns):
        raise SystemExit(f"统计 CSV 需包含列: {required}。请先运行 aggregate_attack_summary_by_eps.py")

    inf_m = _is_inf_eps(df["eps"])
    df_f = df[~inf_m].copy()
    df_i = df[inf_m].copy()

    if len(df_f) == 0 and len(df_i) == 0:
        raise SystemExit("统计表为空")

    out_path = args.out
    if not out_path:
        out_path = os.path.join(
            os.path.dirname(os.path.abspath(args.stats)),
            "plot_mask_token_attack_defense.png",
        )

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    if len(df_f) > 0:
        eps = df_f["eps"].astype(float).values
        mean_def = df_f["平均值"].astype(float).values
        std_def = df_f["标准差"].astype(float).values
        if not args.no_errorbar:
            ax.errorbar(
                eps,
                mean_def,
                yerr=std_def,
                fmt="o-",
                capsize=3,
                markersize=6,
                linewidth=1.5,
                color="#542788",
                ecolor="#9e9ac8",
                elinewidth=1,
                label="Sanitized",
            )
        else:
            ax.plot(eps, mean_def, "o-", markersize=6, linewidth=1.5, color="#542788", label="Sanitized")
        max_eps = float(np.max(eps))
    else:
        max_eps = 0.0

    if len(df_i) > 0:
        row = df_i.iloc[0]
        mean_b = float(row["平均值"])
        std_b = float(row["标准差"])
        if math.isnan(std_b):
            std_b = 0.0
        x_inf_plot = max_eps + 2.0 if len(df_f) > 0 else 2.0
        bl_label = r"No sanitized ($\infty$)"
        if not args.no_errorbar and std_b > 0:
            ax.errorbar(
                [x_inf_plot],
                [mean_b],
                yerr=[std_b],
                fmt="o",
                markersize=7,
                capsize=4,
                color="#d95f0e",
                ecolor="#fdae61",
                elinewidth=1.2,
                label=bl_label,
                zorder=5,
            )
        else:
            ax.scatter(
                [x_inf_plot],
                [mean_b],
                s=55,
                color="#d95f0e",
                edgecolors="white",
                linewidths=0.8,
                zorder=5,
                label=bl_label,
            )

    ax.set_xlabel(r"$\epsilon$ (eps)", fontsize=12)
    ax.set_ylabel("defense rate", fontsize=12)
    ax.set_title("mask token attack", fontsize=13)

    mean_all = df["平均值"].astype(float)
    std_all = df["标准差"].astype(float).fillna(0.0)
    lo = max(0.0, float(np.min(mean_all - std_all)) - 0.02)
    hi = min(1.0, float(np.max(mean_all + std_all)) + 0.02)
    ax.set_ylim(lo, hi)

    if len(df_f) > 0:
        xticks = list(df_f["eps"].astype(float).values)
        tick_labels = [str(int(t)) if abs(t - round(t)) < 1e-9 else str(t) for t in xticks]
        if len(df_i) > 0:
            x_inf_plot = max_eps + 2.0
            xticks = xticks + [x_inf_plot]
            tick_labels = tick_labels + [r"$\infty$"]
            ax.set_xlim(float(np.min(df_f["eps"].astype(float))) - 0.5, x_inf_plot + 1.0)
        ax.set_xticks(xticks)
        ax.set_xticklabels(tick_labels)
    elif len(df_i) > 0:
        ax.set_xticks([2.0])
        ax.set_xticklabels([r"$\infty$"])

    ax.grid(True, linestyle="--", alpha=0.35)
    if len(df_f) > 0 and len(df_i) > 0:
        ax.legend(loc="best", fontsize=9)
    elif len(df_i) > 0 and len(df_f) == 0:
        ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
