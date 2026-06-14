#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 results_statistics_*.csv 绘制 Custext utility（acc vs ε）；无脱敏 baseline 在横轴
以单独一点表示（刻度为 ∞），不画整条水平线。

用法:
  python3 plot_custext_acc_vs_eps.py
  python3 collect_baseline_statistics.py && python3 plot_custext_acc_vs_eps.py
"""

import argparse
import csv
import math
import os

import matplotlib.pyplot as plt
import numpy as np


def load_statistics_csv(path: str):
    eps_list, mean_acc, std_acc = [], [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            eps_list.append(float(row["eps"]))
            mean_acc.append(float(row["平均值"]))
            std_acc.append(float(row["标准差"]))
    return np.array(eps_list), np.array(mean_acc), np.array(std_acc)


def load_baseline_statistics_csv(path: str):
    """读取 collect_baseline_statistics.py 生成的 results_baseline_statistics.csv。"""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        row = next(reader)
    mean_acc = float(row["平均值"])
    std_acc = float(row["标准差"])
    return mean_acc, std_acc


def x_ticks_for_eps(max_eps: float):
    """0–4 步长 1，之后步长 2，直到覆盖 max_eps。"""
    ticks = [0.0, 1.0, 2.0, 4.0]
    e = 6.0
    while e <= max_eps + 1e-9:
        ticks.append(e)
        e += 2.0
    return ticks


def main():
    parser = argparse.ArgumentParser(description="Custext utility: acc vs eps")
    default_csv = os.path.join(
        os.path.dirname(__file__),
        "experiment_results",
        "results_statistics_topk_100.csv",
    )
    default_baseline = os.path.join(
        os.path.dirname(__file__),
        "experiment_results",
        "results_baseline_statistics.csv",
    )
    parser.add_argument("--csv", type=str, default=default_csv, help="Custext 统计结果 CSV")
    parser.add_argument(
        "--baseline-csv",
        type=str,
        default=default_baseline,
        help="baseline 统计 CSV（不存在则只画 Custext 曲线）",
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="不绘制无脱敏 baseline 点",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="输出图片路径（默认与 Custext CSV 同目录）",
    )
    parser.add_argument("--no-errorbar", action="store_true", help="不画标准差误差棒")
    args = parser.parse_args()

    eps, acc_mean, acc_std = load_statistics_csv(args.csv)
    max_eps = float(np.max(eps))

    baseline_mean = baseline_std = None
    if not args.no_baseline and os.path.isfile(args.baseline_csv):
        try:
            baseline_mean, baseline_std = load_baseline_statistics_csv(args.baseline_csv)
        except (StopIteration, KeyError, ValueError) as e:
            print(f"警告: 无法读取 baseline CSV ({args.baseline_csv}): {e}")
    elif not args.no_baseline and not os.path.isfile(args.baseline_csv):
        print(f"提示: 未找到 {args.baseline_csv}，跳过 baseline。可运行: python3 collect_baseline_statistics.py")

    out_path = args.out
    if not out_path:
        base = os.path.splitext(os.path.basename(args.csv))[0]
        out_path = os.path.join(
            os.path.dirname(os.path.abspath(args.csv)),
            f"plot_{base}_custext_utility.png",
        )

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    if not args.no_errorbar:
        ax.errorbar(
            eps,
            acc_mean,
            yerr=acc_std,
            fmt="o-",
            capsize=3,
            markersize=6,
            linewidth=1.5,
            color="#2c7fb8",
            ecolor="#7fcdbb",
            elinewidth=1,
            label="Custext",
        )
    else:
        ax.plot(eps, acc_mean, "o-", markersize=6, linewidth=1.5, color="#2c7fb8", label="Custext")

    # 无脱敏：在数据域用 max_eps+2 作为横坐标，刻度标为 ∞（线性轴无法真正画 x=inf）
    x_inf_plot = max_eps + 2.0
    if baseline_mean is not None:
        bl_label = r"No sanitized ($\infty$)"
        if baseline_std and baseline_std > 0 and not math.isnan(baseline_std):
            ax.errorbar(
                [x_inf_plot],
                [baseline_mean],
                yerr=[baseline_std],
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
                [baseline_mean],
                s=55,
                color="#d95f0e",
                edgecolors="white",
                linewidths=0.8,
                zorder=5,
                label=bl_label,
            )

    ax.set_xlabel(r"$\epsilon$ (eps)", fontsize=12)
    ax.set_ylabel("acc", fontsize=12)
    ax.set_title("Custext utility", fontsize=13)
    y_low = float(np.min(acc_mean))
    y_high = float(np.max(acc_mean))
    if baseline_mean is not None:
        bs = baseline_std if baseline_std and not math.isnan(baseline_std) else 0.0
        y_low = min(y_low, baseline_mean - bs)
        y_high = max(y_high, baseline_mean + bs)
    pad = 0.03
    ax.set_ylim(max(0.0, y_low - pad), min(1.0, y_high + pad))

    xticks = [t for t in x_ticks_for_eps(max_eps) if t <= max_eps + 1e-9]
    tick_labels = [str(int(t)) if t == int(t) else str(t) for t in xticks]
    if baseline_mean is not None:
        xticks = xticks + [x_inf_plot]
        tick_labels = tick_labels + [r"$\infty$"]
        ax.set_xlim(float(np.min(eps)) - 0.5, x_inf_plot + 1.0)
    ax.set_xticks(xticks)
    ax.set_xticklabels(tick_labels)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
