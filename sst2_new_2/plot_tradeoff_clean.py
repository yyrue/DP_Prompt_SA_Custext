#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制 Privacy-Utility Tradeoff 图（MTA 和 KNN 两张子图）

- x轴: Defense Rate (1 - Attack Accuracy)，越高隐私越好
- y轴: Utility (Downstream Accuracy)，越高效用越好
- 理想点在右上角（高隐私 + 高效用）

用法:
  python3 plot_tradeoff_clean.py
  python3 plot_tradeoff_clean.py --mix mix_0.0_20.0
  python3 plot_tradeoff_clean.py --mix mix_0.0_14.0 mix_0.0_20.0 mix_0.0_24.0
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mix", nargs="+", default=None,
                        help="要绘制的 mix 标签列表，如 mix_0.0_20.0；默认自动扫描全部")
    parser.add_argument("--out", type=str, default="./tradeoff_MTA_KNN.png")
    parser.add_argument("--no-errorbar", action="store_true")
    parser.add_argument("--annotate", action="store_true",
                        help="在每个点旁标注 eps 值")
    return parser.parse_args()


def load_custext_utility(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"平均值": "utility_mean", "标准差": "utility_std"})
    df["eps"] = pd.to_numeric(df["eps"], errors="coerce")
    df["utility_mean"] = pd.to_numeric(df["utility_mean"], errors="coerce")
    df["utility_std"] = pd.to_numeric(df["utility_std"], errors="coerce").fillna(0)
    return df.dropna(subset=["eps", "utility_mean"]).sort_values("eps")


def load_custext_mta_privacy(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "defense_rate_mean" not in df.columns:
        if "平均值" in df.columns:
            df = df.rename(columns={"平均值": "defense_rate_mean", "标准差": "defense_rate_std"})
        else:
            raise ValueError(f"Cannot parse MTA privacy CSV: {csv_path}")
    df["eps"] = pd.to_numeric(df["eps"], errors="coerce")
    df["defense_rate_mean"] = pd.to_numeric(df["defense_rate_mean"], errors="coerce")
    df["defense_rate_std"] = pd.to_numeric(df["defense_rate_std"], errors="coerce").fillna(0)
    return df.dropna(subset=["eps", "defense_rate_mean"]).sort_values("eps")


def load_custext_knn_privacy(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["eps"] = pd.to_numeric(df["eps"], errors="coerce")
    df["top10_accuracy_mean"] = pd.to_numeric(df["top10_accuracy_mean"], errors="coerce")
    df["top10_accuracy_std"] = pd.to_numeric(df["top10_accuracy_std"], errors="coerce").fillna(0)
    df["defense_rate_mean"] = 1.0 - df["top10_accuracy_mean"]
    df["defense_rate_std"] = df["top10_accuracy_std"]
    return df.dropna(subset=["eps", "defense_rate_mean"]).sort_values("eps")


def load_mixed_utility(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"平均值": "utility_mean", "标准差": "utility_std"})
    df["eps"] = pd.to_numeric(df["eps"], errors="coerce")
    df["utility_mean"] = pd.to_numeric(df["utility_mean"], errors="coerce")
    df["utility_std"] = pd.to_numeric(df["utility_std"], errors="coerce").fillna(0)
    # 提取 mix_base
    df["mix_base"] = df["mix_dir"].apply(
        lambda x: re.search(r"mix_[0-9.]+_[0-9.]+", str(x)).group(0)
        if re.search(r"mix_[0-9.]+_[0-9.]+", str(x)) else str(x)
    )
    return df.dropna(subset=["eps", "utility_mean"]).sort_values(["mix_base", "eps"])


def load_mixed_mta_privacy(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["eps"] = pd.to_numeric(df["eps_prime"], errors="coerce")
    df["defense_rate_mean"] = pd.to_numeric(df["defense_rate_mean"], errors="coerce")
    df["defense_rate_std"] = pd.to_numeric(df["defense_rate_std"], errors="coerce").fillna(0)
    df["mix_base"] = df["mix_tag"].apply(
        lambda x: re.search(r"mix_[0-9.]+_[0-9.]+", str(x)).group(0)
        if re.search(r"mix_[0-9.]+_[0-9.]+", str(x)) else str(x)
    )
    return df.dropna(subset=["eps", "defense_rate_mean"]).sort_values(["mix_base", "eps"])


def load_mixed_knn_privacy(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["eps"] = pd.to_numeric(df["eps"], errors="coerce")
    df["top10_accuracy_mean"] = pd.to_numeric(df["top10_accuracy_mean"], errors="coerce")
    df["top10_accuracy_std"] = pd.to_numeric(df["top10_accuracy_std"], errors="coerce").fillna(0)
    df["defense_rate_mean"] = 1.0 - df["top10_accuracy_mean"]
    df["defense_rate_std"] = df["top10_accuracy_std"]
    df["mix_base"] = df["mix_dir"].apply(
        lambda x: re.search(r"mix_[0-9.]+_[0-9.]+", str(x)).group(0)
        if re.search(r"mix_[0-9.]+_[0-9.]+", str(x)) else str(x)
    )
    return df.dropna(subset=["eps", "defense_rate_mean"]).sort_values(["mix_base", "eps"])


def merge_tradeoff(utility_df, privacy_df, on_cols=("eps",)):
    """合并 utility 和 privacy，取交集 eps"""
    merged = pd.merge(
        utility_df[list(on_cols) + ["utility_mean", "utility_std"]],
        privacy_df[list(on_cols) + ["defense_rate_mean", "defense_rate_std"]],
        on=list(on_cols),
        how="inner",
    )
    return merged.sort_values(list(on_cols))


def plot_series(ax, defense_rate, utility, eps_vals, label, color, marker,
                defense_std=None, utility_std=None, no_errorbar=False, annotate=False):
    """绘制一条 tradeoff 曲线"""
    x = np.asarray(defense_rate)
    y = np.asarray(utility)

    if no_errorbar or defense_std is None:
        ax.plot(x, y, marker=marker, color=color, linewidth=2, markersize=5, label=label)
    else:
        ax.errorbar(x, y,
                    xerr=np.asarray(defense_std),
                    yerr=np.asarray(utility_std),
                    fmt=f"{marker}-", color=color, linewidth=1.8, markersize=5,
                    capsize=2, capthick=1, label=label)

    if annotate and eps_vals is not None:
        for xi, yi, ep in zip(x, y, eps_vals):
            ax.annotate(f"{ep:g}", (xi, yi), textcoords="offset points",
                        xytext=(4, 4), fontsize=7, alpha=0.8)


def main():
    args = parse_args()
    root = Path("/data/youyaru/CusText-main/CusText/sst2_new_2")

    # ===== 加载 CusText 数据 =====
    custext_utility = load_custext_utility(root / "experiment_results" / "results_statistics_topk_200.csv")

    # MTA privacy
    custext_mta = load_custext_mta_privacy(root / "attack_results" / "attack_summary_by_eps.csv")

    # KNN privacy
    custext_knn = load_custext_knn_privacy(root / "knn_attack_results" / "custext" / "knn_attack_statistics.csv")

    # ===== 加载 Mixed 数据 =====
    mixed_utility = load_mixed_utility(root / "experiment_results_sa" / "results_statistics_all.csv")
    mixed_mta = load_mixed_mta_privacy(root / "attack_results_mixed" / "attack_summary_all_mixes_by_eps_prime.csv")
    mixed_knn = load_mixed_knn_privacy(root / "knn_attack_results" / "mixed" / "knn_attack_statistics_all.csv")

    # ===== 确定要画的 mix 组 =====
    available_mixes = sorted(mixed_utility["mix_base"].unique())
    if args.mix:
        selected_mixes = [m for m in args.mix if m in available_mixes]
    else:
        selected_mixes = available_mixes

    if not selected_mixes:
        print(f"未找到匹配的 mix 数据。可用: {available_mixes}")
        return

    print(f"绘制 CusText + {selected_mixes}")

    # ===== 合并 tradeoff =====
    custext_mta_tradeoff = merge_tradeoff(custext_utility, custext_mta)
    custext_knn_tradeoff = merge_tradeoff(custext_utility, custext_knn)

    # ===== 配色 =====
    colors = plt.cm.tab10.colors
    markers = ["s", "^", "D", "v", "P", "X", "h", "<", ">"]

    # ===== 绘图 =====
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- 左图: MTA Tradeoff ---
    ax_mta = axes[0]
    ax_mta.set_title("Privacy-Utility Tradeoff (Mask Token Attack)-SST2")
    ax_mta.set_xlabel("Privacy (Defense Rate)")
    ax_mta.set_ylabel("Utility (Accuracy)")

    if not custext_mta_tradeoff.empty:
        plot_series(ax_mta,
                    custext_mta_tradeoff["defense_rate_mean"],
                    custext_mta_tradeoff["utility_mean"],
                    custext_mta_tradeoff["eps"],
                    label="CusText",
                    color=colors[0], marker="o",
                    defense_std=custext_mta_tradeoff["defense_rate_std"],
                    utility_std=custext_mta_tradeoff["utility_std"],
                    no_errorbar=args.no_errorbar,
                    annotate=args.annotate)

    for idx, mix_name in enumerate(selected_mixes):
        mix_u = mixed_utility[mixed_utility["mix_base"] == mix_name]
        mix_p = mixed_mta[mixed_mta["mix_base"] == mix_name]
        mix_tradeoff = merge_tradeoff(mix_u, mix_p)
        if mix_tradeoff.empty:
            continue
        plot_series(ax_mta,
                    mix_tradeoff["defense_rate_mean"],
                    mix_tradeoff["utility_mean"],
                    mix_tradeoff["eps"],
                    label=f"SA ({mix_name})",
                    color=colors[(idx + 1) % len(colors)],
                    marker=markers[idx % len(markers)],
                    defense_std=mix_tradeoff["defense_rate_std"],
                    utility_std=mix_tradeoff["utility_std"],
                    no_errorbar=args.no_errorbar,
                    annotate=args.annotate)

    ax_mta.set_xlim(0.5, 1.0)
    ax_mta.set_ylim(0.5, 1.0)
    ax_mta.legend(loc="lower right")
    ax_mta.grid(True, alpha=0.3)

    # --- 右图: KNN Tradeoff ---
    ax_knn = axes[1]
    ax_knn.set_title("Privacy-Utility Tradeoff (KNN Attack)-SST2")
    ax_knn.set_xlabel("Privacy(Defense Rate)")
    ax_knn.set_ylabel("Utility (Accuracy)")

    if not custext_knn_tradeoff.empty:
        plot_series(ax_knn,
                    custext_knn_tradeoff["defense_rate_mean"],
                    custext_knn_tradeoff["utility_mean"],
                    custext_knn_tradeoff["eps"],
                    label="CusText",
                    color=colors[0], marker="o",
                    defense_std=custext_knn_tradeoff["defense_rate_std"],
                    utility_std=custext_knn_tradeoff["utility_std"],
                    no_errorbar=args.no_errorbar,
                    annotate=args.annotate)

    for idx, mix_name in enumerate(selected_mixes):
        mix_u = mixed_utility[mixed_utility["mix_base"] == mix_name]
        mix_p = mixed_knn[mixed_knn["mix_base"] == mix_name]
        mix_tradeoff = merge_tradeoff(mix_u, mix_p)
        if mix_tradeoff.empty:
            continue
        plot_series(ax_knn,
                    mix_tradeoff["defense_rate_mean"],
                    mix_tradeoff["utility_mean"],
                    mix_tradeoff["eps"],
                    label=f"SA ({mix_name})",
                    color=colors[(idx + 1) % len(colors)],
                    marker=markers[idx % len(markers)],
                    defense_std=mix_tradeoff["defense_rate_std"],
                    utility_std=mix_tradeoff["utility_std"],
                    no_errorbar=args.no_errorbar,
                    annotate=args.annotate)

    ax_knn.set_xlim(0.0, 1.0)
    ax_knn.set_ylim(0.5, 1.0)
    ax_knn.legend(loc="lower right")
    ax_knn.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    print(f"图已保存到: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
