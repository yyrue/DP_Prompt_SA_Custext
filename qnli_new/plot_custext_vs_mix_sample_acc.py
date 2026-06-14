#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custext（acc vs ε）与 mix sample amplification、无脱敏 baseline 对比图。

- 横轴：ε（CusText / mix 用 results_statistics 中的 eps；baseline 为 ∞）
- 纵轴：acc（CSV 列「平均值」，误差棒为「标准差」）
- CusText：experiment_results/results_statistics_topk_20.csv（不绘制 eps=3）
- Baseline（no sanitized）：experiment_results/results_baseline_statistics.csv（eps=inf）
- Mix：experiment_results_sa/mix_*/results_statistics_topk_20.csv（可多组）

用法:
  python3 plot_custext_vs_mix_sample_acc.py
  python3 plot_custext_vs_mix_sample_acc.py --out ./plot_custext_vs_mix_sample_acc.png
"""

import argparse
import math
import os
import re
from pathlib import Path
from typing import List, Optional

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


def _fmt_eps_num(x: float) -> str:
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return str(x)


def mix_folder_to_label(folder_name: str) -> str:
    """mix_0.0_14.0_topk_20 -> mix_0_14。"""
    m = re.match(r"^mix_([\d.]+)_([\d.]+)", folder_name)
    if not m:
        return folder_name
    lo, hi = float(m.group(1)), float(m.group(2))
    return f"mix_{_fmt_eps_num(lo)}_{_fmt_eps_num(hi)}"


def discover_mix_csvs(mixed_root: Path) -> List[Path]:
    paths = sorted(mixed_root.glob("mix_*/results_statistics_topk_20.csv"))
    return [p for p in paths if p.is_file()]


def load_stats_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    req = {"eps", "平均值", "标准差"}
    if not req.issubset(df.columns):
        raise SystemExit(f"{path} 需包含列: {req}")
    return df


DEFAULT_FINITE_XTICKS = (0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20)


def main():
    root = Path(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description="Custext vs mix sample: acc vs eps")
    parser.add_argument(
        "--custext",
        type=str,
        default=str(root / "experiment_results" / "results_statistics_topk_20.csv"),
        help="CusText 统计 CSV",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=str(root / "experiment_results" / "results_baseline_statistics.csv"),
        help="无脱敏 baseline 统计 CSV（eps=inf）",
    )
    parser.add_argument(
        "--mixed-root",
        type=str,
        default=str(root / "experiment_results_sa"),
        help="含 mix_* 子目录的根路径",
    )
    parser.add_argument(
        "--mix-csv",
        type=str,
        nargs="*",
        default=None,
        help="显式指定若干 results_statistics_topk_20.csv；默认扫描 mixed-root",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(root / "plot_custext_vs_mix_sample_acc.png"),
        help="输出图片路径",
    )
    parser.add_argument("--no-baseline", action="store_true", help="不画 no sanitized 点")
    parser.add_argument("--no-errorbar", action="store_true", help="不画标准差误差棒")
    args = parser.parse_args()

    custext_path = Path(args.custext).resolve()
    if not custext_path.is_file():
        raise SystemExit(f"未找到 CusText 统计文件: {custext_path}")

    df_c = load_stats_csv(custext_path)
    inf_m = _is_inf_eps(df_c["eps"])
    df_cf = df_c[~inf_m].copy()
    # CusText 曲线不展示 eps=3（与 mix 等设置对齐或按需省略该点）
    eps_f = pd.to_numeric(df_cf["eps"], errors="coerce")
    df_cf = df_cf[~np.isclose(eps_f, 3.0, rtol=0, atol=1e-9)].copy()

    baseline_row: Optional[pd.Series] = None
    baseline_path = Path(args.baseline).resolve()
    if not args.no_baseline and baseline_path.is_file():
        df_b = load_stats_csv(baseline_path)
        inf_b = _is_inf_eps(df_b["eps"])
        if inf_b.any():
            baseline_row = df_b[inf_b].iloc[0]
        elif len(df_b) == 1:
            baseline_row = df_b.iloc[0]
    elif not args.no_baseline:
        print(f"提示: 未找到 baseline 文件 {baseline_path}，跳过 no sanitized")

    mixed_root = Path(args.mixed_root).resolve()
    if args.mix_csv:
        mix_paths = [Path(p).resolve() for p in args.mix_csv]
    else:
        mix_paths = discover_mix_csvs(mixed_root)
    if not mix_paths:
        raise SystemExit(
            f"未找到 mix 统计 CSV（{mixed_root}/mix_*/results_statistics_topk_20.csv）"
        )

    fig, ax = plt.subplots(figsize=(9, 5))

    max_x = 0.0
    x_inf_plot: Optional[float] = None
    color_custext = "#1b7837"
    color_inf = "#762a83"

    if len(df_cf) > 0:
        eps = df_cf["eps"].astype(float).values
        y = df_cf["平均值"].astype(float).values
        yerr = df_cf["标准差"].astype(float).values
        max_x = max(max_x, float(np.nanmax(eps)))
        if not args.no_errorbar:
            ax.errorbar(
                eps,
                y,
                yerr=yerr,
                fmt="o-",
                capsize=3,
                markersize=6,
                linewidth=2.0,
                color=color_custext,
                ecolor="#a6dba0",
                elinewidth=1,
                label="CusText",
                zorder=4,
            )
        else:
            ax.plot(
                eps,
                y,
                "o-",
                markersize=6,
                linewidth=2.0,
                color=color_custext,
                label="CusText",
                zorder=4,
            )

    if baseline_row is not None:
        yb = float(baseline_row["平均值"])
        yerr_b = float(baseline_row["标准差"])
        if math.isnan(yerr_b):
            yerr_b = 0.0
        x_inf = max_x + 2.0 if max_x > 0 else 2.0
        x_inf_plot = x_inf
        if not args.no_errorbar and yerr_b > 0:
            ax.errorbar(
                [x_inf],
                [yb],
                yerr=[yerr_b],
                fmt="s",
                capsize=4,
                markersize=7,
                color=color_inf,
                ecolor="#c2a5cf",
                elinewidth=1.2,
                label="no sanitized",
                zorder=5,
            )
        else:
            ax.scatter(
                [x_inf],
                [yb],
                s=70,
                marker="s",
                color=color_inf,
                edgecolors="white",
                linewidths=0.8,
                zorder=5,
                label="no sanitized",
            )
        max_x = max(max_x, x_inf)

    cmap = plt.get_cmap("tab10")
    for i, csv_path in enumerate(mix_paths):
        if not csv_path.is_file():
            raise SystemExit(f"文件不存在: {csv_path}")
        dm = load_stats_csv(csv_path)
        inf_m = _is_inf_eps(dm["eps"])
        dm = dm[~inf_m]
        folder = csv_path.parent.name
        label = mix_folder_to_label(folder)
        xp = dm["eps"].astype(float).values
        yp = dm["平均值"].astype(float).values
        err = dm["标准差"].astype(float).values
        if len(xp):
            max_x = max(max_x, float(np.nanmax(xp)))
        c = cmap(i % 10)
        if not args.no_errorbar:
            ax.errorbar(
                xp,
                yp,
                yerr=err,
                fmt="o-",
                capsize=2.5,
                markersize=5,
                linewidth=1.4,
                color=c,
                ecolor=c,
                alpha=0.95,
                elinewidth=0.9,
                label=label,
                zorder=3,
            )
        else:
            ax.plot(xp, yp, "o-", markersize=5, linewidth=1.4, color=c, label=label, zorder=3)

    ax.set_xlabel(r"$\epsilon$ (eps)", fontsize=12)
    ax.set_ylabel("acc", fontsize=12)
    ax.set_title("custext vs mixed sample", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best", fontsize=9)

    all_y: List[float] = []
    if len(df_cf) > 0:
        all_y.extend(df_cf["平均值"].astype(float).tolist())
    if baseline_row is not None:
        all_y.append(float(baseline_row["平均值"]))
    for csv_path in mix_paths:
        dm = load_stats_csv(csv_path)
        inf_m = _is_inf_eps(dm["eps"])
        all_y.extend(dm.loc[~inf_m, "平均值"].astype(float).tolist())

    if all_y:
        lo = max(0.0, float(np.min(all_y)) - 0.04)
        hi = min(1.0, float(np.max(all_y)) + 0.04)
        if hi - lo < 0.08:
            mid = 0.5 * (lo + hi)
            lo, hi = max(0.0, mid - 0.2), min(1.0, mid + 0.2)
        ax.set_ylim(lo, hi)

    ax.set_xlim(-0.5, max_x + 0.8)

    tick_pos = [t for t in DEFAULT_FINITE_XTICKS if t <= max_x + 1e-9]
    tick_lbl = [str(int(t)) for t in tick_pos]
    if x_inf_plot is not None:
        tick_pos.append(x_inf_plot)
        tick_lbl.append(r"$\infty$")
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lbl)

    fig.tight_layout()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
