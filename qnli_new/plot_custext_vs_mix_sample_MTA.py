#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custext（单 eps 脱敏）与 mix sample amplification 的 defense rate 对比图。

- 横轴：ε（CusText 用汇总表中的 eps；mix 用 eps_prime）
- 纵轴：defense rate（CusText 用「平均值」列；mix 用 defense_rate_mean）
- CusText：attack_results/attack_summary_by_eps.csv（aggregate_attack_summary_by_eps.py 产出）
- Mix：attack_results_mixed/mix_*/attack_summary_by_eps_prime.csv（可多组）
- 横轴主刻度固定为 0,1,2,3,4,6,8,10,12,14,16,18,20（无 5）；若有 no sanitized 点则右侧追加 ∞

用法:
  python3 plot_custext_vs_mix_sample.py
  python3 plot_custext_vs_mix_sample.py --out ./attack_results_mixed/plot_custext_vs_mix_sample.png
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
    """mix_0.0_14.0 -> mix_0_14（整数端点用整数显示）。"""
    m = re.match(r"^mix_([\d.]+)_([\d.]+)$", folder_name)
    if not m:
        return folder_name
    lo, hi = float(m.group(1)), float(m.group(2))
    return f"mix_{_fmt_eps_num(lo)}_{_fmt_eps_num(hi)}"


def discover_mix_csvs(mixed_root: Path) -> List[Path]:
    paths = sorted(mixed_root.glob("mix_*/attack_summary_by_eps_prime.csv"))
    return [p for p in paths if p.is_file()]


DEFAULT_FINITE_XTICKS = (0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20)


def main():
    root = Path(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description="Custext vs mix sample: defense rate vs eps")
    parser.add_argument(
        "--custext",
        type=str,
        default=str(root / "attack_results" / "attack_summary_by_eps.csv"),
        help="CusText 按 eps 汇总 CSV（含 eps, 平均值, 标准差）",
    )
    parser.add_argument(
        "--mixed-root",
        type=str,
        default=str(root / "attack_results_mixed"),
        help="含 mix_* 子目录的根路径",
    )
    parser.add_argument(
        "--mix-csv",
        type=str,
        nargs="*",
        default=None,
        help="显式指定若干 attack_summary_by_eps_prime.csv；默认自动扫描 mixed-root 下各 mix_*",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(root / "plot_custext_vs_mix_sample_MTA.png"),
        help="输出图片路径",
    )
    parser.add_argument("--no-errorbar", action="store_true", help="不画标准差误差棒")
    args = parser.parse_args()

    custext_path = Path(args.custext).resolve()
    if not custext_path.is_file():
        raise SystemExit(f"未找到 CusText 汇总文件: {custext_path}")

    df_c = pd.read_csv(custext_path)
    req = {"eps", "平均值", "标准差"}
    if not req.issubset(df_c.columns):
        raise SystemExit(f"CusText CSV 需包含列: {req}")

    inf_m = _is_inf_eps(df_c["eps"])
    df_cf = df_c[~inf_m].copy()
    df_ci = df_c[inf_m].copy()

    mixed_root = Path(args.mixed_root).resolve()
    if args.mix_csv:
        mix_paths = [Path(p).resolve() for p in args.mix_csv]
    else:
        mix_paths = discover_mix_csvs(mixed_root)
    if not mix_paths:
        raise SystemExit(f"未找到 mix 汇总 CSV（{mixed_root}/mix_*/attack_summary_by_eps_prime.csv）")

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
            ax.plot(eps, y, "o-", markersize=6, linewidth=2.0, color=color_custext, label="CusText", zorder=4)

    if len(df_ci) > 0:
        row = df_ci.iloc[0]
        yb = float(row["平均值"])
        yerr_b = float(row["标准差"])
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
        dm = pd.read_csv(csv_path)
        need = {"eps_prime", "defense_rate_mean", "defense_rate_std"}
        if not need.issubset(dm.columns):
            raise SystemExit(f"{csv_path} 需包含列: {need}")
        folder = csv_path.parent.name
        label = mix_folder_to_label(folder)
        xp = dm["eps_prime"].astype(float).values
        yp = dm["defense_rate_mean"].astype(float).values
        err = dm["defense_rate_std"].astype(float).values
        max_x = max(max_x, float(np.nanmax(xp)) if len(xp) else 0.0)
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

    ax.set_xlabel(r"$\epsilon$ (eps / $\epsilon'$ for mix)", fontsize=12)
    ax.set_ylabel("defense rate", fontsize=12)
    ax.set_title("custext VS mix sample", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best", fontsize=9)

    all_y: List[float] = []
    if len(df_cf) > 0:
        all_y.extend(df_cf["平均值"].astype(float).tolist())
    if len(df_ci) > 0:
        all_y.append(float(df_ci.iloc[0]["平均值"]))
    for csv_path in mix_paths:
        dm = pd.read_csv(csv_path)
        all_y.extend(dm["defense_rate_mean"].astype(float).tolist())

    if all_y:
        lo = max(0.0, float(np.min(all_y)) - 0.04)
        hi = min(1.0, float(np.max(all_y)) + 0.04)
        if hi - lo < 0.08:
            mid = 0.5 * (lo + hi)
            lo, hi = max(0.0, mid - 0.2), min(1.0, mid + 0.2)
        ax.set_ylim(lo, hi)

    ax.set_xlim(-0.5, max_x + 0.8)

    tick_pos = list(DEFAULT_FINITE_XTICKS)
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
