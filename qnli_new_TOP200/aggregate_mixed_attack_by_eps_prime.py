#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汇总 attack_results_mixed 下各 mix_* 子目录中、不同 seed 的 mask token attack 结果。

对每个子目录：
  - 读取所有 attack_summary_mixed_*_seed_*.csv
  - 按 eps_prime（混合目标预算 ε'）分组
  - 输出 attack_accuracy 在 seed 上的均值与标准差，以及 defense_rate_mean = 1 - attack_accuracy_mean

用法:
  python aggregate_mixed_attack_by_eps_prime.py
  python aggregate_mixed_attack_by_eps_prime.py --root ./attack_results_mixed
"""

import argparse
import os
import re
from pathlib import Path
from typing import List, Optional

import pandas as pd

SUMMARY_GLOB = "attack_summary_mixed_*_seed_*.csv"
SEED_RE = re.compile(r"_seed_(\d+)\.csv$")
OUT_NAME = "attack_summary_by_eps_prime.csv"


def _parse_seed(path: Path) -> Optional[int]:
    m = SEED_RE.search(path.name)
    return int(m.group(1)) if m else None


def aggregate_one_mix_dir(mix_dir: Path) -> Optional[pd.DataFrame]:
    files = sorted(mix_dir.glob(SUMMARY_GLOB))
    if not files:
        return None

    frames = []
    for fp in files:
        seed = _parse_seed(fp)
        if seed is None:
            continue
        df = pd.read_csv(fp)
        if "eps_prime" not in df.columns or "attack_accuracy" not in df.columns:
            continue
        d = df.copy()
        if "seed" not in d.columns:
            d["seed"] = seed
        frames.append(d)

    if not frames:
        return None

    all_df = pd.concat(frames, ignore_index=True)
    all_df["eps_prime"] = pd.to_numeric(all_df["eps_prime"], errors="coerce")
    all_df["attack_accuracy"] = pd.to_numeric(all_df["attack_accuracy"], errors="coerce")
    all_df = all_df.dropna(subset=["eps_prime", "attack_accuracy"])

    grouped = (
        all_df.groupby("eps_prime", as_index=False)
        .agg(
            attack_accuracy_mean=("attack_accuracy", "mean"),
            attack_accuracy_std=("attack_accuracy", "std"),
            n_seeds=("seed", "count"),
        )
        .sort_values("eps_prime")
        .reset_index(drop=True)
    )
    grouped["attack_accuracy_std"] = grouped["attack_accuracy_std"].fillna(0.0)
    grouped["defense_rate_mean"] = 1.0 - grouped["attack_accuracy_mean"]
    grouped["defense_rate_std"] = grouped["attack_accuracy_std"]

    for col in ("eps_low", "eps_high", "mix_tag"):
        if col in all_df.columns:
            grouped[col] = all_df[col].iloc[0]

    out_path = mix_dir / OUT_NAME
    grouped.to_csv(out_path, index=False)
    print(
        f"  [{mix_dir.name}] 写入 {out_path}（{len(frames)} 个 seed 汇总 → {len(grouped)} 行 eps_prime）"
    )
    return grouped


def main():
    parser = argparse.ArgumentParser(description="按 eps_prime 汇总 mixed attack 多 seed 结果")
    default_root = Path(os.path.dirname(os.path.abspath(__file__))) / "attack_results_mixed"
    parser.add_argument(
        "--root",
        type=str,
        default=str(default_root),
        help="attack_results_mixed 根目录",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="只处理名为该字符串的子目录（例如 mix_0.0_14.0）",
    )
    args = parser.parse_args()
    root_path = Path(args.root).resolve()
    if not root_path.is_dir():
        raise SystemExit(f"根目录不存在: {root_path}")

    subdirs = sorted(
        p for p in root_path.iterdir() if p.is_dir() and p.name.startswith("mix_")
    )
    if args.only:
        only_path = root_path / args.only
        if not only_path.is_dir():
            raise SystemExit(f"--only 目录不存在: {only_path}")
        subdirs = [only_path]

    if not subdirs:
        raise SystemExit(f"未找到 mix_* 子目录: {root_path}")

    all_parts: List[pd.DataFrame] = []
    print(f"扫描: {root_path}，共 {len(subdirs)} 个子目录\n")
    for d in subdirs:
        stats = aggregate_one_mix_dir(d)
        if stats is not None:
            stats = stats.copy()
            stats["mix_folder"] = d.name
            all_parts.append(stats)

    if all_parts and not args.only:
        combined = pd.concat(all_parts, ignore_index=True)
        combined_path = root_path / "attack_summary_all_mixes_by_eps_prime.csv"
        combined.to_csv(combined_path, index=False)
        print(f"\n合并表（含 mix_folder）: {combined_path}（{len(combined)} 行）")


if __name__ == "__main__":
    main()
