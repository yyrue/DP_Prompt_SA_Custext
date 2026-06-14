#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汇总 knn_attack_results/ 下各次 KNN attack 的 Top-k accuracy。

用法:
  python collect_knn_attack_results.py
  python collect_knn_attack_results.py --result_dir ./knn_attack_results/mixed
"""

import argparse
import re
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

MIX_LOG_RE = re.compile(
    r"^knn_mixed_([\d.]+)_([\d.]+)_"
)
MIX_PARAM_RE = re.compile(r"^\s*mix=([\d.]+)_([\d.]+)\s*$")


def parse_knn_txt(path: Path) -> dict:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            m = re.match(r"(\w+):\s*(.+)", line)
            if m:
                key, val = m.group(1), m.group(2)
            else:
                pm = MIX_PARAM_RE.match(line)
                if pm:
                    out["eps_low"] = float(pm.group(1))
                    out["eps_high"] = float(pm.group(2))
                    continue
                m = re.match(r"(\w+)=(.+)", line.strip())
                if not m:
                    continue
                key, val = m.group(1), m.group(2)
            try:
                if "." in val:
                    out[key] = float(val)
                else:
                    out[key] = int(val)
            except ValueError:
                out[key] = val
    out["log_file"] = path.name
    return out


def infer_mix_range(row: dict) -> Tuple[Optional[float], Optional[float]]:
    if "eps_low" in row and "eps_high" in row:
        return float(row["eps_low"]), float(row["eps_high"])
    mix_val = row.get("mix")
    if isinstance(mix_val, str) and "_" in mix_val:
        a, b = mix_val.split("_", 1)
        return float(a), float(b)
    log_file = row.get("log_file", "")
    m = MIX_LOG_RE.match(str(log_file))
    if m:
        return float(m.group(1)), float(m.group(2))
    return None, None


def mix_dir_name(eps_low: float, eps_high: float) -> str:
    """与 privatized_dataset_mixed / experiment_results_sa 目录命名一致。"""
    def fmt(x: float) -> str:
        if abs(x - round(x)) < 1e-9:
            return f"{x:.1f}"
        return str(x)

    return f"mix_{fmt(eps_low)}_{fmt(eps_high)}"


def build_stats_df(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("eps", dropna=False).agg(
        top10_accuracy_mean=("top10_accuracy", "mean"),
        top10_accuracy_std=("top10_accuracy", "std"),
        top10_accuracy_changed_mean=("top10_accuracy_changed", "mean"),
        n_runs=("top10_accuracy", "count"),
    )
    out = g.reset_index()
    out["top10_accuracy_std"] = out["top10_accuracy_std"].fillna(0.0)
    return out.sort_values("eps").reset_index(drop=True)


def discover_txt_files(result_dir: Path):
    for p in sorted(result_dir.glob("knn_*.txt")):
        if p.is_file():
            yield p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, default="./knn_attack_results")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    rows = []
    for fp in discover_txt_files(result_dir):
        row = parse_knn_txt(fp)
        if "top10_accuracy" not in row:
            continue
        lo, hi = infer_mix_range(row)
        if lo is not None and hi is not None:
            row["eps_low"] = lo
            row["eps_high"] = hi
            row["mix_dir"] = mix_dir_name(lo, hi)
        rows.append(row)
        tag = f" [{row['mix_dir']}]" if row.get("mix_dir") else ""
        print(f"  {fp.name}: top10={row.get('top10_accuracy'):.4f}{tag}")

    if not rows:
        print("未找到 knn_*.txt 结果")
        return

    df = pd.DataFrame(rows)
    out = args.out or str(result_dir / "knn_attack_summary.csv")
    df.to_csv(out, index=False)
    print(f"\n已保存明细: {out}")

    if "eps" not in df.columns:
        return

    has_mix = "mix_dir" in df.columns and df["mix_dir"].notna().any()

    if has_mix:
        mix_groups = df.groupby("mix_dir", sort=True)
        all_stats = []
        for mix_name, part in mix_groups:
            subdir = result_dir / mix_name
            subdir.mkdir(parents=True, exist_ok=True)

            summary_path = subdir / "knn_attack_summary.csv"
            part.sort_values(["eps", "seed"]).to_csv(summary_path, index=False)

            stats = build_stats_df(part)
            stats.insert(0, "eps_low", part["eps_low"].iloc[0])
            stats.insert(1, "eps_high", part["eps_high"].iloc[0])
            stats_path = subdir / "knn_attack_statistics.csv"
            stats.to_csv(stats_path, index=False)

            print(f"\n{mix_name}:")
            print(f"  明细: {summary_path} ({len(part)} 条)")
            print(f"  统计: {stats_path}")
            print(stats.to_string(index=False))
            s = stats.copy()
            s["mix_dir"] = mix_name
            all_stats.append(s)

        combined_stats = pd.concat(all_stats, ignore_index=True)
        all_path = result_dir / "knn_attack_statistics_all.csv"
        combined_stats.to_csv(all_path, index=False)
        print(f"\n全部 mix 区间汇总: {all_path}")

        # 不再写根目录下仅按 eps 聚合的 knn_attack_statistics.csv（会混淆不同 mix）
        legacy = result_dir / "knn_attack_statistics.csv"
        if legacy.is_file():
            legacy.unlink()
            print(f"已移除易混淆的合并统计: {legacy}")
    else:
        stats = build_stats_df(df)
        stats_path = result_dir / "knn_attack_statistics.csv"
        stats.to_csv(stats_path, index=False)
        print(f"\n按 eps 统计: {stats_path}")
        print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
