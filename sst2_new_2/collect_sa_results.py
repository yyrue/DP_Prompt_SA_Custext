#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sample Amplification 实验结果收集脚本。

从 experiment_results_sa/mix_{eps_low}_{eps_high}_topk_{K}/ 下的
sa_eps_*_topk_*_seed_*.log 提取 test acc，汇总为与 experiment_results
（collect_results.py）相同格式的 CSV：

  - results_summary_topk_{K}.csv      明细：eps, top_k, seed, test_acc, log_file
  - results_statistics_topk_{K}.csv   统计：eps, 平均值, 标准差, 最小值, 最大值, 运行次数

其中 eps 列为目标隐私预算 ε'（日志文件名中的 sa_eps_{ε'}）。

用法:
  # 扫描 experiment_results_sa 下所有 mix_*_topk_* 子目录
  python collect_sa_results.py

  # 只处理某一个 mix 子目录
  python collect_sa_results.py --mix-dir ./experiment_results_sa/mix_0.0_14.0_topk_20

  # 手动指定参数（单目录，兼容旧用法）
  python collect_sa_results.py --eps_low 0 --eps_high 14 --top_k 20 --seeds 42,43
"""

import argparse
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

LOG_GLOB = "sa_eps_*_topk_*_seed_*.log"
MIX_DIR_RE = re.compile(r"^mix_([\d.]+)_([\d.]+)_topk_(\d+)$")
LOG_NAME_RE = re.compile(
    r"^sa_eps_(?P<eps>[\d.]+)_topk_(?P<topk>\d+)_seed_(?P<seed>\d+)\.log$"
)
TEST_ACC_RE = re.compile(r"test acc = (\d+\.\d+)")


def extract_test_acc(log_file: Path) -> Optional[float]:
    try:
        content = log_file.read_text(encoding="utf-8", errors="replace")
        matches = TEST_ACC_RE.findall(content)
        if not matches:
            return None
        return float(matches[-1])
    except OSError as e:
        print(f"  读取失败 {log_file}: {e}")
        return None


def parse_mix_dir_name(name: str) -> Optional[Tuple[float, float, int]]:
    m = MIX_DIR_RE.match(name)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2)), int(m.group(3))


def discover_mix_dirs(base_dir: Path) -> List[Path]:
    return sorted(
        p for p in base_dir.iterdir()
        if p.is_dir() and p.name.startswith("mix_") and "_topk_" in p.name
    )


def infer_from_logs(mix_dir: Path) -> Tuple[List[float], List[int], List[int]]:
    eps_set, seed_set, topk_set = set(), set(), set()
    for fp in mix_dir.glob(LOG_GLOB):
        m = LOG_NAME_RE.match(fp.name)
        if not m:
            continue
        eps_set.add(float(m.group("eps")))
        seed_set.add(int(m.group("seed")))
        topk_set.add(int(m.group("topk")))
    return sorted(eps_set), sorted(seed_set), sorted(topk_set)


def build_stats_df(df: pd.DataFrame) -> pd.DataFrame:
    stats = df.groupby("eps")["test_acc"].agg(["mean", "std", "min", "max", "count"])
    stats = stats.reset_index()
    stats.columns = ["eps", "平均值", "标准差", "最小值", "最大值", "运行次数"]
    stats["标准差"] = stats["标准差"].fillna(0.0)
    stats = stats.sort_values("eps").reset_index(drop=True)
    return stats


def collect_one_mix_dir(
    mix_dir: Path,
    eps_values: Optional[List[float]] = None,
    seeds: Optional[List[int]] = None,
    top_k: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    parsed = parse_mix_dir_name(mix_dir.name)
    if parsed is None:
        print(f"  [跳过] 无法解析目录名: {mix_dir.name}")
        return None
    eps_low, eps_high, dir_top_k = parsed
    top_k = top_k if top_k is not None else dir_top_k

    inferred_eps, inferred_seeds, inferred_topk = infer_from_logs(mix_dir)
    if top_k not in inferred_topk and inferred_topk:
        print(f"  [警告] 目录 topk={dir_top_k}，日志中 topk={inferred_topk}")

    if eps_values is None:
        eps_values = inferred_eps if inferred_eps else list(
            range(int(eps_low), int(eps_high) + 1, 2)
        )
    if seeds is None:
        seeds = inferred_seeds if inferred_seeds else list(range(42, 52))

    print(f"\n{'=' * 60}")
    print(f"收集: {mix_dir.name}  (mix {eps_low}–{eps_high}, top_k={top_k})")
    print(f"  eps (ε'): {eps_values}")
    print(f"  seeds:    {seeds}")
    print(f"{'=' * 60}")

    rows = []
    for eps in eps_values:
        for seed in seeds:
            candidates = [
                mix_dir / f"sa_eps_{int(eps) if eps == int(eps) else eps}_topk_{top_k}_seed_{seed}.log",
                mix_dir / f"sa_eps_{eps}_topk_{top_k}_seed_{seed}.log",
                mix_dir / f"sa_eps_{eps}_seed_{seed}.log",
            ]
            log_file = None
            for c in candidates:
                if c.is_file():
                    log_file = c
                    break
            if log_file is None:
                print(f"  ✗ eps={eps}, seed={seed}: 日志不存在")
                continue

            acc = extract_test_acc(log_file)
            if acc is None:
                hint = ""
                try:
                    head = log_file.read_text(encoding="utf-8", errors="replace")[:8000]
                    if "Traceback" in head or "OutOfMemoryError" in head or "CUDA out of memory" in head:
                        hint = "（日志含异常/中断，未完成训练则无 test acc）"
                except OSError:
                    pass
                print(f"  ✗ eps={eps}, seed={seed}: 未找到 test acc{hint} → {log_file.name}")
                continue

            rows.append({
                "eps": float(eps),
                "top_k": top_k,
                "seed": seed,
                "test_acc": acc,
                "log_file": log_file.name,
            })
            print(f"  ✓ eps={eps}, seed={seed}: test_acc={acc:.4f}")

    if not rows:
        print(f"  [{mix_dir.name}] 未收集到任何结果")
        return None

    df = pd.DataFrame(rows).sort_values(["eps", "seed"]).reset_index(drop=True)

    summary_file = mix_dir / f"results_summary_topk_{top_k}.csv"
    df.to_csv(summary_file, index=False)
    print(f"\n明细已保存: {summary_file}")

    stats = build_stats_df(df)
    stats_file = mix_dir / f"results_statistics_topk_{top_k}.csv"
    stats.to_csv(stats_file, index=False)
    print(f"统计已保存: {stats_file}")
    print(stats.to_string(index=False))

    return df


def main():
    root = Path(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(
        description="收集 SA 实验结果（格式与 experiment_results/collect_results.py 一致）"
    )
    parser.add_argument(
        "--base_result_dir",
        type=str,
        default=str(root / "experiment_results_sa"),
        help="SA 实验结果根目录",
    )
    parser.add_argument(
        "--mix-dir",
        type=str,
        default=None,
        help="只处理单个 mix_*_topk_* 子目录",
    )
    parser.add_argument("--eps_low", type=float, default=None, help="（单目录）混合下界")
    parser.add_argument("--eps_high", type=float, default=None, help="（单目录）混合上界")
    parser.add_argument("--top_k", type=int, default=None, help="top_k（默认从目录名读取）")
    parser.add_argument("--seeds", type=str, default=None, help="seed 列表，逗号分隔")
    parser.add_argument(
        "--eps_values",
        type=str,
        default=None,
        help="目标 ε' 列表，逗号分隔；不指定则从日志文件名自动发现",
    )
    args = parser.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",")] if args.seeds else None
    eps_values = (
        [float(x.strip()) for x in args.eps_values.split(",")]
        if args.eps_values
        else None
    )

    if args.mix_dir:
        mix_dirs = [Path(args.mix_dir).resolve()]
    elif args.eps_low is not None and args.eps_high is not None:
        tk = args.top_k if args.top_k is not None else 20
        mix_dirs = [
            Path(args.base_result_dir).resolve()
            / f"mix_{args.eps_low}_{args.eps_high}_topk_{tk}"
        ]
    else:
        base = Path(args.base_result_dir).resolve()
        if not base.is_dir():
            raise SystemExit(f"目录不存在: {base}")
        mix_dirs = discover_mix_dirs(base)

    if not mix_dirs:
        raise SystemExit("未找到 mix_*_topk_* 子目录")

    print(f"扫描 {len(mix_dirs)} 个子目录 …")
    all_parts = []
    for d in mix_dirs:
        if not d.is_dir():
            print(f"  [跳过] 不存在: {d}")
            continue
        parsed = parse_mix_dir_name(d.name)
        tk = args.top_k if args.top_k is not None else (parsed[2] if parsed else 20)
        df = collect_one_mix_dir(d, eps_values=eps_values, seeds=seeds, top_k=tk)
        if df is not None:
            df = df.copy()
            df["mix_dir"] = d.name
            all_parts.append(df)

    if len(all_parts) > 1:
        base = Path(args.base_result_dir).resolve()
        combined = pd.concat(all_parts, ignore_index=True)
        combined.to_csv(base / "results_summary_all.csv", index=False)
        all_stats = (
            combined.groupby(["mix_dir", "eps", "top_k"])["test_acc"]
            .agg(["mean", "std", "min", "max", "count"])
            .reset_index()
        )
        all_stats.columns = [
            "mix_dir", "eps", "top_k", "平均值", "标准差", "最小值", "最大值", "运行次数",
        ]
        all_stats["标准差"] = all_stats["标准差"].fillna(0.0)
        all_stats.to_csv(base / "results_statistics_all.csv", index=False)
        print(f"\n根目录汇总: {base / 'results_summary_all.csv'}")
        print(f"根目录统计: {base / 'results_statistics_all.csv'}")


if __name__ == "__main__":
    main()
