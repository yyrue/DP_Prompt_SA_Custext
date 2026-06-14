#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 attack_results_mixed/mix_* 目录中的 attack_mixed_*_epsprime_*_seed_*.txt
收集结果，合并写入 attack_summary_mixed_*_seed_*.csv。

规则：CSV 里已有的 eps_prime 行不覆盖；只从 txt 补充缺失的 ε'。
不会处理 attack_mixed_baseline_seed_*.txt。

用法:
  python collect_mixed_attack_from_txt.py --mix-dir ./attack_results_mixed/mix_0.0_14.0
  python collect_mixed_attack_from_txt.py --mixed-root ./attack_results_mixed
"""

import argparse
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

TXT_GLOB = "attack_mixed_*_epsprime_*_seed_*_save_stop_words_*.txt"
SUMMARY_PREFIX = "attack_summary_mixed_"
CSV_COLUMNS = [
    "total_tokens",
    "valid_tokens",
    "correct",
    "attack_accuracy",
    "unmatched_sentences",
    "eps_prime",
    "eps_low",
    "eps_high",
    "seed",
    "mix_tag",
]

RESULT_LINE = re.compile(r"^\s*(\w+):\s*(.+)\s*$")
TXT_NAME = re.compile(
    r"^attack_mixed_.+_epsprime_(?P<eps>\d+)_top_(?P<topk>\d+)_seed_(?P<seed>\d+)"
    r"_save_stop_words_(?P<stop>True|False)\.txt$"
)


def _mix_slug_from_folder(folder_name: str) -> str:
    m = re.match(r"^mix_([\d.]+)_([\d.]+)$", folder_name)
    if not m:
        return folder_name.replace(".", "d")
    return f"mix_{m.group(1).replace('.', 'd')}_{m.group(2).replace('.', 'd')}"


def _mix_tag_from_folder(folder_name: str) -> str:
    m = re.match(r"^mix_([\d.]+)_([\d.]+)$", folder_name)
    if not m:
        return folder_name
    return f"mix_{float(m.group(1))}_{float(m.group(2))}"


def parse_mixed_attack_txt(filepath: Path, mix_folder: str) -> Optional[Dict[str, Any]]:
    text = filepath.read_text(encoding="utf-8")

    results: Dict[str, Any] = {}
    in_results = False
    for line in text.splitlines():
        if line.strip() == "Results:":
            in_results = True
            continue
        if not in_results:
            continue
        m = RESULT_LINE.match(line)
        if not m:
            continue
        key, val = m.group(1), m.group(2).strip()
        try:
            results[key] = float(val) if "." in val else int(val)
        except ValueError:
            results[key] = val

    required_result = {"total_tokens", "valid_tokens", "correct", "attack_accuracy", "unmatched_sentences"}
    if not required_result.issubset(results.keys()):
        return None

    params: Dict[str, Any] = {}
    m_eps = re.search(r"eps_prime\s*\(target\)\s*=\s*([\d.]+)", text)
    if m_eps:
        params["eps_prime"] = float(m_eps.group(1))
    m_mix = re.search(r"mix_eps_low=([\d.]+),\s*mix_eps_high=([\d.]+)", text)
    if m_mix:
        params["eps_low"] = float(m_mix.group(1))
        params["eps_high"] = float(m_mix.group(2))
    m_seed = re.search(r"top_k=(\d+),\s*seed=(\d+)", text)
    if m_seed:
        params["top_k"] = int(m_seed.group(1))
        params["seed"] = int(m_seed.group(2))
    m_emb = re.search(r"embedding_type=(\S+)", text)
    if m_emb:
        params["embedding_type"] = m_emb.group(1)
    m_map = re.search(r"mapping_strategy=(\S+)", text)
    if m_map:
        params["mapping_strategy"] = m_map.group(1)

    name_m = TXT_NAME.match(filepath.name)
    if name_m:
        params.setdefault("eps_prime", float(name_m.group("eps")))
        params.setdefault("top_k", int(name_m.group("topk")))
        params.setdefault("seed", int(name_m.group("seed")))

    if "eps_prime" not in params or "seed" not in params:
        return None

    fm = re.match(r"^mix_([\d.]+)_([\d.]+)$", mix_folder)
    params.setdefault("eps_low", float(fm.group(1)) if fm else 0.0)
    params.setdefault("eps_high", float(fm.group(2)) if fm else params["eps_prime"])
    params.setdefault("embedding_type", "glove_840B-300d")
    params.setdefault("mapping_strategy", "paper")
    params.setdefault("top_k", 20)

    return {
        **{k: results[k] for k in required_result},
        "eps_prime": float(params["eps_prime"]),
        "eps_low": float(params["eps_low"]),
        "eps_high": float(params["eps_high"]),
        "seed": int(params["seed"]),
        "mix_tag": _mix_tag_from_folder(mix_folder),
        "_embedding_type": params["embedding_type"],
        "_mapping_strategy": params["mapping_strategy"],
        "_top_k": int(params["top_k"]),
    }


def summary_csv_path(mix_dir: Path, row: Dict[str, Any]) -> Path:
    mix_slug = _mix_slug_from_folder(mix_dir.name)
    name = (
        f"{SUMMARY_PREFIX}{row['_embedding_type']}_{row['_mapping_strategy']}_"
        f"{mix_slug}_top_{row['_top_k']}_seed_{row['seed']}.csv"
    )
    return mix_dir / name


def merge_rows_into_csv(csv_path: Path, new_rows: List[Dict[str, Any]]) -> Tuple[int, int]:
    """返回 (新增行数, 跳过行数)。"""
    out_rows = [{c: r[c] for c in CSV_COLUMNS} for r in new_rows]

    if csv_path.is_file():
        existing = pd.read_csv(csv_path)
        for c in CSV_COLUMNS:
            if c not in existing.columns:
                existing[c] = None
        existing = existing[CSV_COLUMNS]
        existing_eps = {float(x) for x in existing["eps_prime"].tolist()}
    else:
        existing = pd.DataFrame(columns=CSV_COLUMNS)
        existing_eps = set()

    added, skipped = 0, 0
    to_append = []
    for r in out_rows:
        ep = float(r["eps_prime"])
        if ep in existing_eps:
            skipped += 1
            continue
        to_append.append(r)
        existing_eps.add(ep)
        added += 1

    if to_append:
        merged = pd.concat([existing, pd.DataFrame(to_append)], ignore_index=True)
        merged = merged.sort_values("eps_prime").reset_index(drop=True)
        merged.to_csv(csv_path, index=False)

    return added, skipped


def collect_one_mix_dir(mix_dir: Path) -> None:
    txt_files = sorted(mix_dir.glob(TXT_GLOB))
    if not txt_files:
        print(f"  [{mix_dir.name}] 无 txt，跳过")
        return

    by_csv: Dict[Path, List[Dict[str, Any]]] = {}
    bad = 0
    for fp in txt_files:
        if "baseline" in fp.name:
            continue
        row = parse_mixed_attack_txt(fp, mix_dir.name)
        if row is None:
            bad += 1
            print(f"  [警告] 无法解析: {fp.name}")
            continue
        csv_path = summary_csv_path(mix_dir, row)
        by_csv.setdefault(csv_path, []).append(row)

    for csv_path, rows in sorted(by_csv.items()):
        added, skipped = merge_rows_into_csv(csv_path, rows)
        print(
            f"  [{mix_dir.name}] {csv_path.name}: "
            f"扫描 {len(rows)} 条 txt → 新增 {added} 行, 跳过已有 {skipped} 行"
        )
    if bad:
        print(f"  [{mix_dir.name}] 解析失败 {bad} 个 txt")


def main():
    root = Path(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description="从 mixed attack txt 合并到 per-seed 汇总 CSV")
    parser.add_argument(
        "--mixed-root",
        type=str,
        default=str(root / "attack_results_mixed"),
        help="attack_results_mixed 根目录",
    )
    parser.add_argument(
        "--mix-dir",
        type=str,
        default=None,
        help="只处理单个 mix 子目录",
    )
    args = parser.parse_args()

    if args.mix_dir:
        mix_dirs = [Path(args.mix_dir).resolve()]
    else:
        mixed_root = Path(args.mixed_root).resolve()
        mix_dirs = sorted(p for p in mixed_root.iterdir() if p.is_dir() and p.name.startswith("mix_"))

    if not mix_dirs:
        raise SystemExit("未找到 mix_* 目录")

    print("收集 txt → attack_summary_mixed_*_seed_*.csv（不覆盖已有 eps_prime）\n")
    for d in mix_dirs:
        collect_one_mix_dir(d)
    print("\n完成。若需按 eps 跨 seed 汇总，请再运行: python aggregate_mixed_attack_by_eps_prime.py")


if __name__ == "__main__":
    main()
