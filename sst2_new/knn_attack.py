#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KNN Attack (Song & Raghunathan, 2020) on CusText / Mix 脱敏文本。

对脱敏后每个 token（扰动词），在词表嵌入空间计算与全部词的距离，
取距离最小的 Top-k（默认 k=10）；若原始 token 落在 Top-k 中则计为命中。

Top-k accuracy = 命中数 / 有效 token 数（两词均在 GloVe 词表中且位置对齐）。

用法:
  # CusText 脱敏数据，单个配置
  python knn_attack.py --data_source custext --eps 10 --seed 42

  # Mix 脱敏数据
  python knn_attack.py --data_source mixed --eps 10 --seed 42 --eps_low 0 --eps_high 20

  # 多个 eps
  python knn_attack.py --data_source custext --eps_list 0 2 4 6 8 10 --seed 42

  # 原始数据自检（orig=pert，Top-10 应接近 1.0）
  python knn_attack.py --attack_original --seed 42
"""

from __future__ import annotations

import argparse
import datetime
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="KNN embedding attack (Top-k accuracy)")
    p.add_argument("--dataset", type=str, default="sst2")
    p.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "dev", "test"],
        help="与脱敏 TSV 对应的划分",
    )
    p.add_argument(
        "--embedding_path",
        type=str,
        default="./embeddings/glove_840B-300d.txt",
        help="与 CusText 脱敏一致的 GloVe 文件",
    )
    p.add_argument("--attack_k", type=int, default=10, help="Top-k 邻居数")
    p.add_argument(
        "--data_source",
        type=str,
        default="custext",
        choices=["custext", "mixed"],
        help="custext=privatized_dataset; mixed=privatized_dataset_mixed",
    )
    p.add_argument("--attack_original", action="store_true", default=False,
                   help="用原始文本作扰动侧（自检，Top-k≈1）")
    p.add_argument("--eps", type=float, default=10.0)
    p.add_argument("--eps_list", type=float, nargs="+", default=None)
    p.add_argument("--eps_low", type=float, default=0.0, help="mixed 混合下界")
    p.add_argument("--eps_high", type=float, default=20.0, help="mixed 混合上界")
    p.add_argument("--top_k", type=int, default=20, help="脱敏时的 CusText K（仅用于路径）")
    p.add_argument("--embedding_type", type=str, default="glove_840B-300d")
    p.add_argument("--mapping_strategy", type=str, default="paper")
    p.add_argument("--privatization_strategy", type=str, default="s1")
    p.add_argument("--save_stop_words", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="./knn_attack_results")
    p.add_argument("--max_sentences", type=int, default=0, help="0=不限制，调试用")
    return p


def load_glove_embeddings(path: str) -> Tuple[np.ndarray, List[str], Dict[str, int]]:
    embeddings = []
    idx2word: List[str] = []
    word2idx: Dict[str, int] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) < 301:
                continue
            word = " ".join(tokens[:-300])
            vec = np.asarray([float(x) for x in tokens[-300:]], dtype=np.float64)
            word2idx[word] = len(idx2word)
            idx2word.append(word)
            embeddings.append(vec)
    emb = np.vstack(embeddings)
    print(f"Loaded embeddings: {emb.shape[0]} words, dim={emb.shape[1]}")
    return emb, idx2word, word2idx


class KNNIndex:
    """在 GloVe 空间对扰动词做全词表欧氏距离 Top-k 检索（Song & Raghunathan, 2020）。"""

    def __init__(self, embeddings: np.ndarray, k: int = 10, chunk_size: int = 2048):
        self.embeddings = embeddings
        self.k = min(k, len(embeddings))
        self.chunk_size = chunk_size
        self._emb_norm_sq = (embeddings ** 2).sum(axis=1)

    def topk_words_batch(self, query_indices: np.ndarray) -> List[set]:
        """query_indices: (n,) 词表索引 -> 每个 query 的 Top-k 词索引集合"""
        if len(query_indices) == 0:
            return []
        queries = self.embeddings[query_indices]
        out: List[set] = []
        k = self.k
        for start in range(0, len(queries), self.chunk_size):
            q = queries[start : start + self.chunk_size]
            q_norm_sq = (q ** 2).sum(axis=1, keepdims=True)
            dists = q_norm_sq + self._emb_norm_sq - 2.0 * (q @ self.embeddings.T)
            top_idx = np.argpartition(dists, kth=k - 1, axis=1)[:, :k]
            for row in top_idx:
                out.append(set(row.tolist()))
        return out


def load_original_split(dataset: str, split: str) -> pd.DataFrame:
    path = Path(f"datasets/{dataset}/{split}.tsv")
    if not path.is_file():
        raise FileNotFoundError(f"原始数据不存在: {path}")
    df = pd.read_csv(path, sep="\t", keep_default_na=False)
    df["sentence"] = df["sentence"].fillna("")
    return df


def format_eps_folder_prefix(eps: float, data_source: str) -> str:
    """
    脱敏目录前缀与生成脚本一致：
      - mixed (generate_sample_amplification): eps_0, eps_10（整数）
      - custext (batch_generate_private_data): eps_10.0 等
    """
    if data_source == "mixed":
        return f"eps_{int(round(eps))}"
    if abs(eps - round(eps)) < 1e-9:
        return f"eps_{float(int(round(eps)))}"
    return f"eps_{eps}"


def _priv_folder_name(args, eps: float, seed: int) -> str:
    src = "baseline" if getattr(args, "attack_original", False) else args.data_source
    eps_prefix = format_eps_folder_prefix(eps, src)
    stop = args.save_stop_words
    return (
        f"{eps_prefix}_top_{args.top_k}_{args.privatization_strategy}"
        f"_save_stop_words_{stop}_seed_{seed}"
    )


def resolve_priv_path(args, eps: float, seed: int, split: str) -> Path:
    if args.attack_original:
        return Path(f"datasets/{args.dataset}/{split}.tsv")

    folder = _priv_folder_name(args, eps, seed)
    if args.data_source == "mixed":
        base = Path(
            f"./privatized_dataset_mixed/{args.embedding_type}/{args.mapping_strategy}"
            f"/mix_{args.eps_low}_{args.eps_high}"
        )
    else:
        base = Path(
            f"./privatized_dataset/{args.embedding_type}/{args.mapping_strategy}"
        )

    primary = base / folder / f"{split}.tsv"
    if primary.is_file():
        return primary

    # 兼容另一种 eps 命名（mixed↔custext 或历史数据）
    alt_src = "custext" if args.data_source == "mixed" else "mixed"
    alt_folder = (
        f"{format_eps_folder_prefix(eps, alt_src)}_top_{args.top_k}_{args.privatization_strategy}"
        f"_save_stop_words_{args.save_stop_words}_seed_{seed}"
    )
    alt = base / alt_folder / f"{split}.tsv"
    if alt.is_file():
        return alt
    return primary


def collect_token_pairs(
    orig_df: pd.DataFrame,
    priv_df: pd.DataFrame,
    word2idx: Dict[str, int],
    max_sentences: int = 0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    返回 (orig_idx, pert_idx) 与统计计数。
    仅保留 orig、pert 均在词表中的对齐位置。
    """
    n = len(orig_df)
    if len(priv_df) != n:
        raise ValueError(f"行数不一致: original={n}, privatized={len(priv_df)}")
    if max_sentences > 0:
        n = min(n, max_sentences)

    orig_indices: List[int] = []
    pert_indices: List[int] = []
    stats = {
        "sentences": n,
        "positions_total": 0,
        "positions_changed": 0,
        "skipped_oov_orig": 0,
        "skipped_oov_pert": 0,
        "skipped_length_mismatch": 0,
    }

    for i in range(n):
        o_tokens = str(orig_df.iloc[i]["sentence"]).split()
        p_tokens = str(priv_df.iloc[i]["sentence"]).split()
        if len(o_tokens) != len(p_tokens):
            stats["skipped_length_mismatch"] += abs(len(o_tokens) - len(p_tokens))
            m = min(len(o_tokens), len(p_tokens))
            o_tokens = o_tokens[:m]
            p_tokens = p_tokens[:m]

        for o_tok, p_tok in zip(o_tokens, p_tokens):
            stats["positions_total"] += 1
            if o_tok != p_tok:
                stats["positions_changed"] += 1
            if p_tok not in word2idx:
                stats["skipped_oov_pert"] += 1
                continue
            if o_tok not in word2idx:
                stats["skipped_oov_orig"] += 1
                continue
            orig_indices.append(word2idx[o_tok])
            pert_indices.append(word2idx[p_tok])

    return (
        np.asarray(orig_indices, dtype=np.int64),
        np.asarray(pert_indices, dtype=np.int64),
        stats,
    )


def run_knn_attack(
    orig_df: pd.DataFrame,
    priv_df: pd.DataFrame,
    knn_index: KNNIndex,
    word2idx: Dict[str, int],
    attack_k: int,
    max_sentences: int = 0,
) -> Dict[str, float]:
    orig_idx, pert_idx, stats = collect_token_pairs(
        orig_df, priv_df, word2idx, max_sentences=max_sentences
    )
    valid = len(orig_idx)
    if valid == 0:
        return {
            "attack_k": attack_k,
            "valid_tokens": 0,
            "correct": 0,
            "top10_accuracy": 0.0,
            "top10_accuracy_changed": 0.0,
            "valid_tokens_changed": 0,
            "correct_changed": 0,
            **{f"stat_{k}": v for k, v in stats.items()},
        }

    topk_sets = knn_index.topk_words_batch(pert_idx)
    correct = 0
    correct_changed = 0
    valid_changed = 0

    for o_i, p_i, neighbors in zip(orig_idx, pert_idx, topk_sets):
        if o_i in neighbors:
            correct += 1
        if o_i != p_i:
            valid_changed += 1
            if o_i in neighbors:
                correct_changed += 1

    return {
        "attack_k": attack_k,
        "valid_tokens": valid,
        "correct": correct,
        "top10_accuracy": correct / valid,
        "valid_tokens_changed": valid_changed,
        "correct_changed": correct_changed,
        "top10_accuracy_changed": (
            correct_changed / valid_changed if valid_changed > 0 else float("nan")
        ),
        "defense_rate": 1.0 - (correct / valid),
        "defense_rate_changed": (
            1.0 - (correct_changed / valid_changed) if valid_changed > 0 else float("nan")
        ),
        **{f"stat_{k}": v for k, v in stats.items()},
    }


def save_results(results: Dict, args, eps: Optional[float], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("KNN Attack Results (Song & Raghunathan, 2020)\n")
        f.write("=" * 50 + "\n")
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("Parameters:\n")
        if args.attack_original:
            f.write("  mode=baseline (original as perturbed)\n")
        else:
            f.write(f"  data_source={args.data_source}\n")
            f.write(f"  eps={eps}\n")
            if args.data_source == "mixed":
                f.write(f"  mix={args.eps_low}_{args.eps_high}\n")
        f.write(f"  split={args.split}\n")
        f.write(f"  attack_k={args.attack_k}\n")
        f.write(f"  seed={args.seed}\n")
        f.write(f"  top_k={args.top_k}\n")
        f.write(f"  embedding_type={args.embedding_type}\n")
        f.write(f"  mapping_strategy={args.mapping_strategy}\n")
        f.write(f"  embedding_path={args.embedding_path}\n")
        f.write("=" * 50 + "\n\nResults:\n")
        for key, value in results.items():
            f.write(f"  {key}: {value}\n")
    print(f"结果已保存: {out_path}")


def result_filename(args, eps: Optional[float]) -> str:
    stop = "True" if args.save_stop_words else "False"
    if args.attack_original:
        return f"knn_baseline_{args.split}_seed_{args.seed}.txt"
    if args.data_source == "mixed":
        return (
            f"knn_mixed_{args.eps_low}_{args.eps_high}_{args.embedding_type}_{args.mapping_strategy}"
            f"_eps_{eps}_top_{args.top_k}_seed_{args.seed}_save_stop_words_{stop}.txt"
        )
    return (
        f"knn_{args.embedding_type}_{args.mapping_strategy}"
        f"_eps_{eps}_top_{args.top_k}_seed_{args.seed}_save_stop_words_{stop}.txt"
    )


def run_one_config(
    args,
    embeddings: np.ndarray,
    word2idx: Dict[str, int],
    knn_index: KNNIndex,
    eps: Optional[float],
) -> Optional[Dict]:
    orig_df = load_original_split(args.dataset, args.split)
    priv_path = resolve_priv_path(args, eps if eps is not None else 0.0, args.seed, args.split)

    if args.attack_original:
        priv_df = orig_df.copy()
        tag = "original (sanity check)"
    else:
        if not priv_path.is_file():
            print(f"  ✗ 跳过，文件不存在: {priv_path}")
            return None
        priv_df = pd.read_csv(priv_path, sep="\t", keep_default_na=False)
        priv_df["sentence"] = priv_df["sentence"].fillna("")
        tag = f"{args.data_source}, eps={eps}, path={priv_path}"

    print(f"\n>>> {tag}")
    results = run_knn_attack(
        orig_df,
        priv_df,
        knn_index,
        word2idx,
        attack_k=args.attack_k,
        max_sentences=args.max_sentences,
    )
    results["eps"] = eps
    results["seed"] = args.seed
    results["data_source"] = "baseline" if args.attack_original else args.data_source
    results["split"] = args.split

    print(f"  Top-{args.attack_k} accuracy (all valid):     {results['top10_accuracy']:.4f} "
          f"({results['correct']}/{results['valid_tokens']})")
    if results["valid_tokens_changed"] > 0:
        print(f"  Top-{args.attack_k} accuracy (changed only): {results['top10_accuracy_changed']:.4f} "
              f"({results['correct_changed']}/{results['valid_tokens_changed']})")
    print(f"  defense_rate (1 - top10_acc):              {results['defense_rate']:.4f}")

    out = Path(args.output_dir) / result_filename(args, eps)
    save_results(results, args, eps, out)
    return results


def discover_custext_eps_seeds(
    base: Path, embedding_type: str, mapping_strategy: str, top_k: int, split: str
) -> List[Tuple[float, int]]:
    pat = re.compile(
        rf"^eps_([\d.]+)_top_{top_k}_s1_save_stop_words_(?:True|False)_seed_(\d+)$"
    )
    found = set()
    root = base / embedding_type / mapping_strategy
    if not root.is_dir():
        return []
    for d in root.iterdir():
        if not d.is_dir():
            continue
        m = pat.match(d.name)
        if m and (d / f"{split}.tsv").is_file():
            found.add((float(m.group(1)), int(m.group(2))))
    return sorted(found)


def main():
    args = get_parser().parse_args()
    root = Path(__file__).resolve().parent
    os.chdir(root)

    if not Path(args.embedding_path).is_file():
        raise FileNotFoundError(f"未找到 embedding: {args.embedding_path}")

    embeddings, _, word2idx = load_glove_embeddings(args.embedding_path)
    knn_index = KNNIndex(embeddings, k=args.attack_k)

    eps_list = args.eps_list if args.eps_list else ([args.eps] if not args.attack_original else [None])
    all_results: List[Dict] = []

    if args.attack_original:
        r = run_one_config(args, embeddings, word2idx, knn_index, None)
        if r:
            all_results.append(r)
    else:
        for eps in eps_list:
            r = run_one_config(args, embeddings, word2idx, knn_index, eps)
            if r:
                all_results.append(r)

    if len(all_results) > 1:
        summary = Path(args.output_dir) / (
            f"knn_summary_{args.data_source}_{args.embedding_type}_{args.mapping_strategy}"
            f"_top_{args.top_k}_seed_{args.seed}_{args.split}.csv"
        )
        pd.DataFrame(all_results).to_csv(summary, index=False)
        print(f"\n汇总 CSV: {summary}")


if __name__ == "__main__":
    main()
