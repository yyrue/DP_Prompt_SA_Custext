#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KNN Attack (Song & Raghunathan, 2020) for QNLI.

QNLI 为双文本输入，攻击时对 `question` 和 `sentence` 两列都统计：
对每个扰动词在词嵌入空间做 Top-k 近邻检索，若原词落在近邻集合中则记为命中。
"""

from __future__ import annotations

import argparse
import datetime
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="KNN embedding attack for QNLI")
    p.add_argument("--dataset", type=str, default="qnli")
    p.add_argument("--split", type=str, default="test", choices=["train", "dev", "test"])
    p.add_argument("--embedding_path", type=str, default="./embeddings/glove_840B-300d.txt")
    p.add_argument("--attack_k", type=int, default=10)
    p.add_argument("--data_source", type=str, default="custext", choices=["custext", "mixed"])
    p.add_argument("--attack_original", action="store_true", default=False)
    p.add_argument("--eps", type=float, default=10.0)
    p.add_argument("--eps_list", type=float, nargs="+", default=None)
    p.add_argument("--eps_low", type=float, default=0.0)
    p.add_argument("--eps_high", type=float, default=20.0)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--embedding_type", type=str, default="glove_840B-300d")
    p.add_argument("--mapping_strategy", type=str, default="paper")
    p.add_argument("--privatization_strategy", type=str, default="s1")
    p.add_argument("--save_stop_words", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="./knn_attack_results")
    p.add_argument("--max_sentences", type=int, default=0)
    return p


def load_glove_embeddings(path: str) -> Tuple[np.ndarray, Dict[str, int]]:
    embeddings = []
    word2idx: Dict[str, int] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) < 301:
                continue
            word = " ".join(tokens[:-300])
            vec = np.asarray([float(x) for x in tokens[-300:]], dtype=np.float64)
            word2idx[word] = len(embeddings)
            embeddings.append(vec)
    emb = np.vstack(embeddings)
    print(f"Loaded embeddings: {emb.shape[0]} words, dim={emb.shape[1]}")
    return emb, word2idx


class KNNIndex:
    def __init__(self, embeddings: np.ndarray, k: int = 10, chunk_size: int = 2048):
        self.embeddings = embeddings
        self.k = min(k, len(embeddings))
        self.chunk_size = chunk_size
        self._emb_norm_sq = (embeddings ** 2).sum(axis=1)

    def topk_words_batch(self, query_indices: np.ndarray) -> List[set]:
        if len(query_indices) == 0:
            return []
        queries = self.embeddings[query_indices]
        out: List[set] = []
        for start in range(0, len(queries), self.chunk_size):
            q = queries[start : start + self.chunk_size]
            q_norm_sq = (q ** 2).sum(axis=1, keepdims=True)
            dists = q_norm_sq + self._emb_norm_sq - 2.0 * (q @ self.embeddings.T)
            top_idx = np.argpartition(dists, kth=self.k - 1, axis=1)[:, : self.k]
            for row in top_idx:
                out.append(set(row.tolist()))
        return out


def load_original_split(dataset: str, split: str) -> pd.DataFrame:
    path = Path(f"datasets/{dataset}/{split}.tsv")
    if not path.is_file():
        raise FileNotFoundError(f"原始数据不存在: {path}")
    df = pd.read_csv(path, sep="\t", keep_default_na=False)
    for col in ("question", "sentence"):
        df[col] = df[col].fillna("")
    return df


def format_eps_folder_prefix(eps: float, data_source: str) -> str:
    if data_source == "mixed":
        if abs(eps - round(eps)) < 1e-9:
            return f"eps_{int(round(eps))}"
        return f"eps_{str(eps).replace('.', 'p')}"
    if abs(eps - round(eps)) < 1e-9:
        return f"eps_{float(int(round(eps)))}"
    return f"eps_{eps}"


def resolve_priv_path(args, eps: float, seed: int, split: str) -> Path:
    if args.attack_original:
        return Path(f"datasets/{args.dataset}/{split}.tsv")

    folder = (
        f"{format_eps_folder_prefix(eps, args.data_source)}_top_{args.top_k}_{args.privatization_strategy}"
        f"_save_stop_words_{args.save_stop_words}_seed_{seed}"
    )
    if args.data_source == "mixed":
        base = Path(
            f"./privatized_dataset_mixed/{args.embedding_type}/{args.mapping_strategy}"
            f"/mix_{args.eps_low}_{args.eps_high}"
        )
    else:
        base = Path(f"./privatized_dataset/{args.embedding_type}/{args.mapping_strategy}")
    return base / folder / f"{split}.tsv"


def _collect_from_col(
    orig_text: str,
    pert_text: str,
    word2idx: Dict[str, int],
    stats: Dict[str, int],
    orig_indices: List[int],
    pert_indices: List[int],
):
    o_tokens = str(orig_text).split()
    p_tokens = str(pert_text).split()
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


def collect_token_pairs(
    orig_df: pd.DataFrame, priv_df: pd.DataFrame, word2idx: Dict[str, int], max_sentences: int = 0
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
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
        _collect_from_col(
            orig_df.iloc[i]["question"], priv_df.iloc[i]["question"], word2idx, stats, orig_indices, pert_indices
        )
        _collect_from_col(
            orig_df.iloc[i]["sentence"], priv_df.iloc[i]["sentence"], word2idx, stats, orig_indices, pert_indices
        )

    return np.asarray(orig_indices, dtype=np.int64), np.asarray(pert_indices, dtype=np.int64), stats


def run_knn_attack(
    orig_df: pd.DataFrame,
    priv_df: pd.DataFrame,
    knn_index: KNNIndex,
    word2idx: Dict[str, int],
    attack_k: int,
    max_sentences: int = 0,
) -> Dict[str, float]:
    orig_idx, pert_idx, stats = collect_token_pairs(orig_df, priv_df, word2idx, max_sentences=max_sentences)
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
            "defense_rate": 1.0,
            "defense_rate_changed": float("nan"),
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
        "top10_accuracy_changed": (correct_changed / valid_changed) if valid_changed > 0 else float("nan"),
        "defense_rate": 1.0 - (correct / valid),
        "defense_rate_changed": (1.0 - (correct_changed / valid_changed)) if valid_changed > 0 else float("nan"),
        **{f"stat_{k}": v for k, v in stats.items()},
    }


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


def save_results(results: Dict, args, eps: Optional[float], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("KNN Attack Results (Song & Raghunathan, 2020, QNLI)\n")
        f.write("=" * 50 + "\n")
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if args.attack_original:
            f.write("mode=baseline (original as perturbed)\n")
        else:
            f.write(f"data_source={args.data_source}\n")
            f.write(f"eps={eps}\n")
            if args.data_source == "mixed":
                f.write(f"mix={args.eps_low}_{args.eps_high}\n")
        f.write(f"split={args.split}\n")
        f.write(f"attack_k={args.attack_k}\n")
        f.write(f"seed={args.seed}\n")
        f.write("=" * 50 + "\nResults:\n")
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    print(f"结果已保存: {out_path}")


def run_one_config(args, word2idx: Dict[str, int], knn_index: KNNIndex, eps: Optional[float]) -> Optional[Dict]:
    orig_df = load_original_split(args.dataset, args.split)
    priv_path = resolve_priv_path(args, eps if eps is not None else 0.0, args.seed, args.split)

    if args.attack_original:
        priv_df = orig_df.copy()
    else:
        if not priv_path.is_file():
            print(f"  ✗ 跳过，文件不存在: {priv_path}")
            return None
        priv_df = pd.read_csv(priv_path, sep="\t", keep_default_na=False)
        for col in ("question", "sentence"):
            priv_df[col] = priv_df[col].fillna("")

    results = run_knn_attack(
        orig_df=orig_df,
        priv_df=priv_df,
        knn_index=knn_index,
        word2idx=word2idx,
        attack_k=args.attack_k,
        max_sentences=args.max_sentences,
    )
    results["eps"] = eps
    results["seed"] = args.seed
    results["data_source"] = "baseline" if args.attack_original else args.data_source
    results["split"] = args.split

    print(
        f"Top-{args.attack_k} acc: {results['top10_accuracy']:.4f} "
        f"({results['correct']}/{results['valid_tokens']})"
    )
    out = Path(args.output_dir) / result_filename(args, eps)
    save_results(results, args, eps, out)
    return results


def main():
    args = get_parser().parse_args()
    root = Path(__file__).resolve().parent
    os.chdir(root)

    emb, word2idx = load_glove_embeddings(args.embedding_path)
    knn_index = KNNIndex(emb, k=args.attack_k)

    eps_list = args.eps_list if args.eps_list else ([args.eps] if not args.attack_original else [None])
    all_results: List[Dict] = []
    for eps in eps_list:
        r = run_one_config(args, word2idx, knn_index, eps)
        if r:
            all_results.append(r)

    if len(all_results) > 1:
        summary = Path(args.output_dir) / (
            f"knn_summary_{args.data_source}_{args.embedding_type}_{args.mapping_strategy}"
            f"_top_{args.top_k}_seed_{args.seed}_{args.split}.csv"
        )
        pd.DataFrame(all_results).to_csv(summary, index=False)
        print(f"汇总 CSV: {summary}")


if __name__ == "__main__":
    main()
