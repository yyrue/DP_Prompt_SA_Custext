#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot tradeoff curves for: custext vs selected mixed setting(s).

Examples:
  python3 plot_tradeoff_custext_vs_selected_mixes.py --mode mta --mix mix_0.0_18.0
  python3 plot_tradeoff_custext_vs_selected_mixes.py --mode knn --mix mix_0.0_18.0 mix_0.0_20.0 --no-errorbar
"""

import argparse
import re
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import pandas as pd


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _extract_mix_base(tag: str) -> str:
    match = re.search(r"mix_[0-9.]+_[0-9.]+", str(tag))
    return match.group(0) if match else str(tag)


def _normalize_mix_list(mixes: Iterable[str]) -> List[str]:
    return sorted({_extract_mix_base(m) for m in mixes})


def load_utility_custext(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"eps", "平均值", "标准差"}
    if not required.issubset(df.columns):
        raise ValueError(f"{csv_path} missing columns: {required - set(df.columns)}")

    df = df.rename(columns={"平均值": "utility_mean", "标准差": "utility_std"})
    df["eps"] = _to_numeric(df["eps"])
    df["utility_mean"] = _to_numeric(df["utility_mean"])
    df["utility_std"] = _to_numeric(df["utility_std"]).fillna(0.0)
    return df.dropna(subset=["eps", "utility_mean"])[["eps", "utility_mean", "utility_std"]]


def load_utility_mixed(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"mix_dir", "eps", "平均值", "标准差"}
    if not required.issubset(df.columns):
        raise ValueError(f"{csv_path} missing columns: {required - set(df.columns)}")

    df = df.rename(columns={"平均值": "utility_mean", "标准差": "utility_std"})
    df["eps"] = _to_numeric(df["eps"])
    df["utility_mean"] = _to_numeric(df["utility_mean"])
    df["utility_std"] = _to_numeric(df["utility_std"]).fillna(0.0)
    df["mix_base"] = df["mix_dir"].map(_extract_mix_base)
    return df.dropna(subset=["eps", "utility_mean"])[["mix_base", "eps", "utility_mean", "utility_std"]]


def load_privacy_custext(csv_path: Path, mode: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if mode == "mta":
        required_v1 = {"eps", "defense_rate_mean", "defense_rate_std"}
        required_v2 = {"eps", "平均值", "标准差"}
        if required_v1.issubset(df.columns):
            pass
        elif required_v2.issubset(df.columns):
            df = df.rename(columns={"平均值": "defense_rate_mean", "标准差": "defense_rate_std"})
        else:
            raise ValueError(f"{csv_path} missing required MTA columns")
    else:
        required = {"eps", "top10_accuracy_mean", "top10_accuracy_std"}
        if not required.issubset(df.columns):
            raise ValueError(f"{csv_path} missing columns: {required - set(df.columns)}")
        df["defense_rate_mean"] = 1.0 - _to_numeric(df["top10_accuracy_mean"])
        df["defense_rate_std"] = _to_numeric(df["top10_accuracy_std"]).fillna(0.0)

    df["eps"] = _to_numeric(df["eps"])
    df["defense_rate_mean"] = _to_numeric(df["defense_rate_mean"])
    df["defense_rate_std"] = _to_numeric(df["defense_rate_std"]).fillna(0.0)
    return df.dropna(subset=["eps", "defense_rate_mean"])[["eps", "defense_rate_mean", "defense_rate_std"]]


def load_privacy_mixed(csv_path: Path, mode: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if mode == "mta":
        required = {"mix_tag", "eps_prime", "defense_rate_mean", "defense_rate_std"}
        if not required.issubset(df.columns):
            raise ValueError(f"{csv_path} missing columns: {required - set(df.columns)}")
        df = df.rename(columns={"eps_prime": "eps"})
        df["mix_base"] = df["mix_tag"].map(_extract_mix_base)
    else:
        required = {"mix_dir", "eps", "top10_accuracy_mean", "top10_accuracy_std"}
        if not required.issubset(df.columns):
            raise ValueError(f"{csv_path} missing columns: {required - set(df.columns)}")
        df["mix_base"] = df["mix_dir"].map(_extract_mix_base)
        df["defense_rate_mean"] = 1.0 - _to_numeric(df["top10_accuracy_mean"])
        df["defense_rate_std"] = _to_numeric(df["top10_accuracy_std"]).fillna(0.0)

    df["eps"] = _to_numeric(df["eps"])
    df["defense_rate_mean"] = _to_numeric(df["defense_rate_mean"])
    df["defense_rate_std"] = _to_numeric(df["defense_rate_std"]).fillna(0.0)
    return df.dropna(subset=["eps", "defense_rate_mean"])[
        ["mix_base", "eps", "defense_rate_mean", "defense_rate_std"]
    ]


def annotate_eps(ax: plt.Axes, x, y, eps, x_offset=0.004, y_offset=0.0018, fontsize=8):
    for xv, yv, ev in zip(x, y, eps):
        ax.text(xv + x_offset, yv + y_offset, f"{ev:g}", fontsize=fontsize, alpha=0.9)


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Plot custext vs selected mixed tradeoff curves")
    parser.add_argument("--mode", choices=["mta", "knn"], default="mta", help="Privacy attack mode")
    parser.add_argument("--mix", nargs="+", required=True, help="Target mix name(s), e.g. mix_0.0_18.0")
    parser.add_argument("--annotate-eps", action="store_true", help="Annotate each point with eps")
    parser.add_argument("--no-errorbar", action="store_true", help="Disable error bars")

    parser.add_argument(
        "--custext-utility-csv",
        type=str,
        default=str(root / "experiment_results" / "results_statistics_topk_20.csv"),
    )
    parser.add_argument(
        "--mixed-utility-csv",
        type=str,
        default=str(root / "experiment_results_sa" / "results_statistics_all.csv"),
    )
    parser.add_argument(
        "--custext-privacy-csv",
        type=str,
        default=str(root / "attack_results" / "attack_summary_by_eps.csv"),
        help="For knn mode, this should be knn_attack_results/custext/knn_attack_statistics.csv",
    )
    parser.add_argument(
        "--mixed-privacy-csv",
        type=str,
        default=str(root / "attack_results_mixed" / "attack_summary_all_mixes_by_eps_prime.csv"),
        help="For knn mode, this should be knn_attack_results/mixed/knn_attack_statistics_all.csv",
    )
    parser.add_argument("--out", type=str, default=None, help="Output image path")
    args = parser.parse_args()

    if args.mode == "knn":
        if Path(args.custext_privacy_csv).name == "attack_summary_by_eps.csv":
            args.custext_privacy_csv = str(
                root / "knn_attack_results" / "custext" / "knn_attack_statistics.csv"
            )
        if Path(args.mixed_privacy_csv).name == "attack_summary_all_mixes_by_eps_prime.csv":
            args.mixed_privacy_csv = str(
                root / "knn_attack_results" / "mixed" / "knn_attack_statistics_all.csv"
            )

    selected_mixes = _normalize_mix_list(args.mix)
    mix_suffix = "__".join(selected_mixes)

    if args.out:
        out_path = Path(args.out).resolve()
    else:
        out_dir = root / ("attack_results" if args.mode == "mta" else "knn_attack_results")
        out_path = (out_dir / f"tradeoff_compare_{args.mode}_{mix_suffix}.png").resolve()

    custext_utility_csv = Path(args.custext_utility_csv).resolve()
    mixed_utility_csv = Path(args.mixed_utility_csv).resolve()
    custext_privacy_csv = Path(args.custext_privacy_csv).resolve()
    mixed_privacy_csv = Path(args.mixed_privacy_csv).resolve()
    for p in [custext_utility_csv, mixed_utility_csv, custext_privacy_csv, mixed_privacy_csv]:
        if not p.is_file():
            raise SystemExit(f"CSV not found: {p}")

    custext_utility = load_utility_custext(custext_utility_csv)
    mixed_utility = load_utility_mixed(mixed_utility_csv)
    custext_privacy = load_privacy_custext(custext_privacy_csv, mode=args.mode)
    mixed_privacy = load_privacy_mixed(mixed_privacy_csv, mode=args.mode)

    custext_df = pd.merge(custext_utility, custext_privacy, on="eps", how="inner").sort_values("eps")
    mixed_df = pd.merge(
        mixed_utility,
        mixed_privacy,
        on=["mix_base", "eps"],
        how="inner",
    )

    available_mixes = sorted(mixed_df["mix_base"].dropna().unique().tolist())
    mixed_df = mixed_df[mixed_df["mix_base"].isin(selected_mixes)].copy()
    if mixed_df.empty:
        raise SystemExit(
            "No rows matched for selected --mix values.\n"
            f"Selected: {selected_mixes}\n"
            f"Available: {available_mixes}"
        )

    fig, ax = plt.subplots(figsize=(8.5, 5.6))

    x = custext_df["defense_rate_mean"].to_numpy()
    y = custext_df["utility_mean"].to_numpy()
    eps = custext_df["eps"].to_numpy()
    if args.no_errorbar:
        ax.plot(x, y, "o-", linewidth=2.4, markersize=5.2, label="custext")
    else:
        ax.errorbar(
            x,
            y,
            xerr=custext_df["defense_rate_std"].to_numpy(),
            yerr=custext_df["utility_std"].to_numpy(),
            fmt="o-",
            linewidth=2.4,
            markersize=5.2,
            capsize=3,
            label="custext",
        )
    if args.annotate_eps:
        annotate_eps(ax, x, y, eps, fontsize=8)

    for mix_name, grp in mixed_df.groupby("mix_base"):
        grp = grp.sort_values("eps")
        x = grp["defense_rate_mean"].to_numpy()
        y = grp["utility_mean"].to_numpy()
        eps = grp["eps"].to_numpy()
        if args.no_errorbar:
            ax.plot(x, y, "o-", linewidth=2.0, markersize=4.5, alpha=0.92, label=mix_name)
        else:
            ax.errorbar(
                x,
                y,
                xerr=grp["defense_rate_std"].to_numpy(),
                yerr=grp["utility_std"].to_numpy(),
                fmt="o-",
                linewidth=2.0,
                markersize=4.5,
                capsize=2.5,
                alpha=0.92,
                label=mix_name,
            )
        if args.annotate_eps:
            annotate_eps(ax, x, y, eps, x_offset=0.0035, y_offset=0.0015, fontsize=7)

    ax.set_xlabel("Privacy (Defense Rate)")
    ax.set_ylabel("Utility (Accuracy)")
    ax.set_title(f"Tradeoff: custext vs selected mixed ({args.mode.upper()})")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
