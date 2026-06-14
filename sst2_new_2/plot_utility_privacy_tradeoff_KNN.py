#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot utility-privacy tradeoff for KNN attack.

- x-axis: privacy (defense rate)
- y-axis: utility (accuracy)
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _extract_mix_base(tag: str) -> str:
    match = re.search(r"mix_[0-9.]+_[0-9.]+", str(tag))
    return match.group(0) if match else str(tag)


def annotate_eps_points(
    ax: plt.Axes,
    x_vals,
    y_vals,
    eps_vals,
    *,
    fontsize: int = 8,
    x_offset: float = 0.004,
    y_offset: float = 0.002,
) -> None:
    """Annotate each point with its eps value."""
    for x, y, eps in zip(x_vals, y_vals, eps_vals):
        eps_text = f"{eps:g}"
        ax.text(
            x + x_offset,
            y + y_offset,
            eps_text,
            fontsize=fontsize,
            alpha=0.9,
        )


def load_custext_tradeoff(utility_csv: Path, privacy_csv: Path) -> pd.DataFrame:
    utility_df = pd.read_csv(utility_csv)
    privacy_df = pd.read_csv(privacy_csv)

    required_utility = {"eps", "平均值", "标准差"}
    required_privacy = {"eps", "top10_accuracy_mean", "top10_accuracy_std"}

    if not required_utility.issubset(utility_df.columns):
        raise ValueError(f"{utility_csv} missing columns: {required_utility - set(utility_df.columns)}")
    if not required_privacy.issubset(privacy_df.columns):
        raise ValueError(f"{privacy_csv} missing columns: {required_privacy - set(privacy_df.columns)}")

    utility_df = utility_df.rename(columns={"平均值": "utility_mean", "标准差": "utility_std"})
    utility_df["eps"] = _to_numeric(utility_df["eps"])
    utility_df["utility_mean"] = _to_numeric(utility_df["utility_mean"])
    utility_df["utility_std"] = _to_numeric(utility_df["utility_std"]).fillna(0.0)
    utility_df = utility_df.dropna(subset=["eps", "utility_mean"])

    privacy_df["eps"] = _to_numeric(privacy_df["eps"])
    privacy_df["top10_accuracy_mean"] = _to_numeric(privacy_df["top10_accuracy_mean"])
    privacy_df["top10_accuracy_std"] = _to_numeric(privacy_df["top10_accuracy_std"]).fillna(0.0)
    privacy_df = privacy_df.dropna(subset=["eps", "top10_accuracy_mean"])
    privacy_df["defense_rate_mean"] = 1.0 - privacy_df["top10_accuracy_mean"]
    privacy_df["defense_rate_std"] = privacy_df["top10_accuracy_std"]

    merged = pd.merge(
        utility_df[["eps", "utility_mean", "utility_std"]],
        privacy_df[["eps", "defense_rate_mean", "defense_rate_std"]],
        on="eps",
        how="inner",
    )
    merged["series"] = "custext"
    return merged.sort_values("eps")


def load_mixed_tradeoff(utility_csv: Path, privacy_csv: Path) -> pd.DataFrame:
    utility_df = pd.read_csv(utility_csv)
    privacy_df = pd.read_csv(privacy_csv)

    required_utility = {"mix_dir", "eps", "平均值", "标准差"}
    required_privacy = {"mix_dir", "eps", "top10_accuracy_mean", "top10_accuracy_std"}

    if not required_utility.issubset(utility_df.columns):
        raise ValueError(f"{utility_csv} missing columns: {required_utility - set(utility_df.columns)}")
    if not required_privacy.issubset(privacy_df.columns):
        raise ValueError(f"{privacy_csv} missing columns: {required_privacy - set(privacy_df.columns)}")

    utility_df = utility_df.rename(columns={"平均值": "utility_mean", "标准差": "utility_std"})
    utility_df["eps"] = _to_numeric(utility_df["eps"])
    utility_df["utility_mean"] = _to_numeric(utility_df["utility_mean"])
    utility_df["utility_std"] = _to_numeric(utility_df["utility_std"]).fillna(0.0)
    utility_df["mix_base"] = utility_df["mix_dir"].map(_extract_mix_base)
    utility_df = utility_df.dropna(subset=["eps", "utility_mean"])

    privacy_df["eps"] = _to_numeric(privacy_df["eps"])
    privacy_df["top10_accuracy_mean"] = _to_numeric(privacy_df["top10_accuracy_mean"])
    privacy_df["top10_accuracy_std"] = _to_numeric(privacy_df["top10_accuracy_std"]).fillna(0.0)
    privacy_df["mix_base"] = privacy_df["mix_dir"].map(_extract_mix_base)
    privacy_df = privacy_df.dropna(subset=["eps", "top10_accuracy_mean"])
    privacy_df["defense_rate_mean"] = 1.0 - privacy_df["top10_accuracy_mean"]
    privacy_df["defense_rate_std"] = privacy_df["top10_accuracy_std"]

    merged = pd.merge(
        utility_df[["mix_base", "eps", "utility_mean", "utility_std"]],
        privacy_df[["mix_base", "eps", "defense_rate_mean", "defense_rate_std"]],
        on=["mix_base", "eps"],
        how="inner",
    )
    merged["series"] = merged["mix_base"]
    return merged.sort_values(["mix_base", "eps"])


def main() -> None:
    root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Plot utility-privacy tradeoff curve(s) for KNN attack"
    )
    parser.add_argument(
        "--custext-utility-csv",
        type=str,
        default=str(root / "experiment_results" / "results_statistics_topk_200.csv"),
        help="CuSText utility CSV (acc over eps)",
    )
    parser.add_argument(
        "--mixed-utility-csv",
        type=str,
        default=str(root / "experiment_results_sa" / "results_statistics_all.csv"),
        help="Mixed utility CSV (acc over eps)",
    )
    parser.add_argument(
        "--custext-privacy-csv",
        type=str,
        default=str(root / "knn_attack_results" / "custext" / "knn_attack_statistics.csv"),
        help="CuSText KNN privacy CSV (top10 accuracy over eps)",
    )
    parser.add_argument(
        "--mixed-privacy-csv",
        type=str,
        default=str(root / "knn_attack_results" / "mixed" / "knn_attack_statistics_all.csv"),
        help="Mixed KNN privacy CSV (top10 accuracy over eps)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(root / "knn_attack_results" / "utility_privacy_tradeoff_knn_attack.png"),
        help="Output image path",
    )
    parser.add_argument(
        "--no-errorbar",
        action="store_true",
        help="Disable error bars",
    )
    args = parser.parse_args()

    custext_utility_csv = Path(args.custext_utility_csv).resolve()
    mixed_utility_csv = Path(args.mixed_utility_csv).resolve()
    custext_privacy_csv = Path(args.custext_privacy_csv).resolve()
    mixed_privacy_csv = Path(args.mixed_privacy_csv).resolve()

    for p in [custext_utility_csv, mixed_utility_csv, custext_privacy_csv, mixed_privacy_csv]:
        if not p.is_file():
            raise SystemExit(f"CSV not found: {p}")

    custext_df = load_custext_tradeoff(custext_utility_csv, custext_privacy_csv)
    mixed_df = load_mixed_tradeoff(mixed_utility_csv, mixed_privacy_csv)
    if custext_df.empty and mixed_df.empty:
        raise SystemExit("No matched rows found after merging utility and privacy CSVs.")

    fig, ax = plt.subplots(figsize=(9, 6))

    if not custext_df.empty:
        x = custext_df["defense_rate_mean"].to_numpy()
        y = custext_df["utility_mean"].to_numpy()
        eps = custext_df["eps"].to_numpy()
        if args.no_errorbar:
            ax.plot(
                x,
                y,
                "o-",
                linewidth=2.4,
                markersize=5,
                label="custext",
            )
        else:
            ax.errorbar(
                x,
                y,
                xerr=custext_df["defense_rate_std"].to_numpy(),
                yerr=custext_df["utility_std"].to_numpy(),
                fmt="o-",
                linewidth=2.4,
                markersize=5,
                capsize=3,
                label="custext",
            )
        annotate_eps_points(ax, x, y, eps)

    if not mixed_df.empty:
        for series_name, grp in mixed_df.groupby("series"):
            grp = grp.sort_values("eps")
            x = grp["defense_rate_mean"].to_numpy()
            y = grp["utility_mean"].to_numpy()
            eps = grp["eps"].to_numpy()
            if args.no_errorbar:
                ax.plot(
                    x,
                    y,
                    "o-",
                    linewidth=1.8,
                    markersize=4,
                    alpha=0.92,
                    label=series_name,
                )
            else:
                ax.errorbar(
                    x,
                    y,
                    xerr=grp["defense_rate_std"].to_numpy(),
                    yerr=grp["utility_std"].to_numpy(),
                    fmt="o-",
                    linewidth=1.8,
                    markersize=4,
                    capsize=2.5,
                    alpha=0.92,
                    label=series_name,
                )
            annotate_eps_points(ax, x, y, eps, fontsize=7, x_offset=0.0035, y_offset=0.0015)

    ax.set_xlabel("Privacy (Defense Rate)")
    ax.set_ylabel("Utility (Accuracy)")
    ax.set_title("Utility-Privacy Tradeoff (KNN Attack)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_xlim(0.0, 0.85)
    ax.set_ylim(0.73, 0.94)
    ax.legend()

    fig.tight_layout()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
