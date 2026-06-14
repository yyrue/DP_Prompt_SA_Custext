#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot defense-rate curves for CuSText and mixed settings.

Definition:
  defense_rate = 1 - top10_accuracy_mean
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_curve(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"eps", "top10_accuracy_mean"}
    if not required.issubset(df.columns):
        missing = required.difference(df.columns)
        raise ValueError(f"{csv_path} missing columns: {missing}")

    df["eps"] = pd.to_numeric(df["eps"], errors="coerce")
    df["top10_accuracy_mean"] = pd.to_numeric(df["top10_accuracy_mean"], errors="coerce")
    df = df.dropna(subset=["eps", "top10_accuracy_mean"]).sort_values("eps")
    if df.empty:
        raise ValueError(f"{csv_path} has no valid eps/top10_accuracy_mean rows")

    df["defense_rate"] = 1.0 - df["top10_accuracy_mean"]
    if "top10_accuracy_std" in df.columns:
        df["top10_accuracy_std"] = pd.to_numeric(df["top10_accuracy_std"], errors="coerce").fillna(0.0)
    return df


def main() -> None:
    root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Plot defense rate (1-top10_accuracy_mean) curves for CuSText and mixed settings"
    )
    parser.add_argument(
        "--custext-csv",
        type=str,
        default=str(root / "knn_attack_results" / "custext" / "knn_attack_statistics.csv"),
        help="CuSText knn_attack_statistics.csv path",
    )
    parser.add_argument(
        "--mixed-dir",
        type=str,
        default=str(root / "knn_attack_results" / "mixed"),
        help="Directory containing mix_*/knn_attack_statistics.csv",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(root / "knn_attack_results" / "knn_defense_rate_compare.png"),
        help="Output image path",
    )
    parser.add_argument(
        "--no-errorbar",
        action="store_true",
        help="Disable error bars from top10_accuracy_std",
    )
    args = parser.parse_args()

    custext_csv = Path(args.custext_csv).resolve()
    if not custext_csv.is_file():
        raise SystemExit(f"CuSText CSV not found: {custext_csv}")

    mixed_dir = Path(args.mixed_dir).resolve()
    if not mixed_dir.is_dir():
        raise SystemExit(f"Mixed directory not found: {mixed_dir}")

    mixed_csvs = sorted(mixed_dir.glob("mix_*/knn_attack_statistics.csv"))
    if not mixed_csvs:
        raise SystemExit(f"No mixed CSV found under: {mixed_dir / 'mix_*/knn_attack_statistics.csv'}")

    fig, ax = plt.subplots(figsize=(8, 5))

    custext_df = load_curve(custext_csv)
    if "top10_accuracy_std" in custext_df.columns and not args.no_errorbar:
        ax.errorbar(
            custext_df["eps"].to_numpy(),
            custext_df["defense_rate"].to_numpy(),
            yerr=custext_df["top10_accuracy_std"].to_numpy(),
            fmt="o-",
            linewidth=2.2,
            markersize=5,
            capsize=3,
            label="custext",
        )
    else:
        ax.plot(
            custext_df["eps"].to_numpy(),
            custext_df["defense_rate"].to_numpy(),
            "o-",
            linewidth=2.2,
            markersize=5,
            label="custext",
        )

    for csv_path in mixed_csvs:
        mix_name = csv_path.parent.name
        mix_df = load_curve(csv_path)
        if "top10_accuracy_std" in mix_df.columns and not args.no_errorbar:
            ax.errorbar(
                mix_df["eps"].to_numpy(),
                mix_df["defense_rate"].to_numpy(),
                yerr=mix_df["top10_accuracy_std"].to_numpy(),
                fmt="o-",
                linewidth=1.8,
                markersize=4,
                capsize=2.5,
                alpha=0.9,
                label=mix_name,
            )
        else:
            ax.plot(
                mix_df["eps"].to_numpy(),
                mix_df["defense_rate"].to_numpy(),
                "o-",
                linewidth=1.8,
                markersize=4,
                alpha=0.9,
                label=mix_name,
            )

    ax.set_xlabel("eps")
    ax.set_ylabel("defense rate")
    ax.set_title("Defense Rate vs eps (custext vs mixed)")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()

    fig.tight_layout()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
