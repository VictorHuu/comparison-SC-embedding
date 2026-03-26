#!/usr/bin/env python3
"""
Summarize perturbation regression benchmark outputs with paired statistics.

Expected input files:
- results/perturbation_regression/perturbation_regression_results.csv
- results/perturbation_regression/perturbation_regression_ranking_summary.csv
- results/perturbation_regression/perturbation_regression_fold_results.csv
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PRIMARY_METHOD = "frozen_linear"
SECONDARY_METHOD = "frozen_backbone_trainable_head"
TARGET_METHODS = [PRIMARY_METHOD, SECONDARY_METHOD]
TARGET_EMBEDDING = "difference_v3"
COMPARATORS = ["baseline", "scGPT_human"]
METRICS = ["pearson_r", "mse", "sign_acc"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paired statistics for perturbation regression results")
    p.add_argument("--input_dir", type=str, default="/bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/results/perturbation_regression")
    p.add_argument("--results_csv", type=str, default="perturbation_regression_results.csv")
    p.add_argument("--ranking_csv", type=str, default="perturbation_regression_ranking_summary.csv")
    p.add_argument("--fold_csv", type=str, default="perturbation_regression_fold_results.csv")
    p.add_argument("--n_boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_inputs(input_dir: str, results_csv: str, ranking_csv: str, fold_csv: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load required CSVs with safe empty fallbacks."""
    def _read(path: str, cols: Optional[List[str]] = None) -> pd.DataFrame:
        if not os.path.exists(path):
            return pd.DataFrame(columns=cols if cols is not None else [])
        return pd.read_csv(path)

    results_path = os.path.join(input_dir, results_csv)
    ranking_path = os.path.join(input_dir, ranking_csv)
    fold_path = os.path.join(input_dir, fold_csv)

    results_df = _read(results_path)
    ranking_df = _read(ranking_path)
    fold_df = _read(fold_path)
    return results_df, ranking_df, fold_df


def paired_bootstrap_ci(diffs: np.ndarray, n_boot: int = 10000, seed: int = 42) -> Tuple[float, float]:
    """Nonparametric paired bootstrap CI for mean difference.

    diffs are paired challenger-vs-comparator differences with sign convention
    already applied so that positive means the challenger (difference_v3) is better.
    """
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    n = len(diffs)
    if n == 0:
        return np.nan, np.nan
    if n == 1:
        return float(diffs[0]), float(diffs[0])

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = float(np.mean(diffs[idx]))

    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))
    return ci_lower, ci_upper


def _safe_wilcoxon(diffs: np.ndarray) -> Tuple[float, float]:
    """Paired Wilcoxon signed-rank test with robust NaN fallback."""
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) == 0:
        return np.nan, np.nan
    if np.allclose(diffs, 0.0):
        return np.nan, np.nan
    try:
        res = wilcoxon(diffs, zero_method="wilcox", correction=False, alternative="two-sided", mode="auto")
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return np.nan, np.nan


def _paired_diff_series(challenger: pd.Series, comparator: pd.Series, metric: str) -> pd.Series:
    """Return paired diff with positive = challenger better.

    - For pearson_r and sign_acc (higher is better): challenger - comparator
    - For mse (lower is better): comparator - challenger
    """
    if metric == "mse":
        return comparator - challenger
    return challenger - comparator


def compute_paired_comparison(
    fold_df: pd.DataFrame,
    n_boot: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute paired fold-level comparisons for target methods and comparators."""
    required_cols = {"dataset", "context", "embedding", "method", "fold_id", "pearson_r", "mse", "sign_acc", "n_train", "n_test"}
    if fold_df.empty or not required_cols.issubset(set(fold_df.columns)):
        return pd.DataFrame(columns=[
            "dataset", "method", "metric", "comparator",
            "mean_diff", "median_diff", "ci_lower", "ci_upper",
            "wins", "losses", "ties", "n_folds", "wilcoxon_stat", "wilcoxon_p",
        ])

    out_rows: List[Dict[str, object]] = []
    df = fold_df.copy()
    df = df[df["method"].isin(TARGET_METHODS)]

    key_cols = ["dataset", "context", "method", "fold_id"]

    for (dataset, method), dm in df.groupby(["dataset", "method"]):
        d3 = dm[dm["embedding"] == TARGET_EMBEDDING]
        if d3.empty:
            continue

        for comp in COMPARATORS:
            comp_df = dm[dm["embedding"] == comp]
            if comp_df.empty:
                continue

            merged = d3.merge(
                comp_df,
                on=key_cols,
                suffixes=("_d3", "_comp"),
                how="inner",
            )
            if merged.empty:
                continue

            for metric in METRICS:
                diff = _paired_diff_series(merged[f"{metric}_d3"], merged[f"{metric}_comp"], metric)
                vals = diff.to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]

                n = len(vals)
                if n == 0:
                    continue

                wins = int(np.sum(vals > 0))
                losses = int(np.sum(vals < 0))
                ties = int(np.sum(np.isclose(vals, 0.0)))
                ci_lower, ci_upper = paired_bootstrap_ci(vals, n_boot=n_boot, seed=seed)
                w_stat, w_p = _safe_wilcoxon(vals)

                out_rows.append({
                    "dataset": dataset,
                    "method": method,
                    "metric": metric,
                    "comparator": comp,
                    "mean_diff": float(np.mean(vals)),
                    "median_diff": float(np.median(vals)),
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "wins": wins,
                    "losses": losses,
                    "ties": ties,
                    "n_folds": n,
                    "wilcoxon_stat": w_stat,
                    "wilcoxon_p": w_p,
                })

    out_df = pd.DataFrame(out_rows)
    if out_df.empty:
        return out_df
    return out_df.sort_values(["method", "dataset", "metric", "comparator"]).reset_index(drop=True)


def build_descriptive_summary(results_df: pd.DataFrame, ranking_df: pd.DataFrame) -> pd.DataFrame:
    """Build compact descriptive summary table by method/dataset."""
    rows: List[Dict[str, object]] = []

    # 1) Best embedding per dataset under target methods.
    if not ranking_df.empty and {"summary_type", "dataset", "method", "embedding", "pearson_r", "rank"}.issubset(ranking_df.columns):
        best = ranking_df[
            (ranking_df["summary_type"] == "best_embedding_per_dataset")
            & (ranking_df["method"].isin(TARGET_METHODS))
        ]
        for _, r in best.iterrows():
            rows.append({
                "section": "best_embedding_per_dataset",
                "method": r.get("method", np.nan),
                "dataset": r.get("dataset", np.nan),
                "embedding": r.get("embedding", np.nan),
                "pearson_r": r.get("pearson_r", np.nan),
                "rank": r.get("rank", np.nan),
                "note": "descriptive_only",
            })

        avg = ranking_df[
            (ranking_df["summary_type"] == "average_rank_across_datasets")
            & (ranking_df["method"].isin(TARGET_METHODS))
        ]
        for _, r in avg.iterrows():
            rows.append({
                "section": "average_rank_across_datasets",
                "method": r.get("method", np.nan),
                "dataset": "ALL",
                "embedding": r.get("embedding", np.nan),
                "pearson_r": r.get("pearson_r", np.nan),
                "rank": r.get("rank", np.nan),
                "note": "descriptive_only",
            })

    # 2) Fallback computation from results if ranking file missing/incomplete.
    if not rows and not results_df.empty and {"dataset", "method", "embedding", "pearson_r"}.issubset(results_df.columns):
        mdf = results_df[results_df["method"].isin(TARGET_METHODS)].copy()
        grouped = mdf.groupby(["dataset", "method", "embedding"], as_index=False)["pearson_r"].mean()

        for (dataset, method), g in grouped.groupby(["dataset", "method"]):
            g = g.sort_values("pearson_r", ascending=False).reset_index(drop=True)
            if g.empty:
                continue
            best = g.iloc[0]
            rows.append({
                "section": "best_embedding_per_dataset",
                "method": method,
                "dataset": dataset,
                "embedding": best["embedding"],
                "pearson_r": float(best["pearson_r"]),
                "rank": 1.0,
                "note": "descriptive_only",
            })

            g["rank"] = np.arange(1, len(g) + 1)
            for _, rr in g.iterrows():
                rows.append({
                    "section": "average_rank_across_datasets",
                    "method": method,
                    "dataset": "ALL",
                    "embedding": rr["embedding"],
                    "pearson_r": np.nan,
                    "rank": float(rr["rank"]),
                    "note": "descriptive_only",
                })

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["section", "method", "dataset", "embedding", "pearson_r", "rank", "note"])

    # If fallback created duplicated average-rank rows, average them by embedding/method.
    avg = out[out["section"] == "average_rank_across_datasets"]
    if not avg.empty:
        avg2 = avg.groupby(["section", "method", "embedding", "note"], as_index=False)["rank"].mean()
        avg2["dataset"] = "ALL"
        avg2["pearson_r"] = np.nan
        best = out[out["section"] == "best_embedding_per_dataset"]
        out = pd.concat([
            best[["section", "method", "dataset", "embedding", "pearson_r", "rank", "note"]],
            avg2[["section", "method", "dataset", "embedding", "pearson_r", "rank", "note"]],
        ], ignore_index=True)

    return out.sort_values(["section", "method", "dataset", "rank", "embedding"]).reset_index(drop=True)


def _best_embedding_map(descriptive_df: pd.DataFrame, method: str) -> Dict[str, str]:
    sub = descriptive_df[
        (descriptive_df["section"] == "best_embedding_per_dataset")
        & (descriptive_df["method"] == method)
    ]
    out: Dict[str, str] = {}
    for _, r in sub.iterrows():
        out[str(r["dataset"])] = str(r["embedding"])
    return out


def write_markdown_report(
    out_path: str,
    paired_df: pd.DataFrame,
    descriptive_df: pd.DataFrame,
) -> None:
    """Write conservative, paper-ready markdown summary."""
    best_primary = _best_embedding_map(descriptive_df, PRIMARY_METHOD)
    best_secondary = _best_embedding_map(descriptive_df, SECONDARY_METHOD)

    with open(out_path, "w") as f:
        f.write("# Paper-ready Summary: Perturbation Regression\n\n")
        f.write("## Headline interpretation\n\n")
        f.write("- Primary setting: `frozen_linear`.\n")
        f.write("- Secondary setting: `frozen_backbone_trainable_head`.\n")
        f.write("- Exploratory settings are not used for headline conclusions.\n")
        f.write("- Across datasets, `difference_v3` did **not** show consistent superiority.\n\n")

        f.write("## Descriptive observations by dataset\n\n")
        if "adamson" in best_primary:
            f.write(f"- Adamson (primary/frozen_linear): best embedding = `{best_primary['adamson']}`.\n")
        else:
            f.write("- Adamson (primary/frozen_linear): insufficient rows for a stable best-embedding summary.\n")

        if "norman" in best_primary:
            f.write(f"- Norman (primary/frozen_linear): best embedding = `{best_primary['norman']}`.\n")
        else:
            f.write("- Norman (primary/frozen_linear): insufficient rows for a stable best-embedding summary.\n")

        if "dixit" in best_primary:
            f.write(f"- Dixit (primary/frozen_linear): best embedding = `{best_primary['dixit']}`, but this dataset has small sample size and high variance; treat conclusions as unstable.\n")
        else:
            f.write("- Dixit (primary/frozen_linear): results are high-variance due to small sample size; avoid strong claims.\n")

        f.write("\n")
        f.write("## Paired fold-level comparison notes\n\n")
        if paired_df.empty:
            f.write("No valid fold-level paired comparisons were available from the provided files.\n")
        else:
            f.write("- Fold-level paired differences are reported with bootstrap 95% CI (descriptive uncertainty quantification).\n")
            f.write("- Positive mean_diff is defined as better for `difference_v3` across all metrics (including MSE via sign flip).\n")
            f.write("- Avoid interpreting bootstrap intervals as formal proof by themselves.\n")
            f.write("- Wilcoxon p-values are provided when valid; if missing/NaN, no formal nonparametric inference was possible for that row.\n")

            for method in TARGET_METHODS:
                sub = paired_df[paired_df["method"] == method]
                if sub.empty:
                    continue
                f.write(f"\n### {method}\n\n")
                for dataset in sorted(sub["dataset"].unique()):
                    ds = sub[sub["dataset"] == dataset]
                    if ds.empty:
                        continue
                    # brief direction summary for pearson_r
                    p_rows = ds[ds["metric"] == "pearson_r"]
                    if p_rows.empty:
                        f.write(f"- {dataset}: no pearson_r paired rows available.\n")
                        continue
                    pos = int((p_rows["mean_diff"] > 0).sum())
                    neg = int((p_rows["mean_diff"] < 0).sum())
                    if pos > neg:
                        trend = "difference_v3 appeared competitive in some paired comparisons"
                    elif neg > pos:
                        trend = "difference_v3 was often lower than comparators in paired folds"
                    else:
                        trend = "results were mixed with no clear directional advantage"
                    f.write(f"- {dataset}: {trend}.\n")

        f.write("\n## Conservative conclusion\n\n")
        f.write("- Main claims should be based on `frozen_linear` only.\n")
        f.write("- `frozen_backbone_trainable_head` is secondary and should be discussed separately from the primary probe result.\n")
        f.write("- Current evidence supports mixed performance across datasets rather than a universal winner.\n")
        f.write("- For small/high-variance settings (e.g., Dixit), use cautious language and avoid strong superiority claims.\n")


def main() -> None:
    args = parse_args()
    results_df, ranking_df, fold_df = load_inputs(
        input_dir=args.input_dir,
        results_csv=args.results_csv,
        ranking_csv=args.ranking_csv,
        fold_csv=args.fold_csv,
    )

    paired_df = compute_paired_comparison(fold_df, n_boot=args.n_boot, seed=args.seed)
    descriptive_df = build_descriptive_summary(results_df, ranking_df)

    paired_out = os.path.join(args.input_dir, "paired_comparison_summary.csv")
    desc_out = os.path.join(args.input_dir, "descriptive_ranking_summary.csv")
    md_out = os.path.join(args.input_dir, "paper_ready_summary.md")

    paired_cols = [
        "dataset", "method", "metric", "comparator",
        "mean_diff", "median_diff", "ci_lower", "ci_upper",
        "wins", "losses", "ties", "n_folds", "wilcoxon_stat", "wilcoxon_p",
    ]
    if paired_df.empty:
        pd.DataFrame(columns=paired_cols).to_csv(paired_out, index=False)
    else:
        paired_df[paired_cols].to_csv(paired_out, index=False)

    desc_cols = ["section", "method", "dataset", "embedding", "pearson_r", "rank", "note"]
    if descriptive_df.empty:
        pd.DataFrame(columns=desc_cols).to_csv(desc_out, index=False)
    else:
        descriptive_df[desc_cols].to_csv(desc_out, index=False)

    write_markdown_report(md_out, paired_df, descriptive_df)


if __name__ == "__main__":
    main()
