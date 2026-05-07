#!/usr/bin/env python3
"""Conference-style aggregation tables for Gene-Prompt Expression Completion."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

LOWER_IS_BETTER = {"mse", "mae", "nonzero_mse", "high_expr_mse"}
HIGHER_IS_BETTER = {"r2", "pearson_all", "spearman_all", "nonzero_pearson", "high_expr_pearson", "rank_spearman_mean", "ndcg20_mean", "ndcg50_mean"}
PRIMARY_METRICS = ["mse", "pearson_all", "spearman_all", "r2"]
EMBED_ORDER = ["baseline", "minus", "scGPT_human", "v4_bias_rec_best", "v4_plain_best", "v4_type_pe_best"]


def _available_metrics(df: pd.DataFrame, requested: Iterable[str] = PRIMARY_METRICS) -> list[str]:
    return [m for m in requested if m in df.columns]


def _fmt(value: float, metric: str) -> str:
    if pd.isna(value):
        return "-"
    if metric in LOWER_IS_BETTER:
        return f"{value:.4g}"
    return f"{value:.4f}"


def _metric_is_better(value: float, reference: float, metric: str) -> bool:
    if pd.isna(value) or pd.isna(reference):
        return False
    if metric in LOWER_IS_BETTER:
        return value < reference
    return value > reference


def _style_pivot(pivot: pd.DataFrame, metric: str, baseline: str = "baseline") -> pd.DataFrame:
    styled = pivot.copy().astype(object)
    for idx in pivot.index:
        row = pivot.loc[idx]
        if row.dropna().empty:
            styled.loc[idx] = ["-" for _ in row]
            continue
        best_value = row.min() if metric in LOWER_IS_BETTER else row.max()
        base_value = row.get(baseline, np.nan)
        for col, value in row.items():
            txt = _fmt(value, metric)
            if pd.notna(value) and value == best_value:
                txt = f"**{txt}**"
            elif col != baseline and _metric_is_better(value, base_value, metric):
                txt = f"*{txt}*"
            styled.loc[idx, col] = txt
    return styled


def _markdown_table(styled: pd.DataFrame, index_name: str = "Setting") -> str:
    cols = list(styled.columns)
    lines = ["| " + index_name + " | " + " | ".join(cols) + " |", "|---|" + "|".join(["---:"] * len(cols)) + "|"]
    for idx, row in styled.iterrows():
        lines.append("| " + str(idx) + " | " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def _ordered_embeddings(cols: Iterable[str]) -> list[str]:
    cols = list(cols)
    fixed = [e for e in EMBED_ORDER if e in cols or any(c in EMBED_ORDER for c in cols)]
    return fixed + sorted([e for e in cols if e not in EMBED_ORDER])


def _aggregate_ok(df: pd.DataFrame) -> pd.DataFrame:
    ok = df.copy()
    if "status" in ok.columns:
        ok = ok[ok["status"].astype(str).str.upper().eq("OK")]
    metrics = _available_metrics(ok, list(LOWER_IS_BETTER | HIGHER_IS_BETTER | set(PRIMARY_METRICS)))
    group_cols = [c for c in ["dataset", "embedding", "model", "split_mode", "prompt_ratio"] if c in ok.columns]
    if ok.empty or not metrics or not group_cols:
        return pd.DataFrame()
    return ok.groupby(group_cols, dropna=False)[metrics].agg(["mean", "std", "count"]).reset_index()


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    flat = df.copy()
    flat.columns = ["_".join([str(x) for x in c if str(x)]) if isinstance(c, tuple) else str(c) for c in flat.columns]
    return flat


def _write_embedding_comparison_tables(ok: pd.DataFrame, metrics: list[str], baseline: str = "baseline") -> list[str]:
    """Return one-file markdown tables comparing the six fixed embeddings."""
    lines: list[str] = [
        "## Main embedding comparison tables",
        "",
        "Columns are the six fixed embeddings. **Bold** marks the best embedding in that row; *italic* marks better than the baseline embedding in that row. These tables are intentionally all in this one markdown file for direct paper-style inspection.",
        "",
    ]
    if not {"split_mode", "model", "dataset", "prompt_ratio", "embedding"}.issubset(ok.columns):
        return lines + ["Required columns are missing; cannot build embedding comparison tables.", ""]

    table_specs = [
        ("gene_holdout", "ridge_pair", "Primary headline: gene_holdout + ridge_pair"),
        ("cell_holdout", "ridge_pair", "Secondary: cell_holdout + ridge_pair"),
        ("gene_holdout", "mlp_pair", "Diagnostic high-capacity head: gene_holdout + mlp_pair"),
        ("cell_holdout", "mlp_pair", "Diagnostic high-capacity head: cell_holdout + mlp_pair"),
    ]
    any_table = False
    for split_mode, model, title in table_specs:
        sub = ok[(ok["split_mode"] == split_mode) & (ok["model"] == model)]
        if sub.empty:
            continue
        any_table = True
        lines += [f"### {title}", ""]
        for metric in metrics:
            pivot = sub.pivot_table(index=["dataset", "prompt_ratio"], columns="embedding", values=metric, aggfunc="mean")
            if pivot.empty:
                continue
            pivot = pivot.reindex(columns=_ordered_embeddings(pivot.columns))
            lines += [f"#### {metric}", "", _markdown_table(_style_pivot(pivot, metric, baseline=baseline), index_name="dataset / prompt_ratio"), ""]
    if not any_table:
        lines += ["No ridge_pair/mlp_pair rows were available. Run with `--models ridge_pair` (and optionally `mlp_pair`) to compare embeddings directly.", ""]

    baseline_models = [m for m in ["mean", "knn_prompt"] if "model" in ok.columns and (ok["model"] == m).any()]
    if baseline_models:
        lines += ["## Non-embedding baselines", "", "`mean` and `knn_prompt` do not use gene embeddings; repeated values across embedding columns are expected and should not be interpreted as embedding effects.", ""]
        for model in baseline_models:
            sub = ok[ok["model"] == model]
            metric = "mse" if "mse" in sub.columns else (metrics[0] if metrics else None)
            if metric is None:
                continue
            pivot = sub.pivot_table(index=["dataset", "split_mode", "prompt_ratio"], columns="embedding", values=metric, aggfunc="mean")
            pivot = pivot.reindex(columns=_ordered_embeddings(pivot.columns))
            lines += [f"### {model} ({metric})", "", _markdown_table(_style_pivot(pivot, metric, baseline=baseline), index_name="dataset / split / prompt_ratio"), ""]
    return lines


def _baseline_comparison(ok: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    required = {"dataset", "prompt_ratio", "split_mode", "embedding", "model", "mse"}
    if not required.issubset(ok.columns):
        return pd.DataFrame(), ["## Baseline comparison", "", "Skipped: required columns are missing.", ""]
    rows = []
    for key, g in ok.groupby(["dataset", "prompt_ratio", "split_mode", "embedding"], dropna=False):
        dataset, pr, split, emb = key
        ridge = g[g.model == "ridge_pair"]
        mean = g[g.model == "mean"]
        knn = g[g.model == "knn_prompt"]
        if ridge.empty:
            continue
        ridge_mse = float(ridge.mse.mean())
        mean_mse = float(mean.mse.mean()) if not mean.empty else np.nan
        knn_mse = float(knn.mse.mean()) if not knn.empty else np.nan
        rows.append({
            "dataset": dataset,
            "prompt_ratio": pr,
            "split_mode": split,
            "embedding": emb,
            "ridge_mse": ridge_mse,
            "mean_mse": mean_mse,
            "knn_mse": knn_mse,
            "beats_mean": bool(pd.notna(mean_mse) and ridge_mse < mean_mse),
            "beats_knn": bool(pd.notna(knn_mse) and ridge_mse < knn_mse),
            "embedding_useful_conservative": bool(pd.notna(mean_mse) and pd.notna(knn_mse) and ridge_mse < mean_mse and ridge_mse < knn_mse),
        })
    comp = pd.DataFrame(rows)
    if comp.empty:
        return comp, ["## Baseline comparison", "", "No ridge_pair rows were available for baseline comparison.", ""]
    win = comp.groupby("embedding")["embedding_useful_conservative"].agg(["sum", "count"]).reset_index()
    win["win_rate"] = win["sum"] / win["count"].replace(0, np.nan)
    win = win.sort_values(["win_rate", "sum"], ascending=False)
    lines = ["## Conservative win/loss vs baselines", "", "An embedding is counted as a conservative win only when ridge_pair beats both mean and knn_prompt on MSE for the same dataset/prompt/split.", ""]
    lines.append(_markdown_table(win.set_index("embedding").rename(columns={"sum": "wins", "count": "comparisons"}).round(4), index_name="Embedding"))
    lines.append("")
    return comp, lines


def build_conference_tables(results_csv: str | Path, out_dir: str | Path, baseline: str = "baseline") -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(results_csv)
    ok = df[df["status"].astype(str).str.upper().eq("OK")].copy() if "status" in df.columns else df.copy()
    metrics = _available_metrics(ok)

    lines = [
        "# Gene-Prompt Completion Conference Tables",
        "",
        "Style: **bold** = best embedding within a row; *italic* = better than baseline embedding in the same row. Lower is better for MSE/MAE; higher is better for correlation/R2/ranking metrics.",
        "",
        "## Data included",
        "",
        f"- Input rows: {len(df)}",
        f"- Successful rows: {len(ok)}",
        f"- Metrics shown: {', '.join(metrics) if metrics else 'none'}",
        "",
    ]
    if ok.empty:
        lines += ["No successful rows available.", ""]
    else:
        lines += _write_embedding_comparison_tables(ok, metrics, baseline=baseline)
        _, baseline_lines = _baseline_comparison(ok, out)
        lines += baseline_lines
        lines += [
            "## Interpretation rules",
            "",
            "- The most direct embedding comparison is ridge_pair, because mean and knn_prompt do not use gene embeddings.",
            "- Treat gains that appear only for MSE but not Pearson/Spearman as calibration-only improvements.",
            "- Do not claim broad superiority from one dataset, one prompt ratio, or cell_holdout alone.",
            "",
        ]
    md_path = out / "gene_prompt_completion_conference_tables.md"
    md_path.write_text("\n".join(lines))
    return {"markdown": md_path}


def main() -> None:
    ap = argparse.ArgumentParser(description="Build conference-style Gene-Prompt Completion tables.")
    ap.add_argument("--results", default="results/gene_prompt_completion/gene_prompt_completion_all_results.csv")
    ap.add_argument("--out-dir", default="results/gene_prompt_completion")
    ap.add_argument("--baseline", default="baseline")
    args = ap.parse_args()
    build_conference_tables(args.results, args.out_dir, baseline=args.baseline)


if __name__ == "__main__":
    main()
