#!/usr/bin/env python3
"""Convert project-local scRNA-Seq CSV datasets to .h5ad.

Expected per-dataset directory layout (under --input-root):
  <dataset_name>/ExpressionData.csv
  <dataset_name>/GeneOrdering.csv
  <dataset_name>/PseudoTime.csv

This script intentionally does NOT depend on any scGREAT directory conventions.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


EXPECTED_DATASETS = ["hESC", "hHep", "mDC", "mESC", "mHSC-E", "mHSC-GM", "mHSC-L"]


def _first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for key in candidates:
        col = lower_map.get(key.lower())
        if col is not None:
            return col
    return None


def read_gene_ordering(path: Path) -> list[str]:
    gdf = pd.read_csv(path)
    if gdf.empty:
        raise ValueError(f"GeneOrdering is empty: {path}")

    col = _first_existing_column(
        gdf,
        ["gene", "genes", "gene_name", "gene_symbol", "symbol", "feature", "id"],
    )
    if col is None:
        col = gdf.columns[0]

    genes = (
        gdf[col]
        .astype(str)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan, "None": np.nan})
        .dropna()
        .tolist()
    )
    if not genes:
        raise ValueError(f"No valid genes parsed from {path}")
    return genes


def read_pseudotime(path: Path) -> pd.DataFrame:
    pdf = pd.read_csv(path)
    if pdf.empty:
        raise ValueError(f"PseudoTime is empty: {path}")

    cell_col = _first_existing_column(pdf, ["cell", "cell_id", "cellid", "barcode", "cells"])
    pt_col = _first_existing_column(pdf, ["pseudotime", "pseudo_time", "time", "pt"])

    # Fallback: use first 2 columns.
    if cell_col is None:
        cell_col = pdf.columns[0]
    if pt_col is None:
        if len(pdf.columns) < 2:
            raise ValueError(f"PseudoTime needs at least 2 columns: {path}")
        pt_col = pdf.columns[1]

    out = pdf[[cell_col, pt_col]].copy()
    out.columns = ["cell_id", "pseudotime"]
    out["cell_id"] = out["cell_id"].astype(str).str.strip()
    out["pseudotime"] = pd.to_numeric(out["pseudotime"], errors="coerce")
    out = out.dropna(subset=["cell_id", "pseudotime"])
    out = out.drop_duplicates(subset=["cell_id"], keep="first")
    if out.empty:
        raise ValueError(f"No valid cell_id/pseudotime rows in {path}")
    return out


def read_expression(path: Path) -> pd.DataFrame:
    # Preferred: first column as row labels.
    expr = pd.read_csv(path, index_col=0)
    if expr.empty:
        raise ValueError(f"ExpressionData is empty: {path}")

    # Fallback for cases where first column was actually data.
    if str(expr.index.name).lower() in {"none", "unnamed: 0"} and pd.api.types.is_numeric_dtype(expr.index):
        expr2 = pd.read_csv(path)
        if not expr2.empty and expr2.shape[1] > 1:
            expr = expr2
            expr.index = pd.RangeIndex(expr.shape[0]).astype(str)

    expr.index = expr.index.astype(str).str.strip()
    expr.columns = expr.columns.astype(str).str.strip()
    expr = expr.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return expr


def _overlap_ratio(items: Iterable[str], ref: set[str]) -> float:
    items = list(items)
    if not items:
        return 0.0
    hit = sum(1 for x in items if x in ref)
    return hit / len(items)


def orient_expression_cells_by_genes(expr: pd.DataFrame, genes: list[str]) -> Tuple[pd.DataFrame, str]:
    gene_set = set(genes)

    row_overlap = _overlap_ratio(expr.index, gene_set)
    col_overlap = _overlap_ratio(expr.columns, gene_set)

    # Heuristic 1: label-overlap preferred.
    if row_overlap > col_overlap and row_overlap >= 0.2:
        genes_by_cells = expr
        orientation = "genes_by_cells"
    elif col_overlap > row_overlap and col_overlap >= 0.2:
        genes_by_cells = expr.T
        orientation = "cells_by_genes_transposed"
    else:
        # Heuristic 2: shape against gene count.
        if abs(expr.shape[0] - len(genes)) <= abs(expr.shape[1] - len(genes)):
            genes_by_cells = expr
            orientation = "genes_by_cells_shape_guess"
        else:
            genes_by_cells = expr.T
            orientation = "cells_by_genes_shape_guess_transposed"

    cells_by_genes = genes_by_cells.T

    # If gene names likely absent on columns, overwrite with GeneOrdering.
    if _overlap_ratio(cells_by_genes.columns, gene_set) < 0.2:
        n = min(cells_by_genes.shape[1], len(genes))
        cells_by_genes = cells_by_genes.iloc[:, :n].copy()
        cells_by_genes.columns = genes[:n]

    # Ensure unique gene names for AnnData.
    counts: dict[str, int] = {}
    uniq = []
    for g in cells_by_genes.columns.astype(str):
        k = counts.get(g, 0)
        uniq.append(g if k == 0 else f"{g}_{k}")
        counts[g] = k + 1
    cells_by_genes.columns = uniq

    return cells_by_genes, orientation


def normalize_log1p(x: np.ndarray) -> np.ndarray:
    lib = x.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1.0
    x_norm = x / lib * 1e4
    return np.log1p(x_norm)


def convert_one_dataset(dataset_dir: Path, out_dir: Path) -> tuple[Path, dict[str, object]]:
    name = dataset_dir.name
    expr_path = dataset_dir / "ExpressionData.csv"
    gene_path = dataset_dir / "GeneOrdering.csv"
    pt_path = dataset_dir / "PseudoTime.csv"

    for p in [expr_path, gene_path, pt_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    genes = read_gene_ordering(gene_path)
    pt = read_pseudotime(pt_path)
    expr = read_expression(expr_path)
    cells_x_genes, orientation = orient_expression_cells_by_genes(expr, genes)

    n_cells_before_pt = int(cells_x_genes.shape[0])
    n_genes_before_pt = int(cells_x_genes.shape[1])

    # Align cells with pseudotime.
    cells_x_genes.index = cells_x_genes.index.astype(str).str.strip()
    keep_cells = cells_x_genes.index.intersection(pt["cell_id"])
    if len(keep_cells) == 0:
        raise ValueError(
            f"No overlapping cell IDs between expression and pseudotime for {name}. "
            "Please inspect ID formatting."
        )

    cells_x_genes = cells_x_genes.loc[keep_cells].copy()
    obs = pt.set_index("cell_id").loc[keep_cells].copy()
    obs["dataset"] = name

    raw_x = cells_x_genes.to_numpy(dtype=np.float32, copy=True)
    norm_x = normalize_log1p(raw_x)

    adata = ad.AnnData(X=sparse.csr_matrix(norm_x))
    adata.obs = obs
    adata.obs_names = obs.index.astype(str)
    adata.var_names = pd.Index(cells_x_genes.columns.astype(str), name="gene_symbol")

    adata.layers["counts"] = sparse.csr_matrix(raw_x)
    adata.uns["conversion"] = {
        "source": str(dataset_dir),
        "orientation": orientation,
        "n_cells_before_pt_intersection": n_cells_before_pt,
        "n_cells_after_pt_intersection": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "normalization": "library_size_1e4_then_log1p",
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.h5ad"
    adata.write_h5ad(out_path)

    qc = {
        "dataset": name,
        "orientation": orientation,
        "cells_before_pt_intersection": n_cells_before_pt,
        "cells_after_pt_intersection": int(adata.n_obs),
        "genes_before_pt_intersection": n_genes_before_pt,
        "genes_after_pt_intersection": int(adata.n_vars),
    }
    return out_path, qc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert local scRNA-Seq CSV datasets to h5ad.")
    p.add_argument("--input-root", type=Path, required=True, help="Root folder containing dataset dirs (e.g., scRNA-Seq/).")
    p.add_argument("--output-root", type=Path, default=Path("processed/native"), help="Output folder for generated .h5ad files.")
    p.add_argument("--datasets", type=str, nargs="*", default=EXPECTED_DATASETS, help="Dataset subdir names to convert.")
    p.add_argument(
        "--qc-csv",
        type=Path,
        default=Path("processed/native/conversion_qc.csv"),
        help="Path to write conversion QC summary CSV.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    converted = []
    qc_rows: list[dict[str, object]] = []

    for ds in args.datasets:
        dataset_dir = args.input_root / ds
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

        out, qc = convert_one_dataset(dataset_dir, args.output_root)
        converted.append(out)
        qc_rows.append(qc)
        print(f"[OK] {ds} -> {out}")

    args.qc_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(qc_rows).to_csv(args.qc_csv, index=False)

    print("\nConverted files:")
    for p in converted:
        print(f" - {p}")
    print(f"\nQC summary: {args.qc_csv}")


if __name__ == "__main__":
    main()
