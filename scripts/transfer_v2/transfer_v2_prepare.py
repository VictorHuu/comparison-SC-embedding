#!/usr/bin/env python3
"""Prepare transfer-v2 protocol views from converted h5ad datasets.

Key fix for cross-species runs:
- Uses case-insensitive canonical gene symbols by default (upper-cased),
  which avoids tiny false intersections caused by `Sox2` vs `SOX2` mismatch.
- Emits pair-level diagnostics so you can audit how much canonicalization changes overlap.

Outputs:
- transfer_v2/dataset_stats.csv
- transfer_v2/pair_manifest.csv
- transfer_v2/pair_diagnostics.csv
- transfer_v2/gene_sets/strict_global.txt (if non-empty or strict-mode=global)
- transfer_v2/gene_sets/strict_pairwise/{train}__to__{test}.txt (when strict is pairwise)
- transfer_v2/gene_sets/coverage_matched/{train}__to__{test}.txt
"""

from __future__ import annotations

import argparse
from itertools import permutations
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


def canonical_gene(gene: str, case_mode: str) -> str:
    g = str(gene).strip()
    if case_mode == "upper":
        return g.upper()
    if case_mode == "lower":
        return g.lower()
    return g


def detect_rate_by_canonical(adata: ad.AnnData, case_mode: str) -> pd.Series:
    mat = adata.layers["counts"] if "counts" in adata.layers else adata.X
    if sparse.issparse(mat):
        nz = np.asarray((mat > 0).mean(axis=0)).ravel()
    else:
        nz = (np.asarray(mat) > 0).mean(axis=0)

    canon = [canonical_gene(g, case_mode) for g in adata.var_names.astype(str)]
    s = pd.Series(nz, index=canon, name="detect_rate")
    return s.groupby(level=0).max()


def read_h5ad_dir(root: Path) -> dict[str, ad.AnnData]:
    out: dict[str, ad.AnnData] = {}
    for p in sorted(root.glob("*.h5ad")):
        out[p.stem] = ad.read_h5ad(p)
    if not out:
        raise FileNotFoundError(f"No .h5ad files found under {root}")
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Prepare transfer-v2 protocol gene sets from h5ad files.")
    ap.add_argument("--h5ad-root", type=Path, required=True, help="Folder with per-dataset .h5ad files.")
    ap.add_argument("--out-dir", type=Path, default=Path("transfer_v2"), help="Output folder.")
    ap.add_argument(
        "--coverage-k",
        type=int,
        default=0,
        help="If >0, force coverage-matched gene-set size to K; else use strict size.",
    )
    ap.add_argument(
        "--strict-mode",
        choices=["auto", "global", "pairwise"],
        default="auto",
        help="auto=adaptive, global=all-dataset intersection, pairwise=train-test intersection.",
    )
    ap.add_argument(
        "--auto-global-min-ratio",
        type=float,
        default=0.2,
        help="In auto mode, require strict_global_size / median_dataset_unique_genes >= threshold to use global.",
    )
    ap.add_argument(
        "--case-mode",
        choices=["upper", "lower", "none"],
        default="upper",
        help="Canonicalization mode for overlap computation.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    ds = read_h5ad_dir(args.h5ad_root)

    stats_rows = []
    raw_gene_sets: dict[str, set[str]] = {}
    canon_gene_sets: dict[str, set[str]] = {}
    rates: dict[str, pd.Series] = {}

    for name, adata in ds.items():
        raw = set(map(str, adata.var_names.tolist()))
        canon = set(canonical_gene(g, args.case_mode) for g in raw)

        raw_gene_sets[name] = raw
        canon_gene_sets[name] = canon
        rates[name] = detect_rate_by_canonical(adata, args.case_mode)

        stats_rows.append(
            {
                "dataset": name,
                "n_cells": int(adata.n_obs),
                "n_genes_raw_unique": len(raw),
                "n_genes_canonical_unique": len(canon),
                "pseudotime_present": int("pseudotime" in adata.obs.columns),
            }
        )

    out = args.out_dir
    gdir = out / "gene_sets"
    strict_pair_dir = gdir / "strict_pairwise"
    covdir = gdir / "coverage_matched"
    strict_pair_dir.mkdir(parents=True, exist_ok=True)
    covdir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(stats_rows).sort_values("dataset").to_csv(out / "dataset_stats.csv", index=False)

    strict_global = sorted(set.intersection(*canon_gene_sets.values()))
    strict_global_path = gdir / "strict_global.txt"
    strict_global_n = len(strict_global)
    if strict_global_n > 0:
        strict_global_path.parent.mkdir(parents=True, exist_ok=True)
        strict_global_path.write_text("\n".join(strict_global) + "\n", encoding="utf-8")

    if args.strict_mode == "global":
        if strict_global_n == 0:
            raise ValueError(
                "strict_global intersection is empty under --strict-mode global. "
                "Use --strict-mode pairwise (or auto)."
            )
        strict_mode_used = "global"
    elif args.strict_mode == "pairwise":
        strict_mode_used = "pairwise"
    else:
        med_unique = float(np.median([len(v) for v in canon_gene_sets.values()]))
        global_ratio = (strict_global_n / med_unique) if med_unique > 0 else 0.0
        strict_mode_used = "global" if (strict_global_n > 0 and global_ratio >= args.auto_global_min_ratio) else "pairwise"

    manifest = []
    diagnostics = []

    for tr, te in permutations(sorted(ds.keys()), 2):
        shared_raw = raw_gene_sets[tr].intersection(raw_gene_sets[te])
        shared_canon = sorted(canon_gene_sets[tr].intersection(canon_gene_sets[te]))
        if not shared_canon:
            continue

        if strict_mode_used == "global":
            strict_genes = strict_global
            strict_file = strict_global_path
        else:
            strict_genes = shared_canon
            strict_file = strict_pair_dir / f"{tr}__to__{te}.txt"
            strict_file.write_text("\n".join(strict_genes) + "\n", encoding="utf-8")

        target_k = args.coverage_k if args.coverage_k > 0 else len(strict_genes)
        mean_rate = (rates[tr].reindex(shared_canon).fillna(0.0) + rates[te].reindex(shared_canon).fillna(0.0)) / 2.0
        cov_genes = mean_rate.sort_values(ascending=False).head(min(target_k, len(shared_canon))).index.tolist()

        cov_file = covdir / f"{tr}__to__{te}.txt"
        cov_file.write_text("\n".join(cov_genes) + "\n", encoding="utf-8")

        diagnostics.append(
            {
                "train_dataset": tr,
                "test_dataset": te,
                "n_shared_raw": len(shared_raw),
                "n_shared_canonical": len(shared_canon),
                "canonical_over_raw_ratio": (len(shared_canon) / len(shared_raw)) if len(shared_raw) else np.nan,
                "strict_scope": strict_mode_used,
            }
        )

        manifest.append({"train_dataset": tr, "test_dataset": te, "protocol": "native", "strict_scope": strict_mode_used, "case_mode": args.case_mode, "gene_set_file": "", "n_genes": ""})
        manifest.append({"train_dataset": tr, "test_dataset": te, "protocol": "strict", "strict_scope": strict_mode_used, "case_mode": args.case_mode, "gene_set_file": str(strict_file), "n_genes": len(strict_genes)})
        manifest.append({"train_dataset": tr, "test_dataset": te, "protocol": "coverage_matched", "strict_scope": strict_mode_used, "case_mode": args.case_mode, "gene_set_file": str(cov_file), "n_genes": len(cov_genes)})

    pd.DataFrame(manifest).to_csv(out / "pair_manifest.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(out / "pair_diagnostics.csv", index=False)

    if args.strict_mode == "auto":
        med_unique = float(np.median([len(v) for v in canon_gene_sets.values()]))
        global_ratio = (strict_global_n / med_unique) if med_unique > 0 else 0.0
        if strict_mode_used == "pairwise":
            print(
                "[WARN] auto strict switched to pairwise "
                f"(strict_global_n={strict_global_n}, median_unique={med_unique:.1f}, ratio={global_ratio:.4f}, "
                f"threshold={args.auto_global_min_ratio})."
            )

    print(f"[OK] case_mode={args.case_mode}")
    print(f"[OK] strict_mode_used={strict_mode_used}")
    print(f"[OK] wrote {out / 'dataset_stats.csv'}")
    print(f"[OK] wrote {out / 'pair_diagnostics.csv'}")
    if strict_global_n > 0:
        print(f"[OK] wrote {strict_global_path}")
    print(f"[OK] wrote {out / 'pair_manifest.csv'}")


if __name__ == "__main__":
    main()
