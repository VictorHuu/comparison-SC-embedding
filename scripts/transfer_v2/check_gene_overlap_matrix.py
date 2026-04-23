#!/usr/bin/env python3
"""Generate only the transfer_v2 train/test Jaccard gene-overlap matrix."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate transfer_v2 train/test Jaccard overlap matrix.")
    ap.add_argument("--dataset-stats", type=Path, default=Path("results/transfer_v2/dataset_stats.csv"))
    ap.add_argument("--pair-diagnostics", type=Path, default=Path("results/transfer_v2/pair_diagnostics.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/transfer_v2/gene_overlap_matrices"))
    return ap.parse_args()


def read_dataset_sizes(path: Path) -> Dict[str, int]:
    sizes: Dict[str, int] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sizes[row["dataset"]] = int(row["n_genes_canonical_unique"])
    return sizes


def read_shared(path: Path) -> Dict[Tuple[str, str], int]:
    shared: Dict[Tuple[str, str], int] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tr = row["train_dataset"]
            te = row["test_dataset"]
            shared[(tr, te)] = int(float(row["n_shared_canonical"]))
    return shared


def build_jaccard_matrix(datasets: List[str], sizes: Dict[str, int], shared: Dict[Tuple[str, str], int]) -> Dict[Tuple[str, str], float]:
    mat: Dict[Tuple[str, str], float] = {}
    for tr in datasets:
        for te in datasets:
            if tr == te:
                s = sizes[tr]
            else:
                s = shared.get((tr, te), shared.get((te, tr), 0))
            union = sizes[tr] + sizes[te] - s
            mat[(tr, te)] = (s / union) if union else 0.0
    return mat


def write_matrix_csv(path: Path, datasets: List[str], mat: Dict[Tuple[str, str], float]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset"] + datasets)
        for tr in datasets:
            w.writerow([tr] + [f"{mat[(tr, te)] * 100:.2f}%" for te in datasets])


def write_report(path: Path, datasets: List[str], sizes: Dict[str, int], mat: Dict[Tuple[str, str], float]) -> None:
    lines = ["# transfer_v2 Jaccard 基因重合度矩阵", "", "## 数据集基因数（canonical）", "", "| dataset | n_genes |", "|---|---:|"]
    for d in datasets:
        lines.append(f"| {d} | {sizes[d]} |")
    lines += ["", "## Jaccard overlap = shared / union（百分比）", "", "| train \\ test | " + " | ".join(datasets) + " |", "|---" * (len(datasets) + 1) + "|"]
    for tr in datasets:
        lines.append("| " + tr + " | " + " | ".join(f"{mat[(tr, te)] * 100:.2f}%" for te in datasets) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    sizes = read_dataset_sizes(args.dataset_stats)
    shared = read_shared(args.pair_diagnostics)
    datasets = sorted(sizes)

    jaccard = build_jaccard_matrix(datasets, sizes, shared)

    # clear previous overlap artifacts to keep only jaccard
    for stale in out.glob("base_*.csv"):
        stale.unlink()
    for stale in out.glob("protocol_*_overlap_over_min.csv"):
        stale.unlink()

    write_matrix_csv(out / "base_jaccard.csv", datasets, jaccard)
    write_report(out / "gene_overlap_matrix_report.md", datasets, sizes, jaccard)
    print(f"[OK] wrote {out / 'base_jaccard.csv'}")


if __name__ == "__main__":
    main()
