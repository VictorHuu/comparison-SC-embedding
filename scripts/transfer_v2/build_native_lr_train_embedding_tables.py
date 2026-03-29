#!/usr/bin/env python3
"""Build conference-style markdown tables for transfer_v2 (embedding x train_dataset).

Requested aggregation:
1) filter by (protocol, clf) setting
2) aggregate seeds within each (train_dataset, test_dataset, embedding)
3) aggregate test_dataset within each (train_dataset, embedding)
4) pivot to embedding x train_dataset markdown matrix

Each cell is formatted as: mean ± std, where std is computed across test_dataset
after seed-level averaging for the same (train_dataset, embedding).

Outputs:
- auroc_embedding_x_train_all_settings.md
- auprc_embedding_x_train_all_settings.md
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from math import isclose
from statistics import mean, pstdev


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build AUROC/AUPRC markdown tables (embedding x train_dataset).")
    p.add_argument("--seed-results", default="results/transfer_v2/embedding_transfer_seed_results_v2.csv")
    p.add_argument("--out-dir", default="results/transfer_v2")
    p.add_argument("--protocol", default="native", help="Protocol to process when --all-settings is false.")
    p.add_argument("--clf", default="lr", help="Classifier to process when --all-settings is false.")
    p.add_argument(
        "--all-settings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate tables for all protocol×clf settings found in seed-results.",
    )
    return p.parse_args()


def read_rows(path: str):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt_cell(mu: float | None, sd: float | None) -> str:
    if mu is None or sd is None:
        return "-"
    return f"{mu:.6f} ± {sd:.6f}"


def red(txt: str) -> str:
    return f'<span style="color:red">{txt}</span>'


def render_markdown_matrix(title: str, row_labels: list[str], col_labels: list[str], data: dict[tuple[str, str], tuple[float, float]]) -> str:
    # per-column best mean (ties allowed)
    best_by_col = {}
    for tr in col_labels:
        vals = [data[(emb, tr)][0] for emb in row_labels if (emb, tr) in data]
        best_by_col[tr] = max(vals) if vals else None

    # baseline per column for italic highlighting
    baseline_by_col = {}
    for tr in col_labels:
        baseline_by_col[tr] = data.get(("baseline", tr), (None, None))[0]

    lines = [f"## {title}", "", "| Embedding | " + " | ".join(col_labels) + " |", "|---|" + "|".join(["---:" for _ in col_labels]) + "|"]
    for emb in row_labels:
        cells = []
        for tr in col_labels:
            mu, sd = data.get((emb, tr), (None, None))
            txt = fmt_cell(mu, sd)

            if mu is not None:
                best = best_by_col.get(tr)
                baseline = baseline_by_col.get(tr)
                is_best = (best is not None) and isclose(mu, best, rel_tol=1e-12, abs_tol=1e-12)
                above_baseline = (
                    emb != "baseline"
                    and baseline is not None
                    and (mu > baseline)
                    and (not isclose(mu, baseline, rel_tol=1e-12, abs_tol=1e-12))
                )
                if is_best and above_baseline:
                    txt = f"**{red(txt)}**"
                elif is_best:
                    txt = f"**{txt}**"
                elif above_baseline:
                    txt = red(txt)

            cells.append(txt)
        lines.append("| " + emb + " | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows = read_rows(args.seed_results)
    norm = [{str(k).strip().lower(): v for k, v in r.items()} for r in rows]

    need = {"train_dataset", "test_dataset", "protocol", "embedding", "clf", "seed", "auroc", "auprc"}
    miss = need - set(norm[0].keys()) if norm else need
    if miss:
        raise ValueError(f"Missing required columns: {sorted(miss)}")

    settings_all = sorted({(str(r["protocol"]).strip().lower(), str(r["clf"]).strip().lower()) for r in norm})
    settings = settings_all if args.all_settings else [(args.protocol.lower(), args.clf.lower())]

    au_sections: list[str] = []
    ap_sections: list[str] = []
    generated = 0
    for protocol, clf in settings:
        filt = [
            r
            for r in norm
            if str(r["protocol"]).strip().lower() == protocol
            and str(r["clf"]).strip().lower() == clf
            and str(r["train_dataset"]) != str(r["test_dataset"])
        ]
        if not filt:
            print(f"[WARN] skip setting protocol={protocol}, clf={clf}: no rows")
            continue

        # Step 1: seed aggregation per (train, test, embedding)
        by_pair_emb_au = defaultdict(list)
        by_pair_emb_ap = defaultdict(list)
        for r in filt:
            key = (str(r["train_dataset"]), str(r["test_dataset"]), str(r["embedding"]))
            by_pair_emb_au[key].append(float(r["auroc"]))
            by_pair_emb_ap[key].append(float(r["auprc"]))

        pair_rows = []
        for key in sorted(by_pair_emb_au.keys()):
            tr, te, emb = key
            pair_rows.append(
                {
                    "train_dataset": tr,
                    "test_dataset": te,
                    "embedding": emb,
                    "mean_auroc_seed": mean(by_pair_emb_au[key]),
                    "mean_auprc_seed": mean(by_pair_emb_ap[key]),
                }
            )

        # Step 2: test aggregation per (embedding, train) -> mean/std across test datasets
        by_train_emb_au = defaultdict(list)
        by_train_emb_ap = defaultdict(list)
        for r in pair_rows:
            key = (r["embedding"], r["train_dataset"])
            by_train_emb_au[key].append(float(r["mean_auroc_seed"]))
            by_train_emb_ap[key].append(float(r["mean_auprc_seed"]))

        embeddings = sorted({k[0] for k in by_train_emb_au.keys()})
        trains = sorted({k[1] for k in by_train_emb_au.keys()})

        au_data: dict[tuple[str, str], tuple[float, float]] = {}
        ap_data: dict[tuple[str, str], tuple[float, float]] = {}
        for k, vals in by_train_emb_au.items():
            au_data[k] = (mean(vals), pstdev(vals) if len(vals) > 1 else 0.0)
        for k, vals in by_train_emb_ap.items():
            ap_data[k] = (mean(vals), pstdev(vals) if len(vals) > 1 else 0.0)

        au_sections.append(
            render_markdown_matrix(
            f"{protocol} + {clf} | AUROC matrix (embedding × train_dataset)",
            embeddings,
            trains,
            au_data,
        ))
        ap_sections.append(
            render_markdown_matrix(
            f"{protocol} + {clf} | AUPRC matrix (embedding × train_dataset)",
            embeddings,
            trains,
            ap_data,
        ))

        print(f"[INFO] setting={protocol}+{clf}, shape: {len(embeddings)} x {len(trains)}")
        generated += 1

    if generated == 0:
        raise ValueError("No output tables generated. Check input file and filter settings.")

    au_md = os.path.join(args.out_dir, "auroc_embedding_x_train_all_settings.md")
    ap_md = os.path.join(args.out_dir, "auprc_embedding_x_train_all_settings.md")
    with open(au_md, "w", encoding="utf-8") as f:
        f.write("# AUROC matrices by setting (embedding × train_dataset)\n\n")
        f.write("\n".join(au_sections))
    with open(ap_md, "w", encoding="utf-8") as f:
        f.write("# AUPRC matrices by setting (embedding × train_dataset)\n\n")
        f.write("\n".join(ap_sections))
    print(f"[OK] wrote {au_md}")
    print(f"[OK] wrote {ap_md}")


if __name__ == "__main__":
    main()
