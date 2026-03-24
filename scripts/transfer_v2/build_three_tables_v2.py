#!/usr/bin/env python3
"""Generate exactly 3 tables for v2 transfer embedding comparison.

Outputs (total 3 tables including seed table):
- [existing] embedding_transfer_seed_results_v2.csv
- data_description_table.csv
- winner_table.csv
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

EMB = ["baseline", "minus", "scgpt_human"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed-results", default="transfer/embedding_transfer_seed_results_v2.csv")
    p.add_argument("--quality", default="transfer_v2/pair_diagnostics.csv")
    p.add_argument("--out-dir", default="transfer")
    p.add_argument("--close-margin-ratio", type=float, default=0.20)
    p.add_argument(
        "--no-controlled-subtables",
        action="store_true",
        help="Disable winner sub-tables (default behavior is to emit them).",
    )
    return p.parse_args()


def assign_points(vals):
    ordered = sorted(vals.items(), key=lambda kv: kv[1], reverse=True)
    a, b, c = ordered
    if np.isclose(a[1], b[1]) and np.isclose(b[1], c[1]):
        return {e: 1.0 for e in EMB}
    if np.isclose(a[1], b[1]):
        return {a[0]: 1.5, b[0]: 1.5, c[0]: 0.0}
    if np.isclose(b[1], c[1]):
        return {a[0]: 2.0, b[0]: 0.5, c[0]: 0.5}
    return {a[0]: 2.0, b[0]: 1.0, c[0]: 0.0}


def sanitize_token(s: str) -> str:
    return str(s).strip().lower().replace("/", "_").replace(" ", "_")


def build_winner_from_full(full: pd.DataFrame, close_margin_ratio: float) -> pd.DataFrame:
    slots = pd.concat(
        [
            full[["train_dataset", "test_dataset", "protocol", "clf", "embedding", "mean_auroc"]]
            .rename(columns={"mean_auroc": "value"})
            .assign(metric="auroc"),
            full[["train_dataset", "test_dataset", "protocol", "clf", "embedding", "mean_auprc"]]
            .rename(columns={"mean_auprc": "value"})
            .assign(metric="auprc"),
        ],
        ignore_index=True,
    )

    slot_points = []
    for (tr, te, protocol, clf, metric), g in slots.groupby(["train_dataset", "test_dataset", "protocol", "clf", "metric"]):
        if set(g["embedding"]) != set(EMB):
            continue
        vals = {e: float(g.loc[g["embedding"] == e, "value"].iloc[0]) for e in EMB}
        pts = assign_points(vals)
        slot_points.append(
            {
                "train_dataset": tr,
                "test_dataset": te,
                "baseline_points": pts["baseline"],
                "minus_points": pts["minus"],
                "scgpt_human_points": pts["scgpt_human"],
            }
        )

    if not slot_points:
        return pd.DataFrame(
            columns=[
                "train_dataset",
                "test_dataset",
                "n_slots_used",
                "baseline_points",
                "minus_points",
                "scgpt_human_points",
                "winner",
                "winner_margin",
                "winner_margin_fraction",
            ]
        )

    sp = pd.DataFrame(slot_points)
    winner_rows = []
    for (tr, te), g in sp.groupby(["train_dataset", "test_dataset"]):
        b = g["baseline_points"].sum()
        m = g["minus_points"].sum()
        s = g["scgpt_human_points"].sum()
        n_slots = len(g)
        total = 2.0 * n_slots

        ranking = sorted([("baseline", b), ("minus", m), ("scgpt_human", s)], key=lambda kv: kv[1], reverse=True)
        top_e, top_p = ranking[0]
        second_p = ranking[1][1]
        margin = top_p - second_p
        frac = margin / total if total > 0 else np.nan
        winner = "mixed" if frac <= close_margin_ratio else top_e

        winner_rows.append(
            {
                "train_dataset": tr,
                "test_dataset": te,
                "n_slots_used": n_slots,
                "baseline_points": b,
                "minus_points": m,
                "scgpt_human_points": s,
                "winner": winner,
                "winner_margin": margin,
                "winner_margin_fraction": frac,
            }
        )

    return pd.DataFrame(winner_rows).sort_values(["winner", "winner_margin"], ascending=[True, True])


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.seed_results)
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"train_dataset", "test_dataset", "protocol", "clf", "embedding", "auroc", "auprc"}
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"missing columns in seed results: {sorted(miss)}")

    df = df[df["train_dataset"] != df["test_dataset"]].copy()
    df["embedding"] = df["embedding"].astype(str).str.lower()
    df = df[df["embedding"].isin(EMB)].copy()

    # aggregate seed rows into pair/protocol/clf/embedding means
    full = (
        df.groupby(["train_dataset", "test_dataset", "protocol", "clf", "embedding"], as_index=False)
        .agg(
            n_seeds=("auroc", "size"),
            mean_auroc=("auroc", "mean"),
            std_auroc=("auroc", "std"),
            mean_auprc=("auprc", "mean"),
            std_auprc=("auprc", "std"),
        )
    )

    # table 2) data description table (pair-level metadata)
    desc = (
        full.groupby(["train_dataset", "test_dataset"], as_index=False)
        .agg(
            n_rows=("embedding", "size"),
            embeddings_present=("embedding", lambda s: "|".join(sorted(set(s)))),
            protocols_present=("protocol", lambda s: "|".join(sorted(set(s)))),
            clfs_present=("clf", lambda s: "|".join(sorted(set(s)))),
            min_seeds=("n_seeds", "min"),
            max_seeds=("n_seeds", "max"),
        )
    )

    if os.path.exists(args.quality):
        q = pd.read_csv(args.quality)
        q.columns = [c.strip().lower() for c in q.columns]
        for c in ["quality_flag", "canonical_over_raw_ratio"]:
            if c not in q.columns:
                q[c] = np.nan
        q = q[["train_dataset", "test_dataset", "quality_flag", "canonical_over_raw_ratio"]].copy()
        desc = desc.merge(q, on=["train_dataset", "test_dataset"], how="left")

    # table 3) winner table from aggregated means (slot scoring)
    winner = build_winner_from_full(full, args.close_margin_ratio)

    desc_path = os.path.join(args.out_dir, "data_description_table.csv")
    win_path = os.path.join(args.out_dir, "winner_table.csv")

    desc.to_csv(desc_path, index=False)
    winner.to_csv(win_path, index=False)

    print(f"[OK] wrote {desc_path}")
    print(f"[OK] wrote {win_path}")

    if not args.no_controlled_subtables:
        sub_dir = os.path.join(args.out_dir, "winner_subtables")
        os.makedirs(sub_dir, exist_ok=True)

        for clf, g in full.groupby("clf"):
            w = build_winner_from_full(g, args.close_margin_ratio)
            out = os.path.join(sub_dir, f"winner_by_clf_{sanitize_token(clf)}.csv")
            w.to_csv(out, index=False)
            print(f"[OK] wrote {out}")

        for protocol, g in full.groupby("protocol"):
            w = build_winner_from_full(g, args.close_margin_ratio)
            out = os.path.join(sub_dir, f"winner_by_protocol_{sanitize_token(protocol)}.csv")
            w.to_csv(out, index=False)
            print(f"[OK] wrote {out}")

        for (protocol, clf), g in full.groupby(["protocol", "clf"]):
            w = build_winner_from_full(g, args.close_margin_ratio)
            out = os.path.join(
                sub_dir,
                f"winner_by_protocol_{sanitize_token(protocol)}_clf_{sanitize_token(clf)}.csv",
            )
            w.to_csv(out, index=False)
            print(f"[OK] wrote {out}")


if __name__ == "__main__":
    main()
