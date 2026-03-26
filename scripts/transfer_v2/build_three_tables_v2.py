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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed-results", default="results/transfer_v2/embedding_transfer_seed_results_v2.csv")
    p.add_argument("--quality", default="results/transfer_v2/pair_diagnostics.csv")
    p.add_argument("--out-dir", default="results/transfer_v2")
    p.add_argument("--close-margin-ratio", type=float, default=0.20)
    p.add_argument(
        "--no-controlled-subtables",
        action="store_true",
        help="Disable winner sub-tables (default behavior is to emit them).",
    )
    return p.parse_args()


def assign_points(vals):
    """Assign rank points for arbitrary number of embeddings.

    Highest score gets N-1 points, second gets N-2, ... lowest gets 0.
    Ties receive averaged rank points.
    """
    emb = list(vals.keys())
    n = len(emb)
    if n == 0:
        return {}
    if n == 1:
        return {emb[0]: 0.0}

    ordered = sorted(vals.items(), key=lambda kv: kv[1], reverse=True)
    out = {e: 0.0 for e in emb}
    i = 0
    while i < n:
        j = i
        while j + 1 < n and np.isclose(ordered[j + 1][1], ordered[i][1]):
            j += 1
        # rank points descending: n-1, n-2, ..., 0
        pts = [float(n - 1 - r) for r in range(i, j + 1)]
        avg = float(np.mean(pts))
        for k in range(i, j + 1):
            out[ordered[k][0]] = avg
        i = j + 1
    return out


def sanitize_token(s: str) -> str:
    return str(s).strip().lower().replace("/", "_").replace(" ", "_")


def build_winner_from_full(full: pd.DataFrame, close_margin_ratio: float, emb_list: list[str]) -> pd.DataFrame:
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
        if set(g["embedding"]) != set(emb_list):
            continue
        vals = {e: float(g.loc[g["embedding"] == e, "value"].iloc[0]) for e in emb_list}
        pts = assign_points(vals)
        rec = {"train_dataset": tr, "test_dataset": te}
        for e in emb_list:
            rec[f"{sanitize_token(e)}_points"] = pts[e]
        slot_points.append(rec)

    if not slot_points:
        point_cols = [f"{sanitize_token(e)}_points" for e in emb_list]
        return pd.DataFrame(
            columns=[
                "train_dataset",
                "test_dataset",
                "n_slots_used",
                *point_cols,
                "winner",
                "winner_margin",
                "winner_margin_fraction",
            ]
        )

    sp = pd.DataFrame(slot_points)
    winner_rows = []
    for (tr, te), g in sp.groupby(["train_dataset", "test_dataset"]):
        totals = {e: float(g[f"{sanitize_token(e)}_points"].sum()) for e in emb_list}
        n_slots = len(g)
        total = float(max(len(emb_list) - 1, 1) * n_slots)

        ranking = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
        top_e, top_p = ranking[0]
        second_p = ranking[1][1]
        margin = top_p - second_p
        frac = margin / total if total > 0 else np.nan
        winner = "mixed" if frac <= close_margin_ratio else top_e

        rec = {"train_dataset": tr, "test_dataset": te, "n_slots_used": n_slots}
        for e in emb_list:
            rec[f"{sanitize_token(e)}_points"] = totals[e]
        rec.update({"winner": winner, "winner_margin": margin, "winner_margin_fraction": frac})
        winner_rows.append(rec)

    return pd.DataFrame(winner_rows).sort_values(["winner", "winner_margin"], ascending=[True, True])


def build_vs_baseline_counts(full: pd.DataFrame, emb_list: list[str], group_fields: list[str] | None = None) -> pd.DataFrame:
    """Count win/loss/tie records for each embedding against baseline."""
    group_fields = group_fields or []
    base = "baseline"
    if base not in emb_list:
        return pd.DataFrame(
            columns=[*group_fields, "embedding", "n_slots_compared", "wins_vs_baseline", "losses_vs_baseline", "ties_vs_baseline", "win_rate_vs_baseline"]
        )

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

    key_cols = ["train_dataset", "test_dataset", "protocol", "clf", "metric"]
    rows = []
    for key, g in slots.groupby(key_cols):
        vals = {r["embedding"]: float(r["value"]) for _, r in g.iterrows()}
        if base not in vals:
            continue
        base_v = vals[base]
        ctx = dict(zip(key_cols, key))
        for emb in emb_list:
            if emb == base or emb not in vals:
                continue
            v = vals[emb]
            win = int(v > base_v and not np.isclose(v, base_v))
            loss = int(v < base_v and not np.isclose(v, base_v))
            tie = int(np.isclose(v, base_v))
            rows.append({**ctx, "embedding": emb, "win": win, "loss": loss, "tie": tie})

    if not rows:
        return pd.DataFrame(
            columns=[*group_fields, "embedding", "n_slots_compared", "wins_vs_baseline", "losses_vs_baseline", "ties_vs_baseline", "win_rate_vs_baseline"]
        )

    df = pd.DataFrame(rows)
    agg_fields = [*group_fields, "embedding"]
    out = (
        df.groupby(agg_fields, as_index=False)
        .agg(
            n_slots_compared=("win", "size"),
            wins_vs_baseline=("win", "sum"),
            losses_vs_baseline=("loss", "sum"),
            ties_vs_baseline=("tie", "sum"),
        )
    )
    out["win_rate_vs_baseline"] = out["wins_vs_baseline"] / out["n_slots_compared"].replace(0, np.nan)
    return out.sort_values([*group_fields, "wins_vs_baseline", "win_rate_vs_baseline"], ascending=[*(True for _ in group_fields), False, False])


def build_native_lr_aggregate(full: pd.DataFrame) -> pd.DataFrame:
    """Compact aggregate over dataset pairs for protocol=native, clf=lr."""
    sub = full[(full["protocol"] == "native") & (full["clf"] == "lr")].copy()
    if sub.empty:
        return pd.DataFrame(columns=["embedding", "n_pairs", "agg_mean_auroc", "agg_mean_auprc", "agg_mean_score"])

    sub["pair_key"] = sub["train_dataset"].astype(str) + "->" + sub["test_dataset"].astype(str)
    out = (
        sub.groupby("embedding", as_index=False)
        .agg(
            n_pairs=("pair_key", "nunique"),
            agg_mean_auroc=("mean_auroc", "mean"),
            agg_mean_auprc=("mean_auprc", "mean"),
        )
    )
    out["agg_mean_score"] = (out["agg_mean_auroc"] + out["agg_mean_auprc"]) / 2.0
    return out.sort_values("embedding").reset_index(drop=True)


def write_native_lr_markdown(out_path: str, agg: pd.DataFrame) -> None:
    with open(out_path, "w") as f:
        f.write("# Native + LR aggregated summary (30 pairs)\n\n")
        f.write("| embedding | n_pairs | agg_mean_auroc | agg_mean_auprc | agg_mean_score |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        if agg.empty:
            f.write("| _No data_ |  |  |  |  |\n")
            return

        best_auroc = agg["agg_mean_auroc"].max(skipna=True)
        best_auprc = agg["agg_mean_auprc"].max(skipna=True)
        best_score = agg["agg_mean_score"].max(skipna=True)

        for _, r in agg.iterrows():
            emb = str(r["embedding"])
            n_pairs = int(r["n_pairs"])
            auroc = f"{r['agg_mean_auroc']:.6f}"
            auprc = f"{r['agg_mean_auprc']:.6f}"
            score = f"{r['agg_mean_score']:.6f}"

            if np.isclose(r["agg_mean_auroc"], best_auroc):
                auroc = f"**{auroc}**"
            if np.isclose(r["agg_mean_auprc"], best_auprc):
                auprc = f"**{auprc}**"
            if np.isclose(r["agg_mean_score"], best_score):
                emb = f"**{emb}**"
                score = f"**{score}**"

            f.write(f"| {emb} | {n_pairs} | {auroc} | {auprc} | {score} |\n")


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
    emb_list = sorted(df["embedding"].dropna().unique().tolist())
    if len(emb_list) < 2:
        raise ValueError("Need at least 2 embeddings in seed results to build winner table.")

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
    winner = build_winner_from_full(full, args.close_margin_ratio, emb_list)

    desc_path = os.path.join(args.out_dir, "data_description_table.csv")
    win_path = os.path.join(args.out_dir, "winner_table.csv")

    desc.to_csv(desc_path, index=False)
    winner.to_csv(win_path, index=False)

    print(f"[OK] wrote {desc_path}")
    print(f"[OK] wrote {win_path}")

    # compact native+LR aggregate (conference-style compact table)
    native_lr = build_native_lr_aggregate(full)
    native_lr_csv = os.path.join(args.out_dir, "native_lr_pair30_aggregated_summary.csv")
    native_lr_md = os.path.join(args.out_dir, "native_lr_pair30_aggregated_summary.md")
    native_lr.to_csv(native_lr_csv, index=False, float_format="%.6f")
    write_native_lr_markdown(native_lr_md, native_lr)
    print(f"[OK] wrote {native_lr_csv}")
    print(f"[OK] wrote {native_lr_md}")

    # Extra series: per-embedding counts against baseline
    vb_dir = os.path.join(args.out_dir, "vs_baseline")
    os.makedirs(vb_dir, exist_ok=True)
    vb_all = build_vs_baseline_counts(full, emb_list, group_fields=[])
    vb_all_path = os.path.join(vb_dir, "vs_baseline_overall.csv")
    vb_all.to_csv(vb_all_path, index=False)
    print(f"[OK] wrote {vb_all_path}")

    vb_protocol = build_vs_baseline_counts(full, emb_list, group_fields=["protocol"])
    vb_protocol_path = os.path.join(vb_dir, "vs_baseline_by_protocol.csv")
    vb_protocol.to_csv(vb_protocol_path, index=False)
    print(f"[OK] wrote {vb_protocol_path}")

    vb_clf = build_vs_baseline_counts(full, emb_list, group_fields=["clf"])
    vb_clf_path = os.path.join(vb_dir, "vs_baseline_by_clf.csv")
    vb_clf.to_csv(vb_clf_path, index=False)
    print(f"[OK] wrote {vb_clf_path}")

    vb_protocol_clf = build_vs_baseline_counts(full, emb_list, group_fields=["protocol", "clf"])
    vb_protocol_clf_path = os.path.join(vb_dir, "vs_baseline_by_protocol_clf.csv")
    vb_protocol_clf.to_csv(vb_protocol_clf_path, index=False)
    print(f"[OK] wrote {vb_protocol_clf_path}")

    if not args.no_controlled_subtables:
        sub_dir = os.path.join(args.out_dir, "winner_subtables")
        os.makedirs(sub_dir, exist_ok=True)

        for clf, g in full.groupby("clf"):
            w = build_winner_from_full(g, args.close_margin_ratio, emb_list)
            out = os.path.join(sub_dir, f"winner_by_clf_{sanitize_token(clf)}.csv")
            w.to_csv(out, index=False)
            print(f"[OK] wrote {out}")

        for protocol, g in full.groupby("protocol"):
            w = build_winner_from_full(g, args.close_margin_ratio, emb_list)
            out = os.path.join(sub_dir, f"winner_by_protocol_{sanitize_token(protocol)}.csv")
            w.to_csv(out, index=False)
            print(f"[OK] wrote {out}")

        for (protocol, clf), g in full.groupby(["protocol", "clf"]):
            w = build_winner_from_full(g, args.close_margin_ratio, emb_list)
            out = os.path.join(
                sub_dir,
                f"winner_by_protocol_{sanitize_token(protocol)}_clf_{sanitize_token(clf)}.csv",
            )
            w.to_csv(out, index=False)
            print(f"[OK] wrote {out}")


if __name__ == "__main__":
    main()
