#!/usr/bin/env python3
"""Forensic debugger for embedding aggregation pipeline (v2)."""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

TARGET_EMB = ["baseline", "minus", "scgpt_human","v4_bias_rec_best","v4_plain_best","v4_type_pe_best"]
THRESHOLDS = [0.00, 0.05, 0.10, 0.15, 0.20]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="transfer/embedding_transfer_seed_results_v2.csv")
    p.add_argument("--quality", default="transfer_v2/pair_diagnostics.csv")
    p.add_argument("--summary-results", default="transfer/embedding_transfer_summary_v2.csv", help="Optional summary-level CSV for consistency audit.")
    p.add_argument("--out-dir", default="transfer")
    p.add_argument("--close-margin-ratio", type=float, default=0.20)
    p.add_argument("--high-ratio-threshold", type=float, default=50.0)
    p.add_argument("--disable-protocol-dedup", action="store_true", help="Do not collapse identical protocol slots.")
    return p.parse_args()


def normalize_results(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"train_dataset", "test_dataset", "embedding"}
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"results missing required columns: {sorted(miss)}")

    if "protocol" not in df.columns:
        df["protocol"] = "pooled"
    if "clf" not in df.columns:
        df["clf"] = "pooled"

    # metric column detection
    au_col = next((c for c in ["auroc", "mean_auroc", "auc"] if c in df.columns), None)
    ap_col = next((c for c in ["auprc", "mean_auprc", "ap"] if c in df.columns), None)
    if not au_col or not ap_col:
        raise ValueError("results must include AUROC and AUPRC columns")

    df["embedding_norm"] = df["embedding"].astype(str).str.lower()
    df = df[df["train_dataset"] != df["test_dataset"]].copy()

    # keep target embeddings only
    pres = set(df["embedding_norm"].unique())
    missing_emb = [e for e in TARGET_EMB if e not in pres]
    if missing_emb:
        raise ValueError(f"missing required embeddings: {missing_emb}; present={sorted(pres)}")

    df = df[df["embedding_norm"].isin(TARGET_EMB)].copy()

    long = pd.concat(
        [
            df[["train_dataset", "test_dataset", "protocol", "clf", "embedding_norm", au_col]].rename(columns={au_col: "value"}).assign(metric="auroc"),
            df[["train_dataset", "test_dataset", "protocol", "clf", "embedding_norm", ap_col]].rename(columns={ap_col: "value"}).assign(metric="auprc"),
        ],
        ignore_index=True,
    )
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["value"])
    return long


def normalize_quality(path: str, high_ratio_threshold: float) -> pd.DataFrame:
    q = pd.read_csv(path)
    q.columns = [c.strip().lower() for c in q.columns]
    for c in ["train_dataset", "test_dataset"]:
        if c not in q.columns:
            raise ValueError(f"quality missing {c}")
    if "canonical_over_raw_ratio" not in q.columns:
        q["canonical_over_raw_ratio"] = np.nan

    # If quality_flag missing, derive from ratio threshold instead of forcing all ok.
    if "quality_flag" not in q.columns:
        q["quality_flag"] = np.where(
            pd.to_numeric(q["canonical_over_raw_ratio"], errors="coerce") > high_ratio_threshold,
            "high_ratio",
            "ok",
        )

    def agg_flag(s: pd.Series) -> str:
        vals = set(str(x).lower() for x in s.dropna())
        return "high_ratio" if "high_ratio" in vals else "ok"

    out = q.groupby(["train_dataset", "test_dataset"], as_index=False).agg(
        quality_flag=("quality_flag", agg_flag),
        canonical_over_raw_ratio=("canonical_over_raw_ratio", "mean"),
    )
    return out


def aggregate_raw_rows(df_long: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw rows per pair/protocol/clf/metric/embedding via mean.
    Keeps n_rows_raw for audit.
    """
    gcols = ["train_dataset", "test_dataset", "protocol", "clf", "metric", "embedding_norm"]
    agg = df_long.groupby(gcols, as_index=False).agg(value=("value", "mean"), n_rows_raw=("value", "size"))
    return agg



def build_seed_summary_consistency_audit(seed_agg: pd.DataFrame, summary_agg: pd.DataFrame) -> pd.DataFrame:
    key = ["train_dataset", "test_dataset", "protocol", "clf", "metric", "embedding_norm"]
    m = seed_agg.merge(summary_agg, on=key, how="outer", suffixes=("_seed", "_summary"), indicator=True)
    m["abs_diff"] = (m["value_seed"] - m["value_summary"]).abs()
    m["status"] = np.where(
        m["_merge"] != "both",
        "missing_side",
        np.where(m["abs_diff"].fillna(0) <= 1e-8, "match", "mismatch"),
    )
    return m

def build_slot_matrix(agg: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # completeness audit first (before filtering)
    key = ["train_dataset", "test_dataset", "protocol", "clf", "metric"]
    completeness = agg.groupby(key).agg(
        embeddings_present=("embedding_norm", lambda s: "|".join(sorted(set(s)))),
        n_embeddings=("embedding_norm", lambda s: len(set(s))),
    ).reset_index()

    pivot = agg.pivot_table(index=key, columns="embedding_norm", values="value", aggfunc="mean").reset_index()
    for e in TARGET_EMB:
        if e not in pivot.columns:
            pivot[e] = np.nan

    pivot["complete_slot"] = pivot[TARGET_EMB].notna().all(axis=1)
    return pivot, completeness


def deduplicate_slots(slot_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Deduplicate ONLY when duplicated across protocol with identical 3-embedding vector.

    Never dedup across clf or metric.
    """
    rows = []
    audit_rows = []

    for (tr, te, clf, metric), g in slot_df.groupby(["train_dataset", "test_dataset", "clf", "metric"]):
        g = g.copy()
        g["sig"] = g[TARGET_EMB].round(12).astype(str).agg("|".join, axis=1)

        for sig, sg in g.groupby("sig"):
            n_before = len(sg)
            keep = sg.iloc[0].copy()
            keep["dedup_group_id"] = f"{tr}->{te}|{clf}|{metric}|{sig}"
            keep["protocols_collapsed"] = "|".join(sorted(set(sg["protocol"].astype(str))))
            keep["n_rows_after_dedup"] = 1
            rows.append(keep)

            audit_rows.append(
                {
                    "train_dataset": tr,
                    "test_dataset": te,
                    "clf": clf,
                    "metric": metric,
                    "dedup_group_id": keep["dedup_group_id"],
                    "n_rows_before_dedup": n_before,
                    "n_rows_after_dedup": 1,
                    "n_duplicate_groups_removed": max(0, n_before - 1),
                    "protocols_collapsed": keep["protocols_collapsed"],
                    "comment": "collapsed_identical_triplet" if n_before > 1 else "no_collapse",
                }
            )

    dedup = pd.DataFrame(rows)
    dedup_audit = pd.DataFrame(audit_rows)
    return dedup, dedup_audit


def tie_type_from_values(vals: Dict[str, float]) -> str:
    a, b, c = sorted(vals.values(), reverse=True)
    if np.isclose(a, b) and np.isclose(b, c):
        return "three_way_tie"
    if np.isclose(a, b):
        return "two_way_tie_first"
    if np.isclose(b, c):
        return "two_way_tie_second"
    return "no_tie"


def assign_points(vals: Dict[str, float]) -> Dict[str, float]:
    ordered = sorted(vals.items(), key=lambda kv: kv[1], reverse=True)
    a, b, c = ordered
    if np.isclose(a[1], b[1]) and np.isclose(b[1], c[1]):
        return {e: 1.0 for e in TARGET_EMB}
    if np.isclose(a[1], b[1]):
        return {a[0]: 1.5, b[0]: 1.5, c[0]: 0.0}
    if np.isclose(b[1], c[1]):
        return {a[0]: 2.0, b[0]: 0.5, c[0]: 0.5}
    return {a[0]: 2.0, b[0]: 1.0, c[0]: 0.0}


def build_slot_level_diagnostics(dedup: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, r in dedup.iterrows():
        vals = {e: float(r[e]) for e in TARGET_EMB}
        pts = assign_points(vals)
        tie = tie_type_from_values(vals)
        winner = max(vals.items(), key=lambda kv: kv[1])[0] if tie != "three_way_tie" else "mixed"

        out.append(
            {
                "train_dataset": r["train_dataset"],
                "test_dataset": r["test_dataset"],
                "protocol": r["protocol"],
                "clf": r["clf"],
                "metric": r["metric"],
                "slot_id": f"{r['train_dataset']}->{r['test_dataset']}|{r['protocol']}|{r['clf']}|{r['metric']}",
                "baseline_value": vals["baseline"],
                "minus_value": vals["minus"],
                "scgpt_human_value": vals["scgpt_human"],
                "baseline_points": pts["baseline"],
                "minus_points": pts["minus"],
                "scgpt_human_points": pts["scgpt_human"],
                "slot_winner": winner,
                "tie_type": tie,
                "dedup_group_id": r["dedup_group_id"],
            }
        )

    s = pd.DataFrame(out)
    s = s.merge(quality, on=["train_dataset", "test_dataset"], how="left")
    s["quality_flag"] = s["quality_flag"].fillna("unknown")
    # reorder cols
    cols = [
        "train_dataset", "test_dataset", "quality_flag", "protocol", "clf", "metric", "slot_id",
        "baseline_value", "minus_value", "scgpt_human_value",
        "baseline_points", "minus_points", "scgpt_human_points",
        "slot_winner", "tie_type", "dedup_group_id",
    ]
    return s[cols]


def build_pair_level_diagnostics(slot_diag: pd.DataFrame, close_margin: float) -> pd.DataFrame:
    rows = []
    for (tr, te), g in slot_diag.groupby(["train_dataset", "test_dataset"]):
        b = float(g["baseline_points"].sum())
        m = float(g["minus_points"].sum())
        s = float(g["scgpt_human_points"].sum())
        n_slots = int(len(g))
        total = 2.0 * n_slots

        ranking = sorted([("baseline", b), ("minus", m), ("scgpt_human", s)], key=lambda kv: kv[1], reverse=True)
        top_e, top_p = ranking[0]
        second_e, second_p = ranking[1]
        margin = top_p - second_p
        frac = (margin / total) if total > 0 else np.nan

        if n_slots == 0:
            winner = "insufficient"
            consistency = "insufficient"
            note = "no_valid_slots"
        elif frac <= close_margin:
            winner = "mixed"
            consistency = "mixed"
            note = "mixed_due_to_threshold"
        else:
            winner = top_e
            consistency = "clear"
            note = "clear_winner"

        qflag = g["quality_flag"].iloc[0] if "quality_flag" in g.columns else "unknown"
        ratio = g["canonical_over_raw_ratio"].iloc[0] if "canonical_over_raw_ratio" in g.columns else np.nan

        rows.append(
            {
                "train_dataset": tr,
                "test_dataset": te,
                "quality_flag": qflag,
                "canonical_over_raw_ratio": ratio,
                "n_slots_used": n_slots,
                "total_possible_points": total,
                "baseline_points": b,
                "minus_points": m,
                "scgpt_human_points": s,
                "top_embedding_by_points": top_e,
                "second_embedding_by_points": second_e,
                "winner": winner,
                "winner_margin": margin,
                "winner_margin_fraction": frac,
                "consistency_flag": consistency,
                "is_mixed_due_to_threshold": int(note == "mixed_due_to_threshold"),
                "note": note,
            }
        )

    pair = pd.DataFrame(rows)
    pair = pair.sort_values(["quality_flag", "winner_margin"], ascending=[True, True])
    return pair


def audit_pair_coverage(raw_long: pd.DataFrame, dedup_diag: pd.DataFrame, quality: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[Tuple[str, str]]]]:
    result_pairs = set(zip(raw_long["train_dataset"], raw_long["test_dataset"]))
    quality_pairs = set(zip(quality["train_dataset"], quality["test_dataset"]))

    datasets = sorted(set(raw_long["train_dataset"]).union(set(raw_long["test_dataset"]))
                      .union(set(quality["train_dataset"]))
                      .union(set(quality["test_dataset"])))
    expected = {(a, b) for a in datasets for b in datasets if a != b}

    # per-pair details
    raw_info = raw_long.groupby(["train_dataset", "test_dataset"]).agg(
        n_rows_raw=("value", "size"),
        embeddings_present=("embedding_norm", lambda s: "|".join(sorted(set(s)))),
        protocols_present=("protocol", lambda s: "|".join(sorted(set(map(str, s))))),
        clfs_present=("clf", lambda s: "|".join(sorted(set(map(str, s))))),
        metrics_present=("metric", lambda s: "|".join(sorted(set(map(str, s))))),
    ).reset_index()

    dedup_info = dedup_diag.groupby(["train_dataset", "test_dataset"]).agg(n_rows_after_dedup=("slot_id", "size")).reset_index()
    qinfo = quality.copy()

    merged = pd.DataFrame(sorted(expected), columns=["train_dataset", "test_dataset"])
    merged = merged.merge(raw_info, on=["train_dataset", "test_dataset"], how="left")
    merged = merged.merge(dedup_info, on=["train_dataset", "test_dataset"], how="left")
    merged = merged.merge(qinfo, on=["train_dataset", "test_dataset"], how="left")

    merged["pair_present_in_results"] = merged.apply(lambda r: (r["train_dataset"], r["test_dataset"]) in result_pairs, axis=1)
    merged["pair_present_in_quality_table"] = merged.apply(lambda r: (r["train_dataset"], r["test_dataset"]) in quality_pairs, axis=1)
    merged["merged_successfully"] = merged["pair_present_in_results"] & merged["pair_present_in_quality_table"]

    merged["quality_flag"] = merged["quality_flag"].fillna("missing")
    merged["n_rows_raw"] = merged["n_rows_raw"].fillna(0).astype(int)
    merged["n_rows_after_dedup"] = merged["n_rows_after_dedup"].fillna(0).astype(int)
    for c in ["embeddings_present", "protocols_present", "clfs_present", "metrics_present"]:
        merged[c] = merged[c].fillna("")

    merged["missing_embedding_flag"] = ~merged["embeddings_present"].apply(lambda s: set(TARGET_EMB).issubset(set(s.split("|")) if s else set()))

    cols = [
        "train_dataset", "test_dataset", "pair_present_in_results", "pair_present_in_quality_table", "merged_successfully",
        "quality_flag", "n_rows_raw", "n_rows_after_dedup", "embeddings_present", "protocols_present", "clfs_present", "metrics_present", "missing_embedding_flag",
    ]

    missing = {
        "missing_in_results": sorted(list(expected - result_pairs)),
        "missing_in_quality": sorted(list(expected - quality_pairs)),
        "results_not_in_quality": sorted(list(result_pairs - quality_pairs)),
        "quality_not_in_results": sorted(list(quality_pairs - result_pairs)),
    }

    return merged[cols], missing


def run_threshold_sensitivity(pair_diag: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for th in THRESHOLDS:
        tmp = pair_diag.copy()
        tmp["winner_th"] = np.where(tmp["n_slots_used"] == 0, "insufficient", np.where(tmp["winner_margin_fraction"] <= th, "mixed", tmp["top_embedding_by_points"]))
        tmp["clear_th"] = (tmp["winner_th"].isin(TARGET_EMB)).astype(int)

        for emb in TARGET_EMB:
            rows.append(
                {
                    "threshold": th,
                    "embedding": emb,
                    "pair_majority_wins": int((tmp["winner_th"] == emb).sum()),
                    "pair_clear_wins": int(((tmp["winner_th"] == emb) & (tmp["clear_th"] == 1)).sum()),
                    "wins_on_ok_pairs": int(((tmp["winner_th"] == emb) & (tmp["quality_flag"] == "ok")).sum()),
                    "wins_on_high_ratio_pairs": int(((tmp["winner_th"] == emb) & (tmp["quality_flag"] == "high_ratio")).sum()),
                }
            )
    return pd.DataFrame(rows)


def high_ratio_details(pair_diag: pd.DataFrame, close_margin: float) -> pd.DataFrame:
    hr = pair_diag[pair_diag["quality_flag"] == "high_ratio"].copy()
    if hr.empty:
        cols = ["train_dataset", "test_dataset", "n_slots_used", "baseline_points", "minus_points", "scgpt_human_points", "winner", "winner_margin", "winner_margin_fraction", "reason_no_winner"]
        return pd.DataFrame(columns=cols)

    def reason(row):
        if row["n_slots_used"] == 0:
            return "missing_data"
        if row["winner"] == "mixed":
            return "threshold_too_strict_or_all_mixed"
        if row["winner"] == "insufficient":
            return "insufficient"
        return "has_winner"

    hr["reason_no_winner"] = hr.apply(reason, axis=1)
    return hr[["train_dataset", "test_dataset", "n_slots_used", "baseline_points", "minus_points", "scgpt_human_points", "winner", "winner_margin", "winner_margin_fraction", "reason_no_winner"]]


def print_debug_report(pair_cov: pd.DataFrame, pair_diag: pd.DataFrame, threshold_df: pd.DataFrame, missing: Dict[str, List[Tuple[str, str]]]) -> None:
    total_expected = len(pair_cov)
    total_merged = int(pair_cov["merged_successfully"].sum())
    ok_n = int((pair_diag["quality_flag"] == "ok").sum())
    hr_n = int((pair_diag["quality_flag"] == "high_ratio").sum())
    miss_emb_pairs = int(pair_cov["missing_embedding_flag"].sum())
    zero_slots = int((pair_diag["n_slots_used"] == 0).sum())
    mixed_n = int((pair_diag["winner"] == "mixed").sum())

    th0 = threshold_df[threshold_df["threshold"] == 0.0].sort_values(["pair_majority_wins", "wins_on_ok_pairs"], ascending=False)
    lead0 = th0.iloc[0]["embedding"] if not th0.empty else "n/a"

    th_ok = threshold_df[threshold_df["threshold"] == 0.20].sort_values(["wins_on_ok_pairs", "pair_majority_wins"], ascending=False)
    lead_ok = th_ok.iloc[0]["embedding"] if not th_ok.empty else "n/a"

    print("=== DEBUG REPORT ===")
    print(f"total expected directed non-self pairs: {total_expected}")
    print(f"total merged pairs (results∩quality): {total_merged}")
    print(f"ok pairs: {ok_n}; high_ratio pairs: {hr_n}")
    print(f"pairs with missing embeddings: {miss_emb_pairs}")
    print(f"pairs with zero valid slots: {zero_slots}")
    print(f"pairs labeled mixed: {mixed_n}")

    if hr_n > 0:
        hr_wins = pair_diag[(pair_diag["quality_flag"] == "high_ratio") & (pair_diag["winner"].isin(TARGET_EMB))]
        if hr_wins.empty:
            print("high_ratio pairs have zero wins because: no pair achieved non-mixed winner under threshold and/or missing slots.")

    print(f"leader at threshold=0.00: {lead0}")
    print(f"leader on ok_only proxy (threshold=0.20 by wins_on_ok_pairs): {lead_ok}")

    print("Missing pair diagnostics:")
    for k, vals in missing.items():
        print(f"- {k}: {len(vals)}")


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    raw = normalize_results(args.results)
    quality = normalize_quality(args.quality, args.high_ratio_threshold)

    agg = aggregate_raw_rows(raw)

    summary_audit = pd.DataFrame()
    if args.summary_results and os.path.exists(args.summary_results):
        summary_long = normalize_results(args.summary_results)
        summary_agg = aggregate_raw_rows(summary_long)
        summary_audit = build_seed_summary_consistency_audit(agg, summary_agg)

    slot_matrix, completeness = build_slot_matrix(agg)

    # keep only complete slots for scoring; incomplete are audited via coverage/completeness
    complete_slots = slot_matrix[slot_matrix["complete_slot"]].copy()
    if args.disable_protocol_dedup:
        dedup = complete_slots.copy()
        dedup["dedup_group_id"] = dedup.apply(lambda r: f"{r.train_dataset}->{r.test_dataset}|{r.protocol}|{r.clf}|{r.metric}", axis=1)
        dedup["protocols_collapsed"] = dedup["protocol"].astype(str)
        dedup["n_rows_after_dedup"] = 1
        dedup_audit_detail = dedup[["train_dataset","test_dataset","clf","metric","dedup_group_id","protocols_collapsed"]].copy()
        dedup_audit_detail["n_rows_before_dedup"] = 1
        dedup_audit_detail["n_rows_after_dedup"] = 1
        dedup_audit_detail["n_duplicate_groups_removed"] = 0
        dedup_audit_detail["comment"] = "dedup_disabled"
    else:
        dedup, dedup_audit_detail = deduplicate_slots(complete_slots)

    slot_diag = build_slot_level_diagnostics(dedup, quality)
    pair_diag = build_pair_level_diagnostics(slot_diag, args.close_margin_ratio)
    pair_cov, missing = audit_pair_coverage(raw, slot_diag, quality)

    # pair-level missing embedding flag refinement using completeness
    comp_pair = completeness.groupby(["train_dataset", "test_dataset"]).agg(
        any_incomplete=("n_embeddings", lambda s: int(any(v < 3 for v in s)))
    ).reset_index()
    pair_cov = pair_cov.merge(comp_pair, on=["train_dataset", "test_dataset"], how="left")
    pair_cov["missing_embedding_flag"] = pair_cov["missing_embedding_flag"] | pair_cov["any_incomplete"].fillna(0).astype(bool)
    pair_cov.drop(columns=["any_incomplete"], inplace=True)

    hr_diag = high_ratio_details(pair_diag, args.close_margin_ratio)
    th = run_threshold_sensitivity(pair_diag)

    # dedup per pair summary
    dedup_audit = dedup_audit_detail.groupby(["train_dataset", "test_dataset"], as_index=False).agg(
        n_rows_before_dedup=("n_rows_before_dedup", "sum"),
        n_rows_after_dedup=("n_rows_after_dedup", "sum"),
        n_duplicate_groups_removed=("n_duplicate_groups_removed", "sum"),
        protocols_collapsed=("protocols_collapsed", lambda s: "||".join(sorted(set(s)))),
        comment=("comment", lambda s: "collapsed" if any(x == "collapsed_identical_triplet" for x in s) else "no_collapse"),
    )

    # outputs
    pair_cov.to_csv(os.path.join(args.out_dir, "pair_coverage_audit.csv"), index=False)
    pair_diag.to_csv(os.path.join(args.out_dir, "pair_level_diagnostics.csv"), index=False)
    slot_diag.to_csv(os.path.join(args.out_dir, "slot_level_diagnostics.csv"), index=False)
    hr_diag.to_csv(os.path.join(args.out_dir, "high_ratio_pair_diagnostics.csv"), index=False)
    th.to_csv(os.path.join(args.out_dir, "threshold_sensitivity.csv"), index=False)
    dedup_audit.to_csv(os.path.join(args.out_dir, "dedup_audit.csv"), index=False)
    if not summary_audit.empty:
        summary_audit.to_csv(os.path.join(args.out_dir, "seed_summary_consistency_audit.csv"), index=False)

    # print heavily collapsed pairs
    heavy = dedup_audit.sort_values("n_duplicate_groups_removed", ascending=False).head(10)
    print("Top collapsed pairs (dedup):")
    if not heavy.empty:
        print(heavy[["train_dataset", "test_dataset", "n_duplicate_groups_removed", "comment"]].to_string(index=False))
    else:
        print("(none)")

    print_debug_report(pair_cov, pair_diag, th, missing)


if __name__ == "__main__":
    main()
