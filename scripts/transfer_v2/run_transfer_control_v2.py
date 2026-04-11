#!/usr/bin/env python3
"""Full protocol diagnostics for transfer-v2 confound control.

Compares native / strict / coverage_matched / topology_matched with per-direction
statistics and shift plots (histograms + ECDF).
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run transfer-v2 diagnostics.")
    p.add_argument("--h5ad-root", default="processed/native")
    p.add_argument("--pair-manifest", default="results/transfer_v2/pair_manifest.csv")
    p.add_argument("--pair-diag-csv", default="results/transfer_v2/pair_diagnostics.csv")
    p.add_argument("--out-dir", default="results/transfer_v2")
    return p.parse_args()


def read_csv(path: str):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def canonical(g: str, mode: str = "upper") -> str:
    s = str(g).strip()
    return s.upper() if mode == "upper" else s.lower() if mode == "lower" else s


def normalize_edges_table(obj, gene_to_idx):
    if isinstance(obj, dict):
        tf_raw = obj.get("tf", obj.get("source", []))
        tg_raw = obj.get("tg", obj.get("target", []))
        y_raw = obj.get("y", obj.get("label", []))
    else:
        arr = np.asarray(obj)
        tf_raw, tg_raw, y_raw = arr[:, 0], arr[:, 1], arr[:, 2]

    def idx(x):
        s = str(x)
        if s in gene_to_idx:
            return gene_to_idx[s]
        return int(float(s))

    return np.array([idx(x) for x in tf_raw]), np.array([idx(x) for x in tg_raw]), np.array([int(float(v)) for v in y_raw])


def get_split(adata, key):
    if key in adata.uns:
        return adata.uns[key]
    if "edge_splits" in adata.uns and isinstance(adata.uns["edge_splits"], dict) and key in adata.uns["edge_splits"]:
        return adata.uns["edge_splits"][key]
    raise KeyError(f"Missing edge split '{key}' in adata.uns.")


def mat_from_adata(adata: ad.AnnData):
    mat = adata.layers["counts"] if "counts" in adata.layers else adata.X
    if sparse.issparse(mat):
        return mat.tocsr()
    return np.asarray(mat, dtype=np.float32)


def build_proxy_edge_splits_from_h5ad(adata: ad.AnnData, max_genes: int = 256, max_edges_each: int = 12000, seed: int = 0):
    """Fallback split builder for h5ad files without explicit Train/Validation/Test."""
    if "pseudotime" not in adata.obs.columns:
        raise KeyError("Missing explicit edge splits and obs['pseudotime']; cannot build proxy splits.")

    mat = mat_from_adata(adata)
    n_cells, n_genes = adata.n_obs, adata.n_vars
    if n_genes < 30:
        raise ValueError(f"Too few genes ({n_genes}) to construct proxy edge splits.")

    if sparse.issparse(mat):
        mean = np.asarray(mat.mean(axis=0)).ravel()
        mean2 = np.asarray(mat.power(2).mean(axis=0)).ravel()
        var = np.maximum(mean2 - mean**2, 0.0)
    else:
        var = np.var(mat, axis=0)

    k = int(min(max_genes, n_genes))
    sel = np.argsort(-var)[:k]
    rng = np.random.default_rng(seed)
    cell_idx = np.sort(rng.choice(np.arange(n_cells), size=min(n_cells, 3000), replace=False))
    if sparse.issparse(mat):
        X = mat[cell_idx][:, sel].toarray().astype(np.float32)
    else:
        X = np.asarray(mat[cell_idx][:, sel], dtype=np.float32)

    X -= X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    X /= sd
    C = np.corrcoef(X, rowvar=False)
    iu, ju = np.triu_indices(k, k=1)
    s = np.abs(C[iu, ju])
    hi = float(np.quantile(s, 0.90))
    lo = float(np.quantile(s, 0.10))
    pos_idx = np.where(s >= hi)[0]
    neg_idx = np.where(s <= lo)[0]
    n_take = int(min(len(pos_idx), len(neg_idx), max_edges_each))
    if n_take < 100:
        raise ValueError("Failed to build sufficient proxy edge splits from h5ad.")

    pos_pick = rng.choice(pos_idx, size=n_take, replace=False)
    neg_pick = rng.choice(neg_idx, size=n_take, replace=False)
    tf_sel = np.concatenate([iu[pos_pick], iu[neg_pick]])
    tg_sel = np.concatenate([ju[pos_pick], ju[neg_pick]])
    y = np.concatenate([np.ones(n_take, dtype=int), np.zeros(n_take, dtype=int)])
    tf = sel[tf_sel]
    tg = sel[tg_sel]
    perm = rng.permutation(len(y))
    tf, tg, y = tf[perm], tg[perm], y[perm]

    n = len(y)
    n_train = int(0.6 * n)
    n_val = int(0.2 * n)
    tr = slice(0, n_train)
    va = slice(n_train, n_train + n_val)
    te = slice(n_train + n_val, n)
    return (tf[tr], tg[tr], y[tr]), (tf[va], tg[va], y[va]), (tf[te], tg[te], y[te])


def dataset_struct(adata):
    genes = np.asarray(adata.var_names.astype(str))
    m = {g: i for i, g in enumerate(genes)}
    has_explicit = (
        ("Train_set" in adata.uns and "Validation_set" in adata.uns and "Test_set" in adata.uns)
        or (
            "edge_splits" in adata.uns
            and isinstance(adata.uns["edge_splits"], dict)
            and all(k in adata.uns["edge_splits"] for k in ["Train_set", "Validation_set", "Test_set"])
        )
    )
    if has_explicit:
        tr_tf, tr_tg, tr_y = normalize_edges_table(get_split(adata, "Train_set"), m)
        va_tf, va_tg, va_y = normalize_edges_table(get_split(adata, "Validation_set"), m)
        te_tf, te_tg, te_y = normalize_edges_table(get_split(adata, "Test_set"), m)
    else:
        (tr_tf, tr_tg, tr_y), (va_tf, va_tg, va_y), (te_tf, te_tg, te_y) = build_proxy_edge_splits_from_h5ad(adata)
    tf = np.concatenate([tr_tf, va_tf, te_tf])
    tg = np.concatenate([tr_tg, va_tg, te_tg])
    y = np.concatenate([tr_y, va_y, te_y])
    n = len(genes)
    deg = np.zeros(n)
    trf = np.zeros(n)
    tef = np.zeros(n)
    pos = np.zeros(n)
    cnt = np.zeros(n)
    for a, b, lab in zip(tf, tg, y):
        deg[a] += 1
        deg[b] += 1
        pos[a] += lab
        pos[b] += lab
        cnt[a] += 1
        cnt[b] += 1
    for a, b in zip(np.concatenate([tr_tf, va_tf]), np.concatenate([tr_tg, va_tg])):
        trf[a] += 1
        trf[b] += 1
    for a, b in zip(te_tf, te_tg):
        tef[a] += 1
        tef[b] += 1
    tf_proxy = np.zeros(n)
    tf_proxy[np.unique(tf)] = 1
    return {
        "genes": genes.tolist(),
        "degree": deg,
        "train_node_freq": trf,
        "test_node_freq": tef,
        "tf_proxy": tf_proxy,
        "pos_edge_ratio": np.divide(pos, cnt, out=np.zeros_like(pos), where=cnt > 0),
        "global_pos_edge_ratio": float(y.mean()) if len(y) else np.nan,
    }


def subset_stats(meta, idx, canonical_over_raw_ratio):
    deg = meta["degree"][idx]
    trf = meta["train_node_freq"][idx]
    tef = meta["test_node_freq"][idx]
    tfp = meta["tf_proxy"][idx]
    hub_cut = np.quantile(meta["degree"], 0.95) if len(meta["degree"]) else 0.0
    hub_fraction = float((deg >= hub_cut).mean()) if len(deg) else np.nan
    return {
        "n_genes": len(idx),
        "canonical_over_raw_ratio": canonical_over_raw_ratio,
        "mean_degree": float(np.mean(deg)) if len(deg) else np.nan,
        "median_degree": float(np.median(deg)) if len(deg) else np.nan,
        "mean_train_node_freq": float(np.mean(trf)) if len(trf) else np.nan,
        "median_train_node_freq": float(np.median(trf)) if len(trf) else np.nan,
        "mean_test_node_freq": float(np.mean(tef)) if len(tef) else np.nan,
        "median_test_node_freq": float(np.median(tef)) if len(tef) else np.nan,
        "tf_proxy_fraction": float(np.mean(tfp)) if len(tfp) else np.nan,
        "pos_edge_ratio": float(np.mean(meta["pos_edge_ratio"][idx])) if len(idx) else np.nan,
        "hub_fraction": hub_fraction,
    }


def ecdf(vals):
    x = np.sort(np.asarray(vals))
    if len(x) == 0:
        return np.array([0.0]), np.array([0.0])
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def resolve_existing_path(user_path: str, candidates: list[str], label: str, required: bool = True) -> str:
    if user_path:
        if os.path.exists(user_path):
            return user_path
        raise FileNotFoundError(f"{label} not found: {user_path}")
    for p in candidates:
        if p and os.path.exists(p):
            return p
    if required:
        raise FileNotFoundError(f"{label} not found. Tried: {candidates}")
    return ""


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    plot_dir = os.path.join(args.out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Align with other transfer_v2 scripts: diagnostics inputs default to <out-dir>/...
    manifest_path = resolve_existing_path(args.pair_manifest, [args.pair_manifest], label="pair manifest", required=True)
    pair_diag_path = resolve_existing_path(args.pair_diag_csv, [args.pair_diag_csv], label="pair diagnostics", required=False)
    manifest = read_csv(manifest_path)
    pair_diag = { (r["train_dataset"], r["test_dataset"]): r for r in read_csv(pair_diag_path) } if pair_diag_path else {}

    all_ds = sorted({r["train_dataset"] for r in manifest}.union({r["test_dataset"] for r in manifest}))
    ds = {d: dataset_struct(ad.read_h5ad(os.path.join(args.h5ad_root, f"{d}.h5ad"))) for d in all_ds}

    rows = []
    by_pair_protocol = defaultdict(dict)

    for r in manifest:
        tr, te, protocol = r["train_dataset"], r["test_dataset"], r["protocol"]
        if protocol not in {"native", "strict", "coverage_matched", "topology_matched"}:
            continue
        case_mode = r.get("case_mode", "upper")
        tr_map = {canonical(g, case_mode): i for i, g in enumerate(ds[tr]["genes"])}
        te_map = {canonical(g, case_mode): i for i, g in enumerate(ds[te]["genes"])}
        shared = sorted(set(tr_map).intersection(set(te_map)))

        if protocol == "native":
            tr_idx = np.array(list(tr_map.values()), dtype=int)
            te_idx = np.array(list(te_map.values()), dtype=int)
        else:
            allowed = set(shared)
            gsf = r.get("gene_set_file", "")
            if gsf and os.path.exists(gsf):
                with open(gsf, encoding="utf-8") as f:
                    allowed = {x.strip() for x in f if x.strip()}
            use = sorted(allowed.intersection(set(shared)))
            tr_idx = np.array([tr_map[c] for c in use], dtype=int)
            te_idx = np.array([te_map[c] for c in use], dtype=int)

        ratio = np.nan
        if (tr, te) in pair_diag:
            try:
                ratio = float(pair_diag[(tr, te)].get("canonical_over_raw_ratio", "nan"))
            except Exception:
                ratio = np.nan

        tr_stats = subset_stats(ds[tr], tr_idx, ratio)
        te_stats = subset_stats(ds[te], te_idx, ratio)
        for side, st in [("train", tr_stats), ("test", te_stats)]:
            row = {"train_dataset": tr, "test_dataset": te, "direction": f"{tr}->{te}", "protocol": protocol, "side": side}
            row.update(st)
            rows.append(row)
            by_pair_protocol[(tr, te, side)][protocol] = st

        for metric, arr_name in [("degree", "degree"), ("train_node_freq", "train_node_freq"), ("test_node_freq", "test_node_freq")]:
            tr_vals = ds[tr][arr_name][tr_idx]
            te_vals = ds[te][arr_name][te_idx]
            plt.figure(figsize=(6, 4))
            if len(tr_vals):
                plt.hist(tr_vals, bins=30, alpha=0.5, label=f"train-{protocol}")
            if len(te_vals):
                plt.hist(te_vals, bins=30, alpha=0.5, label=f"test-{protocol}")
            plt.legend()
            plt.title(f"{tr}->{te} {protocol} {metric} histogram")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"hist_{tr}__to__{te}__{protocol}__{metric}.png"), dpi=160)
            plt.close()

            plt.figure(figsize=(6, 4))
            x1, y1 = ecdf(tr_vals)
            x2, y2 = ecdf(te_vals)
            plt.plot(x1, y1, label=f"train-{protocol}")
            plt.plot(x2, y2, label=f"test-{protocol}")
            plt.legend()
            plt.title(f"{tr}->{te} {protocol} {metric} ECDF")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"ecdf_{tr}__to__{te}__{protocol}__{metric}.png"), dpi=160)
            plt.close()

    out_csv = os.path.join(args.out_dir, "transfer_control_v2_diagnostics.csv")
    write_csv(out_csv, rows, list(rows[0].keys()) if rows else ["train_dataset", "test_dataset", "direction", "protocol", "side"])

    delta_rows = []
    for key, pdata in by_pair_protocol.items():
        tr, te, side = key
        if "native" not in pdata:
            continue
        base = pdata["native"]
        for p in ["strict", "coverage_matched", "topology_matched"]:
            if p not in pdata:
                continue
            cur = pdata[p]
            for m in ["n_genes", "mean_degree", "mean_train_node_freq", "mean_test_node_freq", "tf_proxy_fraction", "pos_edge_ratio", "hub_fraction"]:
                b = float(base[m]) if str(base[m]) != "nan" else np.nan
                c = float(cur[m]) if str(cur[m]) != "nan" else np.nan
                d = c - b if np.isfinite(b) and np.isfinite(c) else np.nan
                pct = (100.0 * d / b) if np.isfinite(d) and abs(b) > 1e-12 else np.nan
                delta_rows.append({"train_dataset": tr, "test_dataset": te, "side": side, "compare_protocol": p, "metric": m, "delta_abs": d, "delta_pct": pct})

    delta_csv = os.path.join(args.out_dir, "transfer_control_v2_protocol_deltas.csv")
    write_csv(delta_csv, delta_rows, list(delta_rows[0].keys()) if delta_rows else ["train_dataset", "test_dataset", "side", "compare_protocol", "metric", "delta_abs", "delta_pct"])

    print(f"[OK] wrote {out_csv}")
    print(f"[OK] wrote {delta_csv}")


if __name__ == "__main__":
    main()
