#!/usr/bin/env python3
"""Embedding transfer benchmark v2 with protocol-confound diagnostics.

This script intentionally focuses on benchmark trustworthiness, not score optimization.
It adds controls for topology/frequency/coverage leakage and reports transfer direction
(`train_dataset -> test_dataset`) explicitly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean, pstdev

import anndata as ad
import numpy as np
from scipy import sparse


@dataclass
class SplitValidation:
    tf_overlap: int
    target_overlap: int
    gene_overlap: int
    valid: bool


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run embedding transfer benchmark (v2, h5ad-only, edge-level).")
    p.add_argument("--base-dir", default="/bigdata2/hyt/projects/scbenchmark", help="Directory containing vocab/checkpoints.")
    p.add_argument("--h5ad-root", default="processed/native", help="Directory containing per-dataset .h5ad files.")
    p.add_argument("--pair-manifest", default="results/transfer_v2/pair_manifest.csv")
    p.add_argument("--out-dir", default="results/transfer_v2")
    p.add_argument("--embeddings-config", default="", help="Optional JSON file: {name:{path,key},...}")
    p.add_argument("--classifiers", nargs="*", default=["lr", "mlp"])
    p.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--split-mode", choices=["edge_disjoint", "tf_disjoint", "target_disjoint", "gene_disjoint"], default="edge_disjoint")
    p.add_argument("--skip-invalid-splits", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--oov-policy", choices=["zero", "random_fixed", "mean_embedding", "skip_pair"], default="mean_embedding")
    p.add_argument("--topology-bins", type=int, default=4)
    p.add_argument("--precision-k", type=int, default=200)
    p.add_argument(
        "--resample-lr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Bootstrap-resample train edges for LR (v1 style).",
    )
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


def infer_fields(rows):
    keys = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    return keys


def render_progress(done: int, total: int, prefix: str = "Progress") -> None:
    width = 30
    ratio = 1.0 if total <= 0 else max(0.0, min(1.0, done / total))
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    msg = f"\r{prefix}: [{bar}] {done}/{total} ({ratio * 100:5.1f}%)"
    sys.stderr.write(msg)
    if done >= total:
        sys.stderr.write("\n")
    sys.stderr.flush()


def canonical(g: str, mode: str) -> str:
    s = str(g).strip()
    if mode == "upper":
        return s.upper()
    if mode == "lower":
        return s.lower()
    return s


def default_embeddings_config(base_dir: str) -> dict[str, dict[str, str]]:
    return {
        "minus": {"path": f"{base_dir}/save_pretrain/minus/best_model.pt", "key": "module.embedding.weight"},
        "baseline": {"path": f"{base_dir}/save_pretrain/baseline/best_model.pt", "key": "module.embedding.weight"},
        "scGPT_human": {"path": f"{base_dir}/save_pretrain/scGPT_human/best_model.pt", "key": "encoder.embedding.weight"},
        "v4_bias_rec_best": {"path": f"{base_dir}/save_pretrain/v4_bias_rec_best/best_model.pt", "key": "embedding.weight"},
        "v4_plain_best": {"path": f"{base_dir}/save_pretrain/v4_plain_best/best_model.pt", "key": "embedding.weight"},
        "v4_type_pe_best": {"path": f"{base_dir}/save_pretrain/v4_type_pe_best/best_model.pt", "key": "embedding.weight"},
    }


def load_embedding(path: str, key: str):
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if key in ckpt:
        return ckpt[key].detach().cpu().numpy()
    for nk in ["state_dict", "model_state_dict", "model"]:
        if nk in ckpt and isinstance(ckpt[nk], dict) and key in ckpt[nk]:
            return ckpt[nk][key].detach().cpu().numpy()
    raise KeyError(f"missing key {key} in {path}")


def fit_eval(Xtr, ytr, Xte, yte, clf, seed, resample_lr=True):
    from sklearn.calibration import calibration_curve
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    if clf == "lr" and resample_lr:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(ytr), size=len(ytr))
        Xtr = Xtr[idx]
        ytr = ytr[idx]

    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr)
    Xte = sc.transform(Xte)

    if clf == "lr":
        m = LogisticRegression(max_iter=1000, n_jobs=1, C=1.0, random_state=seed)
    else:
        m = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, early_stopping=True, random_state=seed)

    m.fit(Xtr, ytr)
    ptr = m.predict_proba(Xtr)[:, 1] if hasattr(m, "predict_proba") else m.decision_function(Xtr)
    pte = m.predict_proba(Xte)[:, 1] if hasattr(m, "predict_proba") else m.decision_function(Xte)

    def pr_at_k(y_true, y_score, k):
        if len(y_true) == 0:
            return np.nan, np.nan
        kk = min(k, len(y_true))
        top = np.argsort(-y_score)[:kk]
        hits = y_true[top].sum()
        return float(hits / kk), float(hits / max(1, y_true.sum()))

    yhat = (pte >= 0.5).astype(int)
    pk, rk = pr_at_k(yte, pte, k=min(200, len(yte)))
    calib = brier_score_loss(yte, np.clip(pte, 0, 1)) if np.all((pte >= 0.0) & (pte <= 1.0)) else np.nan
    _ = calibration_curve(yte, np.clip(pte, 0, 1), n_bins=10, strategy="uniform") if np.isfinite(calib) else None

    return {
        "auroc": roc_auc_score(yte, pte),
        "auprc": average_precision_score(yte, pte),
        "f1": f1_score(yte, yhat, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(yte, yhat),
        "precision_at_k": pk,
        "recall_at_k": rk,
        "calibration_brier": calib,
        "train_scores": ptr,
        "test_scores": pte,
    }


def summarize(rows):
    grp = defaultdict(list)
    for r in rows:
        k = (r["train_dataset"], r["test_dataset"], r["protocol"], r["embedding"], r["clf"])
        grp[k].append(r)

    out = []
    metric_cols = ["auroc", "auprc", "f1", "balanced_accuracy", "precision_at_k", "recall_at_k", "calibration_brier"]
    for k, vals in sorted(grp.items()):
        item = {
            "train_dataset": k[0],
            "test_dataset": k[1],
            "protocol": k[2],
            "embedding": k[3],
            "clf": k[4],
            "n": len(vals),
        }
        for m in metric_cols:
            arr = [float(v[m]) for v in vals if str(v[m]) != "nan"]
            item[f"mean_{m}"] = mean(arr) if arr else np.nan
            item[f"std_{m}"] = pstdev(arr) if len(arr) > 1 else 0.0
        out.append(item)
    return out


def _require_split_table(adata: ad.AnnData, split_key: str):
    if split_key in adata.uns:
        return adata.uns[split_key]
    if "edge_splits" in adata.uns and isinstance(adata.uns["edge_splits"], dict) and split_key in adata.uns["edge_splits"]:
        return adata.uns["edge_splits"][split_key]
    raise KeyError(f"Missing edge split '{split_key}' in adata.uns.")


def _normalize_edges_table(obj, gene_to_idx: dict[str, int]):
    if isinstance(obj, dict):
        tf_raw = obj.get("tf", obj.get("source", obj.get("src", [])))
        tg_raw = obj.get("tg", obj.get("target", obj.get("dst", [])))
        y_raw = obj.get("y", obj.get("label", obj.get("labels", [])))
    else:
        arr = np.asarray(obj)
        tf_raw, tg_raw, y_raw = arr[:, 0], arr[:, 1], arr[:, 2]

    def to_idx(x):
        if isinstance(x, (int, np.integer)):
            return int(x)
        sx = str(x).strip()
        if sx.isdigit() or (sx.startswith("-") and sx[1:].isdigit()):
            return int(sx)
        return int(gene_to_idx[sx])

    tf = np.array([to_idx(x) for x in tf_raw], dtype=int)
    tg = np.array([to_idx(x) for x in tg_raw], dtype=int)
    y = np.array([int(float(v)) for v in y_raw], dtype=int)
    return tf, tg, y


def _mat_from_adata(adata: ad.AnnData):
    mat = adata.layers["counts"] if "counts" in adata.layers else adata.X
    if sparse.issparse(mat):
        return mat.tocsr()
    return np.asarray(mat, dtype=np.float32)


def _build_proxy_edge_splits_from_h5ad(adata: ad.AnnData, max_genes: int = 256, max_edges_each: int = 12000, seed: int = 0):
    """Fallback split construction when explicit Train/Val/Test edge splits are absent.

    Uses variance-selected genes + correlation quantiles to derive positive/negative pairs,
    then creates train/val/test partitions.
    """
    if "pseudotime" not in adata.obs.columns:
        raise KeyError(
            "Missing explicit edge splits and obs['pseudotime']; cannot build proxy Train/Validation/Test splits."
        )

    mat = _mat_from_adata(adata)
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


def _balance_binary_edges(tf: np.ndarray, tg: np.ndarray, y: np.ndarray, rng: np.random.Generator):
    """Keep class balance as much as possible after split constraints."""
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return tf, tg, y
    n = min(len(pos), len(neg))
    sel = np.concatenate([rng.choice(pos, size=n, replace=False), rng.choice(neg, size=n, replace=False)])
    sel = rng.permutation(sel)
    return tf[sel], tg[sel], y[sel]


def _split_items_three_way(items: np.ndarray, rng: np.random.Generator):
    items = np.array(items, dtype=int)
    rng.shuffle(items)
    n = len(items)
    n_train = int(0.6 * n)
    n_val = int(0.2 * n)
    return set(items[:n_train].tolist()), set(items[n_train : n_train + n_val].tolist()), set(items[n_train + n_val :].tolist())


def _assign_edges_by_mode(all_tf: np.ndarray, all_tg: np.ndarray, all_y: np.ndarray, mode: str, rng: np.random.Generator):
    """Rebuild Train/Validation/Test under requested disjointness mode.

    Modes:
    - edge_disjoint: keeps original partitioning behavior (caller provides original splits).
    - tf_disjoint: enforces disjoint TF identities across train/val/test.
    - target_disjoint: enforces disjoint target identities across train/val/test.
    - gene_disjoint: enforces both endpoints from disjoint gene subsets.
    """
    if mode == "edge_disjoint":
        return None

    if mode == "tf_disjoint":
        tr_set, va_set, te_set = _split_items_three_way(np.unique(all_tf), rng)
        tr_idx = np.where(np.isin(all_tf, list(tr_set)))[0]
        va_idx = np.where(np.isin(all_tf, list(va_set)))[0]
        te_idx = np.where(np.isin(all_tf, list(te_set)))[0]
    elif mode == "target_disjoint":
        tr_set, va_set, te_set = _split_items_three_way(np.unique(all_tg), rng)
        tr_idx = np.where(np.isin(all_tg, list(tr_set)))[0]
        va_idx = np.where(np.isin(all_tg, list(va_set)))[0]
        te_idx = np.where(np.isin(all_tg, list(te_set)))[0]
    else:  # gene_disjoint
        genes = np.unique(np.concatenate([all_tf, all_tg]))
        tr_set, va_set, te_set = _split_items_three_way(genes, rng)
        tr_idx = np.where(np.isin(all_tf, list(tr_set)) & np.isin(all_tg, list(tr_set)))[0]
        va_idx = np.where(np.isin(all_tf, list(va_set)) & np.isin(all_tg, list(va_set)))[0]
        te_idx = np.where(np.isin(all_tf, list(te_set)) & np.isin(all_tg, list(te_set)))[0]

    tr_tf, tr_tg, tr_y = _balance_binary_edges(all_tf[tr_idx], all_tg[tr_idx], all_y[tr_idx], rng)
    va_tf, va_tg, va_y = _balance_binary_edges(all_tf[va_idx], all_tg[va_idx], all_y[va_idx], rng)
    te_tf, te_tg, te_y = _balance_binary_edges(all_tf[te_idx], all_tg[te_idx], all_y[te_idx], rng)
    return (tr_tf, tr_tg, tr_y), (va_tf, va_tg, va_y), (te_tf, te_tg, te_y)


def _split_diag_row(name: str, mode: str, tr_tf, tr_tg, tr_y, va_tf, va_tg, va_y, te_tf, te_tg, te_y):
    def ratio(y):
        return float(np.mean(y)) if len(y) else np.nan
    def deg_stats(tf, tg):
        if len(tf) == 0:
            return np.nan, np.nan
        nodes = np.concatenate([tf, tg])
        _, counts = np.unique(nodes, return_counts=True)
        return float(np.mean(counts)), float(np.median(counts))
    tr_genes = set(np.concatenate([tr_tf, tr_tg]).tolist()) if len(tr_tf) else set()
    va_genes = set(np.concatenate([va_tf, va_tg]).tolist()) if len(va_tf) else set()
    te_genes = set(np.concatenate([te_tf, te_tg]).tolist()) if len(te_tf) else set()
    md_tr, med_tr = deg_stats(tr_tf, tr_tg)
    md_va, med_va = deg_stats(va_tf, va_tg)
    md_te, med_te = deg_stats(te_tf, te_tg)
    valid = int(
        len(tr_y) >= 20 and len(va_y) >= 20 and len(te_y) >= 20
        and len(np.unique(tr_y)) > 1 and len(np.unique(va_y)) > 1 and len(np.unique(te_y)) > 1
    )
    return {
        "dataset": name,
        "split_mode": mode,
        "n_train_edges": len(tr_y),
        "n_val_edges": len(va_y),
        "n_test_edges": len(te_y),
        "n_train_genes": len(tr_genes),
        "n_val_genes": len(va_genes),
        "n_test_genes": len(te_genes),
        "tf_overlap_train_test": len(set(tr_tf.tolist()).intersection(set(te_tf.tolist()))),
        "target_overlap_train_test": len(set(tr_tg.tolist()).intersection(set(te_tg.tolist()))),
        "gene_overlap_train_test": len(tr_genes.intersection(te_genes)),
        "train_pos_ratio": ratio(tr_y),
        "val_pos_ratio": ratio(va_y),
        "test_pos_ratio": ratio(te_y),
        "train_mean_degree": md_tr,
        "train_median_degree": med_tr,
        "val_mean_degree": md_va,
        "val_median_degree": med_va,
        "test_mean_degree": md_te,
        "test_median_degree": med_te,
        "valid": valid,
    }


def prepare_dataset(adata: ad.AnnData, split_mode: str = "edge_disjoint", split_seed: int = 0, dataset_name: str = ""):
    genes = np.asarray(adata.var_names.astype(str))
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    has_explicit = (
        ("Train_set" in adata.uns and "Validation_set" in adata.uns and "Test_set" in adata.uns)
        or (
            "edge_splits" in adata.uns
            and isinstance(adata.uns["edge_splits"], dict)
            and all(k in adata.uns["edge_splits"] for k in ["Train_set", "Validation_set", "Test_set"])
        )
    )
    if has_explicit:
        tr_tf, tr_tg, tr_y = _normalize_edges_table(_require_split_table(adata, "Train_set"), gene_to_idx)
        va_tf, va_tg, va_y = _normalize_edges_table(_require_split_table(adata, "Validation_set"), gene_to_idx)
        te_tf, te_tg, te_y = _normalize_edges_table(_require_split_table(adata, "Test_set"), gene_to_idx)
    else:
        (tr_tf, tr_tg, tr_y), (va_tf, va_tg, va_y), (te_tf, te_tg, te_y) = _build_proxy_edge_splits_from_h5ad(adata)
    if split_mode != "edge_disjoint":
        all_tf = np.concatenate([tr_tf, va_tf, te_tf])
        all_tg = np.concatenate([tr_tg, va_tg, te_tg])
        all_y = np.concatenate([tr_y, va_y, te_y])
        rng = np.random.default_rng(split_seed)
        rebuilt = _assign_edges_by_mode(all_tf, all_tg, all_y, split_mode, rng)
        if rebuilt is not None:
            (tr_tf, tr_tg, tr_y), (va_tf, va_tg, va_y), (te_tf, te_tg, te_y) = rebuilt

    train_tf = np.concatenate([tr_tf, va_tf])
    train_tg = np.concatenate([tr_tg, va_tg])
    train_y = np.concatenate([tr_y, va_y])
    diag = _split_diag_row(dataset_name, split_mode, tr_tf, tr_tg, tr_y, va_tf, va_tg, va_y, te_tf, te_tg, te_y)
    return {
        "genes": genes.tolist(),
        "train_tf": train_tf,
        "train_tg": train_tg,
        "train_y": train_y,
        "test_tf": te_tf,
        "test_tg": te_tg,
        "test_y": te_y,
        "train_tf_raw": tr_tf,
        "train_tg_raw": tr_tg,
        "val_tf": va_tf,
        "val_tg": va_tg,
        "split_diag": diag,
    }


def pair_features(lookup, tf, tg):
    a = lookup[tf]
    b = lookup[tg]
    had = a * b
    cos = np.sum(a * b, axis=1, keepdims=True) / ((np.linalg.norm(a, axis=1, keepdims=True) + 1e-8) * (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8))
    l2 = np.linalg.norm(a - b, axis=1, keepdims=True)
    return np.concatenate([a, b, had, cos, l2], axis=1)


def map_pairs_to_local(pairs_tf, pairs_tg, pairs_y, genes, gene_to_local):
    tf2, tg2, y2 = [], [], []
    for a, b, l in zip(pairs_tf, pairs_tg, pairs_y):
        ga, gb = genes[a], genes[b]
        if ga in gene_to_local and gb in gene_to_local:
            tf2.append(gene_to_local[ga])
            tg2.append(gene_to_local[gb])
            y2.append(l)
    return np.array(tf2), np.array(tg2), np.array(y2)


def _gene_fingerprint(genes: list[str]) -> str:
    payload = "||".join(sorted(genes))
    return hashlib.md5(payload.encode()).hexdigest()


def build_gene_meta(ds):
    n = len(ds["genes"])
    deg = np.zeros(n, dtype=float)
    trf = np.zeros(n, dtype=float)
    tef = np.zeros(n, dtype=float)
    pos = np.zeros(n, dtype=float)
    cnt = np.zeros(n, dtype=float)

    for tf, tg, y in zip(ds["train_tf"], ds["train_tg"], ds["train_y"]):
        deg[tf] += 1
        deg[tg] += 1
        trf[tf] += 1
        trf[tg] += 1
        pos[tf] += y
        pos[tg] += y
        cnt[tf] += 1
        cnt[tg] += 1
    for tf, tg in zip(ds["test_tf"], ds["test_tg"]):
        deg[tf] += 1
        deg[tg] += 1
        tef[tf] += 1
        tef[tg] += 1
    tf_proxy = np.zeros(n, dtype=float)
    tf_proxy[np.unique(np.concatenate([ds["train_tf"], ds["test_tf"]]))] = 1.0
    return {
        "degree": deg,
        "train_node_freq": trf,
        "test_node_freq": tef,
        "tf_proxy": tf_proxy,
        "pos_edge_ratio": np.divide(pos, cnt, out=np.zeros_like(pos), where=cnt > 0),
    }


def choose_topology_matched(train_pool, test_pool, tr_meta, te_meta, k, seed, bins=4):
    rng = np.random.default_rng(seed)

    def signatures(pool, meta):
        metrics = ["degree", "train_node_freq", "test_node_freq", "pos_edge_ratio"]
        sigs = {}
        qbins = {}
        for m in metrics:
            vals = np.array([meta[m][g] for g in pool], dtype=float)
            cuts = np.quantile(vals, np.linspace(0, 1, bins + 1)) if len(vals) > 0 else np.array([0, 1])
            cuts = np.unique(cuts)
            qbins[m] = cuts
        for g in pool:
            key = []
            for m in metrics:
                cuts = qbins[m]
                v = meta[m][g]
                b = int(np.searchsorted(cuts, v, side="right") - 1)
                key.append(min(max(0, b), max(0, len(cuts) - 2)))
            key.append(int(meta["tf_proxy"][g] > 0))
            sigs[g] = tuple(key)
        return sigs

    tr_sig = signatures(train_pool, tr_meta)
    te_sig = signatures(test_pool, te_meta)

    tr_grp = defaultdict(list)
    te_grp = defaultdict(list)
    for g in train_pool:
        tr_grp[tr_sig[g]].append(g)
    for g in test_pool:
        te_grp[te_sig[g]].append(g)

    shared_keys = sorted(set(tr_grp).intersection(set(te_grp)))
    capacities = {k0: min(len(tr_grp[k0]), len(te_grp[k0])) for k0 in shared_keys}
    total = sum(capacities.values())
    if total == 0:
        return sorted(rng.choice(train_pool, size=min(k, len(train_pool)), replace=False).tolist()), sorted(rng.choice(test_pool, size=min(k, len(test_pool)), replace=False).tolist())

    target = min(k, total)
    chosen_tr, chosen_te = [], []
    for k0 in shared_keys:
        take = int(round(target * capacities[k0] / total))
        take = min(take, capacities[k0])
        if take > 0:
            chosen_tr.extend(rng.choice(np.array(tr_grp[k0], dtype=int), size=take, replace=False).tolist())
            chosen_te.extend(rng.choice(np.array(te_grp[k0], dtype=int), size=take, replace=False).tolist())

    while len(chosen_tr) < target:
        for k0 in shared_keys:
            rem_tr = [g for g in tr_grp[k0] if g not in chosen_tr]
            rem_te = [g for g in te_grp[k0] if g not in chosen_te]
            if rem_tr and rem_te:
                chosen_tr.append(rem_tr[0])
                chosen_te.append(rem_te[0])
                if len(chosen_tr) >= target:
                    break

    return sorted(chosen_tr), sorted(chosen_te)


def check_disjoint(ds, mode: str) -> SplitValidation:
    tr_tf, tr_tg = ds["train_tf"], ds["train_tg"]
    te_tf, te_tg = ds["test_tf"], ds["test_tg"]
    tr_nodes = set(np.concatenate([tr_tf, tr_tg]).tolist())
    te_nodes = set(np.concatenate([te_tf, te_tg]).tolist())
    tf_overlap = len(set(tr_tf.tolist()).intersection(set(te_tf.tolist())))
    target_overlap = len(set(tr_tg.tolist()).intersection(set(te_tg.tolist())))
    gene_overlap = len(tr_nodes.intersection(te_nodes))
    valid = True
    if mode == "tf_disjoint":
        valid = tf_overlap == 0
    elif mode == "target_disjoint":
        valid = target_overlap == 0
    elif mode == "gene_disjoint":
        valid = gene_overlap == 0
    return SplitValidation(tf_overlap=tf_overlap, target_overlap=target_overlap, gene_overlap=gene_overlap, valid=valid)


def get_embedding_vector(emb, idx, policy, emb_mean, rng):
    if idx is not None and 0 <= idx < emb.shape[0]:
        return emb[idx]
    if policy == "zero":
        return np.zeros((emb.shape[1],), dtype=emb.dtype)
    if policy == "random_fixed":
        return rng.normal(size=(emb.shape[1],)).astype(emb.dtype)
    if policy == "mean_embedding":
        return emb_mean
    return None


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    seed_csv = os.path.join(args.out_dir, "embedding_transfer_seed_results_v2.csv")
    sum_csv = os.path.join(args.out_dir, "embedding_transfer_summary_v2.csv")

    with open(f"{args.base_dir}/vocab.json", encoding="utf-8") as f:
        vocab = json.load(f)

    manifest = [r for r in read_csv(args.pair_manifest) if r["protocol"] in {"native", "strict", "coverage_matched", "topology_matched"}]
    datasets = sorted(set([r["train_dataset"] for r in manifest] + [r["test_dataset"] for r in manifest]))

    ds = {}
    ds_split_rows = []
    for d in datasets:
        p = os.path.join(args.h5ad_root, f"{d}.h5ad")
        seed = int(hashlib.md5(f"{d}::{args.split_mode}".encode()).hexdigest()[:8], 16)
        ds[d] = prepare_dataset(ad.read_h5ad(p), split_mode=args.split_mode, split_seed=seed, dataset_name=d)
        ds_split_rows.append(ds[d]["split_diag"])

    if args.embeddings_config:
        with open(args.embeddings_config, encoding="utf-8") as f:
            emb_cfg = json.load(f)
    else:
        emb_cfg = default_embeddings_config(args.base_dir)

    emb_cfg = {k: v for k, v in emb_cfg.items() if os.path.exists(v.get("path", ""))}
    emb_map = {name: load_embedding(cfg["path"], cfg["key"]) for name, cfg in emb_cfg.items()}

    ds_meta = {name: build_gene_meta(v) for name, v in ds.items()}
    seed_rows, score_rows, match_diag_rows, split_rows = [], [], [], []
    protocol_debug_rows, invalid_rows, pair_protocol_rows = [], [], []
    strict_fingerprints = {}

    total_jobs = len(manifest)
    render_progress(0, total_jobs, prefix="transfer_v2")
    for i_row, row in enumerate(manifest, start=1):
        tr, te, protocol = row["train_dataset"], row["test_dataset"], row["protocol"]
        case_mode = row.get("case_mode", "upper")
        trd, ted = ds[tr], ds[te]
        split_stat = check_disjoint(trd, args.split_mode)
        split_rows.append({"dataset": tr, "split_mode": args.split_mode, "tf_overlap": split_stat.tf_overlap, "target_overlap": split_stat.target_overlap, "gene_overlap": split_stat.gene_overlap, "valid": int(split_stat.valid)})
        if not split_stat.valid and args.skip_invalid_splits:
            render_progress(i_row, total_jobs, prefix="transfer_v2")
            continue

        tr_can = {canonical(g, case_mode): g for g in trd["genes"]}
        te_can = {canonical(g, case_mode): g for g in ted["genes"]}
        native_tr_can = sorted(tr_can.keys())
        native_te_can = sorted(te_can.keys())
        shared_can = sorted(set(native_tr_can).intersection(set(native_te_can)))

        for emb_name, emb in emb_map.items():
            rng = np.random.default_rng(int(hashlib.md5(f"{tr}::{te}::{emb_name}".encode()).hexdigest()[:8], 16))
            emb_mean = emb.mean(axis=0)
            max_idx = emb.shape[0]

            train_pool = [tr_can[c] for c in native_tr_can]
            test_pool = [te_can[c] for c in native_te_can]
            if protocol == "native":
                tr_raw, te_raw = train_pool, test_pool
            else:
                allowed = set(shared_can)
                if row.get("gene_set_file") and os.path.exists(row["gene_set_file"]):
                    with open(row["gene_set_file"], encoding="utf-8") as f:
                        allowed = {x.strip() for x in f if x.strip()}
                if protocol == "strict":
                    use = sorted(allowed.intersection(set(shared_can)))
                    tr_raw = [tr_can[c] for c in use if c in tr_can]
                    te_raw = [te_can[c] for c in use if c in te_can]
                elif protocol == "coverage_matched":
                    k_target = len(allowed)
                    # coverage_matched controls subset size, not strict identity matching.
                    tr_cands = [tr_can[c] for c in native_tr_can]
                    te_cands = [te_can[c] for c in native_te_can]
                    k = min(k_target, len(tr_cands), len(te_cands))
                    tr_raw = sorted(rng.choice(np.array(tr_cands, dtype=object), size=k, replace=False).tolist())
                    te_raw = sorted(rng.choice(np.array(te_cands, dtype=object), size=k, replace=False).tolist())
                else:
                    k_target = len(allowed)
                    tr_cands = [tr_can[c] for c in shared_can if c in tr_can]
                    te_cands = [te_can[c] for c in shared_can if c in te_can]
                    tr_idx = [trd["genes"].index(g) for g in tr_cands]
                    te_idx = [ted["genes"].index(g) for g in te_cands]
                    sel_tr_idx, sel_te_idx = choose_topology_matched(tr_idx, te_idx, ds_meta[tr], ds_meta[te], k=min(k_target, len(tr_idx), len(te_idx)), seed=int(rng.integers(0, 1 << 31)), bins=args.topology_bins)
                    tr_raw = [trd["genes"][i] for i in sel_tr_idx]
                    te_raw = [ted["genes"][i] for i in sel_te_idx]
                    match_diag_rows.append({"train_dataset": tr, "test_dataset": te, "embedding": emb_name, "before_n": min(len(tr_idx), len(te_idx)), "after_n": min(len(sel_tr_idx), len(sel_te_idx)), "retained_fraction": (min(len(sel_tr_idx), len(sel_te_idx)) / max(1, min(len(tr_idx), len(te_idx))))})

            tr_fp = _gene_fingerprint(tr_raw)
            te_fp = _gene_fingerprint(te_raw)
            strict_key = (tr, te, emb_name)
            if protocol == "strict":
                strict_fingerprints[strict_key] = (tr_fp, te_fp)
            if protocol == "coverage_matched" and strict_key in strict_fingerprints and strict_fingerprints[strict_key] == (tr_fp, te_fp):
                invalid_rows.append(
                    {
                        "train_dataset": tr,
                        "test_dataset": te,
                        "protocol": protocol,
                        "embedding": emb_name,
                        "clf": "all",
                        "seed": -1,
                        "status": "invalid",
                        "reason": "coverage_identical_to_strict",
                    }
                )

            protocol_debug_rows.append(
                {
                    "train_dataset": tr,
                    "test_dataset": te,
                    "protocol": protocol,
                    "embedding": emb_name,
                    "seed": -1,
                    "train_gene_count": len(tr_raw),
                    "test_gene_count": len(te_raw),
                    "train_gene_fingerprint": tr_fp,
                    "test_gene_fingerprint": te_fp,
                    "train_first20_sorted": "|".join(sorted(tr_raw)[:20]),
                    "test_first20_sorted": "|".join(sorted(te_raw)[:20]),
                }
            )

            tr_vecs, te_vecs = [], []
            tr_oov = te_oov = 0
            for g in tr_raw:
                idx = int(vocab[g]) if g in vocab else None
                v = get_embedding_vector(emb, idx, args.oov_policy, emb_mean, rng)
                if v is None:
                    tr_oov += 1
                    continue
                tr_vecs.append((g, v))
                if idx is None or idx >= max_idx:
                    tr_oov += 1
            for g in te_raw:
                idx = int(vocab[g]) if g in vocab else None
                v = get_embedding_vector(emb, idx, args.oov_policy, emb_mean, rng)
                if v is None:
                    te_oov += 1
                    continue
                te_vecs.append((g, v))
                if idx is None or idx >= max_idx:
                    te_oov += 1

            tr_raw = [g for g, _ in tr_vecs]
            te_raw = [g for g, _ in te_vecs]
            if len(tr_raw) < 20 or len(te_raw) < 20:
                continue
            gmap_tr = {g: i for i, g in enumerate(tr_raw)}
            gmap_te = {g: i for i, g in enumerate(te_raw)}

            tr_tf, tr_tg, tr_y = map_pairs_to_local(trd["train_tf"], trd["train_tg"], trd["train_y"], trd["genes"], gmap_tr)
            te_tf, te_tg, te_y = map_pairs_to_local(ted["test_tf"], ted["test_tg"], ted["test_y"], ted["genes"], gmap_te)
            pair_protocol_rows.append(
                {
                    "train_dataset": tr,
                    "test_dataset": te,
                    "protocol": protocol,
                    "embedding": emb_name,
                    "n_train_edges": len(tr_y),
                    "n_test_edges": len(te_y),
                    "n_train_pos": int((tr_y == 1).sum()),
                    "n_test_pos": int((te_y == 1).sum()),
                    "n_train_neg": int((tr_y == 0).sum()),
                    "n_test_neg": int((te_y == 0).sum()),
                    "n_train_genes": len(tr_raw),
                    "n_test_genes": len(te_raw),
                    "gene_overlap": len(set(tr_raw).intersection(set(te_raw))),
                    "tf_overlap": len(set(np.array(tr_raw)[np.unique(tr_tf)].tolist()).intersection(set(np.array(te_raw)[np.unique(te_tf)].tolist()))) if len(tr_tf) and len(te_tf) else 0,
                    "target_overlap": len(set(np.array(tr_raw)[np.unique(tr_tg)].tolist()).intersection(set(np.array(te_raw)[np.unique(te_tg)].tolist()))) if len(tr_tg) and len(te_tg) else 0,
                    "train_pos_ratio": float(np.mean(tr_y)) if len(tr_y) else np.nan,
                    "test_pos_ratio": float(np.mean(te_y)) if len(te_y) else np.nan,
                }
            )
            if len(tr_y) < 20 or len(te_y) < 20:
                invalid_rows.append({"train_dataset": tr, "test_dataset": te, "protocol": protocol, "embedding": emb_name, "clf": "all", "seed": -1, "status": "invalid", "reason": "empty_split"})
                continue
            if len(np.unique(tr_y)) < 2 or len(np.unique(te_y)) < 2:
                invalid_rows.append({"train_dataset": tr, "test_dataset": te, "protocol": protocol, "embedding": emb_name, "clf": "all", "seed": -1, "status": "invalid", "reason": "one_class_test"})
                continue

            emb_lookup_tr = np.stack([v for _, v in tr_vecs], axis=0)
            emb_lookup_te = np.stack([v for _, v in te_vecs], axis=0)
            Xtr, Xte = pair_features(emb_lookup_tr, tr_tf, tr_tg), pair_features(emb_lookup_te, te_tf, te_tg)
            if float(np.nanvar(Xtr)) < 1e-12 or float(np.nanvar(Xte)) < 1e-12:
                invalid_rows.append({"train_dataset": tr, "test_dataset": te, "protocol": protocol, "embedding": emb_name, "clf": "all", "seed": -1, "status": "invalid", "reason": "constant_features"})
                continue
            if not np.all(np.isfinite(Xtr)) or not np.all(np.isfinite(Xte)):
                invalid_rows.append({"train_dataset": tr, "test_dataset": te, "protocol": protocol, "embedding": emb_name, "clf": "all", "seed": -1, "status": "invalid", "reason": "nan_or_inf_features"})
                continue

            for clf in args.classifiers:
                for seed in args.seeds:
                    try:
                        out = fit_eval(Xtr, tr_y, Xte, te_y, clf, seed, resample_lr=args.resample_lr)
                    except Exception as e:
                        invalid_rows.append(
                            {
                                "train_dataset": tr,
                                "test_dataset": te,
                                "protocol": protocol,
                                "embedding": emb_name,
                                "clf": clf,
                                "seed": seed,
                                "status": "invalid",
                                "reason": f"fit_eval_error:{type(e).__name__}",
                            }
                        )
                        continue
                    p_tr, p_te = out["train_scores"], out["test_scores"]
                    if len(np.unique(np.round(p_te, 12))) <= 1:
                        invalid_rows.append({"train_dataset": tr, "test_dataset": te, "protocol": protocol, "embedding": emb_name, "clf": clf, "seed": seed, "status": "invalid", "reason": "constant_predictions"})
                        continue
                    if not np.all(np.isfinite(p_tr)) or not np.all(np.isfinite(p_te)):
                        invalid_rows.append({"train_dataset": tr, "test_dataset": te, "protocol": protocol, "embedding": emb_name, "clf": clf, "seed": seed, "status": "invalid", "reason": "nan_or_inf_predictions"})
                        continue
                    pos_tr = float(p_tr[tr_y == 1].mean())
                    neg_tr = float(p_tr[tr_y == 0].mean())
                    pos_te = float(p_te[te_y == 1].mean())
                    neg_te = float(p_te[te_y == 0].mean())
                    m_tr, m_te = pos_tr - neg_tr, pos_te - neg_te
                    d_pos, d_neg, d_mar = pos_te - pos_tr, neg_te - neg_tr, m_te - m_tr
                    if abs(d_pos) > 1.25 * abs(d_neg):
                        mode = "positive-collapse dominated"
                    elif abs(d_neg) > 1.25 * abs(d_pos):
                        mode = "negative-drift dominated"
                    else:
                        mode = "mixed"

                    seed_rows.append({
                        "train_dataset": tr,
                        "test_dataset": te,
                        "direction": f"{tr}->{te}",
                        "protocol": protocol,
                        "embedding": emb_name,
                        "clf": clf,
                        "seed": seed,
                        "n_common_genes": min(len(tr_raw), len(te_raw)),
                        "retained_fraction": min(len(tr_raw), len(te_raw)) / max(1, min(len(train_pool), len(test_pool))),
                        "n_train_edges": len(tr_y),
                        "n_test_edges": len(te_y),
                        "oov_train_count": tr_oov,
                        "oov_test_count": te_oov,
                        "oov_train_rate": tr_oov / max(1, len(tr_raw)),
                        "oov_test_rate": te_oov / max(1, len(te_raw)),
                        "oov_policy": args.oov_policy,
                        "auroc": f"{out['auroc']:.6f}",
                        "auprc": f"{out['auprc']:.6f}",
                        "f1": f"{out['f1']:.6f}",
                        "balanced_accuracy": f"{out['balanced_accuracy']:.6f}",
                        "precision_at_k": f"{out['precision_at_k']:.6f}",
                        "recall_at_k": f"{out['recall_at_k']:.6f}",
                        "calibration_brier": f"{out['calibration_brier']:.6f}",
                    })
                    score_rows.append({
                        "train_dataset": tr,
                        "test_dataset": te,
                        "direction": f"{tr}->{te}",
                        "protocol": protocol,
                        "embedding": emb_name,
                        "clf": clf,
                        "seed": seed,
                        "mean_pos_score_source": pos_tr,
                        "mean_neg_score_source": neg_tr,
                        "margin_source": m_tr,
                        "mean_pos_score_target": pos_te,
                        "mean_neg_score_target": neg_te,
                        "margin_target": m_te,
                        "delta_pos": d_pos,
                        "delta_neg": d_neg,
                        "delta_margin": d_mar,
                        "abs_shift_pos": abs(d_pos),
                        "abs_shift_neg": abs(d_neg),
                        "abs_shift_margin": abs(d_mar),
                        "failure_type": mode,
                    })
                    if tr == "mESC" or te == "mESC":
                        protocol_debug_rows.append(
                            {
                                "train_dataset": tr,
                                "test_dataset": te,
                                "protocol": protocol,
                                "embedding": emb_name,
                                "clf": clf,
                                "seed": seed,
                                "train_gene_count": len(tr_raw),
                                "test_gene_count": len(te_raw),
                                "train_gene_fingerprint": tr_fp,
                                "test_gene_fingerprint": te_fp,
                                "train_first20_sorted": "|".join(sorted(tr_raw)[:20]),
                                "test_first20_sorted": "|".join(sorted(te_raw)[:20]),
                                "n_train_edges": len(tr_y),
                                "n_test_edges": len(te_y),
                                "n_train_pos": int((tr_y == 1).sum()),
                                "n_test_pos": int((te_y == 1).sum()),
                                "n_train_neg": int((tr_y == 0).sum()),
                                "n_test_neg": int((te_y == 0).sum()),
                                "Xtr_shape": f"{Xtr.shape[0]}x{Xtr.shape[1]}",
                                "Xte_shape": f"{Xte.shape[0]}x{Xte.shape[1]}",
                                "Xtr_var": float(np.var(Xtr)),
                                "Xte_var": float(np.var(Xte)),
                                "pred_mean": float(np.mean(p_te)),
                                "pred_std": float(np.std(p_te)),
                                "pred_min": float(np.min(p_te)),
                                "pred_max": float(np.max(p_te)),
                                "pred_unique": int(len(np.unique(np.round(p_te, 12)))),
                                "status": "ok",
                                "reason": "",
                            }
                        )
        render_progress(i_row, total_jobs, prefix="transfer_v2")

    write_csv(seed_csv, seed_rows, infer_fields(seed_rows) if seed_rows else [])
    sum_rows = summarize(seed_rows)
    write_csv(sum_csv, sum_rows, infer_fields(sum_rows) if sum_rows else [])
    print(f"[OK] wrote {seed_csv}")
    print(f"[OK] wrote {sum_csv}")


if __name__ == "__main__":
    main()
