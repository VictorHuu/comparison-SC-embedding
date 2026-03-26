#!/usr/bin/env python3
"""Embedding transfer benchmark v2 (h5ad-only, v1-aligned edge-level).

Task definition (edge-level transfer; same evaluation logic as v1):
- Train on source dataset edges (Train_set + Validation_set).
- Test on target dataset edges (Test_set).
- Edge features are built from gene embeddings: [a, b, a*b, cosine(a,b), ||a-b||_2].

Data requirement:
- Each dataset .h5ad must include edge splits in `adata.uns`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from statistics import mean, pstdev

import anndata as ad
import numpy as np
from scipy import sparse

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run embedding transfer benchmark (v2, h5ad-only, edge-level).")
    p.add_argument("--base-dir", default="/bigdata2/hyt/projects/scbenchmark", help="Directory containing vocab/checkpoints.")
    p.add_argument("--h5ad-root", default="processed/native", help="Directory containing per-dataset .h5ad files.")
    p.add_argument("--pair-manifest", default="transfer_v2/pair_manifest.csv")
    p.add_argument("--out-dir", default="transfer")
    p.add_argument("--embeddings-config", default="", help="Optional JSON file: {name:{path,key},...}")
    p.add_argument("--classifiers", nargs="*", default=["lr", "mlp"])
    p.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
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
    """v2-native defaults (no dependency on v1 script files)."""
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
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
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
    p = m.predict_proba(Xte)[:, 1] if hasattr(m, "predict_proba") else m.decision_function(Xte)
    return roc_auc_score(yte, p), average_precision_score(yte, p)


def summarize(rows):
    grp = defaultdict(list)
    for r in rows:
        k = (r["train_dataset"], r["test_dataset"], r["protocol"], r["embedding"], r["clf"])
        grp[k].append(r)

    out = []
    for k, vals in sorted(grp.items()):
        au = [float(v["auroc"]) for v in vals]
        ap = [float(v["auprc"]) for v in vals]
        out.append(
            {
                "train_dataset": k[0],
                "test_dataset": k[1],
                "protocol": k[2],
                "embedding": k[3],
                "clf": k[4],
                "mean_auroc": mean(au),
                "std_auroc": pstdev(au) if len(au) > 1 else 0.0,
                "mean_auprc": mean(ap),
                "std_auprc": pstdev(ap) if len(ap) > 1 else 0.0,
                "n": len(vals),
            }
        )
    return out


def _require_split_table(adata: ad.AnnData, split_key: str):
    if split_key in adata.uns:
        return adata.uns[split_key]
    if "edge_splits" in adata.uns and isinstance(adata.uns["edge_splits"], dict) and split_key in adata.uns["edge_splits"]:
        return adata.uns["edge_splits"][split_key]
    raise KeyError(
        f"Missing edge split '{split_key}' in adata.uns. "
        "Expected either adata.uns['Train_set'/'Validation_set'/'Test_set'] or adata.uns['edge_splits'][...]."
    )


def _mat_from_adata(adata: ad.AnnData):
    mat = adata.layers["counts"] if "counts" in adata.layers else adata.X
    if sparse.issparse(mat):
        return mat.tocsr()
    return np.asarray(mat, dtype=np.float32)


def _build_proxy_edge_splits_from_h5ad(adata: ad.AnnData, max_genes: int = 256, max_edges_each: int = 12000, seed: int = 0):
    """Fallback edge split construction from h5ad expression only (no v1 dependency)."""
    if "pseudotime" not in adata.obs.columns:
        raise KeyError(
            "Missing explicit edge splits and missing obs['pseudotime']; "
            "cannot construct proxy edge splits."
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
    if n_cells > 3000:
        cell_idx = np.sort(rng.choice(np.arange(n_cells), size=3000, replace=False))
    else:
        cell_idx = np.arange(n_cells)

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
    if len(s) < 100:
        raise ValueError("Not enough gene pairs to build proxy edge splits.")

    hi = float(np.quantile(s, 0.90))
    lo = float(np.quantile(s, 0.10))
    pos_idx = np.where(s >= hi)[0]
    neg_idx = np.where(s <= lo)[0]
    if len(pos_idx) < 200 or len(neg_idx) < 200:
        hi = float(np.quantile(s, 0.80))
        lo = float(np.quantile(s, 0.20))
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


def _normalize_edges_table(obj, gene_to_idx: dict[str, int]):
    """Return tf_idx, tg_idx, y as numpy arrays.

    Accepted forms:
    - dict with keys among {tf,tg,y} or aliases {source,target,label}
    - 2D array/list with >=3 columns [tf, tg, y]
    """

    if isinstance(obj, dict):
        tf_raw = obj.get("tf", obj.get("source", obj.get("src", [])))
        tg_raw = obj.get("tg", obj.get("target", obj.get("dst", [])))
        y_raw = obj.get("y", obj.get("label", obj.get("labels", [])))
    else:
        arr = np.asarray(obj)
        if arr.ndim != 2 or arr.shape[1] < 3:
            raise ValueError("Edge split table must be dict-like or 2D array with >=3 columns.")
        tf_raw, tg_raw, y_raw = arr[:, 0], arr[:, 1], arr[:, 2]

    def to_idx(x):
        if isinstance(x, (int, np.integer)):
            return int(x)
        sx = str(x).strip()
        if sx.isdigit() or (sx.startswith("-") and sx[1:].isdigit()):
            return int(sx)
        if sx in gene_to_idx:
            return int(gene_to_idx[sx])
        raise KeyError(f"Cannot map edge endpoint '{x}' to gene index.")

    tf = np.array([to_idx(x) for x in tf_raw], dtype=int)
    tg = np.array([to_idx(x) for x in tg_raw], dtype=int)
    y = np.array([int(float(v)) for v in y_raw], dtype=int)

    if not (len(tf) == len(tg) == len(y)):
        raise ValueError("Edge split columns have inconsistent lengths.")
    return tf, tg, y


def prepare_dataset(adata: ad.AnnData):
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

    return {
        "genes": genes.tolist(),
        "train_tf": np.concatenate([tr_tf, va_tf]),
        "train_tg": np.concatenate([tr_tg, va_tg]),
        "train_y": np.concatenate([tr_y, va_y]),
        "test_tf": te_tf,
        "test_tg": te_tg,
        "test_y": te_y,
    }


def pair_features(lookup, tf, tg):
    a = lookup[tf]
    b = lookup[tg]
    had = a * b
    cos = np.sum(a * b, axis=1, keepdims=True) / (
        (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8) * (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    )
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


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    seed_csv = os.path.join(args.out_dir, "embedding_transfer_seed_results_v2.csv")
    sum_csv = os.path.join(args.out_dir, "embedding_transfer_summary_v2.csv")
    report_md = os.path.join(args.out_dir, "embedding_transfer_report_v2.md")

    with open(f"{args.base_dir}/vocab.json", encoding="utf-8") as f:
        vocab = json.load(f)

    manifest = [r for r in read_csv(args.pair_manifest) if r["protocol"] in {"native", "strict", "coverage_matched"}]
    datasets = sorted(set([r["train_dataset"] for r in manifest] + [r["test_dataset"] for r in manifest]))

    ds = {}
    for d in datasets:
        p = os.path.join(args.h5ad_root, f"{d}.h5ad")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing dataset h5ad: {p}")
        ds[d] = prepare_dataset(ad.read_h5ad(p))

    if args.embeddings_config:
        with open(args.embeddings_config, encoding="utf-8") as f:
            emb_cfg = json.load(f)
    else:
        emb_cfg = default_embeddings_config(args.base_dir)

    emb_cfg = {k: v for k, v in emb_cfg.items() if os.path.exists(v.get("path", ""))}
    if not emb_cfg:
        raise FileNotFoundError(
            "No embedding checkpoints found. Provide --embeddings-config or place defaults under "
            "--base-dir/save_pretrain/{minus,baseline,scGPT_human,v4_bias_rec_best,v4_plain_best,v4_type_pe_best}/best_model.pt."
        )

    emb_map = {name: load_embedding(cfg["path"], cfg["key"]) for name, cfg in emb_cfg.items()}

    seed_rows = []

    total_jobs = len(manifest)
    render_progress(0, total_jobs, prefix="transfer_v2")
    for i_row, row in enumerate(manifest, start=1):
        tr = row["train_dataset"]
        te = row["test_dataset"]
        protocol = row["protocol"]
        case_mode = row.get("case_mode", "upper")

        trd = ds[tr]
        ted = ds[te]

        tr_can = {}
        for g in trd["genes"]:
            c = canonical(g, case_mode)
            if c not in tr_can:
                tr_can[c] = g
        te_can = {}
        for g in ted["genes"]:
            c = canonical(g, case_mode)
            if c not in te_can:
                te_can[c] = g

        native_tr_can = sorted(tr_can.keys())
        native_te_can = sorted(te_can.keys())
        shared_can = sorted(set(native_tr_can).intersection(set(native_te_can)))

        for emb_name, emb in emb_map.items():
            max_idx = emb.shape[0]
            # Native: train/test use their own available genes (v1-aligned), not forced to shared intersection.
            if protocol == "native":
                tr_raw = [tr_can[c] for c in native_tr_can if tr_can[c] in vocab and int(vocab[tr_can[c]]) < max_idx]
                te_raw = [te_can[c] for c in native_te_can if te_can[c] in vocab and int(vocab[te_can[c]]) < max_idx]
                if len(tr_raw) < 20 or len(te_raw) < 20:
                    continue
            else:
                with open(row.get("gene_set_file", ""), encoding="utf-8") as f:
                    allowed = set(x.strip() for x in f if x.strip())

                if protocol == "strict":
                    # strict: use exactly allowed ∩ shared canonical genes on both sides
                    strict_can = sorted(allowed.intersection(set(shared_can)))
                    tr_raw, te_raw = [], []
                    for c in strict_can:
                        gtr, gte = tr_can[c], te_can[c]
                        if gtr in vocab and gte in vocab and int(vocab[gtr]) < max_idx and int(vocab[gte]) < max_idx:
                            tr_raw.append(gtr)
                            te_raw.append(gte)
                    if len(tr_raw) < 20:
                        continue
                else:
                    # coverage_matched: use K=len(file) as matched target size, but sample independently
                    # from native train/test pools to avoid collapsing to strict.
                    k_target = len(allowed)
                    tr_pool = [tr_can[c] for c in native_tr_can if tr_can[c] in vocab and int(vocab[tr_can[c]]) < max_idx]
                    te_pool = [te_can[c] for c in native_te_can if te_can[c] in vocab and int(vocab[te_can[c]]) < max_idx]
                    k = min(k_target, len(tr_pool), len(te_pool))
                    if k < 20:
                        continue
                    seed_key = f"{tr}::{te}::{emb_name}::{k}::coverage_matched"
                    seed_int = int(hashlib.md5(seed_key.encode()).hexdigest()[:8], 16)
                    rng = np.random.default_rng(seed_int)
                    tr_raw = sorted(rng.choice(np.array(tr_pool, dtype=object), size=k, replace=False).tolist())
                    te_raw = sorted(rng.choice(np.array(te_pool, dtype=object), size=k, replace=False).tolist())

            gmap_tr = {g: i for i, g in enumerate(tr_raw)}
            gmap_te = {g: i for i, g in enumerate(te_raw)}

            tr_tf, tr_tg, tr_y = map_pairs_to_local(trd["train_tf"], trd["train_tg"], trd["train_y"], trd["genes"], gmap_tr)
            te_tf, te_tg, te_y = map_pairs_to_local(ted["test_tf"], ted["test_tg"], ted["test_y"], ted["genes"], gmap_te)

            if len(tr_y) < 20 or len(te_y) < 20:
                continue

            emb_lookup_tr = np.stack([emb[vocab[g]] for g in tr_raw], axis=0)
            emb_lookup_te = np.stack([emb[vocab[g]] for g in te_raw], axis=0)
            Xtr = pair_features(emb_lookup_tr, tr_tf, tr_tg)
            Xte = pair_features(emb_lookup_te, te_tf, te_tg)

            for clf in args.classifiers:
                for seed in args.seeds:
                    if len(np.unique(tr_y)) < 2 or len(np.unique(te_y)) < 2:
                        continue
                    au, ap = fit_eval(Xtr, tr_y, Xte, te_y, clf, seed, resample_lr=args.resample_lr)
                    seed_rows.append(
                        {
                            "train_dataset": tr,
                            "test_dataset": te,
                            "protocol": protocol,
                            "embedding": emb_name,
                            "clf": clf,
                            "seed": seed,
                            "n_common_genes": min(len(tr_raw), len(te_raw)),
                            "n_train_edges": len(tr_y),
                            "n_test_edges": len(te_y),
                            "auroc": f"{au:.6f}",
                            "auprc": f"{ap:.6f}",
                        }
                    )
        render_progress(i_row, total_jobs, prefix="transfer_v2")

    write_csv(
        seed_csv,
        seed_rows,
        [
            "train_dataset",
            "test_dataset",
            "protocol",
            "embedding",
            "clf",
            "seed",
            "n_common_genes",
            "n_train_edges",
            "n_test_edges",
            "auroc",
            "auprc",
        ],
    )
    sum_rows = summarize(seed_rows)
    write_csv(
        sum_csv,
        sum_rows,
        [
            "train_dataset",
            "test_dataset",
            "protocol",
            "embedding",
            "clf",
            "mean_auroc",
            "std_auroc",
            "mean_auprc",
            "std_auprc",
            "n",
        ],
    )

    with open(report_md, "w", encoding="utf-8") as f:
        f.write("# embedding_transfer_report_v2\n\n")
        f.write("## 实验设置（h5ad-only, v1-aligned edge-level）\n\n")
        f.write(f"- h5ad_root: {args.h5ad_root}\n")
        f.write(f"- pair_manifest: {args.pair_manifest}\n")
        f.write(f"- embeddings: {list(emb_map.keys())}\n")
        f.write(f"- classifiers: {args.classifiers}\n")
        f.write(f"- seeds: {args.seeds}\n")
        f.write(f"- resample_lr: {args.resample_lr}\n\n")
        f.write("## 输出\n\n")
        f.write(f"- `{seed_csv}`\n")
        f.write(f"- `{sum_csv}`\n")

    print(f"[OK] wrote {seed_csv}")
    print(f"[OK] wrote {sum_csv}")
    print(f"[OK] wrote {report_md}")


if __name__ == "__main__":
    main()
