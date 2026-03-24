#!/usr/bin/env python3
"""Embedding transfer benchmark v2 (h5ad-native; no scGREAT dependency).

Task definition (cell-level transfer):
- Build cell embeddings by expression-weighted averaging of gene embeddings.
- Train on source dataset cells; test on target dataset cells.
- Labels come from pseudotime median split per dataset.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
from collections import defaultdict
from statistics import mean, pstdev

import anndata as ad
import numpy as np
from scipy import sparse

LEGACY_SCRIPTS = ["analyze_grn_transferability.py", "run_transfer_control.py"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run embedding transfer benchmark (v2, h5ad-native).")
    p.add_argument("--base-dir", default="/bigdata2/hyt/projects/scbenchmark", help="Directory containing vocab/checkpoints.")
    p.add_argument("--h5ad-root", default="processed/native", help="Directory containing per-dataset .h5ad files.")
    p.add_argument("--pair-manifest", default="transfer_v2/pair_manifest.csv")
    p.add_argument("--out-dir", default="transfer")
    p.add_argument("--embeddings-config", default="", help="Optional JSON file: {name:{path,key},...}")
    p.add_argument("--classifiers", nargs="*", default=["lr", "mlp"])
    p.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
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


def canonical(g: str, mode: str) -> str:
    s = str(g).strip()
    if mode == "upper":
        return s.upper()
    if mode == "lower":
        return s.lower()
    return s


def load_legacy_embeddings(base_dir: str) -> dict[str, dict[str, str]]:
    merged: dict[str, dict[str, str]] = {}
    for script in LEGACY_SCRIPTS:
        if not os.path.exists(script):
            continue
        tree = ast.parse(open(script, encoding="utf-8").read(), filename=script)
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id == "EMBEDDINGS":
                        val = eval(compile(ast.Expression(node.value), script, "eval"), {"BASE_DIR": base_dir}, {})
                        if isinstance(val, dict):
                            merged.update(val)
    return merged


def load_embedding(path: str, key: str):
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if key in ckpt:
        return ckpt[key].detach().cpu().numpy()
    for nk in ["state_dict", "model_state_dict", "model"]:
        if nk in ckpt and isinstance(ckpt[nk], dict) and key in ckpt[nk]:
            return ckpt[nk][key].detach().cpu().numpy()
    raise KeyError(f"missing key {key} in {path}")


def fit_eval(Xtr, ytr, Xte, yte, clf, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

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
        out.append({
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
        })
    return out


def train_test_idx(n: int, seed: int, ratio: float = 0.8):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    cut = int(n * ratio)
    return idx[:cut], idx[cut:]


def prepare_dataset(adata: ad.AnnData):
    if "pseudotime" not in adata.obs.columns:
        raise ValueError("h5ad missing obs['pseudotime']")

    y = np.asarray(adata.obs["pseudotime"].to_numpy(dtype=float))
    med = np.nanmedian(y)
    y_bin = (y >= med).astype(int)

    mat = adata.layers["counts"] if "counts" in adata.layers else adata.X
    if sparse.issparse(mat):
        X = mat.tocsr().astype(np.float32)
    else:
        X = np.asarray(mat, dtype=np.float32)

    genes = np.asarray(adata.var_names.astype(str))
    return X, genes, y_bin


def select_gene_indices(genes: np.ndarray, case_mode: str):
    mp = {}
    for i, g in enumerate(genes):
        cg = canonical(g, case_mode)
        if cg not in mp:
            mp[cg] = i
    return mp


def cell_embed(X, gene_idx, emb_mat):
    # X: cells x genes (dense or csr), gene_idx: indices selected in X, emb_mat: selected_genes x dim
    if sparse.issparse(X):
        W = X[:, gene_idx].toarray()
    else:
        W = X[:, gene_idx]
    W = np.asarray(W, dtype=np.float32)
    s = W.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    W = W / s
    return W @ emb_mat


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
        emb_cfg = load_legacy_embeddings(args.base_dir)

    emb_cfg = {k: v for k, v in emb_cfg.items() if os.path.exists(v.get("path", ""))}
    if not emb_cfg:
        raise FileNotFoundError("No embedding checkpoints found. Use --embeddings-config or restore legacy EMBEDDINGS.")

    emb_map = {name: load_embedding(cfg["path"], cfg["key"]) for name, cfg in emb_cfg.items()}

    seed_rows = []

    for row in manifest:
        tr = row["train_dataset"]
        te = row["test_dataset"]
        protocol = row["protocol"]
        case_mode = row.get("case_mode", "upper")

        Xtr_all, gtr, ytr_all = ds[tr]
        Xte_all, gte, yte_all = ds[te]

        if protocol == "native":
            allowed = set(canonical(g, case_mode) for g in gtr).intersection(set(canonical(g, case_mode) for g in gte))
        else:
            with open(row.get("gene_set_file", ""), encoding="utf-8") as f:
                allowed = set(x.strip() for x in f if x.strip())

        idx_tr = select_gene_indices(gtr, case_mode)
        idx_te = select_gene_indices(gte, case_mode)
        common = sorted(allowed.intersection(idx_tr.keys()).intersection(idx_te.keys()))
        if len(common) < 20:
            continue

        # choose genes mappable to vocab first
        common_vocab = [g for g in common if g in vocab]
        if len(common_vocab) < 20:
            continue

        tr_gene_idx = np.array([idx_tr[g] for g in common_vocab], dtype=int)
        te_gene_idx = np.array([idx_te[g] for g in common_vocab], dtype=int)

        for emb_name, emb in emb_map.items():
            emb_idx = np.array([vocab[g] for g in common_vocab], dtype=int)
            emb_mat = emb[emb_idx]

            Ztr = cell_embed(Xtr_all, tr_gene_idx, emb_mat)
            Zte = cell_embed(Xte_all, te_gene_idx, emb_mat)

            for clf in args.classifiers:
                for seed in args.seeds:
                    tr_i, _ = train_test_idx(Ztr.shape[0], seed)
                    _, te_i = train_test_idx(Zte.shape[0], seed)
                    if len(np.unique(ytr_all[tr_i])) < 2 or len(np.unique(yte_all[te_i])) < 2:
                        continue

                    au, ap = fit_eval(Ztr[tr_i], ytr_all[tr_i], Zte[te_i], yte_all[te_i], clf, seed)
                    seed_rows.append({
                        "train_dataset": tr,
                        "test_dataset": te,
                        "protocol": protocol,
                        "embedding": emb_name,
                        "clf": clf,
                        "seed": seed,
                        "n_common_genes": len(common_vocab),
                        "n_train_cells": len(tr_i),
                        "n_test_cells": len(te_i),
                        "auroc": f"{au:.6f}",
                        "auprc": f"{ap:.6f}",
                    })

    write_csv(seed_csv, seed_rows, ["train_dataset", "test_dataset", "protocol", "embedding", "clf", "seed", "n_common_genes", "n_train_cells", "n_test_cells", "auroc", "auprc"])
    sum_rows = summarize(seed_rows)
    write_csv(sum_csv, sum_rows, ["train_dataset", "test_dataset", "protocol", "embedding", "clf", "mean_auroc", "std_auroc", "mean_auprc", "std_auprc", "n"])

    with open(report_md, "w", encoding="utf-8") as f:
        f.write("# embedding_transfer_report_v2\n\n")
        f.write("## 实验设置\n\n")
        f.write(f"- h5ad_root: {args.h5ad_root}\n")
        f.write(f"- pair_manifest: {args.pair_manifest}\n")
        f.write(f"- embeddings: {list(emb_map.keys())}\n")
        f.write(f"- classifiers: {args.classifiers}\n")
        f.write(f"- seeds: {args.seeds}\n\n")
        f.write("## 输出\n\n")
        f.write(f"- `{seed_csv}`\n")
        f.write(f"- `{sum_csv}`\n")

    print(f"[OK] wrote {seed_csv}")
    print(f"[OK] wrote {sum_csv}")
    print(f"[OK] wrote {report_md}")


if __name__ == "__main__":
    main()
