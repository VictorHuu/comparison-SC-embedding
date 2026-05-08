#!/usr/bin/env python3
"""Leak-aware supervised TF->target GRN link-prediction utilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

BASE_DIR = "/bigdata2/hyt/projects/scbenchmark"
VOCAB_PATH = f"{BASE_DIR}/vocab.json"
EMBEDDINGS = {
    "minus": {"path": f"{BASE_DIR}/save_pretrain/minus/best_model.pt", "key": "module.embedding.weight"},
    "baseline": {"path": f"{BASE_DIR}/save_pretrain/baseline/best_model.pt", "key": "module.embedding.weight"},
    "scGPT_human": {"path": f"{BASE_DIR}/save_pretrain/scGPT_human/best_model.pt", "key": "encoder.embedding.weight"},
    "v4_bias_rec_best": {"path": f"{BASE_DIR}/save_pretrain/v4_bias_rec_best/best_model.pt", "key": "embedding.weight"},
    "v4_plain_best": {"path": f"{BASE_DIR}/save_pretrain/v4_plain_best/best_model.pt", "key": "encoder.embedding.weight"},
    "v4_type_pe_best": {"path": f"{BASE_DIR}/save_pretrain/v4_type_pe_best/best_model.pt", "key": "embedding.weight"},
}
REQUIRED_SPLITS = ("Train_set", "Validation_set", "Test_set")


def load_vocab(path: str = VOCAB_PATH) -> Dict[str, int]:
    with open(path) as f:
        return json.load(f)


def load_embedding(name: str, registry: Dict[str, Dict[str, str]] = EMBEDDINGS) -> np.ndarray:
    cfg = registry[name]
    obj = torch.load(cfg["path"], map_location="cpu", weights_only=False)
    if cfg["key"] not in obj:
        raise KeyError(f"{name}: missing checkpoint key {cfg['key']} in {cfg['path']}")
    return obj[cfg["key"]].detach().cpu().numpy().astype(np.float32)


def detect_tf_target_label_columns(df: pd.DataFrame) -> Tuple[str, str, str | None]:
    norm = {str(c).lower().replace("_", "").replace("-", "").replace(".", ""): c for c in df.columns}
    tf = next((norm[k] for k in ["tf", "transcriptionfactor", "gene1", "regulator", "source"] if k in norm), None)
    tg = next((norm[k] for k in ["target", "targetgene", "gene2", "target_gene", "sink"] if k in norm), None)
    lab = next((norm[k] for k in ["label", "y", "ispositive", "edge", "class"] if k in norm), None)
    if tf is None or tg is None:
        if df.shape[1] >= 2:
            tf, tg = df.columns[0], df.columns[1]
        else:
            raise ValueError("edge table needs at least TF and target columns")
    return tf, tg, lab


def _read_edge_csv(path: Path, genes: List[str] | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    # scGREAT split files are frequently index + integer TF/target/label columns.
    if df.shape[1] >= 4 and str(df.columns[0]).lower().startswith("unnamed"):
        df = df.iloc[:, 1:]
    if df.shape[1] >= 3 and np.issubdtype(df.iloc[:, 0].dtype, np.number) and np.issubdtype(df.iloc[:, 1].dtype, np.number):
        out = pd.DataFrame({"tf": df.iloc[:, 0].astype(int), "target": df.iloc[:, 1].astype(int), "label": df.iloc[:, 2].astype(int)})
        if genes is not None:
            out["tf"] = out["tf"].map(lambda i: genes[i] if 0 <= int(i) < len(genes) else None)
            out["target"] = out["target"].map(lambda i: genes[i] if 0 <= int(i) < len(genes) else None)
            out = out.dropna(subset=["tf", "target"])
        return out[["tf", "target", "label"]].drop_duplicates()
    tf, tg, lab = detect_tf_target_label_columns(df)
    out = pd.DataFrame({"tf": df[tf].astype(str), "target": df[tg].astype(str)})
    out["label"] = df[lab].astype(int) if lab else 1
    return out[["tf", "target", "label"]].drop_duplicates()


def load_dataset_edges(dataset_dir: str | Path) -> Tuple[str, List[str], Dict[str, pd.DataFrame]]:
    ds = Path(dataset_dir)
    target_path = ds / "Target.csv"
    genes = pd.read_csv(target_path)["Gene"].astype(str).tolist() if target_path.exists() and "Gene" in pd.read_csv(target_path, nrows=1).columns else []
    splits = {}
    for split in REQUIRED_SPLITS:
        path = ds / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(f"missing required GRN split file: {path}")
        splits[split] = _read_edge_csv(path, genes if genes else None)
    if not genes:
        genes = sorted(set(pd.concat(splits.values())["tf"]).union(set(pd.concat(splits.values())["target"])))
    return ds.name, genes, splits


def discover_grn_datasets(base_dir: str) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    roots = [Path(base_dir), Path(base_dir) / "scGREAT", Path("/bigdata2/hyt/projects/scGREAT")]
    rows, missing = [], []
    seen = set()
    for root in roots:
        if not root.exists():
            continue
        for d in [p for p in root.rglob("*") if p.is_dir()]:
            files = {p.name for p in d.iterdir() if p.is_file()}
            has_any = any(f"{s}.csv" in files for s in REQUIRED_SPLITS)
            if not has_any:
                continue
            miss = [f"{s}.csv" for s in REQUIRED_SPLITS if f"{s}.csv" not in files]
            key = str(d.resolve())
            if key in seen:
                continue
            seen.add(key)
            if miss:
                missing.append({"asset_type": "dataset", "asset_name": d.name, "status": "MISSING_GRN_SPLITS", "notes": ";".join(miss), "path": str(d)})
            else:
                rows.append({"dataset": d.name, "dataset_dir": str(d), "status": "OK", "notes": "Train/Validation/Test present"})
    if not rows:
        missing.append({"asset_type": "dataset", "asset_name": "GRN edge splits", "status": "MISSING_REQUIRED_GRN_EDGE_FILES", "notes": "Need Train_set.csv, Validation_set.csv, Test_set.csv", "path": base_dir})
    return rows, missing


def candidate_space(genes: Iterable[str], positives: pd.DataFrame, vocab: Dict[str, int]) -> Tuple[List[str], List[str], set[Tuple[str, str]]]:
    gene_set = {str(g) for g in genes if str(g) in vocab}
    pos = positives[(positives.tf.isin(gene_set)) & (positives.target.isin(gene_set))]
    tfs = sorted(set(pos.tf) & gene_set)
    targets = sorted(gene_set)
    all_pos = set(zip(pos.tf.astype(str), pos.target.astype(str)))
    return tfs, targets, all_pos


def split_positive_edges(pos: pd.DataFrame, mode: str, seed: int) -> Dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    pos = pos[["tf", "target"]].drop_duplicates().reset_index(drop=True)
    if mode == "edge_holdout":
        idx = rng.permutation(len(pos)); ntr = int(0.6 * len(idx)); nva = int(0.2 * len(idx))
        return {"train": pos.iloc[idx[:ntr]].copy(), "val": pos.iloc[idx[ntr:ntr+nva]].copy(), "test": pos.iloc[idx[ntr+nva:]].copy()}
    key = "tf" if mode == "tf_holdout" else "target"
    vals = np.array(sorted(pos[key].unique()), dtype=object); rng.shuffle(vals)
    ntr = int(0.6 * len(vals)); nva = int(0.2 * len(vals))
    parts = {"train": set(vals[:ntr]), "val": set(vals[ntr:ntr+nva]), "test": set(vals[ntr+nva:])}
    return {k: pos[pos[key].isin(v)].copy() for k, v in parts.items()}


def _sample_from_pool(pool: List[Tuple[str, str]], n: int, rng: np.random.Generator) -> List[Tuple[str, str]]:
    if n <= 0 or not pool:
        return []
    n = min(n, len(pool))
    idx = rng.choice(len(pool), size=n, replace=False)
    return [pool[int(i)] for i in idx]


def sample_negatives(pos_split: Dict[str, pd.DataFrame], tfs: List[str], targets: List[str], all_pos: set[Tuple[str, str]], ratio: float, protocol: str, seed: int) -> Tuple[Dict[str, pd.DataFrame], List[Dict[str, object]]]:
    rng = np.random.default_rng(seed)
    used_neg = set()
    pos_all = pd.concat(pos_split.values(), ignore_index=True)
    tf_degree = pos_all.tf.value_counts().to_dict()
    target_degree = pos_all.target.value_counts().to_dict()
    diag, out = [], {}
    for split, pdf in pos_split.items():
        n_need = int(np.ceil(len(pdf) * ratio))
        pool = [(tf, tg) for tf in tfs for tg in targets if tf != tg and (tf, tg) not in all_pos and (tf, tg) not in used_neg]
        if protocol == "degree_matched_negative" and len(pdf):
            chosen = []
            by_tf = {tf: [p for p in pool if p[0] == tf] for tf in set(pdf.tf)}
            for tf, grp in pdf.groupby("tf"):
                chosen += _sample_from_pool(by_tf.get(tf, []), int(np.ceil(len(grp) * ratio)), rng)
            if len(chosen) < n_need:
                chosen += _sample_from_pool([p for p in pool if p not in set(chosen)], n_need - len(chosen), rng)
        else:
            chosen = _sample_from_pool(pool, n_need, rng)
        used_neg.update(chosen)
        ndf = pd.DataFrame(chosen, columns=["tf", "target"]) if chosen else pd.DataFrame(columns=["tf", "target"])
        out[split] = ndf
        y_pos, y_neg = len(pdf), len(ndf)
        diag.append({"split": split, "negative_sampling": protocol, "negative_ratio": ratio, "n_positive": y_pos, "n_negative": y_neg, "positive_ratio": y_pos / max(1, y_pos + y_neg), "n_tfs": len(tfs), "n_targets": len(targets), "mean_tf_degree": np.mean([tf_degree.get(tf, 0) for tf in tfs]) if tfs else np.nan, "mean_target_degree": np.mean([target_degree.get(tg, 0) for tg in targets]) if targets else np.nan})
    return out, diag


def make_labeled(pos: pd.DataFrame, neg: pd.DataFrame) -> pd.DataFrame:
    p = pos.copy(); p["label"] = 1
    n = neg.copy(); n["label"] = 0
    return pd.concat([p, n], ignore_index=True).drop_duplicates(["tf", "target", "label"])


def _topology_features(edges: pd.DataFrame, train_edges: pd.DataFrame) -> np.ndarray:
    pos = train_edges[train_edges.label == 1] if "label" in train_edges else train_edges
    tf_deg = pos.tf.value_counts().to_dict()
    tg_deg = pos.target.value_counts().to_dict()
    node_freq = pd.concat([pos.tf, pos.target]).value_counts().to_dict() if len(pos) else {}
    return np.asarray([[tf_deg.get(r.tf, 0), tg_deg.get(r.target, 0), node_freq.get(r.tf, 0), node_freq.get(r.target, 0)] for r in edges.itertuples()], dtype=np.float32)


def pair_features(edges: pd.DataFrame, emb: np.ndarray, vocab: Dict[str, int], mode: str = "embedding_pair", train_edges: pd.DataFrame | None = None) -> np.ndarray:
    ti = edges.tf.map(vocab).to_numpy(); gi = edges.target.map(vocab).to_numpy()
    etf, etg = emb[ti], emb[gi]
    dot = np.sum(etf * etg, axis=1, keepdims=True)
    denom = (np.linalg.norm(etf, axis=1, keepdims=True) * np.linalg.norm(etg, axis=1, keepdims=True) + 1e-8)
    cos = dot / denom
    euc = np.linalg.norm(etf - etg, axis=1, keepdims=True)
    if mode == "similarity_only":
        return np.concatenate([cos, dot, euc], axis=1).astype(np.float32)
    base = np.concatenate([etf, etg, etf * etg, np.abs(etf - etg), cos], axis=1).astype(np.float32)
    if mode == "topology_features_optional" and train_edges is not None:
        return np.concatenate([base, _topology_features(edges, train_edges)], axis=1).astype(np.float32)
    return base


def fit_predict(train_edges: pd.DataFrame, test_edges: pd.DataFrame, emb: np.ndarray, vocab: Dict[str, int], model: str, seed: int, feature_mode: str = "embedding_pair") -> np.ndarray:
    Xtr = pair_features(train_edges, emb, vocab, feature_mode, train_edges); ytr = train_edges.label.astype(int).to_numpy()
    Xte = pair_features(test_edges, emb, vocab, feature_mode, train_edges)
    scaler = StandardScaler().fit(Xtr); Xtr = scaler.transform(Xtr); Xte = scaler.transform(Xte)
    if model == "mlp":
        clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200, early_stopping=True, random_state=seed)
    elif model == "elasticnet_lr":
        clf = LogisticRegression(max_iter=1000, penalty="elasticnet", solver="saga", l1_ratio=0.5, n_jobs=1, random_state=seed)
    else:
        clf = LogisticRegression(max_iter=1000, n_jobs=1, random_state=seed)
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


def score_metrics(y: np.ndarray, s: np.ndarray) -> Dict[str, float]:
    out = {"auroc": np.nan, "auprc": np.nan, "precision_at_k": np.nan, "recall_at_k": np.nan, "f1_at_threshold": np.nan, "pos_score_mean": np.nan, "neg_score_mean": np.nan, "pos_score_median": np.nan, "neg_score_median": np.nan}
    if len(y) == 0 or len(np.unique(y)) < 2:
        return out
    out["auroc"] = float(roc_auc_score(y, s)); out["auprc"] = float(average_precision_score(y, s))
    k = max(1, int(np.sum(y == 1))); order = np.argsort(-s)[:k]
    out["precision_at_k"] = float(np.mean(y[order] == 1)); out["recall_at_k"] = float(np.sum(y[order] == 1) / max(1, np.sum(y == 1)))
    prec, rec, thr = precision_recall_curve(y, s)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12); imax = int(np.nanargmax(f1))
    threshold = float(thr[min(imax, len(thr)-1)]) if len(thr) else 0.5
    out["f1_at_threshold"] = float(f1_score(y, (s >= threshold).astype(int), zero_division=0))
    out["pos_score_mean"] = float(np.mean(s[y == 1])); out["neg_score_mean"] = float(np.mean(s[y == 0]))
    out["pos_score_median"] = float(np.median(s[y == 1])); out["neg_score_median"] = float(np.median(s[y == 0]))
    return out
