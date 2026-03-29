#!/usr/bin/env python3
"""
Conservative, leak-free perturbation regression benchmark.

Design principles:
- PRIMARY comparison: frozen_linear (probe setting).
- SECONDARY adaptation: frozen_backbone_trainable_head.
- EXPLORATORY only: full_finetune_embedding_head (disabled by default).

Important: implementation is embedding-agnostic and does not assume any embedding
should win.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BASE_DIR = '/bigdata2/hyt/projects/scbenchmark'
OUTPUT_DIR = '/bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/results/perturbation_regression'
os.makedirs(OUTPUT_DIR, exist_ok=True)

VOCAB_PATH = f"{BASE_DIR}/vocab.json"
PERTURB_DATA_DIR = f"{BASE_DIR}/data/downstreams/perturbation/processed_data"
DATASETS = ["adamson", "dixit", "norman"]

EMBEDDINGS = {
    "minus": {
        "path": f"{BASE_DIR}/save_pretrain/minus/best_model.pt",
        "key": "module.embedding.weight",
    },
    "baseline": {
        "path": f"{BASE_DIR}/save_pretrain/baseline/best_model.pt",
        "key": "module.embedding.weight",
    },
    "scGPT_human": {
        "path": f"{BASE_DIR}/save_pretrain/scGPT_human/best_model.pt",
        "key": "encoder.embedding.weight",
    },
    "v4_bias_rec_best": {
        "path": f"{BASE_DIR}/save_pretrain/v4_bias_rec_best/best_model.pt",
        "key": "embedding.weight",
    },
    "v4_plain_best": {
        "path": f"{BASE_DIR}/save_pretrain/v4_plain_best/best_model.pt",
        "key": "encoder.embedding.weight",
    },
    "v4_type_pe_best": {
        "path": f"{BASE_DIR}/save_pretrain/v4_type_pe_best/best_model.pt",
        "key": "embedding.weight",
    }
}

SEED = 42
TOP_K = 256
MIN_CELLS_PER_PERT = 5
MIN_CONTEXT_PERT_GENES = 10
MIN_NN_TRAIN = 20


@dataclass
class ContextSamples:
    """Per-context perturbation gene samples.

    Each sample corresponds to one perturbation gene and its delta vector
    (mean perturbed profile - mean control profile).
    """

    context: str
    gene_ids: np.ndarray
    deltas_full: np.ndarray


@dataclass
class FoldMetric:
    """Single fold metric payload for downstream paired analysis."""

    fold_id: int
    pearson_r: float
    mse: float
    sign_acc: float
    n_train: int
    n_test: int


@dataclass
class EvalResult:
    """Aggregated summary plus fold-level metrics."""

    pearson_r: float
    pearson_r_std: float
    mse: float
    mse_std: float
    sign_acc: float
    sign_acc_std: float
    n_folds: int
    target_dim: int
    fold_metrics: List[FoldMetric]


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)


def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in {"1", "true", "t", "yes", "y"}:
        return True
    if v in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean value: {v}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Leak-free perturbation regression benchmark")
    p.add_argument("--top_k", type=int, default=TOP_K)
    p.add_argument("--enable_full_finetune", type=str2bool, default=False,
                   help="Enable exploratory full embedding fine-tuning (default: false).")
    p.add_argument("--enable_sign_reg", type=str2bool, default=False,
                   help="Optional sign regularization for neural head (default: false).")
    p.add_argument("--sign_reg_weight", type=float, default=0.05)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--finetune_epochs", type=int, default=200)
    p.add_argument("--finetune_lr", type=float, default=1e-3)
    p.add_argument("--finetune_weight_decay", type=float, default=1e-4)
    return p.parse_args()


def load_vocab() -> Dict[str, int]:
    with open(VOCAB_PATH) as f:
        return json.load(f)


def load_checkpoint_embedding(path: str, key: str) -> np.ndarray:
    """Load embedding with robust checkpoint key resolution."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    candidate_keys = [
        key,
        "encoder.embedding.weight",
        "module.embedding.weight",
        "embedding.weight",
    ]
    for k in candidate_keys:
        if k and k in ckpt:
            return ckpt[k].detach().cpu().numpy().astype(np.float32)

    for nested_key in ["state_dict", "model_state_dict", "model"]:
        if nested_key in ckpt and isinstance(ckpt[nested_key], dict):
            sd = ckpt[nested_key]
            for k in candidate_keys:
                if k and k in sd:
                    return sd[k].detach().cpu().numpy().astype(np.float32)

    raise KeyError(f"No embedding weight key found. Tried: {candidate_keys}")


def load_perturb_data(ds_name: str) -> Dict[str, object]:
    path = os.path.join(PERTURB_DATA_DIR, f"{ds_name}_data.pt")
    d = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "raw": d,
        "genes_list": d["genes"],
        "expr_list": d["expressions"],
        "base_idx": d["base_idx"],
        "single_ctrl": d["single_ctrl"],
    }


def build_dense_profiles(data: Dict[str, object], vocab_size: int) -> np.ndarray:
    """Precompute dense profile for every cell once."""
    genes_list = data["genes_list"]
    expr_list = data["expr_list"]
    n_cells = len(genes_list)
    dense = np.zeros((n_cells, vocab_size), dtype=np.float32)
    for i in range(n_cells):
        g = np.asarray(genes_list[i], dtype=np.int64)
        e = np.asarray(expr_list[i], dtype=np.float32)
        valid = (g >= 0) & (g < vocab_size)
        if valid.any():
            np.add.at(dense[i], g[valid], e[valid])
    return dense


def _extract_cell_types(raw_dict: Dict[str, object], n_cells: int) -> Optional[np.ndarray]:
    candidate_keys = [
        "cell_type", "cell_types", "celltype", "celltypes",
        "cls_name", "cls", "batch_name", "batch",
    ]
    for k in candidate_keys:
        if k not in raw_dict:
            continue
        v = raw_dict[k]
        if isinstance(v, (list, tuple, np.ndarray)) and len(v) == n_cells:
            return np.array(v)
    return None


def _build_context_samples(
    dense_profiles: np.ndarray,
    single_ctrl: Sequence[int],
    ctrl_indices: List[int],
    pert_indices: List[int],
    min_cells_per_pert: int,
) -> Optional[ContextSamples]:
    if len(ctrl_indices) < 20 or len(pert_indices) < 50:
        return None

    mean_ctrl = dense_profiles[np.asarray(ctrl_indices, dtype=np.int64)].mean(axis=0)

    by_pert_gene: Dict[int, List[int]] = defaultdict(list)
    for i in pert_indices:
        gid = int(single_ctrl[i])
        if gid >= 0:
            by_pert_gene[gid].append(i)

    valid_pert_genes = [g for g, idxs in by_pert_gene.items() if len(idxs) >= min_cells_per_pert]
    if len(valid_pert_genes) < MIN_CONTEXT_PERT_GENES:
        return None

    gene_ids: List[int] = []
    deltas: List[np.ndarray] = []
    for g in valid_pert_genes:
        idxs = np.asarray(by_pert_gene[g], dtype=np.int64)
        mean_pert = dense_profiles[idxs].mean(axis=0)
        deltas.append(mean_pert - mean_ctrl)
        gene_ids.append(g)

    return ContextSamples(
        context="",
        gene_ids=np.asarray(gene_ids, dtype=np.int64),
        deltas_full=np.stack(deltas, axis=0).astype(np.float32),
    )


def collect_context_samples(
    data: Dict[str, object],
    dense_profiles: np.ndarray,
    min_cells_per_pert: int = MIN_CELLS_PER_PERT,
) -> List[ContextSamples]:
    """Collect cell-type contexts; if none valid, fallback to dataset::all."""
    base_idx = data["base_idx"]
    single_ctrl = data["single_ctrl"]

    ctrl_all = [i for i, b in enumerate(base_idx) if b == 1]
    pert_all = [i for i, b in enumerate(base_idx) if b == 0 and int(single_ctrl[i]) >= 0]

    n_cells = len(base_idx)
    cell_types = _extract_cell_types(data["raw"], n_cells)

    out: List[ContextSamples] = []
    if cell_types is not None:
        for ct in sorted(set(cell_types.tolist())):
            ct_idx = set(np.where(cell_types == ct)[0].tolist())
            ctrl_ct = [i for i in ctrl_all if i in ct_idx]
            pert_ct = [i for i in pert_all if i in ct_idx]
            cs = _build_context_samples(dense_profiles, single_ctrl, ctrl_ct, pert_ct, min_cells_per_pert)
            if cs is not None:
                cs.context = f"celltype::{ct}"
                out.append(cs)

    if not out:
        cs = _build_context_samples(dense_profiles, single_ctrl, ctrl_all, pert_all, min_cells_per_pert)
        if cs is not None:
            cs.context = "dataset::all"
            out.append(cs)

    return out


def make_cv_splitter(n_pert_genes: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Split strategy:
    - n<25: LeaveOneOut
    - n>=25: 5-fold KFold (or n_splits=min(5,n))
    """
    idx = np.arange(n_pert_genes)
    if n_pert_genes < 2:
        return []

    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    if n_pert_genes < 25:
        loo = LeaveOneOut()
        for tr, te in loo.split(idx):
            splits.append((tr, te))
        return splits

    kf = KFold(n_splits=min(5, n_pert_genes), shuffle=True, random_state=SEED)
    for tr, te in kf.split(idx):
        splits.append((tr, te))
    return splits


def select_top_genes_from_train(y_train_full: np.ndarray, top_k: int) -> np.ndarray:
    """Leak-free top-gene selection from training fold only."""
    score = np.mean(np.abs(y_train_full), axis=0)
    k = min(top_k, y_train_full.shape[1])
    return np.argsort(-score)[:k]


def _pearson_mean(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    rs: List[float] = []
    for i in range(len(y_true)):
        if np.std(y_true[i]) > 1e-8 and np.std(y_pred[i]) > 1e-8:
            rs.append(float(pearsonr(y_true[i], y_pred[i])[0]))
    return float(np.mean(rs)) if rs else float("nan")


def _sign_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.sign(y_pred) == np.sign(y_true)))


def _aggregate(folds: List[FoldMetric], target_dim: int) -> EvalResult:
    p = np.asarray([f.pearson_r for f in folds], dtype=np.float64)
    m = np.asarray([f.mse for f in folds], dtype=np.float64)
    s = np.asarray([f.sign_acc for f in folds], dtype=np.float64)
    return EvalResult(
        pearson_r=float(np.nanmean(p)),
        pearson_r_std=float(np.nanstd(p)),
        mse=float(np.nanmean(m)),
        mse_std=float(np.nanstd(m)),
        sign_acc=float(np.nanmean(s)),
        sign_acc_std=float(np.nanstd(s)),
        n_folds=len(folds),
        target_dim=target_dim,
        fold_metrics=folds,
    )


def evaluate_frozen_linear(
    X: np.ndarray,
    Y_full: np.ndarray,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    top_k: int,
) -> Optional[EvalResult]:
    """PRIMARY: frozen linear probe with fold-local standardization and top-k selection."""
    folds: List[FoldMetric] = []
    final_target_dim = min(top_k, Y_full.shape[1])

    for fold_id, (tr_idx, te_idx) in enumerate(cv_splits):
        X_tr, X_te = X[tr_idx], X[te_idx]
        Y_tr_full, Y_te_full = Y_full[tr_idx], Y_full[te_idx]

        top_idx = select_top_genes_from_train(Y_tr_full, top_k)
        Y_tr, Y_te = Y_tr_full[:, top_idx], Y_te_full[:, top_idx]

        xs = StandardScaler()
        ys = StandardScaler()
        X_tr_s = xs.fit_transform(X_tr)
        X_te_s = xs.transform(X_te)
        Y_tr_s = ys.fit_transform(Y_tr)
        Y_te_s = ys.transform(Y_te)

        model = Ridge(alpha=1.0)
        model.fit(X_tr_s, Y_tr_s)
        pred = model.predict(X_te_s)

        folds.append(FoldMetric(
            fold_id=fold_id,
            pearson_r=_pearson_mean(Y_te_s, pred),
            mse=float(np.mean((pred - Y_te_s) ** 2)),
            sign_acc=_sign_acc(Y_te_s, pred),
            n_train=len(tr_idx),
            n_test=len(te_idx),
        ))
        final_target_dim = len(top_idx)

    if not folds:
        return None
    return _aggregate(folds, final_target_dim)


def evaluate_frozen_mlp(
    X: np.ndarray,
    Y_full: np.ndarray,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    top_k: int,
) -> Optional[EvalResult]:
    """Optional frozen nonlinear probe; skipped for small-sample settings."""
    if not cv_splits:
        return None

    # Explicit small-sample conservative policy
    # Skip MLP-style probe entirely when n_pert_genes < 25.
    n_pert = Y_full.shape[0]
    if n_pert < 25:
        return None

    if min(len(tr) for tr, _ in cv_splits) < MIN_NN_TRAIN:
        return None

    folds: List[FoldMetric] = []
    final_target_dim = min(top_k, Y_full.shape[1])

    for fold_id, (tr_idx, te_idx) in enumerate(cv_splits):
        X_tr, X_te = X[tr_idx], X[te_idx]
        Y_tr_full, Y_te_full = Y_full[tr_idx], Y_full[te_idx]

        top_idx = select_top_genes_from_train(Y_tr_full, top_k)
        Y_tr, Y_te = Y_tr_full[:, top_idx], Y_te_full[:, top_idx]

        xs = StandardScaler()
        ys = StandardScaler()
        X_tr_s = xs.fit_transform(X_tr)
        X_te_s = xs.transform(X_te)
        Y_tr_s = ys.fit_transform(Y_tr)
        Y_te_s = ys.transform(Y_te)

        model = MLPRegressor(
            hidden_layer_sizes=(64,),
            activation="relu",
            alpha=1e-4,
            max_iter=500,
            early_stopping=False,
            random_state=SEED,
        )
        model.fit(X_tr_s, Y_tr_s)
        pred = model.predict(X_te_s)

        folds.append(FoldMetric(
            fold_id=fold_id,
            pearson_r=_pearson_mean(Y_te_s, pred),
            mse=float(np.mean((pred - Y_te_s) ** 2)),
            sign_acc=_sign_acc(Y_te_s, pred),
            n_train=len(tr_idx),
            n_test=len(te_idx),
        ))
        final_target_dim = len(top_idx)

    if not folds:
        return None
    return _aggregate(folds, final_target_dim)


class EmbeddingHeadRegressor(nn.Module):
    """Simple MLP head: embedding -> Linear -> ReLU -> Linear -> output."""

    def __init__(self, emb_matrix: np.ndarray, out_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        n_vocab, d = emb_matrix.shape
        self.embedding = nn.Embedding(n_vocab, d)
        self.embedding.weight.data.copy_(torch.tensor(emb_matrix, dtype=torch.float32))
        self.head = nn.Sequential(
            nn.Linear(d, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, gene_ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.embedding(gene_ids))


def evaluate_finetune_model(
    gene_ids: np.ndarray,
    Y_full: np.ndarray,
    emb_matrix: np.ndarray,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    top_k: int,
    freeze_backbone: bool,
    hidden_dim: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    enable_sign_reg: bool,
    sign_reg_weight: float,
) -> Optional[EvalResult]:
    """Neural adaptation setting with fold-local target selection and MSE default."""
    if not cv_splits:
        return None

    if min(len(tr) for tr, _ in cv_splits) < MIN_NN_TRAIN:
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds: List[FoldMetric] = []
    final_target_dim = min(top_k, Y_full.shape[1])

    for fold_id, (tr_idx, te_idx) in enumerate(cv_splits):
        gid_tr, gid_te = gene_ids[tr_idx], gene_ids[te_idx]
        Y_tr_full, Y_te_full = Y_full[tr_idx], Y_full[te_idx]

        top_idx = select_top_genes_from_train(Y_tr_full, top_k)
        Y_tr, Y_te = Y_tr_full[:, top_idx], Y_te_full[:, top_idx]

        ys = StandardScaler()
        Y_tr_s = ys.fit_transform(Y_tr).astype(np.float32)
        Y_te_s = ys.transform(Y_te).astype(np.float32)

        model = EmbeddingHeadRegressor(emb_matrix, out_dim=len(top_idx), hidden_dim=hidden_dim).to(device)
        if freeze_backbone:
            model.embedding.weight.requires_grad = False

        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=weight_decay,
        )
        mse_loss = nn.MSELoss()

        gid_tr_t = torch.tensor(gid_tr, dtype=torch.long, device=device)
        y_tr_t = torch.tensor(Y_tr_s, dtype=torch.float32, device=device)

        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            pred = model(gid_tr_t)
            loss = mse_loss(pred, y_tr_t)

            if enable_sign_reg:
                sign_target = torch.sign(y_tr_t)
                sign_pred = torch.tanh(pred)
                sign_loss = torch.mean((sign_pred - sign_target) ** 2)
                loss = loss + sign_reg_weight * sign_loss

            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            gid_te_t = torch.tensor(gid_te, dtype=torch.long, device=device)
            pred_te = model(gid_te_t).cpu().numpy()

        folds.append(FoldMetric(
            fold_id=fold_id,
            pearson_r=_pearson_mean(Y_te_s, pred_te),
            mse=float(np.mean((pred_te - Y_te_s) ** 2)),
            sign_acc=_sign_acc(Y_te_s, pred_te),
            n_train=len(tr_idx),
            n_test=len(te_idx),
        ))
        final_target_dim = len(top_idx)

    if not folds:
        return None
    return _aggregate(folds, final_target_dim)


def result_to_summary_row(
    dataset: str,
    context: str,
    embedding: str,
    method: str,
    setting_group: str,
    n_pert_genes: int,
    res: EvalResult,
) -> Dict[str, object]:
    return {
        "dataset": dataset,
        "context": context,
        "embedding": embedding,
        "method": method,
        "setting_group": setting_group,
        "n_pert_genes": n_pert_genes,
        "target_dim": res.target_dim,
        "pearson_r": res.pearson_r,
        "pearson_r_std": res.pearson_r_std,
        "mse": res.mse,
        "mse_std": res.mse_std,
        "sign_acc": res.sign_acc,
        "sign_acc_std": res.sign_acc_std,
        "n_folds": res.n_folds,
    }


def result_to_fold_rows(
    dataset: str,
    context: str,
    embedding: str,
    method: str,
    folds: List[FoldMetric],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for f in folds:
        rows.append({
            "dataset": dataset,
            "context": context,
            "embedding": embedding,
            "method": method,
            "fold_id": f.fold_id,
            "pearson_r": f.pearson_r,
            "mse": f.mse,
            "sign_acc": f.sign_acc,
            "n_train": f.n_train,
            "n_test": f.n_test,
        })
    return rows


def run_benchmark(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    log("=" * 92)
    log("Perturbation Regression Benchmark (Conservative + Leak-Free)")
    log("PRIMARY: frozen_linear (probe-based comparison)")
    log("SECONDARY: frozen_backbone_trainable_head (adaptation benchmark)")
    log("EXPLORATORY: full_finetune_embedding_head (disabled by default)")
    log("NOTE: Mixed evidence across datasets is expected; implementation is embedding-agnostic.")
    log("=" * 92)

    vocab = load_vocab()
    vocab_size = len(vocab)

    summary_rows: List[Dict[str, object]] = []
    fold_rows: List[Dict[str, object]] = []

    for ds in DATASETS:
        log(f"\n[Dataset] {ds}")
        try:
            data = load_perturb_data(ds)
        except Exception as e:
            log(f"  ! Failed to load dataset: {e}")
            continue

        dense_profiles = build_dense_profiles(data, vocab_size)
        contexts = collect_context_samples(data, dense_profiles, min_cells_per_pert=MIN_CELLS_PER_PERT)
        if not contexts:
            log("  ! No valid context found. Skip dataset.")
            continue

        for emb_name, emb_cfg in EMBEDDINGS.items():
            try:
                emb_matrix = load_checkpoint_embedding(emb_cfg["path"], emb_cfg["key"])
            except Exception as e:
                log(f"  ! Failed to load embedding {emb_name}: {e}")
                continue

            for cs in contexts:
                n_pert = len(cs.gene_ids)
                cv_splits = make_cv_splitter(n_pert)
                if not cv_splits:
                    log(f"  - {ds}/{cs.context}/{emb_name}: no valid CV split")
                    continue

                X = emb_matrix[cs.gene_ids]
                Y_full = cs.deltas_full

                # PRIMARY setting
                res_linear = evaluate_frozen_linear(X, Y_full, cv_splits, top_k=args.top_k)
                if res_linear is not None:
                    method = "frozen_linear"
                    summary_rows.append(result_to_summary_row(ds, cs.context, emb_name, method, "frozen_probe", n_pert, res_linear))
                    fold_rows.extend(result_to_fold_rows(ds, cs.context, emb_name, method, res_linear.fold_metrics))

                # Optional nonlinear frozen probe; conservative skip on small n.
                res_mlp = evaluate_frozen_mlp(X, Y_full, cv_splits, top_k=args.top_k)
                if res_mlp is not None:
                    method = "frozen_mlp"
                    summary_rows.append(result_to_summary_row(ds, cs.context, emb_name, method, "frozen_probe", n_pert, res_mlp))
                    fold_rows.extend(result_to_fold_rows(ds, cs.context, emb_name, method, res_mlp.fold_metrics))
                elif n_pert < 25:
                    log(f"  - {ds}/{cs.context}/{emb_name}: skip frozen_mlp (n_pert_genes={n_pert} < 25)")

                # SECONDARY adaptation setting
                res_fbth = evaluate_finetune_model(
                    gene_ids=cs.gene_ids,
                    Y_full=Y_full,
                    emb_matrix=emb_matrix,
                    cv_splits=cv_splits,
                    top_k=args.top_k,
                    freeze_backbone=True,
                    hidden_dim=args.hidden_dim,
                    epochs=args.finetune_epochs,
                    lr=args.finetune_lr,
                    weight_decay=args.finetune_weight_decay,
                    enable_sign_reg=args.enable_sign_reg,
                    sign_reg_weight=args.sign_reg_weight,
                )
                if res_fbth is not None:
                    method = "frozen_backbone_trainable_head"
                    summary_rows.append(result_to_summary_row(ds, cs.context, emb_name, method, "finetune", n_pert, res_fbth))
                    fold_rows.extend(result_to_fold_rows(ds, cs.context, emb_name, method, res_fbth.fold_metrics))
                else:
                    log(f"  - {ds}/{cs.context}/{emb_name}: skip frozen_backbone_trainable_head (min train < {MIN_NN_TRAIN})")

                # EXPLORATORY setting (disabled by default)
                if args.enable_full_finetune:
                    res_full = evaluate_finetune_model(
                        gene_ids=cs.gene_ids,
                        Y_full=Y_full,
                        emb_matrix=emb_matrix,
                        cv_splits=cv_splits,
                        top_k=args.top_k,
                        freeze_backbone=False,
                        hidden_dim=args.hidden_dim,
                        epochs=args.finetune_epochs,
                        lr=args.finetune_lr,
                        weight_decay=args.finetune_weight_decay,
                        enable_sign_reg=args.enable_sign_reg,
                        sign_reg_weight=args.sign_reg_weight,
                    )
                    if res_full is not None:
                        method = "full_finetune_embedding_head"
                        summary_rows.append(result_to_summary_row(ds, cs.context, emb_name, method, "exploratory", n_pert, res_full))
                        fold_rows.extend(result_to_fold_rows(ds, cs.context, emb_name, method, res_full.fold_metrics))

    summary_df = pd.DataFrame(summary_rows)
    folds_df = pd.DataFrame(fold_rows)
    return summary_df, folds_df


def main() -> None:
    args = parse_args()
    summary_df, folds_df = run_benchmark(args)

    summary_csv = os.path.join(OUTPUT_DIR, "perturbation_regression_results.csv")
    folds_csv = os.path.join(OUTPUT_DIR, "perturbation_regression_fold_results.csv")

    summary_cols = [
        "dataset", "context", "embedding", "method", "setting_group",
        "n_pert_genes", "target_dim",
        "pearson_r", "pearson_r_std", "mse", "mse_std", "sign_acc", "sign_acc_std", "n_folds",
    ]
    fold_cols = [
        "dataset", "context", "embedding", "method", "fold_id",
        "pearson_r", "mse", "sign_acc", "n_train", "n_test",
    ]

    if summary_df.empty:
        pd.DataFrame(columns=summary_cols).to_csv(summary_csv, index=False)
    else:
        summary_df[summary_cols].to_csv(summary_csv, index=False)

    if folds_df.empty:
        pd.DataFrame(columns=fold_cols).to_csv(folds_csv, index=False)
    else:
        folds_df[fold_cols].to_csv(folds_csv, index=False)

    log(f"Saved summary CSV: {summary_csv}")
    log(f"Saved fold-level CSV: {folds_csv}")
    log("Interpretation note: frozen_linear is the primary probe-based comparison.")
    log("Interpretation note: frozen_backbone_trainable_head is a secondary adaptation benchmark.")
    log("Interpretation note: full_finetune_embedding_head is exploratory and disabled by default.")


if __name__ == "__main__":
    main()
