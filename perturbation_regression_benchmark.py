#!/usr/bin/env python3
"""
Perturbation Regression Benchmark
=================================
New downstream task: formulate perturbation evaluation as regression.

For each perturbation gene g in a dataset:
  1) Build mean control expression profile (gene-level, vocab space)
  2) Build mean perturbed profile for g
  3) Target y_g = (mean_perturbed_g - mean_control) on top-K informative genes
  4) Input x_g = embedding(g)

We evaluate three settings:
  - frozen_linear: fixed gene embedding + linear ridge regressor
  - frozen_mlp: fixed gene embedding + non-linear MLP probe
  - trainable_embedding: trainable embedding + dual-stream non-linear head

Metrics:
  - Pearson r (per-sample, averaged)
  - MSE (standardized target space)
"""

import os
import json
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from datetime import datetime

warnings.filterwarnings("ignore")

BASE_DIR = '/bigdata2/hyt/projects/scbenchmark'
OUTPUT_DIR = '/bigdata2/hyt/projects/grn_benchmark'
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOG_FILE = os.path.join(OUTPUT_DIR, 'perturbation_regression.log')
VOCAB_PATH = f'{BASE_DIR}/vocab.json'
PERTURB_DATA_DIR = f'{BASE_DIR}/data/downstreams/perturbation/processed_data'
DATASETS = ['adamson', 'dixit', 'norman']

EMBEDDINGS = {
    'difference_v3': {
        'path': f'{BASE_DIR}/save_pretrain/difference_aligned_v3/best_model.pt',
        'key': 'module.embedding.weight',
    },
    'baseline': {
        'path': f'{BASE_DIR}/save_pretrain/baseline/best_model.pt',
        'key': 'module.embedding.weight',
    },
    'scGPT_human': {
        'path': f'{BASE_DIR}/save_pretrain/scGPT_human/best_model.pt',
        'key': 'encoder.embedding.weight',
    },
}


def log(msg):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')


def load_vocab():
    with open(VOCAB_PATH) as f:
        return json.load(f)


def load_checkpoint_embedding(path, key):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    # Robust key loading: handle different checkpoint layouts.
    candidate_keys = [
        key,
        'encoder.embedding.weight',
        'module.embedding.weight',
        'embedding.weight',
    ]
    for k in candidate_keys:
        if k and k in ckpt:
            return ckpt[k].detach().cpu().numpy().astype(np.float32)

    # Common nested format
    for nested_key in ['state_dict', 'model_state_dict', 'model']:
        if nested_key in ckpt and isinstance(ckpt[nested_key], dict):
            sd = ckpt[nested_key]
            for k in candidate_keys:
                if k and k in sd:
                    return sd[k].detach().cpu().numpy().astype(np.float32)

    raise KeyError(f'No embedding weight key found. Tried: {candidate_keys}')


def load_perturb_data(ds_name):
    path = os.path.join(PERTURB_DATA_DIR, f'{ds_name}_data.pt')
    d = torch.load(path, map_location='cpu', weights_only=False)
    return {
        'raw': d,
        'genes_list': d['genes'],
        'expr_list': d['expressions'],
        'base_idx': d['base_idx'],
        'single_ctrl': d['single_ctrl'],
    }


def _dense_profile(genes_arr, expr_arr, vocab_size):
    v = np.zeros(vocab_size, dtype=np.float32)
    g = np.asarray(genes_arr, dtype=np.int64)
    e = np.asarray(expr_arr, dtype=np.float32)
    valid = (g >= 0) & (g < vocab_size)
    if valid.sum() == 0:
        return v
    g, e = g[valid], e[valid]
    np.add.at(v, g, e)
    return v


def _build_gene_level_samples_for_indices(
    data, vocab_size, ctrl_indices, pert_indices, min_cells=5, top_k=256
):
    genes_list = data['genes_list']
    expr_list = data['expr_list']
    single_ctrl = data['single_ctrl']

    if len(ctrl_indices) < 20 or len(pert_indices) < 50:
        return None

    ctrl_profiles = [_dense_profile(genes_list[i], expr_list[i], vocab_size) for i in ctrl_indices]
    mean_ctrl = np.mean(np.stack(ctrl_profiles, axis=0), axis=0)

    pert_groups = defaultdict(list)
    for i in pert_indices:
        pert_groups[int(single_ctrl[i])].append(i)

    valid_genes = [g for g, idxs in pert_groups.items() if len(idxs) >= min_cells]
    if len(valid_genes) < 10:
        return None

    all_deltas = []
    all_gene_ids = []
    for g in valid_genes:
        p_profiles = [_dense_profile(genes_list[i], expr_list[i], vocab_size) for i in pert_groups[g]]
        mean_pert = np.mean(np.stack(p_profiles, axis=0), axis=0)
        all_deltas.append(mean_pert - mean_ctrl)
        all_gene_ids.append(g)

    all_deltas = np.stack(all_deltas, axis=0)  # (n_pert_genes, vocab_size)

    gene_score = np.mean(np.abs(all_deltas), axis=0)
    top_idx = np.argsort(-gene_score)[:top_k]

    Y = all_deltas[:, top_idx]
    G = np.array(all_gene_ids, dtype=np.int64)
    return {'gene_ids': G, 'targets': Y, 'top_gene_idx': top_idx, 'n_ctrl': len(ctrl_indices)}


def _extract_cell_types(raw_dict, n_cells):
    """Best-effort extraction of per-cell cell-type labels from the .pt payload."""
    candidate_keys = [
        'cell_type', 'cell_types', 'celltype', 'celltypes',
        'cls_name', 'cls', 'batch_name', 'batch',
    ]
    for k in candidate_keys:
        if k not in raw_dict:
            continue
        v = raw_dict[k]
        if isinstance(v, (list, tuple, np.ndarray)) and len(v) == n_cells:
            return np.array(v)
    return None


def build_gene_level_samples(data, vocab_size, min_cells=5, top_k=256):
    base_idx = data['base_idx']
    single_ctrl = data['single_ctrl']

    ctrl_indices_all = [i for i, b in enumerate(base_idx) if b == 1]
    pert_indices_all = [i for i, b in enumerate(base_idx) if b == 0 and single_ctrl[i] >= 0]

    # 1) Try cell-type-specific slices if cell-type labels are available.
    cell_types = _extract_cell_types(data['raw'], len(base_idx))
    out = []
    if cell_types is not None:
        for ct in sorted(set(cell_types.tolist())):
            ct_idx = set(np.where(cell_types == ct)[0].tolist())
            ctrl_ct = [i for i in ctrl_indices_all if i in ct_idx]
            pert_ct = [i for i in pert_indices_all if i in ct_idx]
            s = _build_gene_level_samples_for_indices(
                data, vocab_size=vocab_size, ctrl_indices=ctrl_ct, pert_indices=pert_ct,
                min_cells=min_cells, top_k=top_k
            )
            if s is not None:
                s['context'] = f'celltype::{ct}'
                out.append(s)

    # 2) Fallback to dataset-level if no cell-type slice is valid.
    if not out:
        s = _build_gene_level_samples_for_indices(
            data, vocab_size=vocab_size, ctrl_indices=ctrl_indices_all, pert_indices=pert_indices_all,
            min_cells=min_cells, top_k=top_k
        )
        if s is not None:
            s['context'] = 'dataset::all'
            out.append(s)

    return out


def _pearson_mean(y_true, y_pred):
    rs = []
    for i in range(len(y_true)):
        if np.std(y_true[i]) > 1e-8 and np.std(y_pred[i]) > 1e-8:
            rs.append(pearsonr(y_true[i], y_pred[i])[0])
    return float(np.mean(rs)) if rs else 0.0


def run_frozen_linear(gene_ids, targets, emb_matrix):
    X = emb_matrix[gene_ids]
    kf = KFold(n_splits=min(5, len(gene_ids)), shuffle=True, random_state=42)

    rs, mses = [], []
    for tr, te in kf.split(X):
        sx = StandardScaler()
        sy = StandardScaler()
        X_tr = sx.fit_transform(X[tr])
        X_te = sx.transform(X[te])
        Y_tr = sy.fit_transform(targets[tr])
        Y_te = sy.transform(targets[te])

        reg = Ridge(alpha=1.0)
        reg.fit(X_tr, Y_tr)
        Y_pred = reg.predict(X_te)

        rs.append(_pearson_mean(Y_te, Y_pred))
        mses.append(float(np.mean((Y_te - Y_pred) ** 2)))

    return {
        'pearson_r': float(np.mean(rs)),
        'pearson_r_std': float(np.std(rs)),
        'mse': float(np.mean(mses)),
        'mse_std': float(np.std(mses)),
    }


def run_frozen_mlp(gene_ids, targets, emb_matrix):
    X = emb_matrix[gene_ids]
    kf = KFold(n_splits=min(5, len(gene_ids)), shuffle=True, random_state=42)

    rs, mses = [], []
    for tr, te in kf.split(X):
        sx = StandardScaler()
        sy = StandardScaler()
        X_tr = sx.fit_transform(X[tr])
        X_te = sx.transform(X[te])
        Y_tr = sy.fit_transform(targets[tr])
        Y_te = sy.transform(targets[te])

        reg = MLPRegressor(
            hidden_layer_sizes=(256, 128),
            activation='relu',
            solver='adam',
            alpha=1e-4,
            batch_size='auto',
            learning_rate_init=1e-3,
            max_iter=500,
            early_stopping=True,
            random_state=42,
        )
        reg.fit(X_tr, Y_tr)
        Y_pred = reg.predict(X_te)

        rs.append(_pearson_mean(Y_te, Y_pred))
        mses.append(float(np.mean((Y_te - Y_pred) ** 2)))

    return {
        'pearson_r': float(np.mean(rs)),
        'pearson_r_std': float(np.std(rs)),
        'mse': float(np.mean(mses)),
        'mse_std': float(np.std(mses)),
    }


class DualStreamEmbeddingRegressor(nn.Module):
    def __init__(self, init_emb, out_dim):
        super().__init__()
        num_genes, emb_dim = init_emb.shape
        self.embedding = nn.Embedding(num_genes, emb_dim)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.from_numpy(init_emb))
        # Split representation into similarity and directional components.
        self.sim_proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim // 2),
        )
        self.dir_proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Tanh(),
            nn.Linear(emb_dim, emb_dim // 2),
        )
        self.head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, out_dim),
        )

    def forward(self, gene_ids):
        e = self.embedding(gene_ids)
        sim = self.sim_proj(e)
        direction = self.dir_proj(e)
        z = torch.cat([sim, direction], dim=-1)
        return self.head(z)


def run_trainable_embedding(gene_ids, targets, emb_matrix, epochs=300, lr=1e-3, wd=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kf = KFold(n_splits=min(5, len(gene_ids)), shuffle=True, random_state=42)

    rs, mses = [], []
    for tr, te in kf.split(gene_ids):
        sy = StandardScaler()
        Y_tr = sy.fit_transform(targets[tr]).astype(np.float32)
        Y_te = sy.transform(targets[te]).astype(np.float32)

        model = DualStreamEmbeddingRegressor(emb_matrix, out_dim=targets.shape[1]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        loss_mse = nn.MSELoss()
        loss_bce = nn.BCEWithLogitsLoss()

        tr_gene = torch.from_numpy(gene_ids[tr]).long().to(device)
        tr_y = torch.from_numpy(Y_tr).float().to(device)

        model.train()
        for _ in range(epochs):
            pred = model(tr_gene)
            # Direction-aware asymmetric objective:
            # 1) regression on signed delta
            # 2) sign consistency auxiliary loss
            sign_target = (tr_y > 0).float()
            mse = loss_mse(pred, tr_y)
            sign_loss = loss_bce(pred, sign_target)
            # asymmetric penalty for wrong-sign large-magnitude targets
            wrong_sign = torch.relu(-(pred * tr_y))
            asym = wrong_sign.mean()
            loss = mse + 0.2 * sign_loss + 0.1 * asym
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            te_gene = torch.from_numpy(gene_ids[te]).long().to(device)
            pred = model(te_gene).cpu().numpy()

        rs.append(_pearson_mean(Y_te, pred))
        mses.append(float(np.mean((Y_te - pred) ** 2)))

    return {
        'pearson_r': float(np.mean(rs)),
        'pearson_r_std': float(np.std(rs)),
        'mse': float(np.mean(mses)),
        'mse_std': float(np.std(mses)),
    }


def main():
    log('=' * 70)
    log('Perturbation Regression Benchmark (new task)')
    log('=' * 70)

    vocab = load_vocab()
    vocab_size = len(vocab)
    log(f'Vocab size: {vocab_size}')

    loaded_embs = {}
    for name, cfg in EMBEDDINGS.items():
        try:
            m = load_checkpoint_embedding(cfg['path'], cfg['key'])
            loaded_embs[name] = m
            log(f'Loaded {name}: {m.shape}')
        except Exception as e:
            log(f'Failed {name}: {e}')

    all_rows = []

    for ds in DATASETS:
        pt_path = os.path.join(PERTURB_DATA_DIR, f'{ds}_data.pt')
        if not os.path.exists(pt_path):
            log(f'{ds}: not found, skip')
            continue

        log('\n' + '-' * 70)
        log(f'Dataset: {ds}')
        data = load_perturb_data(ds)

        samples = build_gene_level_samples(data, vocab_size=vocab_size, min_cells=5, top_k=256)
        if not samples:
            log('Insufficient data after filtering (including fallback), skip')
            continue

        for s in samples:
            gene_ids = s['gene_ids']
            targets = s['targets']
            ctx = s['context']
            log(f"[{ctx}] Perturbation genes used: {len(gene_ids)}, target dim: {targets.shape[1]}, ctrl cells: {s['n_ctrl']}")

            for emb_name, emb_matrix in loaded_embs.items():
                if emb_matrix.shape[0] <= np.max(gene_ids):
                    log(f'  {emb_name}: embedding vocab too small, skip')
                    continue

                log(f'  {emb_name} | frozen_linear ...')
                res_frozen = run_frozen_linear(gene_ids, targets, emb_matrix)
                log(f"    r={res_frozen['pearson_r']:.4f}±{res_frozen['pearson_r_std']:.4f}, "
                    f"mse={res_frozen['mse']:.4f}±{res_frozen['mse_std']:.4f}")
                all_rows.append({
                    'dataset': ds,
                    'context': ctx,
                    'embedding': emb_name,
                    'method': 'frozen_linear',
                    'n_pert_genes': len(gene_ids),
                    'target_dim': targets.shape[1],
                    **res_frozen,
                })

                log(f'  {emb_name} | frozen_mlp ...')
                res_mlp = run_frozen_mlp(gene_ids, targets, emb_matrix)
                log(f"    r={res_mlp['pearson_r']:.4f}±{res_mlp['pearson_r_std']:.4f}, "
                    f"mse={res_mlp['mse']:.4f}±{res_mlp['mse_std']:.4f}")
                all_rows.append({
                    'dataset': ds,
                    'context': ctx,
                    'embedding': emb_name,
                    'method': 'frozen_mlp',
                    'n_pert_genes': len(gene_ids),
                    'target_dim': targets.shape[1],
                    **res_mlp,
                })

                log(f'  {emb_name} | trainable_embedding ...')
                res_trainable = run_trainable_embedding(gene_ids, targets, emb_matrix)
                log(f"    r={res_trainable['pearson_r']:.4f}±{res_trainable['pearson_r_std']:.4f}, "
                    f"mse={res_trainable['mse']:.4f}±{res_trainable['mse_std']:.4f}")
                all_rows.append({
                    'dataset': ds,
                    'context': ctx,
                    'embedding': emb_name,
                    'method': 'trainable_embedding',
                    'n_pert_genes': len(gene_ids),
                    'target_dim': targets.shape[1],
                    **res_trainable,
                })

    if all_rows:
        df = pd.DataFrame(all_rows)
        out_csv = os.path.join(OUTPUT_DIR, 'perturbation_regression_results.csv')
        df.to_csv(out_csv, index=False)
        log(f'\nSaved: {out_csv}')
    else:
        log('No results produced.')

    log('Done.')


if __name__ == '__main__':
    main()
