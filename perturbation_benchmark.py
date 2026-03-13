#!/usr/bin/env python3
"""
Perturbation Benchmark - Embedding Only
=========================================
Evaluate gene embeddings on perturbation prediction tasks.

Task A: Perturbation Classification
  - Given a perturbed cell's expression, classify which gene was perturbed
  - Cell representation: weighted average of gene embeddings
  - Metric: accuracy, macro F1

Task B: Perturbation Effect Similarity
  - Does embedding similarity predict similarity of perturbation effects?
  - For each pair of perturbed genes, compare:
    cosine_sim(emb_g1, emb_g2) vs correlation(delta_g1, delta_g2)
  - Metric: Spearman correlation

Task C: Perturbation Direction Prediction
  - Use perturbed gene embedding to predict expression change direction
  - Linear regression: emb_perturb -> delta (top 20 DEGs)
  - Metric: Pearson R, MSE
"""

import os, sys, json, warnings
import numpy as np
import torch
import pandas as pd
from datetime import datetime
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import cosine
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore")

# =============================================================
# Config
# =============================================================
BASE_DIR = '/root/autodl-tmp/scbenchmark'
OUTPUT_DIR = '/root/autodl-tmp/perturbation_benchmark'
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOG_FILE = os.path.join(OUTPUT_DIR, 'perturbation.log')
VOCAB_PATH = f'{BASE_DIR}/vocab.json'

PERTURB_DATA_DIR = f'{BASE_DIR}/data/downstreams/perturbation/processed_data'
DATASETS = ['adamson', 'dixit', 'norman']

EMBEDDINGS = {
    'difference_v3': {
        'path': f'{BASE_DIR}/save_pretrain/difference_aligned_v3/best_model.pt',
        'key': 'module.embedding.weight',
        'type': 'checkpoint',
    },
    'baseline': {
        'path': f'{BASE_DIR}/save_pretrain/baseline/best_model.pt',
        'key': 'module.embedding.weight',
        'type': 'checkpoint',
    },
    'scGPT_human': {
        'path': '/root/autodl-tmp/scGPT_human/best_model.pt',
        'key': 'encoder.embedding.weight',
        'type': 'checkpoint',
    },
    'GF-12L95M': {
        'dir': '/root/autodl-tmp/gene_embeddings/intersect/GF-12L95M',
        'type': 'geneformer',
    },
}


def log(msg):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')


# =============================================================
# Loading
# =============================================================
def load_vocab():
    with open(VOCAB_PATH) as f:
        return json.load(f)


def load_checkpoint_embedding(path, key):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    return ckpt[key].detach().numpy()


def load_gf_embedding(emb_dir, name='GF-12L95M'):
    emb_path = os.path.join(emb_dir, f'{name}_emb.csv')
    gl_path = os.path.join(emb_dir, f'{name}_genelist.txt')
    emb = pd.read_csv(emb_path, header=None).values.astype(np.float32)
    with open(gl_path) as f:
        genelist = [line.strip() for line in f]
    return emb, genelist


def build_symbol_to_entrez():
    mapping_file = os.path.join(OUTPUT_DIR, 'gene_symbol_to_entrez.json')
    if os.path.exists(mapping_file):
        with open(mapping_file) as f:
            return json.load(f)
    alt_path = '/root/autodl-tmp/embedding_benchmark/gene_symbol_to_entrez.json'
    if os.path.exists(alt_path):
        import shutil
        shutil.copy2(alt_path, mapping_file)
        with open(mapping_file) as f:
            return json.load(f)
    alt2 = '/root/autodl-tmp/grn_benchmark/gene_symbol_to_entrez.json'
    if os.path.exists(alt2):
        import shutil
        shutil.copy2(alt2, mapping_file)
        with open(mapping_file) as f:
            return json.load(f)
    return {}


def load_perturb_data(ds_name):
    """Load perturbation dataset. Returns dict with parsed fields."""
    path = os.path.join(PERTURB_DATA_DIR, f'{ds_name}_data.pt')
    d = torch.load(path, map_location='cpu', weights_only=False)

    genes_list = d['genes']       # list of arrays (gene token IDs per cell)
    expr_list = d['expressions']  # list of arrays (expression values per cell)
    base_idx = d['base_idx']      # list of ints (1=control, 0=perturbed)
    single_ctrl = d['single_ctrl']  # list of ints (-1=control, else=perturbed gene vocab idx)

    n_cells = len(genes_list)

    # Separate control and perturbed cells
    ctrl_indices = [i for i in range(n_cells) if base_idx[i] == 1]
    pert_indices = [i for i in range(n_cells) if base_idx[i] == 0]

    # Get unique perturbation genes
    pert_gene_ids = set()
    for i in pert_indices:
        pg = single_ctrl[i]
        if pg >= 0:
            pert_gene_ids.add(pg)

    return {
        'genes_list': genes_list,
        'expr_list': expr_list,
        'base_idx': base_idx,
        'single_ctrl': single_ctrl,
        'ctrl_indices': ctrl_indices,
        'pert_indices': pert_indices,
        'pert_gene_ids': sorted(pert_gene_ids),
        'n_cells': n_cells,
    }


# =============================================================
# Build cell representations using gene embeddings
# =============================================================
def build_cell_repr(genes_arr, expr_arr, emb_matrix):
    """Build cell representation: weighted average of gene embeddings."""
    emb_dim = emb_matrix.shape[1]
    # genes_arr is array of vocab indices, expr_arr is expression values
    genes = np.array(genes_arr)
    expr = np.array(expr_arr)

    # Filter to valid indices
    valid = (genes >= 0) & (genes < emb_matrix.shape[0])
    genes = genes[valid]
    expr = expr[valid]

    if len(genes) == 0:
        return np.zeros(emb_dim, dtype=np.float32)

    # Weighted average
    weights = np.abs(expr)
    w_sum = weights.sum()
    if w_sum < 1e-8:
        return np.zeros(emb_dim, dtype=np.float32)

    embs = emb_matrix[genes]  # (n_genes, emb_dim)
    cell_repr = (embs * weights[:, None]).sum(axis=0) / w_sum
    return cell_repr.astype(np.float32)


def build_cell_repr_gf(genes_arr, expr_arr, vocab, gf_emb, gf_genelist, s2e):
    """Build cell representation using Geneformer embeddings."""
    emb_dim = gf_emb.shape[1]
    idx2gene = {v: k for k, v in vocab.items()}
    e2gf = {eid: i for i, eid in enumerate(gf_genelist)}

    genes = np.array(genes_arr)
    expr = np.array(expr_arr)

    weighted_sum = np.zeros(emb_dim, dtype=np.float32)
    w_sum = 0.0

    for g_idx, e_val in zip(genes, expr):
        if g_idx < 0 or g_idx >= len(idx2gene):
            continue
        gene_name = idx2gene.get(int(g_idx))
        if gene_name and gene_name in s2e:
            eid = s2e[gene_name]
            if eid in e2gf:
                w = abs(float(e_val))
                weighted_sum += gf_emb[e2gf[eid]] * w
                w_sum += w

    if w_sum < 1e-8:
        return np.zeros(emb_dim, dtype=np.float32)
    return (weighted_sum / w_sum).astype(np.float32)


# =============================================================
# Task A: Perturbation Classification
# =============================================================
def run_perturbation_classification(data, emb_matrix, emb_name, vocab=None,
                                     gf_emb=None, gf_genelist=None, s2e=None):
    """Classify which gene was perturbed from cell expression profile."""
    pert_indices = data['pert_indices']
    single_ctrl = data['single_ctrl']
    genes_list = data['genes_list']
    expr_list = data['expr_list']

    # Build labels
    labels = []
    for i in pert_indices:
        labels.append(single_ctrl[i])
    labels = np.array(labels)

    # Filter to perturbations with enough cells (>=5)
    from collections import Counter
    counts = Counter(labels)
    valid_labels = {k for k, v in counts.items() if v >= 5}
    mask = np.array([l in valid_labels for l in labels])

    if mask.sum() < 50:
        return None

    pert_idx_filtered = [pert_indices[i] for i in range(len(pert_indices)) if mask[i]]
    labels_filtered = labels[mask]

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(labels_filtered)
    n_classes = len(le.classes_)

    # Build cell representations
    X = []
    is_gf = (emb_matrix is None)
    for idx in pert_idx_filtered:
        if is_gf:
            cell = build_cell_repr_gf(genes_list[idx], expr_list[idx],
                                       vocab, gf_emb, gf_genelist, s2e)
        else:
            cell = build_cell_repr(genes_list[idx], expr_list[idx], emb_matrix)
        X.append(cell)
    X = np.array(X)

    # 5-fold CV with LR
    results = {}
    for clf_name, clf_fn in [('lr', lambda: LogisticRegression(max_iter=1000, n_jobs=1, C=1.0)),
                              ('mlp', lambda: MLPClassifier(hidden_layer_sizes=(256,128),
                                                            max_iter=300, early_stopping=True,
                                                            random_state=42))]:
        try:
            n_splits = min(5, min(Counter(y).values()))
            if n_splits < 2:
                continue
            skf = StratifiedKFold(n_splits=min(5, n_splits), shuffle=True, random_state=42)
            accs, f1s = [], []
            for train_idx, test_idx in skf.split(X, y):
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X[train_idx])
                X_te = scaler.transform(X[test_idx])
                clf = clf_fn()
                clf.fit(X_tr, y[train_idx])
                y_pred = clf.predict(X_te)
                accs.append(accuracy_score(y[test_idx], y_pred))
                f1s.append(f1_score(y[test_idx], y_pred, average='macro'))
            results[clf_name] = {
                'accuracy': np.mean(accs),
                'acc_std': np.std(accs),
                'f1_macro': np.mean(f1s),
                'f1_std': np.std(f1s),
            }
        except Exception as e:
            log(f"    {clf_name} error: {e}")

    return {
        'n_cells': len(pert_idx_filtered),
        'n_classes': n_classes,
        'results': results,
    }


# =============================================================
# Helper: build cell repr for a subset of cells
# =============================================================
def _build_reprs(indices, genes_list, expr_list, emb_matrix, vocab=None,
                 gf_emb=None, gf_genelist=None, s2e=None, max_cells=500):
    """Build cell representations for a subset of cell indices."""
    is_gf = (emb_matrix is None)
    reprs = []
    for idx in indices[:max_cells]:
        if is_gf:
            r = build_cell_repr_gf(genes_list[idx], expr_list[idx],
                                    vocab, gf_emb, gf_genelist, s2e)
        else:
            r = build_cell_repr(genes_list[idx], expr_list[idx], emb_matrix)
        reprs.append(r)
    return np.array(reprs)


# =============================================================
# Task B: Perturbation Effect Similarity (in embedding space)
# =============================================================
def run_perturbation_similarity(data, emb_matrix, vocab, emb_name,
                                 gf_emb=None, gf_genelist=None, s2e=None):
    """Test if embedding similarity predicts perturbation effect similarity.
    Uses embedding-space cell representations (handles variable-length genes).
    """
    pert_indices = data['pert_indices']
    ctrl_indices = data['ctrl_indices']
    single_ctrl = data['single_ctrl']
    genes_list = data['genes_list']
    expr_list = data['expr_list']

    # Build mean control cell repr in embedding space
    ctrl_reprs = _build_reprs(ctrl_indices, genes_list, expr_list, emb_matrix,
                               vocab=vocab, gf_emb=gf_emb, gf_genelist=gf_genelist,
                               s2e=s2e, max_cells=500)
    mean_ctrl_repr = ctrl_reprs.mean(axis=0)

    # Group by perturbation gene
    from collections import defaultdict
    pert_groups = defaultdict(list)
    for i in pert_indices:
        pg = single_ctrl[i]
        if pg >= 0:
            pert_groups[pg].append(i)

    valid_pert_genes = [pg for pg, idxs in pert_groups.items() if len(idxs) >= 5]
    if len(valid_pert_genes) < 10:
        return None

    # Compute delta in embedding space per pert gene
    deltas = {}
    for pg in valid_pert_genes:
        pert_reprs = _build_reprs(pert_groups[pg], genes_list, expr_list, emb_matrix,
                                   vocab=vocab, gf_emb=gf_emb, gf_genelist=gf_genelist,
                                   s2e=s2e, max_cells=100)
        deltas[pg] = pert_reprs.mean(axis=0) - mean_ctrl_repr

    # Get perturbation gene embeddings
    is_gf = (emb_matrix is None)
    idx2gene = {v: k for k, v in vocab.items()} if vocab else {}
    e2gf = {eid: i for i, eid in enumerate(gf_genelist)} if gf_genelist else {}

    def get_emb(gene_vocab_idx):
        if is_gf:
            gene_name = idx2gene.get(int(gene_vocab_idx))
            if gene_name and gene_name in s2e:
                eid = s2e[gene_name]
                if eid in e2gf:
                    return gf_emb[e2gf[eid]]
            return None
        else:
            if 0 <= gene_vocab_idx < emb_matrix.shape[0]:
                return emb_matrix[gene_vocab_idx]
            return None

    # Compute pairwise similarities
    emb_sims = []
    delta_corrs = []
    genes_with_emb = [pg for pg in valid_pert_genes if get_emb(pg) is not None]

    if len(genes_with_emb) < 5:
        return None

    for i in range(len(genes_with_emb)):
        for j in range(i+1, len(genes_with_emb)):
            g1, g2 = genes_with_emb[i], genes_with_emb[j]
            e1, e2 = get_emb(g1), get_emb(g2)

            # Embedding cosine similarity
            n1 = np.linalg.norm(e1)
            n2 = np.linalg.norm(e2)
            if n1 < 1e-8 or n2 < 1e-8:
                continue
            emb_sim = np.dot(e1, e2) / (n1 * n2)

            # Perturbation effect similarity (in embedding space)
            d1, d2 = deltas[g1], deltas[g2]
            if np.std(d1) < 1e-8 or np.std(d2) < 1e-8:
                continue
            delta_corr = pearsonr(d1, d2)[0]

            emb_sims.append(emb_sim)
            delta_corrs.append(delta_corr)

    if len(emb_sims) < 10:
        return None

    emb_sims = np.array(emb_sims)
    delta_corrs = np.array(delta_corrs)

    spearman_r, spearman_p = spearmanr(emb_sims, delta_corrs)
    pearson_r, pearson_p = pearsonr(emb_sims, delta_corrs)

    return {
        'n_pairs': len(emb_sims),
        'n_genes': len(genes_with_emb),
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
    }


# =============================================================
# Task C: Perturbation Direction Prediction (in embedding space)
# =============================================================
def run_perturbation_direction(data, emb_matrix, vocab, emb_name,
                                gf_emb=None, gf_genelist=None, s2e=None):
    """Predict perturbation effect in embedding space from gene embedding."""
    pert_indices = data['pert_indices']
    ctrl_indices = data['ctrl_indices']
    single_ctrl = data['single_ctrl']
    genes_list = data['genes_list']
    expr_list = data['expr_list']

    # Build mean control cell repr
    ctrl_reprs = _build_reprs(ctrl_indices, genes_list, expr_list, emb_matrix,
                               vocab=vocab, gf_emb=gf_emb, gf_genelist=gf_genelist,
                               s2e=s2e, max_cells=500)
    mean_ctrl_repr = ctrl_reprs.mean(axis=0)

    # Group by perturbation gene
    from collections import defaultdict
    pert_groups = defaultdict(list)
    for i in pert_indices:
        pg = single_ctrl[i]
        if pg >= 0:
            pert_groups[pg].append(i)

    valid_pert_genes = [pg for pg, idxs in pert_groups.items() if len(idxs) >= 5]
    if len(valid_pert_genes) < 10:
        return None

    # Get embedding and delta for each perturbation gene
    is_gf = (emb_matrix is None)
    idx2gene = {v: k for k, v in vocab.items()} if vocab else {}
    e2gf = {eid: i for i, eid in enumerate(gf_genelist)} if gf_genelist else {}

    def get_emb(gene_vocab_idx):
        if is_gf:
            gene_name = idx2gene.get(int(gene_vocab_idx))
            if gene_name and gene_name in s2e:
                eid = s2e[gene_name]
                if eid in e2gf:
                    return gf_emb[e2gf[eid]]
            return None
        else:
            if 0 <= gene_vocab_idx < emb_matrix.shape[0]:
                return emb_matrix[gene_vocab_idx]
            return None

    all_embs = []
    all_deltas = []
    valid_genes = []
    for pg in valid_pert_genes:
        emb = get_emb(pg)
        if emb is None:
            continue
        pert_reprs = _build_reprs(pert_groups[pg], genes_list, expr_list, emb_matrix,
                                   vocab=vocab, gf_emb=gf_emb, gf_genelist=gf_genelist,
                                   s2e=s2e, max_cells=100)
        delta = pert_reprs.mean(axis=0) - mean_ctrl_repr
        all_embs.append(emb)
        all_deltas.append(delta)
        valid_genes.append(pg)

    if len(valid_genes) < 10:
        return None

    all_embs = np.array(all_embs)      # (n_pert, emb_dim)
    all_deltas = np.array(all_deltas)  # (n_pert, emb_dim)

    # Ridge regression: emb -> delta (5-fold CV)
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=min(5, len(valid_genes)), shuffle=True, random_state=42)

    pearson_rs = []
    mses = []
    for train_idx, test_idx in kf.split(all_embs):
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        X_tr = scaler_x.fit_transform(all_embs[train_idx])
        X_te = scaler_x.transform(all_embs[test_idx])
        Y_tr = scaler_y.fit_transform(all_deltas[train_idx])
        Y_te = scaler_y.transform(all_deltas[test_idx])

        reg = Ridge(alpha=1.0)
        reg.fit(X_tr, Y_tr)
        Y_pred = reg.predict(X_te)

        # Per-sample Pearson R
        for i in range(len(Y_te)):
            if np.std(Y_te[i]) > 1e-8 and np.std(Y_pred[i]) > 1e-8:
                r = pearsonr(Y_te[i], Y_pred[i])[0]
                pearson_rs.append(r)
        mse = np.mean((Y_pred - Y_te) ** 2)
        mses.append(mse)

    return {
        'n_pert_genes': len(valid_genes),
        'pearson_r': np.mean(pearson_rs) if pearson_rs else 0,
        'pearson_r_std': np.std(pearson_rs) if pearson_rs else 0,
        'mse': np.mean(mses),
    }


# =============================================================
# Main
# =============================================================
def main():
    log("=" * 70)
    log("Perturbation Benchmark - Embedding Only")
    log("=" * 70)

    # Load vocab
    vocab = load_vocab()
    log(f"Vocab: {len(vocab)} genes")

    # Load embeddings
    loaded_embs = {}
    for name, cfg in EMBEDDINGS.items():
        if cfg['type'] == 'checkpoint':
            mat = load_checkpoint_embedding(cfg['path'], cfg['key'])
            loaded_embs[name] = {'matrix': mat, 'type': 'checkpoint'}
            log(f"Loaded {name}: {mat.shape}")
        else:
            emb, gl = load_gf_embedding(cfg['dir'])
            loaded_embs[name] = {'matrix': emb, 'genelist': gl, 'type': 'geneformer'}
            log(f"Loaded {name}: {emb.shape}")

    s2e = build_symbol_to_entrez()
    all_results = []

    for ds_name in DATASETS:
        pt_path = os.path.join(PERTURB_DATA_DIR, f'{ds_name}_data.pt')
        if not os.path.exists(pt_path):
            log(f"\n{ds_name}: NOT FOUND, skipping")
            continue

        log(f"\n{'='*70}")
        log(f"Dataset: {ds_name}")
        log(f"{'='*70}")

        data = load_perturb_data(ds_name)
        log(f"  Cells: {data['n_cells']} (control={len(data['ctrl_indices'])}, "
            f"perturbed={len(data['pert_indices'])})")
        log(f"  Unique perturbation genes: {len(data['pert_gene_ids'])}")

        for emb_name, emb_data in loaded_embs.items():
            log(f"\n  --- {emb_name} ---")

            if emb_data['type'] == 'checkpoint':
                emb_matrix = emb_data['matrix']
                gf_args = dict(gf_emb=None, gf_genelist=None, s2e=None)
            else:
                emb_matrix = None
                gf_args = dict(gf_emb=emb_data['matrix'],
                              gf_genelist=emb_data['genelist'], s2e=s2e)

            # Task A: Classification
            log(f"    Task A: Perturbation Classification...")
            res_cls = run_perturbation_classification(
                data, emb_matrix, emb_name, vocab=vocab,
                **gf_args)
            if res_cls:
                for clf_name, metrics in res_cls['results'].items():
                    log(f"      {clf_name}: acc={metrics['accuracy']:.4f}±{metrics['acc_std']:.4f}, "
                        f"f1={metrics['f1_macro']:.4f}±{metrics['f1_std']:.4f} "
                        f"({res_cls['n_cells']} cells, {res_cls['n_classes']} classes)")
                    all_results.append({
                        'dataset': ds_name, 'task': 'classification',
                        'embedding': emb_name, 'clf': clf_name,
                        'accuracy': metrics['accuracy'], 'acc_std': metrics['acc_std'],
                        'f1_macro': metrics['f1_macro'], 'f1_std': metrics['f1_std'],
                        'n_cells': res_cls['n_cells'], 'n_classes': res_cls['n_classes'],
                    })
            else:
                log(f"      Skipped (not enough data)")

            # Task B: Effect Similarity
            log(f"    Task B: Perturbation Effect Similarity...")
            res_sim = run_perturbation_similarity(
                data, emb_matrix, vocab, emb_name, **gf_args)
            if res_sim:
                log(f"      Spearman r={res_sim['spearman_r']:.4f} (p={res_sim['spearman_p']:.2e}), "
                    f"Pearson r={res_sim['pearson_r']:.4f} "
                    f"({res_sim['n_genes']} genes, {res_sim['n_pairs']} pairs)")
                all_results.append({
                    'dataset': ds_name, 'task': 'similarity',
                    'embedding': emb_name,
                    'spearman_r': res_sim['spearman_r'],
                    'pearson_r': res_sim['pearson_r'],
                    'n_genes': res_sim['n_genes'],
                    'n_pairs': res_sim['n_pairs'],
                })
            else:
                log(f"      Skipped (not enough data)")

            # Task C: Direction Prediction
            log(f"    Task C: Perturbation Direction Prediction...")
            res_dir = run_perturbation_direction(
                data, emb_matrix, vocab, emb_name, **gf_args)
            if res_dir:
                log(f"      Pearson r={res_dir['pearson_r']:.4f}±{res_dir['pearson_r_std']:.4f}, "
                    f"MSE={res_dir['mse']:.4f} "
                    f"({res_dir['n_pert_genes']} pert genes)")
                all_results.append({
                    'dataset': ds_name, 'task': 'direction',
                    'embedding': emb_name,
                    'pearson_r': res_dir['pearson_r'],
                    'pearson_r_std': res_dir['pearson_r_std'],
                    'mse': res_dir['mse'],
                    'n_pert_genes': res_dir['n_pert_genes'],
                })
            else:
                log(f"      Skipped (not enough data)")

    # Summary
    log(f"\n{'='*70}")
    log("SUMMARY")
    log(f"{'='*70}")

    if all_results:
        df = pd.DataFrame(all_results)

        # Task A summary
        cls_df = df[df['task'] == 'classification']
        if len(cls_df) > 0:
            log("\n--- Task A: Perturbation Classification (LR) ---")
            lr_df = cls_df[cls_df['clf'] == 'lr']
            emb_names = list(EMBEDDINGS.keys())
            log(f"{'Dataset':<15} " + " ".join(f"{n:<25}" for n in emb_names))
            log("-" * (15 + 26 * len(emb_names)))
            for ds in DATASETS:
                row = f"{ds:<15} "
                for emb in emb_names:
                    m = lr_df[(lr_df['dataset'] == ds) & (lr_df['embedding'] == emb)]
                    if len(m) > 0:
                        row += f"acc={m.iloc[0]['accuracy']:.4f} f1={m.iloc[0]['f1_macro']:.4f}  "
                    else:
                        row += f"{'N/A':<25} "
                log(row)

        # Task B summary
        sim_df = df[df['task'] == 'similarity']
        if len(sim_df) > 0:
            log("\n--- Task B: Perturbation Effect Similarity ---")
            log(f"{'Dataset':<15} " + " ".join(f"{n:<20}" for n in emb_names))
            log("-" * (15 + 21 * len(emb_names)))
            for ds in DATASETS:
                row = f"{ds:<15} "
                for emb in emb_names:
                    m = sim_df[(sim_df['dataset'] == ds) & (sim_df['embedding'] == emb)]
                    if len(m) > 0:
                        row += f"ρ={m.iloc[0]['spearman_r']:.4f}           "
                    else:
                        row += f"{'N/A':<20} "
                log(row)

        # Task C summary
        dir_df = df[df['task'] == 'direction']
        if len(dir_df) > 0:
            log("\n--- Task C: Perturbation Direction Prediction ---")
            log(f"{'Dataset':<15} " + " ".join(f"{n:<20}" for n in emb_names))
            log("-" * (15 + 21 * len(emb_names)))
            for ds in DATASETS:
                row = f"{ds:<15} "
                for emb in emb_names:
                    m = dir_df[(dir_df['dataset'] == ds) & (dir_df['embedding'] == emb)]
                    if len(m) > 0:
                        row += f"r={m.iloc[0]['pearson_r']:.4f}            "
                    else:
                        row += f"{'N/A':<20} "
                log(row)

        csv_path = os.path.join(OUTPUT_DIR, 'perturbation_results.csv')
        df.to_csv(csv_path, index=False)
        log(f"\nResults saved to {csv_path}")

    log("\nDone!")


if __name__ == '__main__':
    main()
