#!/usr/bin/env python3
"""
Gene Embedding Benchmark
========================
Fair comparison of gene embeddings on downstream tasks.
Standard weighted average aggregation (no post-hoc fusion).

Embeddings:
  - difference_v3 (60697 x 256)
  - baseline (60697 x 256)
  - scGPT_human (60697 x 512)
  - GF-12L95M (Geneformer V2 12L, 11355 x 512, Entrez IDs)

Tasks:
  A. Cell Type Annotation
  B. Perturbation Classification

Evaluation: 5-fold stratified CV, LR + MLP
"""

import os, sys, json, time, gzip, warnings, urllib.request
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore")

# =============================================================
# Configuration
# =============================================================
BASE_DIR = '/root/autodl-tmp/scbenchmark'
OUTPUT_DIR = '/root/autodl-tmp/embedding_benchmark'
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOG_FILE = os.path.join(OUTPUT_DIR, 'benchmark.log')

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
        'path': '/root/autodl-tmp/scGPT_human/best_model.pt',
        'key': 'encoder.embedding.weight',
    },
}

GF_CONFIG = {
    'dir': '/root/autodl-tmp/gene_embeddings/intersect/GF-12L95M',
    'name': 'GF-12L95M',
}

CLS_DATA_DIR = f'{BASE_DIR}/data/downstreams/classification/processed_data'
PERTURB_DATA_DIR = f'{BASE_DIR}/data/downstreams/perturbation/processed_data'

ANNOTATION_DATASETS = ['Myeloid', 'Multiple_Sclerosis', 'pancread', 'lupus']
PERTURBATION_DATASETS = ['adamson', 'dixit', 'norman']


# =============================================================
# Logging
# =============================================================
def log(msg):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')


# =============================================================
# Loading Functions
# =============================================================
def load_checkpoint_embedding(path, key):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    return ckpt[key].detach().numpy()


def load_csv_embedding(emb_dir, name):
    emb_path = os.path.join(emb_dir, f'{name}_emb.csv')
    gl_path = os.path.join(emb_dir, f'{name}_genelist.txt')
    emb = pd.read_csv(emb_path, header=None).values.astype(np.float32)
    with open(gl_path) as f:
        genelist = [line.strip() for line in f]
    return emb, genelist


def load_dataset(pt_path):
    d = torch.load(pt_path, map_location='cpu', weights_only=False)
    return d['genes'], d['expressions'], d['cls_name']


def load_vocab(vocab_path):
    with open(vocab_path) as f:
        return json.load(f)


# =============================================================
# Gene ID Mapping (for Geneformer)
# =============================================================
def build_symbol_to_entrez():
    """Download NCBI gene_info and build symbol -> entrezID mapping"""
    mapping_file = os.path.join(OUTPUT_DIR, 'gene_symbol_to_entrez.json')
    if os.path.exists(mapping_file):
        log("Loading cached gene symbol -> Entrez mapping...")
        with open(mapping_file) as f:
            return json.load(f)

    log("Downloading NCBI Homo_sapiens.gene_info.gz ...")
    url = 'https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz'
    local_path = os.path.join(OUTPUT_DIR, 'Homo_sapiens.gene_info.gz')

    try:
        urllib.request.urlretrieve(url, local_path)
    except Exception as e:
        log(f"  Failed to download: {e}")
        return None

    symbol_to_entrez = {}
    with gzip.open(local_path, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 5:
                continue
            entrez_id = parts[1]
            symbol = parts[2]
            synonyms = parts[4]
            symbol_to_entrez[symbol] = entrez_id
            if synonyms != '-':
                for syn in synonyms.split('|'):
                    if syn not in symbol_to_entrez:
                        symbol_to_entrez[syn] = entrez_id

    with open(mapping_file, 'w') as f:
        json.dump(symbol_to_entrez, f)
    log(f"  Built mapping: {len(symbol_to_entrez)} entries")
    return symbol_to_entrez


def build_vocab_to_gf_index(vocab, gf_genelist, symbol_to_entrez):
    """Build vocab_index -> gf_embedding_index mapping"""
    rev_vocab = {v: k for k, v in vocab.items()}
    entrez_to_gf = {eid: i for i, eid in enumerate(gf_genelist)}

    mapping = {}
    for idx, symbol in rev_vocab.items():
        if symbol in symbol_to_entrez:
            eid = symbol_to_entrez[symbol]
            if eid in entrez_to_gf:
                mapping[idx] = entrez_to_gf[eid]
    return mapping


# =============================================================
# Cell Representation
# =============================================================
def build_cell_repr(genes_list, expr_list, emb_matrix, max_genes=512):
    """Standard weighted average: cell = sum(expr_i * emb_i) / sum(expr_i)"""
    n_cells = len(genes_list)
    emb_dim = emb_matrix.shape[1]
    result = np.zeros((n_cells, emb_dim), dtype=np.float32)

    for i in range(n_cells):
        g = np.array(genes_list[i])
        e = np.array(expr_list[i], dtype=np.float32)
        if len(g) > max_genes:
            idx = np.argsort(-e)[:max_genes]
            g, e = g[idx], e[idx]
        valid = (g >= 0) & (g < emb_matrix.shape[0])
        g, e = g[valid], e[valid]
        if len(g) == 0:
            continue
        w = e / (e.sum() + 1e-8)
        result[i] = (emb_matrix[g] * w[:, None]).sum(0)

    return result


def build_cell_repr_gf(genes_list, expr_list, vocab_to_gf, gf_emb, max_genes=512):
    """Build cell repr using Geneformer embedding with gene ID mapping"""
    n_cells = len(genes_list)
    emb_dim = gf_emb.shape[1]
    result = np.zeros((n_cells, emb_dim), dtype=np.float32)
    coverage = 0

    for i in range(n_cells):
        g = np.array(genes_list[i])
        e = np.array(expr_list[i], dtype=np.float32)
        if len(g) > max_genes:
            idx = np.argsort(-e)[:max_genes]
            g, e = g[idx], e[idx]

        valid_mask = np.array([int(gi) in vocab_to_gf for gi in g])
        if valid_mask.sum() == 0:
            continue

        g_valid = g[valid_mask]
        e_valid = e[valid_mask]
        gf_indices = np.array([vocab_to_gf[int(gi)] for gi in g_valid])

        gene_embs = gf_emb[gf_indices]
        w = e_valid / (e_valid.sum() + 1e-8)
        result[i] = (gene_embs * w[:, None]).sum(0)
        coverage += 1

    return result, coverage


# =============================================================
# Classification
# =============================================================
def run_classification(X, y, clf_type='lr', n_splits=5, random_state=42):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Check minimum class size for stratified CV
    from collections import Counter
    class_counts = Counter(y_enc)
    min_count = min(class_counts.values())
    if min_count < n_splits:
        n_splits = max(2, min_count)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    accs, f1_macros, f1_weights = [], [], []

    for train_idx, test_idx in skf.split(X, y_enc):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_enc[train_idx], y_enc[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        if clf_type == 'lr':
            clf = LogisticRegression(max_iter=1000, random_state=random_state, n_jobs=-1)
        else:
            clf = MLPClassifier(
                hidden_layer_sizes=(256, 128), max_iter=300,
                random_state=random_state, early_stopping=True
            )

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        accs.append(accuracy_score(y_test, y_pred))
        f1_macros.append(f1_score(y_test, y_pred, average='macro', zero_division=0))
        f1_weights.append(f1_score(y_test, y_pred, average='weighted', zero_division=0))

    return {
        'accuracy': (np.mean(accs), np.std(accs)),
        'f1_macro': (np.mean(f1_macros), np.std(f1_macros)),
        'f1_weighted': (np.mean(f1_weights), np.std(f1_weights)),
    }


# =============================================================
# Evaluate one embedding on one dataset
# =============================================================
def evaluate_embedding(emb_name, X, labels, ds_name, task, n_cells, n_classes, all_results):
    for clf_type in ['lr', 'mlp']:
        t0 = time.time()
        res = run_classification(X, labels, clf_type=clf_type)
        elapsed = time.time() - t0

        log(f"  {emb_name:20s} | {clf_type:3s} | "
            f"acc={res['accuracy'][0]:.4f}+-{res['accuracy'][1]:.4f} | "
            f"f1m={res['f1_macro'][0]:.4f}+-{res['f1_macro'][1]:.4f} | "
            f"f1w={res['f1_weighted'][0]:.4f}+-{res['f1_weighted'][1]:.4f} | "
            f"{elapsed:.1f}s")

        all_results.append({
            'task': task,
            'dataset': ds_name,
            'embedding': emb_name,
            'classifier': clf_type,
            'n_cells': n_cells,
            'n_classes': n_classes,
            'accuracy_mean': round(res['accuracy'][0], 4),
            'accuracy_std': round(res['accuracy'][1], 4),
            'f1_macro_mean': round(res['f1_macro'][0], 4),
            'f1_macro_std': round(res['f1_macro'][1], 4),
            'f1_weighted_mean': round(res['f1_weighted'][0], 4),
            'f1_weighted_std': round(res['f1_weighted'][1], 4),
        })


# =============================================================
# Run one task across all embeddings
# =============================================================
def run_task(task_name, datasets, data_dir, embeddings, gf_emb, vocab_to_gf, all_results):
    log(f"\n{'=' * 70}")
    log(f"Task: {task_name}")
    log(f"{'=' * 70}")

    for ds_name in datasets:
        pt_path = os.path.join(data_dir, f'{ds_name}_data.pt')
        if not os.path.exists(pt_path):
            # Try classification dir as fallback
            alt = os.path.join(CLS_DATA_DIR, f'{ds_name}_data.pt')
            if os.path.exists(alt):
                pt_path = alt
            else:
                log(f"\n  {ds_name}: NOT FOUND, skipping")
                continue

        try:
            genes, expressions, cls_names = load_dataset(pt_path)
        except Exception as e:
            log(f"\n  {ds_name}: LOAD ERROR: {e}")
            continue

        labels = np.array(cls_names)
        n_classes = len(set(cls_names))
        log(f"\n--- {ds_name}: {len(labels)} cells, {n_classes} classes ---")

        # Cache cell representations per embedding
        for emb_name, emb_matrix in embeddings.items():
            try:
                t0 = time.time()
                X = build_cell_repr(genes, expressions, emb_matrix)
                log(f"  {emb_name} repr built: {X.shape} ({time.time()-t0:.1f}s)")
                evaluate_embedding(emb_name, X, labels, ds_name, task_name,
                                   len(labels), n_classes, all_results)
            except Exception as e:
                log(f"  ERROR on {emb_name}/{ds_name}: {e}")

        # Geneformer
        if gf_emb is not None and vocab_to_gf is not None:
            try:
                t0 = time.time()
                X_gf, cov = build_cell_repr_gf(genes, expressions, vocab_to_gf, gf_emb)
                log(f"  GF-12L95M repr built: {X_gf.shape}, coverage={cov}/{len(labels)} ({time.time()-t0:.1f}s)")
                if cov < len(labels) * 0.1:
                    log(f"  WARNING: Very low coverage, skipping GF-12L95M for {ds_name}")
                else:
                    evaluate_embedding('GF-12L95M', X_gf, labels, ds_name, task_name,
                                       len(labels), n_classes, all_results)
            except Exception as e:
                log(f"  ERROR on GF-12L95M/{ds_name}: {e}")


# =============================================================
# Main
# =============================================================
def main():
    log("=" * 70)
    log("Gene Embedding Benchmark")
    log(f"Started: {datetime.now()}")
    log("=" * 70)

    # --- Load vocab ---
    vocab = load_vocab(f'{BASE_DIR}/vocab.json')
    log(f"Vocab: {len(vocab)} genes")

    # --- Check if scGPT_human uses same vocab ---
    scgpt_vocab = load_vocab('/root/autodl-tmp/scGPT_human/vocab.json')
    common = set(vocab.keys()) & set(scgpt_vocab.keys())
    log(f"Vocab overlap with scGPT_human: {len(common)}/{len(vocab)}")
    if len(common) < len(vocab) * 0.9:
        log("WARNING: Significant vocab mismatch between your model and scGPT_human!")

    # --- Load checkpoint embeddings ---
    embeddings = {}
    for name, cfg in EMBEDDINGS.items():
        log(f"Loading {name}...")
        try:
            emb = load_checkpoint_embedding(cfg['path'], cfg['key'])
            embeddings[name] = emb
            log(f"  Shape: {emb.shape}")
        except Exception as e:
            log(f"  FAILED: {e}")

    # --- Load Geneformer ---
    gf_emb = None
    vocab_to_gf = None
    try:
        gf_dir = GF_CONFIG['dir']
        gf_name = GF_CONFIG['name']
        gf_emb, gf_genelist = load_csv_embedding(gf_dir, gf_name)
        log(f"Loaded {gf_name}: emb={gf_emb.shape}, genelist={len(gf_genelist)}")

        symbol_to_entrez = build_symbol_to_entrez()
        if symbol_to_entrez:
            vocab_to_gf = build_vocab_to_gf_index(vocab, gf_genelist, symbol_to_entrez)
            log(f"  Vocab->GF mapping: {len(vocab_to_gf)} genes mapped")
        else:
            log("  Gene mapping failed, GF-12L95M will be skipped")
            gf_emb = None
    except Exception as e:
        log(f"Failed to load Geneformer: {e}")

    all_results = []

    # --- Task A: Annotation ---
    run_task('annotation', ANNOTATION_DATASETS, CLS_DATA_DIR,
             embeddings, gf_emb, vocab_to_gf, all_results)

    # --- Task B: Perturbation Classification ---
    run_task('perturbation_cls', PERTURBATION_DATASETS, CLS_DATA_DIR,
             embeddings, gf_emb, vocab_to_gf, all_results)

    # --- Save results ---
    results_df = pd.DataFrame(all_results)
    csv_path = os.path.join(OUTPUT_DIR, 'benchmark_results.csv')
    results_df.to_csv(csv_path, index=False)
    log(f"\nResults saved to {csv_path}")

    # --- Summary Table ---
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)

    for task in ['annotation', 'perturbation_cls']:
        task_df = results_df[results_df['task'] == task]
        if len(task_df) == 0:
            continue
        log(f"\n{'=' * 50}")
        log(f"Task: {task}")
        log(f"{'=' * 50}")

        for ds in task_df['dataset'].unique():
            ds_df = task_df[task_df['dataset'] == ds]
            log(f"\n--- {ds} ---")
            log(f"{'Embedding':<20} {'Clf':<5} {'Accuracy':>14} {'F1-macro':>14} {'F1-weighted':>14}")
            log("-" * 72)
            for _, row in ds_df.iterrows():
                log(f"{row['embedding']:<20} {row['classifier']:<5} "
                    f"{row['accuracy_mean']:.4f}+-{row['accuracy_std']:.4f}  "
                    f"{row['f1_macro_mean']:.4f}+-{row['f1_macro_std']:.4f}  "
                    f"{row['f1_weighted_mean']:.4f}+-{row['f1_weighted_std']:.4f}")

    log(f"\nBenchmark complete! {datetime.now()}")
    return results_df


if __name__ == '__main__':
    main()
