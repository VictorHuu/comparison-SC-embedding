#!/usr/bin/env python3
"""
GRN Benchmark - Embedding Only
===============================
Predict TF-Target regulatory relationships using ONLY gene embeddings.
No expression data. Uses scGREAT's datasets as ground truth.

For each gene pair (TF, Target):
  features = [emb_TF; emb_Target; emb_TF * emb_Target; cosine_sim]
  (concatenation + hadamard product + cosine similarity)

Classifiers: Logistic Regression, MLP
Metrics: AUROC, AUPRC
"""

import os, sys, json, gzip, warnings
import numpy as np
import pandas as pd
import torch
import urllib.request
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")

# =============================================================
# Config
# =============================================================
BASE_DIR = '/root/autodl-tmp/scbenchmark'
SCGREAT_DIR = '/root/autodl-tmp/scGREAT'
OUTPUT_DIR = '/root/autodl-tmp/grn_benchmark'
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOG_FILE = os.path.join(OUTPUT_DIR, 'grn_emb_only.log')
VOCAB_PATH = f'{BASE_DIR}/vocab.json'

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

DATASETS = ['hESC500', 'mESC500']


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
    url = 'https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz'
    local_path = os.path.join(OUTPUT_DIR, 'Homo_sapiens.gene_info.gz')
    urllib.request.urlretrieve(url, local_path)
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
    return symbol_to_entrez


def get_dataset_genes(dataset_name):
    target_path = os.path.join(SCGREAT_DIR, dataset_name, 'Target.csv')
    df = pd.read_csv(target_path)
    return df['Gene'].tolist()


def load_grn_split(dataset_name, split_name):
    """Load train/val/test split. Returns (TF_indices, Target_indices, labels)."""
    path = os.path.join(SCGREAT_DIR, dataset_name, f'{split_name}.csv')
    df = pd.read_csv(path, index_col=0, header=0)
    tf_idx = df.iloc[:, 0].values.astype(int)
    tgt_idx = df.iloc[:, 1].values.astype(int)
    labels = df.iloc[:, 2].values.astype(int)
    return tf_idx, tgt_idx, labels


# =============================================================
# Build gene embedding lookup for a dataset
# =============================================================
def build_gene_emb_lookup(emb_matrix, vocab, dataset_genes):
    """For checkpoint embeddings: gene_index -> embedding vector."""
    n_genes = len(dataset_genes)
    emb_dim = emb_matrix.shape[1]
    lookup = np.zeros((n_genes, emb_dim), dtype=np.float32)
    mapped = 0
    for i, gene in enumerate(dataset_genes):
        if gene in vocab:
            lookup[i] = emb_matrix[vocab[gene]]
            mapped += 1
    return lookup, mapped


def build_gene_emb_lookup_gf(gf_emb, gf_genelist, symbol_to_entrez, dataset_genes):
    """For Geneformer: gene_index -> embedding vector."""
    n_genes = len(dataset_genes)
    emb_dim = gf_emb.shape[1]
    lookup = np.zeros((n_genes, emb_dim), dtype=np.float32)
    entrez_to_gf = {eid: i for i, eid in enumerate(gf_genelist)}
    mapped = 0
    for i, gene in enumerate(dataset_genes):
        if gene in symbol_to_entrez:
            eid = symbol_to_entrez[gene]
            if eid in entrez_to_gf:
                lookup[i] = gf_emb[entrez_to_gf[eid]]
                mapped += 1
    return lookup, mapped


# =============================================================
# Build pair features from embeddings
# =============================================================
def build_pair_features(lookup, tf_indices, tgt_indices):
    """
    For each (TF, Target) pair, compute:
      [emb_TF | emb_Target | emb_TF * emb_Target | cosine_sim]
    """
    emb_tf = lookup[tf_indices]     # (N, dim)
    emb_tgt = lookup[tgt_indices]   # (N, dim)

    # Hadamard product
    hadamard = emb_tf * emb_tgt     # (N, dim)

    # Cosine similarity
    norm_tf = np.linalg.norm(emb_tf, axis=1, keepdims=True) + 1e-8
    norm_tgt = np.linalg.norm(emb_tgt, axis=1, keepdims=True) + 1e-8
    cosine = np.sum(emb_tf * emb_tgt, axis=1, keepdims=True) / (norm_tf * norm_tgt)

    # L2 distance
    l2_dist = np.linalg.norm(emb_tf - emb_tgt, axis=1, keepdims=True)

    # Concatenate all features
    features = np.concatenate([emb_tf, emb_tgt, hadamard, cosine, l2_dist], axis=1)
    return features


# =============================================================
# Evaluate
# =============================================================
def evaluate(train_X, train_y, test_X, test_y, clf_name='lr'):
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.transform(test_X)

    if clf_name == 'lr':
        clf = LogisticRegression(max_iter=1000, n_jobs=1, C=1.0)
    else:
        clf = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500,
                            early_stopping=True, random_state=42)

    clf.fit(train_X, train_y)

    if hasattr(clf, 'predict_proba'):
        probs = clf.predict_proba(test_X)[:, 1]
    else:
        probs = clf.decision_function(test_X)

    auroc = roc_auc_score(test_y, probs)
    auprc = average_precision_score(test_y, probs)
    return auroc, auprc


# =============================================================
# Main
# =============================================================
def main():
    log("=" * 70)
    log("GRN Benchmark - Embedding Only (No Expression)")
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

    symbol_to_entrez = build_symbol_to_entrez()

    # Also add random baseline
    log("Adding random embedding baseline (256-dim)...")

    # Results storage
    all_results = []

    for ds_name in DATASETS:
        ds_path = os.path.join(SCGREAT_DIR, ds_name)
        if not os.path.exists(ds_path):
            log(f"Dataset {ds_name}: NOT FOUND, skipping")
            continue

        dataset_genes = get_dataset_genes(ds_name)
        log(f"\n{'='*70}")
        log(f"Dataset: {ds_name} ({len(dataset_genes)} genes)")
        log(f"{'='*70}")

        # Load splits
        train_tf, train_tgt, train_y = load_grn_split(ds_name, 'Train_set')
        val_tf, val_tgt, val_y = load_grn_split(ds_name, 'Validation_set')
        test_tf, test_tgt, test_y = load_grn_split(ds_name, 'Test_set')

        log(f"Train: {len(train_y)} pairs (pos={train_y.sum()}, neg={len(train_y)-train_y.sum()})")
        log(f"Val:   {len(val_y)} pairs (pos={val_y.sum()}, neg={len(val_y)-val_y.sum()})")
        log(f"Test:  {len(test_y)} pairs (pos={test_y.sum()}, neg={len(test_y)-test_y.sum()})")

        # Combine train+val for training
        all_train_tf = np.concatenate([train_tf, val_tf])
        all_train_tgt = np.concatenate([train_tgt, val_tgt])
        all_train_y = np.concatenate([train_y, val_y])

        log(f"\n{'Embedding':<20} {'Clf':<5} {'Coverage':>10} {'AUROC':>12} {'AUPRC':>12}")
        log("-" * 65)

        for emb_name, emb_data in loaded_embs.items():
            # Build lookup
            if emb_data['type'] == 'checkpoint':
                lookup, mapped = build_gene_emb_lookup(
                    emb_data['matrix'], vocab, dataset_genes
                )
            else:
                lookup, mapped = build_gene_emb_lookup_gf(
                    emb_data['matrix'], emb_data['genelist'],
                    symbol_to_entrez, dataset_genes
                )

            coverage = f"{mapped}/{len(dataset_genes)}"

            # Build features
            train_X = build_pair_features(lookup, all_train_tf, all_train_tgt)
            test_X = build_pair_features(lookup, test_tf, test_tgt)

            for clf_name in ['lr', 'mlp']:
                try:
                    auroc, auprc = evaluate(train_X, all_train_y, test_X, test_y, clf_name)
                    log(f"{emb_name:<20} {clf_name:<5} {coverage:>10} {auroc:>11.4f} {auprc:>11.4f}")
                    all_results.append({
                        'dataset': ds_name, 'embedding': emb_name,
                        'clf': clf_name, 'coverage': coverage,
                        'auroc': auroc, 'auprc': auprc,
                    })
                except Exception as e:
                    log(f"{emb_name:<20} {clf_name:<5} {coverage:>10} ERROR: {e}")

        # Random embedding baseline
        np.random.seed(42)
        random_lookup = np.random.randn(len(dataset_genes), 256).astype(np.float32)
        train_X = build_pair_features(random_lookup, all_train_tf, all_train_tgt)
        test_X = build_pair_features(random_lookup, test_tf, test_tgt)
        for clf_name in ['lr', 'mlp']:
            try:
                auroc, auprc = evaluate(train_X, all_train_y, test_X, test_y, clf_name)
                log(f"{'random_256':<20} {clf_name:<5} {'910/910':>10} {auroc:>11.4f} {auprc:>11.4f}")
                all_results.append({
                    'dataset': ds_name, 'embedding': 'random_256',
                    'clf': clf_name, 'coverage': f"{len(dataset_genes)}/{len(dataset_genes)}",
                    'auroc': auroc, 'auprc': auprc,
                })
            except Exception as e:
                log(f"{'random_256':<20} {clf_name:<5} ERROR: {e}")

    # Summary
    log(f"\n{'='*70}")
    log("SUMMARY")
    log(f"{'='*70}")

    for ds_name in DATASETS:
        ds_results = [r for r in all_results if r['dataset'] == ds_name]
        if not ds_results:
            continue
        log(f"\n--- {ds_name} ---")
        log(f"{'Embedding':<20} {'Clf':<5} {'Coverage':>10} {'AUROC':>12} {'AUPRC':>12}")
        log("-" * 65)
        for r in ds_results:
            log(f"{r['embedding']:<20} {r['clf']:<5} {r['coverage']:>10} {r['auroc']:>11.4f} {r['auprc']:>11.4f}")

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, 'grn_emb_only_results.csv')
    pd.DataFrame(all_results).to_csv(csv_path, index=False)
    log(f"\nResults saved to {csv_path}")
    log("Done!")


if __name__ == '__main__':
    main()
