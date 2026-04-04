#!/usr/bin/env python3
"""
GRN Benchmark on BEELINE datasets - Embedding Only
====================================================
Downloads BEELINE benchmark data from Zenodo, processes all datasets,
and evaluates gene embeddings on GRN prediction.

Cell types: hESC, hHep, mDC, mESC, mHSC-E, mHSC-GM, mHSC-L
Network types: Specific ChIP-seq, Non-Specific ChIP-seq, STRING
Gene counts: 500, 1000
"""

import os, sys, json, gzip, zipfile, warnings, random
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
BASE_DIR = '/bigdata2/hyt/projects/scbenchmark'
BEELINE_DIR = '/bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/BEELINE'
SCGREAT_DIR = '/bigdata2/hyt/projects/scGREAT'
OUTPUT_DIR = '/bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/grn_benchmark'
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOG_FILE = os.path.join(OUTPUT_DIR, 'grn_beeline.log')
VOCAB_PATH = f'{BASE_DIR}/vocab.json'

EMBEDDINGS = {
    # 'difference_v3': {
    #     'path': f'{BASE_DIR}/save_pretrain/difference_aligned_v3/best_model.pt',
    #     'key': 'module.embedding.weight',
    #     'type': 'checkpoint',
    # },
    'minus': {
        'path': f'{BASE_DIR}/save_pretrain/minus/best_model.pt',
        'key': 'module.embedding.weight',
        'type': 'checkpoint',
    },
    'baseline': {
        'path': f'{BASE_DIR}/save_pretrain/baseline/best_model.pt',
        'key': 'module.embedding.weight',
        'type': 'checkpoint',
    },
    'scGPT_human': {
        'path': f'{BASE_DIR}/save_pretrain/scGPT_human/best_model.pt',
        'key': 'encoder.embedding.weight',
        'type': 'checkpoint',
    },
    'v4_bias_rec_best': {
        'path': f'{BASE_DIR}/save_pretrain/v4_bias_rec_best/best_model.pt',
        'key': 'embedding.weight',
        'type': 'checkpoint',
    },
    'v4_plain_best': {
        'path': f'{BASE_DIR}/save_pretrain/v4_plain_best/best_model.pt',
        'key': 'embedding.weight',
        'type': 'checkpoint',
    },
    'v4_type_pe_best': {
        'path': f'{BASE_DIR}/save_pretrain/v4_type_pe_best/best_model.pt',
        'key': 'embedding.weight',
        'type': 'checkpoint',
    },
    'GF-12L95M': {
        'dir': '/bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding/gene_embeddings/intersect/GF-12L95M',
        'type': 'geneformer',
    },
}

# Cell type -> expression dir, species, specific network file
CELL_CONFIGS = {
    'hESC': {
        'expr_dir': 'BEELINE-data/inputs/scRNA-Seq/hESC',
        'species': 'human',
        'specific_net': 'hESC-ChIP-seq-network.csv',
    },
    'hHep': {
        'expr_dir': 'BEELINE-data/inputs/scRNA-Seq/hHep',
        'species': 'human',
        'specific_net': 'HepG2-ChIP-seq-network.csv',
    },
    'mDC': {
        'expr_dir': 'BEELINE-data/inputs/scRNA-Seq/mDC',
        'species': 'mouse',
        'specific_net': 'mDC-ChIP-seq-network.csv',
    },
    'mESC': {
        'expr_dir': 'BEELINE-data/inputs/scRNA-Seq/mESC',
        'species': 'mouse',
        'specific_net': 'mESC-ChIP-seq-network.csv',
    },
    'mHSC-E': {
        'expr_dir': 'BEELINE-data/inputs/scRNA-Seq/mHSC-E',
        'species': 'mouse',
        'specific_net': 'mHSC-ChIP-seq-network.csv',
    },
    'mHSC-GM': {
        'expr_dir': 'BEELINE-data/inputs/scRNA-Seq/mHSC-GM',
        'species': 'mouse',
        'specific_net': 'mHSC-ChIP-seq-network.csv',
    },
    'mHSC-L': {
        'expr_dir': 'BEELINE-data/inputs/scRNA-Seq/mHSC-L',
        'species': 'mouse',
        'specific_net': 'mHSC-ChIP-seq-network.csv',
    },
}

# Network types per species
NETWORK_TYPES = {
    'human': {
        'Specific': None,   # filled per cell type
        'Non-Specific': 'Non-specific-ChIP-seq-network.csv',
        'STRING': 'STRING-network.csv',
    },
    'mouse': {
        'Specific': None,
        'Non-Specific': 'Non-Specific-ChIP-seq-network.csv',
        'STRING': 'STRING-network.csv',
    },
}

N_HVGS = [500, 1000]


def log(msg):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')


def first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def resolve_expression_path(cell_type, cfg):
    """
    Resolve ExpressionData.csv path with local override support.
    Priority:
      1) BEELINE_EXPR_ROOT/<cell_type>/ExpressionData.csv (env)
      2) ./scRNA-Seq/<cell_type>/ExpressionData.csv (current working tree)
      3) legacy BEELINE zip layout path
    """
    env_root = os.environ.get('BEELINE_EXPR_ROOT', '').strip()
    return first_existing([
        os.path.join(env_root, cell_type, 'ExpressionData.csv') if env_root else None,
        os.path.join(os.getcwd(), 'scRNA-Seq', cell_type, 'ExpressionData.csv'),
        os.path.join(BEELINE_DIR, cfg['expr_dir'], 'ExpressionData.csv'),
    ])


def resolve_tf_list_path(species):
    return first_existing([
        os.path.join(BEELINE_DIR, f'{species}-tfs.csv'),
        os.path.join(BEELINE_DIR, 'BEELINE-data', 'inputs', 'TFs', f'{species}-tfs.csv'),
    ])


def resolve_network_root():
    return first_existing([
        os.path.join(BEELINE_DIR, 'Networks'),
        os.path.join(BEELINE_DIR, 'BEELINE-Networks'),
    ])


# =============================================================
# Download BEELINE data
# =============================================================
def download_beeline():
    data_check = os.path.join(BEELINE_DIR, 'BEELINE-data')
    net_check = os.path.join(BEELINE_DIR, 'Networks')
    if os.path.exists(data_check) and os.path.exists(net_check):
        log("BEELINE data already present.")
        return

    os.makedirs(BEELINE_DIR, exist_ok=True)
    base_url = 'https://zenodo.org/records/3701939/files'
    for fname in ['BEELINE-data.zip', 'BEELINE-Networks.zip']:
        fpath = os.path.join(BEELINE_DIR, fname)
        if not os.path.exists(fpath):
            log(f"Downloading {fname}...")
            urllib.request.urlretrieve(f'{base_url}/{fname}?download=1', fpath)
            log(f"  Downloaded: {os.path.getsize(fpath) / 1e6:.1f} MB")

    for fname in ['BEELINE-data.zip', 'BEELINE-Networks.zip']:
        fpath = os.path.join(BEELINE_DIR, fname)
        log(f"Extracting {fname}...")
        with zipfile.ZipFile(fpath, 'r') as z:
            z.extractall(BEELINE_DIR)
    log("BEELINE data ready.")


# =============================================================
# Loading embeddings
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
    alt_path = '/bigdata2/hyt/projects/embedding_benchmark/gene_symbol_to_entrez.json'
    if os.path.exists(alt_path):
        import shutil
        shutil.copy2(alt_path, mapping_file)
        with open(mapping_file) as f:
            return json.load(f)
    return {}


# =============================================================
# Build dataset from BEELINE expression + network
# =============================================================
def build_beeline_dataset(expr_path, net_path, tf_list_path, n_hvg=500):
    """
    Build gene list and positive pairs from BEELINE raw data.
    Returns (gene_list, gene_to_idx, pos_pairs_set, tf_indices) or None.
    """
    # Read expression: rows=genes, cols=cells
    expr = pd.read_csv(expr_path, index_col=0, header=0)
    all_genes = list(expr.index)
    all_genes_set = set(all_genes)

    # Read network: Gene1,Gene2
    net = pd.read_csv(net_path)
    tf_col, tgt_col = net.columns[0], net.columns[1]

    # Filter to genes in expression data
    net = net[net[tf_col].isin(all_genes_set) & net[tgt_col].isin(all_genes_set)]
    if len(net) < 10:
        return None

    # Read TF list
    tf_df = pd.read_csv(tf_list_path)
    known_tfs = set(tf_df.iloc[:, 0].tolist())

    # Select top n_hvg genes by variance (same as scGREAT)
    gene_var = expr.var(axis=1).sort_values(ascending=False)
    hvg = set(gene_var.head(n_hvg).index.tolist())

    gene_list = sorted(hvg)
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}
    gene_set = set(gene_list)

    # Filter network to selected genes
    net_filt = net[net[tf_col].isin(gene_set) & net[tgt_col].isin(gene_set)]
    if len(net_filt) < 10:
        return None

    # Positive pairs as (TF_idx, Target_idx)
    pos_pairs = set()
    for _, row in net_filt.iterrows():
        tf_idx = gene_to_idx[row[tf_col]]
        tgt_idx = gene_to_idx[row[tgt_col]]
        pos_pairs.add((tf_idx, tgt_idx))

    # TF indices (those that appear as Gene1 in positive network)
    tf_indices = sorted(set(gene_to_idx[g] for g in net_filt[tf_col].unique()))

    return gene_list, gene_to_idx, pos_pairs, tf_indices


# =============================================================
# Hard Negative Split (from scGREAT pre-processing)
# =============================================================
def hard_negative_split(pos_pairs, gene_indices, tf_indices, seed=42):
    """Generate train/test splits with hard negative sampling."""
    random.seed(seed)
    np.random.seed(seed)

    gene_set = np.array(gene_indices)

    # Build positive dict: TF -> list of targets
    pos_dict = {}
    for tf, tgt in pos_pairs:
        if tf not in pos_dict:
            pos_dict[tf] = []
        pos_dict[tf].append(tgt)

    # Build negative dict: for each TF, genes NOT in its targets
    neg_dict = {}
    for tf in tf_indices:
        if tf in pos_dict:
            pos_items = set(pos_dict[tf])
            pos_items.add(tf)
            neg_dict[tf] = [g for g in gene_set if g not in pos_items]
        else:
            neg_dict[tf] = [g for g in gene_set if g != tf]

    # Split positives: 67% train, 10% val, 23% test
    ratio = 0.67
    train_pos, val_pos, test_pos = {}, {}, {}
    for k, targets in pos_dict.items():
        targets = list(targets)
        np.random.shuffle(targets)
        if len(targets) == 1:
            if np.random.uniform() <= 0.5:
                train_pos[k] = targets
            else:
                test_pos[k] = targets
        elif len(targets) == 2:
            train_pos[k] = [targets[0]]
            test_pos[k] = [targets[1]]
        else:
            n_train = int(len(targets) * ratio)
            n_val = int(len(targets) * 0.1)
            train_pos[k] = targets[:n_train]
            val_pos[k] = targets[n_train:n_train + n_val]
            test_pos[k] = targets[n_train + n_val:]

    # Split negatives similarly
    train_neg, val_neg, test_neg = {}, {}, {}
    for k in pos_dict.keys():
        negs = list(neg_dict.get(k, []))
        np.random.shuffle(negs)
        n = len(negs)
        n_train = int(n * ratio)
        n_val = int(n * 0.1)
        train_neg[k] = negs[:n_train]
        val_neg[k] = negs[n_train:n_train + n_val]
        test_neg[k] = negs[n_train + n_val:]

    def build_split(pos_d, neg_d):
        pairs, labels = [], []
        for k, targets in pos_d.items():
            for t in targets:
                pairs.append([k, t])
                labels.append(1)
        for k, targets in neg_d.items():
            for t in targets:
                pairs.append([k, t])
                labels.append(0)
        if len(pairs) == 0:
            return np.array([]).reshape(0, 2).astype(int), np.array([]).astype(int)
        pairs = np.array(pairs, dtype=int)
        labels = np.array(labels, dtype=int)
        return pairs, labels

    train_pairs, train_labels = build_split(train_pos, train_neg)
    val_pairs, val_labels = build_split(val_pos, val_neg)
    test_pairs, test_labels = build_split(test_pos, test_neg)

    return (train_pairs, train_labels), (val_pairs, val_labels), (test_pairs, test_labels)


# =============================================================
# Features & evaluation
# =============================================================
def build_gene_lookup(emb_matrix, vocab, gene_list):
    n = len(gene_list)
    d = emb_matrix.shape[1]
    lookup = np.zeros((n, d), dtype=np.float32)
    mapped = 0
    for i, g in enumerate(gene_list):
        if g in vocab:
            lookup[i] = emb_matrix[vocab[g]]
            mapped += 1
    return lookup, mapped


def build_gene_lookup_gf(gf_emb, gf_genelist, s2e, gene_list):
    n = len(gene_list)
    d = gf_emb.shape[1]
    lookup = np.zeros((n, d), dtype=np.float32)
    e2gf = {eid: i for i, eid in enumerate(gf_genelist)}
    mapped = 0
    for i, g in enumerate(gene_list):
        if g in s2e:
            eid = s2e[g]
            if eid in e2gf:
                lookup[i] = gf_emb[e2gf[eid]]
                mapped += 1
    return lookup, mapped


def build_pair_features(lookup, pairs):
    if len(pairs) == 0:
        return np.array([]).reshape(0, lookup.shape[1] * 3 + 2)
    tf_emb = lookup[pairs[:, 0]]
    tgt_emb = lookup[pairs[:, 1]]
    hadamard = tf_emb * tgt_emb
    norm_tf = np.linalg.norm(tf_emb, axis=1, keepdims=True) + 1e-8
    norm_tgt = np.linalg.norm(tgt_emb, axis=1, keepdims=True) + 1e-8
    cosine = np.sum(tf_emb * tgt_emb, axis=1, keepdims=True) / (norm_tf * norm_tgt)
    l2 = np.linalg.norm(tf_emb - tgt_emb, axis=1, keepdims=True)
    return np.concatenate([tf_emb, tgt_emb, hadamard, cosine, l2], axis=1)


def evaluate(train_X, train_y, test_X, test_y, clf_name='lr'):
    if len(test_X) == 0 or len(train_X) == 0:
        return None, None
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.transform(test_X)

    if clf_name == 'lr':
        clf = LogisticRegression(max_iter=1000, n_jobs=1, C=1.0)
    else:
        clf = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500,
                            early_stopping=True, random_state=42)
    clf.fit(train_X, train_y)
    probs = clf.predict_proba(test_X)[:, 1]
    auroc = roc_auc_score(test_y, probs)
    auprc = average_precision_score(test_y, probs)
    return auroc, auprc


# =============================================================
# Load scGREAT pre-processed datasets
# =============================================================
def load_scgreat_dataset(ds_name):
    ds_dir = os.path.join(SCGREAT_DIR, ds_name)
    if not os.path.exists(ds_dir):
        return None

    target_path = os.path.join(ds_dir, 'Target.csv')
    if not os.path.exists(target_path):
        return None

    df = pd.read_csv(target_path)
    gene_list = df['Gene'].tolist()

    splits = {}
    for split_name in ['Train_set', 'Validation_set', 'Test_set']:
        path = os.path.join(ds_dir, f'{split_name}.csv')
        if not os.path.exists(path):
            return None
        sdf = pd.read_csv(path, index_col=0, header=0)
        pairs = sdf.iloc[:, :2].values.astype(int)
        labels = sdf.iloc[:, 2].values.astype(int)
        splits[split_name] = (pairs, labels)

    return gene_list, splits


# =============================================================
# Run evaluation on one dataset
# =============================================================
def run_one_dataset(ds_name, gene_list, train_pairs, train_labels,
                    test_pairs, test_labels, loaded_embs, vocab, s2e):
    """Evaluate all embeddings on one dataset. Returns list of result dicts."""
    results = []

    log(f"\n{'='*70}")
    log(f"Dataset: {ds_name} ({len(gene_list)} genes)")
    log(f"{'='*70}")
    log(f"Train: {len(train_labels)} (pos={train_labels.sum()})")
    log(f"Test:  {len(test_labels)} (pos={test_labels.sum()})")

    log(f"\n{'Embedding':<20} {'Clf':<5} {'Coverage':>12} {'AUROC':>8} {'AUPRC':>8}")
    log("-" * 60)

    for emb_name, emb_data in loaded_embs.items():
        if emb_data['type'] == 'checkpoint':
            lookup, mapped = build_gene_lookup(emb_data['matrix'], vocab, gene_list)
        else:
            lookup, mapped = build_gene_lookup_gf(
                emb_data['matrix'], emb_data['genelist'], s2e, gene_list)

        cov = f"{mapped}/{len(gene_list)}"
        train_X = build_pair_features(lookup, train_pairs)
        test_X = build_pair_features(lookup, test_pairs)

        for clf_name in ['lr', 'mlp']:
            try:
                auroc, auprc = evaluate(train_X, train_labels, test_X, test_labels, clf_name)
                if auroc is not None:
                    log(f"{emb_name:<20} {clf_name:<5} {cov:>12} {auroc:>7.4f} {auprc:>7.4f}")
                    results.append({
                        'dataset': ds_name, 'embedding': emb_name,
                        'clf': clf_name, 'coverage': cov,
                        'auroc': auroc, 'auprc': auprc,
                    })
            except Exception as e:
                log(f"{emb_name:<20} {clf_name:<5} ERROR: {e}")

    return results


# =============================================================
# Main
# =============================================================
def main():
    log("=" * 70)
    log("GRN BEELINE Full Benchmark - Embedding Only")
    log("=" * 70)

    # Load vocab & embeddings
    vocab = load_vocab()
    log(f"Vocab: {len(vocab)} genes")

    loaded_embs = {}
    for name, cfg in EMBEDDINGS.items():
        try:
            if cfg['type'] == 'checkpoint':
                mat = load_checkpoint_embedding(cfg['path'], cfg['key'])
                loaded_embs[name] = {'matrix': mat, 'type': 'checkpoint'}
                log(f"Loaded {name}: {mat.shape}")
            else:
                emb, gl = load_gf_embedding(cfg['dir'])
                loaded_embs[name] = {'matrix': emb, 'genelist': gl, 'type': 'geneformer'}
                log(f"Loaded {name}: {emb.shape}")
        except Exception as e:
            log(f"SKIP embedding {name}: {e}")

    if not loaded_embs:
        log("No embedding loaded successfully. Exiting.")
        return

    s2e = build_symbol_to_entrez()

    # Download BEELINE
    download_beeline()

    all_results = []

    # =========================================================
    # Part 1: scGREAT pre-processed datasets (hESC500, mESC500)
    # =========================================================
    log("\n--- Part 1: scGREAT pre-processed datasets ---")
    if os.path.exists(SCGREAT_DIR):
        for d in sorted(os.listdir(SCGREAT_DIR)):
            ds_dir = os.path.join(SCGREAT_DIR, d)
            if os.path.isdir(ds_dir) and os.path.exists(os.path.join(ds_dir, 'Target.csv')):
                result = load_scgreat_dataset(d)
                if result is None:
                    continue
                gene_list, splits = result
                train_p, train_l = splits['Train_set']
                val_p, val_l = splits['Validation_set']
                test_p, test_l = splits['Test_set']

                # Combine train+val
                if len(val_p) > 0:
                    all_train_p = np.vstack([train_p, val_p])
                    all_train_l = np.concatenate([train_l, val_l])
                else:
                    all_train_p, all_train_l = train_p, train_l

                res = run_one_dataset(
                    f"{d} [scGREAT]", gene_list,
                    all_train_p, all_train_l, test_p, test_l,
                    loaded_embs, vocab, s2e)
                all_results.extend(res)

    # =========================================================
    # Part 2: BEELINE raw datasets (all cell types × network types × gene counts)
    # =========================================================
    log("\n--- Part 2: BEELINE datasets (from raw data) ---")

    for cell_type, cfg in CELL_CONFIGS.items():
        expr_path = resolve_expression_path(cell_type, cfg)
        if expr_path is None or not os.path.exists(expr_path):
            log(f"  {cell_type}: ExpressionData.csv not found, skipping")
            continue

        species = cfg['species']
        tf_list_path = resolve_tf_list_path(species)
        if tf_list_path is None:
            log(f"  {cell_type}: TF list for {species} not found, skipping")
            continue
        network_root = resolve_network_root()
        if network_root is None:
            log(f"  {cell_type}: network root not found, skipping")
            continue

        # Build network type -> file path mapping
        net_types = {}
        # Specific network
        specific_file = os.path.join(network_root, species, cfg['specific_net'])
        if os.path.exists(specific_file):
            net_types['Specific'] = specific_file

        # Non-Specific and STRING
        for net_name, net_file in NETWORK_TYPES[species].items():
            if net_name == 'Specific':
                continue
            fpath = os.path.join(network_root, species, net_file)
            if os.path.exists(fpath):
                net_types[net_name] = fpath

        log(f"\n  Cell type: {cell_type} ({species}), networks: {list(net_types.keys())}")

        for net_name, net_path in net_types.items():
            for n_hvg in N_HVGS:
                ds_name = f"{cell_type}_{net_name}_{n_hvg}"

                try:
                    result = build_beeline_dataset(expr_path, net_path, tf_list_path, n_hvg=n_hvg)
                    if result is None:
                        log(f"  {ds_name}: too few edges after filtering, skipping")
                        continue

                    gene_list, gene_to_idx, pos_pairs, tf_indices = result
                    gene_indices = list(range(len(gene_list)))

                    (train_p, train_l), (val_p, val_l), (test_p, test_l) = \
                        hard_negative_split(pos_pairs, gene_indices, tf_indices)

                    if len(test_p) < 10 or len(train_p) < 10:
                        log(f"  {ds_name}: too few samples after split, skipping")
                        continue

                    # Combine train+val
                    if len(val_p) > 0:
                        all_train_p = np.vstack([train_p, val_p])
                        all_train_l = np.concatenate([train_l, val_l])
                    else:
                        all_train_p, all_train_l = train_p, train_l

                    res = run_one_dataset(
                        ds_name, gene_list,
                        all_train_p, all_train_l, test_p, test_l,
                        loaded_embs, vocab, s2e)
                    all_results.extend(res)

                except Exception as e:
                    log(f"  {ds_name}: ERROR - {e}")

    # =========================================================
    # Final Summary
    # =========================================================
    log(f"\n{'='*70}")
    log("FINAL SUMMARY (LR results)")
    log(f"{'='*70}")

    if all_results:
        df = pd.DataFrame(all_results)
        lr_df = df[df['clf'] == 'lr']

        emb_names = list(EMBEDDINGS.keys())
        header = f"{'Dataset':<35} " + " ".join(f"{n:<16}" for n in emb_names)
        log(f"\n{header}")
        log("-" * (35 + 17 * len(emb_names)))

        for ds in lr_df['dataset'].unique():
            ds_data = lr_df[lr_df['dataset'] == ds]
            row = f"{ds:<35} "
            for emb_name in emb_names:
                emb_row = ds_data[ds_data['embedding'] == emb_name]
                if len(emb_row) > 0:
                    auroc = emb_row.iloc[0]['auroc']
                    auprc = emb_row.iloc[0]['auprc']
                    row += f"{auroc:.4f}/{auprc:.4f}  "
                else:
                    row += f"{'N/A':<16} "
            log(row)

        # Also print MLP summary
        log(f"\n{'='*70}")
        log("FINAL SUMMARY (MLP results)")
        log(f"{'='*70}")
        mlp_df = df[df['clf'] == 'mlp']
        log(f"\n{header}")
        log("-" * (35 + 17 * len(emb_names)))
        for ds in mlp_df['dataset'].unique():
            ds_data = mlp_df[mlp_df['dataset'] == ds]
            row = f"{ds:<35} "
            for emb_name in emb_names:
                emb_row = ds_data[ds_data['embedding'] == emb_name]
                if len(emb_row) > 0:
                    auroc = emb_row.iloc[0]['auroc']
                    auprc = emb_row.iloc[0]['auprc']
                    row += f"{auroc:.4f}/{auprc:.4f}  "
                else:
                    row += f"{'N/A':<16} "
            log(row)

        # Average across datasets
        log(f"\n{'='*70}")
        log("AVERAGE ACROSS BEELINE DATASETS (LR)")
        log(f"{'='*70}")
        beeline_lr = lr_df[~lr_df['dataset'].str.contains('scGREAT')]
        if len(beeline_lr) > 0:
            for emb_name in emb_names:
                emb_data = beeline_lr[beeline_lr['embedding'] == emb_name]
                if len(emb_data) > 0:
                    mean_auroc = emb_data['auroc'].mean()
                    mean_auprc = emb_data['auprc'].mean()
                    std_auroc = emb_data['auroc'].std()
                    std_auprc = emb_data['auprc'].std()
                    log(f"  {emb_name:<20} AUROC: {mean_auroc:.4f}±{std_auroc:.4f}  AUPRC: {mean_auprc:.4f}±{std_auprc:.4f}  (n={len(emb_data)})")

        csv_path = os.path.join(OUTPUT_DIR, 'grn_beeline_full_results.csv')
        df.to_csv(csv_path, index=False)
        log(f"\nResults saved to {csv_path}")

    log("\nDone!")


if __name__ == '__main__':
    main()
