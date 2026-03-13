#!/usr/bin/env python3
"""
Setup scGREAT GRN experiments with custom gene embeddings.
==========================================================
For each embedding (difference_v3, baseline, scGPT_human, GF-12L95M),
generate biovect.npy files matching scGREAT's expected format,
patch scGREAT code for variable embed_size, then run GRN experiments.

Usage: python setup_scgreat.py [--run]
  Without --run: only generates embedding files + patches code
  With --run: generates embeddings + patches + runs all experiments
"""

import os, sys, json, gzip, subprocess, shutil
import numpy as np
import pandas as pd
import torch
import urllib.request
from datetime import datetime

# =============================================================
# Config
# =============================================================
BASE_DIR = '/root/autodl-tmp/scbenchmark'
SCGREAT_DIR = '/root/autodl-tmp/scGREAT'
OUTPUT_DIR = '/root/autodl-tmp/grn_benchmark'
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOG_FILE = os.path.join(OUTPUT_DIR, 'setup.log')

VOCAB_PATH = f'{BASE_DIR}/vocab.json'

# Embedding sources
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

# scGREAT datasets
HUMAN_DATASETS = ['hESC500', 'hHEP500']
MOUSE_DATASETS = ['mESC500', 'mHSC-E500', 'mHSC-GM500', 'mHSC-L500']


def log(msg):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')


# =============================================================
# Step 1: Clone scGREAT and patch code
# =============================================================
def clone_scgreat():
    if os.path.exists(SCGREAT_DIR):
        log(f"scGREAT already exists at {SCGREAT_DIR}")
        return
    log("Cloning scGREAT repository...")
    subprocess.run(
        ['git', 'clone', 'https://github.com/WangyuchenCS/scGREAT.git', SCGREAT_DIR],
        check=True
    )
    log("Clone complete.")


def patch_scgreat():
    """
    Patch scGREAT source code to support variable embed_size and data_dir.

    Original issues:
    1. model.py: nn.Linear(1536, 1024) hardcodes 1536 = 2*768
    2. main.py: hardcodes 'biovect768.npy'
    3. demo.py: hardcodes data_dir = 'mESC500'
    """
    log("Patching scGREAT source code...")

    # --- Patch model.py ---
    # Fix: nn.Linear(1536, 1024) -> nn.Linear(2 * embed_size, 1024)
    # Also the [1:] slice on biovect means we need a dummy row 0
    model_py = os.path.join(SCGREAT_DIR, 'model.py')
    with open(model_py, 'w') as f:
        f.write('''import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class scGREAT(nn.Module):
    def __init__(self, expression_data_shape, embed_size, num_layers, num_head, biobert_embedding_path):
        super(scGREAT, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_head, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.biobert = np.load(biobert_embedding_path)[1:]
        self.biobert_embedding = nn.Embedding.from_pretrained(torch.from_numpy(self.biobert))
        self.position_embedding = nn.Embedding(2, embed_size)

        self.encoder512 = nn.Linear(expression_data_shape[1], 512)
        self.encoder768 = nn.Linear(512, embed_size)

        # PATCHED: was hardcoded nn.Linear(1536, 1024) assuming embed_size=768
        self.flatten = nn.Flatten()
        self.linear1024 = nn.Linear(2 * embed_size, 1024)
        self.layernorm1024 = nn.LayerNorm(1024)
        self.batchnorm1024 = nn.BatchNorm1d(1024)

        self.linear512 = nn.Linear(1024, 512)
        self.layernorm512 = nn.LayerNorm(512)
        self.batchnorm512 = nn.BatchNorm1d(512)

        self.linear256 = nn.Linear(512, 256)
        self.layernorm256 = nn.LayerNorm(256)
        self.batchnorm256 = nn.BatchNorm1d(256)

        self.linear2 = nn.Linear(256, 1)
        self.actf = nn.PReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.softmax = nn.Softmax(dim=1)
        self.pool = nn.AvgPool1d(kernel_size=4, stride=4)

    def forward(self, gene_pair_index, expr_embedding):
        bs = expr_embedding.shape[0]
        position = torch.Tensor([0, 1] * bs).reshape(bs, -1).to(torch.int32)
        position = position.to(self.device)
        p_e = self.position_embedding(position)
        expr_embedding = expr_embedding.to(self.device)
        gene_pair_index = gene_pair_index.to(self.device)

        out_expr_e = self.encoder512(expr_embedding)
        out_expr_e = F.leaky_relu(self.encoder768(out_expr_e))
        b_e = self.biobert_embedding(gene_pair_index)
        input_ = torch.add(out_expr_e, torch.add(b_e, p_e))
        out = self.transformer_encoder(input_)
        out = self.flatten(out)

        out = self.linear1024(out)
        out = self.dropout(out)
        out = self.actf(out)

        r = out.unsqueeze(1)
        r = self.pool(r)
        r = r.squeeze(1)

        out = self.linear512(out)
        out = self.dropout(out)
        out = self.actf(out)

        out = self.linear256(out) + r
        out = self.dropout(out)
        out = self.actf(out)

        outs = self.linear2(out)
        outs = nn.Sigmoid()(outs)

        return outs
''')
    log("  Patched model.py (nn.Linear(2*embed_size, 1024))")

    # --- Patch main.py ---
    # Fix: parameterize biovect filename based on embed_size
    main_py = os.path.join(SCGREAT_DIR, 'main.py')
    with open(main_py, 'w') as f:
        f.write('''import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from dataset import Dataset
from model import scGREAT
from train_val import train, validate


def main(data_dir, args):
    expression_data_path = data_dir + '/BL--ExpressionData.csv'
    # PATCHED: use embed_size to find biovect file
    biovect_e_path = data_dir + f'/biovect{args.embed_size}.npy'
    train_data_path = data_dir + '/Train_set.csv'
    val_data_path = data_dir + '/Validation_set.csv'
    test_data_path = data_dir + '/Test_set.csv'
    expression_data = np.array(pd.read_csv(expression_data_path, index_col=0, header=0))

    # Data Preprocessing
    standard = StandardScaler()
    scaled_df = standard.fit_transform(expression_data.T)
    expression_data = scaled_df.T
    expression_data_shape = expression_data.shape

    train_dataset = Dataset(train_data_path, expression_data)
    val_dataset = Dataset(val_data_path, expression_data)
    test_dataset = Dataset(test_data_path, expression_data)

    Batch_size = args.batch_size
    Embed_size = args.embed_size
    Num_layers = args.num_layers
    Num_head = args.num_head
    LR = args.lr
    EPOCHS = args.epochs
    step_size = args.step_size
    gamma = args.gamma

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=Batch_size,
                                               shuffle=True,
                                               drop_last=False,
                                               num_workers=0)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=Batch_size,
                                             shuffle=True,
                                             drop_last=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=Batch_size,
                                              shuffle=True,
                                              drop_last=False)

    print(f'biovect path: {biovect_e_path}')
    T = scGREAT(expression_data_shape, Embed_size, Num_layers, Num_head, biovect_e_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    T = T.to(device)
    optimizer = torch.optim.Adam(T.parameters(), lr=LR)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    loss_func = nn.BCELoss()

    best_val_auc = 0
    best_test_auc = 0
    best_test_aupr = 0

    for epoch in range(1, EPOCHS + 1):
        train(T, train_loader, loss_func, optimizer, epoch, scheduler, args)
        AUC_val, AUPR_val = validate(T, val_loader, loss_func)
        print('-' * 100)
        print('| end of epoch {:3d} |valid AUROC {:8.3f} | valid AUPRC {:8.3f}'.format(epoch, AUC_val, AUPR_val))
        print('-' * 100)
        AUC_test, AUPR_test = validate(T, test_loader, loss_func)
        print('| end of epoch {:3d} |test  AUROC {:8.3f} | test  AUPRC {:8.3f}'.format(epoch, AUC_test, AUPR_test))
        print('-' * 100)

        if AUC_val > best_val_auc:
            best_val_auc = AUC_val
            best_test_auc = AUC_test
            best_test_aupr = AUPR_test

        if AUC_val < 0.501:
            print("AUC_val<0.501 !!")
            break

    print('=' * 100)
    print(f'BEST test AUROC: {best_test_auc:.4f} | BEST test AUPRC: {best_test_aupr:.4f}')
    print('=' * 100)
    return best_test_auc, best_test_aupr
''')
    log("  Patched main.py (biovect{embed_size}.npy, best metric tracking)")

    # --- Patch demo.py ---
    # Fix: add --data_dir argument, support multiple runs
    demo_py = os.path.join(SCGREAT_DIR, 'demo.py')
    with open(demo_py, 'w') as f:
        f.write('''from main import main
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True, help='Path to experiment data directory')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--embed_size', type=int, default=768, help='Embedding size')
parser.add_argument('--num_layers', type=int, default=2, help='Number of layers')
parser.add_argument('--num_head', type=int, default=4, help='Number of attention heads')
parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate')
parser.add_argument('--epochs', type=int, default=80, help='Number of epochs')
parser.add_argument('--step_size', type=int, default=10, help='Step size for LR scheduler')
parser.add_argument('--gamma', type=float, default=0.999, help='Gamma for LR scheduler')
parser.add_argument('--scheduler_flag', type=bool, default=True, help='Enable/disable scheduler')
parser.add_argument('--n_runs', type=int, default=5, help='Number of runs for averaging')

args = parser.parse_args()

print(f'data_dir: {args.data_dir}')
print(f'embed_size: {args.embed_size}')

import numpy as np

all_auc = []
all_aupr = []
for run in range(args.n_runs):
    print(f'\\n{"="*60}')
    print(f'Run {run+1}/{args.n_runs}')
    print(f'{"="*60}')
    auc, aupr = main(args.data_dir, args)
    all_auc.append(auc)
    all_aupr.append(aupr)

print(f'\\n{"="*80}')
print(f'Final Results ({args.n_runs} runs):')
print(f'  AUROC: {np.mean(all_auc):.4f} +/- {np.std(all_auc):.4f}')
print(f'  AUPRC: {np.mean(all_aupr):.4f} +/- {np.std(all_aupr):.4f}')
print(f'{"="*80}')
''')
    log("  Patched demo.py (--data_dir, --n_runs, mean+std reporting)")

    log("All patches applied.")


# =============================================================
# Step 2: Load embeddings
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
    """Build gene symbol -> Entrez ID mapping from NCBI"""
    mapping_file = os.path.join(OUTPUT_DIR, 'gene_symbol_to_entrez.json')
    if os.path.exists(mapping_file):
        log("Loading cached gene symbol -> Entrez mapping...")
        with open(mapping_file) as f:
            return json.load(f)

    alt_path = '/root/autodl-tmp/embedding_benchmark/gene_symbol_to_entrez.json'
    if os.path.exists(alt_path):
        log("Copying gene mapping from embedding_benchmark...")
        shutil.copy2(alt_path, mapping_file)
        with open(mapping_file) as f:
            return json.load(f)

    log("Downloading NCBI Homo_sapiens.gene_info.gz ...")
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
    log(f"  Built mapping: {len(symbol_to_entrez)} entries")
    return symbol_to_entrez


# =============================================================
# Step 3: Generate biovect.npy for each embedding x dataset
# =============================================================
def get_dataset_genes(dataset_name):
    """Read gene list from scGREAT dataset's Target.csv (columns: ,Gene,index)"""
    target_path = os.path.join(SCGREAT_DIR, dataset_name, 'Target.csv')
    if not os.path.exists(target_path):
        return None
    df = pd.read_csv(target_path)
    genes = df['Gene'].tolist()
    return genes


def create_biovect_from_checkpoint(emb_matrix, vocab, dataset_genes, emb_name, dataset_name):
    """
    Create biovect.npy for checkpoint-based embeddings.
    Note: scGREAT model does np.load(path)[1:], so we prepend a dummy row 0.
    """
    n_genes = len(dataset_genes)
    emb_dim = emb_matrix.shape[1]
    # Row 0 = dummy (will be sliced off by model.py's [1:])
    # Rows 1..n_genes = actual gene embeddings
    biovect = np.zeros((n_genes + 1, emb_dim), dtype=np.float32)

    mapped = 0
    missing = []
    for i, gene in enumerate(dataset_genes):
        if gene in vocab:
            idx = vocab[gene]
            biovect[i + 1] = emb_matrix[idx]
            mapped += 1
        else:
            missing.append(gene)

    coverage = mapped / n_genes * 100
    log(f"  {emb_name} x {dataset_name}: {mapped}/{n_genes} genes mapped ({coverage:.1f}%)")
    if missing:
        log(f"    Missing ({len(missing)}): {missing[:10]}{'...' if len(missing) > 10 else ''}")

    return biovect, mapped, n_genes


def create_biovect_from_geneformer(gf_emb, gf_genelist, symbol_to_entrez, dataset_genes, dataset_name):
    """
    Create biovect.npy for Geneformer embedding.
    Prepends a dummy row 0 for scGREAT's [1:] slicing.
    """
    n_genes = len(dataset_genes)
    emb_dim = gf_emb.shape[1]
    biovect = np.zeros((n_genes + 1, emb_dim), dtype=np.float32)

    entrez_to_gf = {eid: i for i, eid in enumerate(gf_genelist)}

    mapped = 0
    missing = []
    for i, gene in enumerate(dataset_genes):
        if gene in symbol_to_entrez:
            entrez_id = symbol_to_entrez[gene]
            if entrez_id in entrez_to_gf:
                gf_idx = entrez_to_gf[entrez_id]
                biovect[i + 1] = gf_emb[gf_idx]
                mapped += 1
            else:
                missing.append(gene)
        else:
            missing.append(gene)

    coverage = mapped / n_genes * 100
    log(f"  GF-12L95M x {dataset_name}: {mapped}/{n_genes} genes mapped ({coverage:.1f}%)")
    if missing:
        log(f"    Missing ({len(missing)}): {missing[:10]}{'...' if len(missing) > 10 else ''}")

    return biovect, mapped, n_genes


def setup_experiment_dir(emb_name, dataset_name, biovect, emb_dim):
    """
    Create experiment directory with symlinks to original data + custom biovect.
    Structure: OUTPUT_DIR / emb_name / dataset_name /
    """
    exp_dir = os.path.join(OUTPUT_DIR, emb_name, dataset_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Save biovect with embed_dim in filename (matches patched main.py)
    biovect_path = os.path.join(exp_dir, f'biovect{emb_dim}.npy')
    np.save(biovect_path, biovect)
    log(f"    Saved {biovect_path} shape={biovect.shape}")

    # Symlink all other data files from original scGREAT dataset
    src_dir = os.path.join(SCGREAT_DIR, dataset_name)
    if not os.path.exists(src_dir):
        log(f"    WARNING: source dir {src_dir} not found")
        return exp_dir

    for fname in os.listdir(src_dir):
        if fname.startswith('biovect'):
            continue
        src = os.path.join(src_dir, fname)
        dst = os.path.join(exp_dir, fname)
        if os.path.exists(dst) or os.path.islink(dst):
            os.remove(dst)
        if os.path.isfile(src):
            os.symlink(src, dst)

    return exp_dir


# =============================================================
# Step 4: Generate run scripts
# =============================================================
def generate_run_script(all_datasets):
    """Generate a single shell script to run all experiments."""
    script_path = os.path.join(OUTPUT_DIR, 'run_all.sh')

    lines = [
        '#!/bin/bash',
        f'# scGREAT GRN benchmark - generated {datetime.now()}',
        f'# Experiments: {len(EMBEDDINGS) + 1} embeddings x {len(all_datasets)} datasets',
        '',
        f'SCGREAT_DIR="{SCGREAT_DIR}"',
        f'OUTPUT_DIR="{OUTPUT_DIR}"',
        '',
        'pip install torch scikit-learn pandas numpy 2>/dev/null',
        '',
        'RESULTS_FILE="$OUTPUT_DIR/results_summary.txt"',
        'echo "scGREAT GRN Benchmark Results" > "$RESULTS_FILE"',
        'echo "==============================" >> "$RESULTS_FILE"',
        'echo "" >> "$RESULTS_FILE"',
        '',
    ]

    # All embedding configs: 4 custom + 1 BioBERT original
    emb_configs = []
    for emb_name, emb_cfg in EMBEDDINGS.items():
        if emb_cfg['type'] == 'checkpoint':
            emb_dim = 512 if emb_name == 'scGPT_human' else 256
        else:
            emb_dim = 512
        emb_configs.append((emb_name, emb_dim))
    emb_configs.append(('BioBERT_original', 768))

    for emb_name, emb_dim in emb_configs:
        for ds in all_datasets:
            exp_dir = os.path.join(OUTPUT_DIR, emb_name, ds)
            log_path = os.path.join(exp_dir, 'train.log')

            lines.append(f'echo ""')
            lines.append(f'echo "=========================================="')
            lines.append(f'echo "  {emb_name} x {ds} (dim={emb_dim})"')
            lines.append(f'echo "=========================================="')

            # num_head must divide embed_size evenly
            # 256: 4 heads OK (64 per head)
            # 512: 4 heads OK (128 per head)
            # 768: 4 heads OK (192 per head)
            num_head = 4

            lines.append(
                f'cd "$SCGREAT_DIR" && python demo.py '
                f'--data_dir {exp_dir} '
                f'--embed_size {emb_dim} '
                f'--num_layers 2 '
                f'--num_head {num_head} '
                f'--epochs 50 '
                f'--batch_size 64 '
                f'--lr 1e-4 '
                f'--n_runs 5 '
                f'2>&1 | tee {log_path}'
            )
            lines.append(f'echo "{emb_name} x {ds}: done" >> "$RESULTS_FILE"')
            lines.append(f'tail -3 {log_path} >> "$RESULTS_FILE"')
            lines.append('')

    lines.append('echo ""')
    lines.append('echo "All experiments complete!"')
    lines.append('echo "Results summary: $RESULTS_FILE"')

    with open(script_path, 'w') as f:
        f.write('\n'.join(lines))
    os.chmod(script_path, 0o755)
    log(f"Generated run script: {script_path}")
    return script_path


def setup_biobert_baseline(datasets):
    """Symlink the original scGREAT biovect768 as 'BioBERT_original' baseline."""
    for ds in datasets:
        exp_dir = os.path.join(OUTPUT_DIR, 'BioBERT_original', ds)
        os.makedirs(exp_dir, exist_ok=True)
        src_dir = os.path.join(SCGREAT_DIR, ds)
        if not os.path.exists(src_dir):
            continue
        for fname in os.listdir(src_dir):
            src = os.path.join(src_dir, fname)
            dst = os.path.join(exp_dir, fname)
            if os.path.exists(dst) or os.path.islink(dst):
                os.remove(dst)
            if os.path.isfile(src):
                os.symlink(src, dst)
        log(f"  BioBERT_original x {ds}: using original biovect768.npy")


# =============================================================
# Main
# =============================================================
def main():
    log("=" * 60)
    log("scGREAT GRN Benchmark Setup")
    log("=" * 60)

    # 1. Clone + patch
    clone_scgreat()
    patch_scgreat()

    # 2. Check available datasets
    available_datasets = []
    for ds in HUMAN_DATASETS + MOUSE_DATASETS:
        ds_path = os.path.join(SCGREAT_DIR, ds)
        if os.path.exists(ds_path):
            available_datasets.append(ds)
            genes = get_dataset_genes(ds)
            n_genes = len(genes) if genes else 0
            log(f"  Dataset {ds}: {n_genes} genes")
        else:
            log(f"  Dataset {ds}: NOT FOUND")

    human_avail = [ds for ds in HUMAN_DATASETS if ds in available_datasets]
    mouse_avail = [ds for ds in MOUSE_DATASETS if ds in available_datasets]
    log(f"\nHuman datasets: {human_avail}")
    log(f"Mouse datasets: {mouse_avail}")
    log("Note: Mouse datasets may have low gene coverage with human embeddings\n")

    all_datasets = human_avail + mouse_avail

    # 3. Load vocab
    log("Loading vocab...")
    vocab = load_vocab()
    log(f"  Vocab size: {len(vocab)}")

    # 4. Load embeddings
    log("\nLoading embeddings...")
    loaded_embs = {}
    for emb_name, cfg in EMBEDDINGS.items():
        if cfg['type'] == 'checkpoint':
            log(f"  Loading {emb_name} from {cfg['path']}...")
            emb_matrix = load_checkpoint_embedding(cfg['path'], cfg['key'])
            loaded_embs[emb_name] = {
                'matrix': emb_matrix,
                'dim': emb_matrix.shape[1],
                'type': 'checkpoint',
            }
            log(f"    Shape: {emb_matrix.shape}")
        else:
            log(f"  Loading {emb_name} from {cfg['dir']}...")
            gf_emb, gf_genelist = load_gf_embedding(cfg['dir'])
            loaded_embs[emb_name] = {
                'matrix': gf_emb,
                'genelist': gf_genelist,
                'dim': gf_emb.shape[1],
                'type': 'geneformer',
            }
            log(f"    Shape: {gf_emb.shape}, genes: {len(gf_genelist)}")

    # 5. Build symbol->entrez for Geneformer
    symbol_to_entrez = build_symbol_to_entrez()

    # 6. Generate biovect files
    log("\n" + "=" * 60)
    log("Generating embedding files for scGREAT...")
    log("=" * 60)

    coverage_report = []
    for ds in all_datasets:
        dataset_genes = get_dataset_genes(ds)
        if dataset_genes is None:
            log(f"  Skipping {ds}: no Target.csv found")
            continue

        log(f"\nDataset: {ds} ({len(dataset_genes)} genes)")
        for emb_name, emb_data in loaded_embs.items():
            if emb_data['type'] == 'checkpoint':
                biovect, mapped, total = create_biovect_from_checkpoint(
                    emb_data['matrix'], vocab, dataset_genes, emb_name, ds
                )
            else:
                biovect, mapped, total = create_biovect_from_geneformer(
                    emb_data['matrix'], emb_data['genelist'],
                    symbol_to_entrez, dataset_genes, ds
                )

            emb_dim = emb_data['dim']
            setup_experiment_dir(emb_name, ds, biovect, emb_dim)
            coverage_report.append({
                'embedding': emb_name,
                'dataset': ds,
                'mapped': mapped,
                'total': total,
                'coverage': mapped / total * 100,
                'dim': emb_dim,
            })

    # 7. BioBERT baseline
    log("\nSetting up BioBERT original baseline...")
    setup_biobert_baseline(all_datasets)

    # 8. Coverage summary
    log("\n" + "=" * 60)
    log("Gene Coverage Summary")
    log("=" * 60)
    log(f"{'Embedding':<20} {'Dataset':<15} {'Mapped':>8} {'Total':>8} {'Coverage':>10} {'Dim':>5}")
    log("-" * 70)
    for r in coverage_report:
        log(f"{r['embedding']:<20} {r['dataset']:<15} {r['mapped']:>8} {r['total']:>8} {r['coverage']:>9.1f}% {r['dim']:>5}")

    # 9. Run script
    log("\n" + "=" * 60)
    log("Generating run script...")
    log("=" * 60)
    run_script = generate_run_script(all_datasets)

    log("\n" + "=" * 60)
    log("Setup complete!")
    log(f"Experiment dirs: {OUTPUT_DIR}/<embedding>/<dataset>/")
    log(f"Run all experiments: bash {run_script}")
    log("=" * 60)

    # 10. Optionally run
    if '--run' in sys.argv:
        log("\nStarting experiments...")
        subprocess.run(['bash', run_script])


if __name__ == '__main__':
    main()
