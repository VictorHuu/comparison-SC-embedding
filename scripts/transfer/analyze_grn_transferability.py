#!/usr/bin/env python3
"""
Focused GRN transferability experiment (minimal but convincing).

Outputs:
1) raw seed results CSV
2) summary CSV (mean/std + paired deltas)
3) markdown report
"""

import os
import csv
import json
from statistics import mean, pstdev

# Heavy scientific imports are loaded lazily in main(),
# so this script can still write fallback deliverables in minimal environments.

BASE_DIR = '/bigdata2/hyt/projects/scbenchmark'
SCGREAT_DIR = '/bigdata2/hyt/projects/scGREAT'
OUT_DIR = 'transfer'
os.makedirs(OUT_DIR, exist_ok=True)

RAW_CSV = os.path.join(OUT_DIR, 'transfer_seed_results.csv')
SUMMARY_CSV = os.path.join(OUT_DIR, 'transfer_seed_summary.csv')
REPORT_MD = os.path.join(OUT_DIR, 'transferability_report.md')

EMBEDDINGS = {
    'difference_v3': {'path': f'{BASE_DIR}/save_pretrain/difference_aligned_v3/best_model.pt', 'key': 'module.embedding.weight'},
    'baseline': {'path': f'{BASE_DIR}/save_pretrain/baseline/best_model.pt', 'key': 'module.embedding.weight'},
    'scGPT_human': {'path': f'{BASE_DIR}/save_pretrain/scGPT_human/best_model.pt', 'key': 'encoder.embedding.weight'},
}

DATASETS = ['hESC500', 'mESC500']
SEEDS = [0, 1, 2, 3, 4]


def _exists_required():
    return os.path.isdir(SCGREAT_DIR) and os.path.exists(f'{BASE_DIR}/vocab.json')


def load_vocab():
    with open(f'{BASE_DIR}/vocab.json') as f:
        return json.load(f)


def load_embedding(path, key):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if key in ckpt:
        return ckpt[key].detach().cpu().numpy()
    for nk in ['state_dict', 'model_state_dict', 'model']:
        if nk in ckpt and isinstance(ckpt[nk], dict) and key in ckpt[nk]:
            return ckpt[nk][key].detach().cpu().numpy()
    raise KeyError(f'missing key: {key}')


def read_target_genes(ds):
    p = os.path.join(SCGREAT_DIR, ds, 'Target.csv')
    genes = []
    with open(p, newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            genes.append(row['Gene'])
    return genes


def read_split(ds, split):
    p = os.path.join(SCGREAT_DIR, ds, f'{split}.csv')
    tf, tgt, y = [], [], []
    with open(p, newline='') as f:
        r = csv.reader(f)
        header = next(r)
        for row in r:
            tf.append(int(row[1]))
            tgt.append(int(row[2]))
            y.append(int(row[3]))
    return np.array(tf), np.array(tgt), np.array(y)


def build_lookup(emb, vocab, genes):
    d = emb.shape[1]
    lookup = np.zeros((len(genes), d), dtype=np.float32)
    mapped = 0
    for i, g in enumerate(genes):
        if g in vocab:
            lookup[i] = emb[vocab[g]]
            mapped += 1
    return lookup, mapped


def pair_features(lookup, tf, tgt):
    a = lookup[tf]
    b = lookup[tgt]
    had = a * b
    cos = np.sum(a * b, axis=1, keepdims=True) / ((np.linalg.norm(a, axis=1, keepdims=True) + 1e-8) * (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8))
    l2 = np.linalg.norm(a - b, axis=1, keepdims=True)
    return np.concatenate([a, b, had, cos, l2], axis=1)


def fit_eval(Xtr, ytr, Xte, yte, clf, seed):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xte = scaler.transform(Xte)
    if clf == 'lr':
        m = LogisticRegression(max_iter=1000, n_jobs=1, C=1.0, random_state=seed)
    else:
        m = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, early_stopping=True, random_state=seed)
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1] if hasattr(m, 'predict_proba') else m.decision_function(Xte)
    return roc_auc_score(yte, p), average_precision_score(yte, p), p


def bootstrap_train_resample(X, y, seed):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=len(y))
    return X[idx], y[idx]


def write_empty_outputs(reason):
    with open(RAW_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['train_dataset','test_dataset','embedding','clf','seed','coverage_train','coverage_test','auroc','auprc'])
    with open(SUMMARY_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['scope','train_dataset','test_dataset','embedding','clf','mean_auroc','std_auroc','mean_auprc','std_auprc','delta_vs_baseline_auroc','delta_vs_baseline_auprc','delta_vs_scgpt_auroc','delta_vs_scgpt_auprc'])
    with open(REPORT_MD, 'w') as f:
        f.write('# GRN Transferability Report\n\n')
        f.write('## 实验目的\n\n验证 difference_v3 的跨数据集可迁移性是否稳定。\n\n')
        f.write('## 实验设置\n\n')
        f.write('- 目标设置：hESC500↔mESC500 transfer，embedding={difference_v3, baseline, scGPT_human}，clf={LR, MLP}，seeds=5。\n\n')
        f.write('## 主表A：MLP mean±std\n\n')
        f.write('（当前环境未产出，见 `transfer_seed_summary.csv`）\n\n')
        f.write('## 主表B：LR corrected repeated runs\n\n')
        f.write('（当前环境未产出，见 `transfer_seed_summary.csv`）\n\n')
        f.write('## 辅助表：strict common-gene evaluation\n\n')
        f.write('（当前环境未产出，见 `transfer_seed_summary.csv`）\n\n')
        f.write('## 结论\n\n当前证据不足，尚不能判断 transfer gain 是否稳定。\n\n')
        f.write('## Limitations\n\n')
        f.write(f'- 本次环境无法访问必需数据目录：`{reason}`，未执行数值实验。\n')
        f.write('\n## 代码变更文件\n\n')
        f.write('- `analyze_grn_transferability.py`：重复 seed、paired delta、strict common-gene 与报告导出主脚本。\n')


def aggregate(rows):
    # key -> list rows
    grp = {}
    for r in rows:
        k = (r['train_dataset'], r['test_dataset'], r['embedding'], r['clf'])
        grp.setdefault(k, []).append(r)
    out = []
    for k, vals in grp.items():
        au = [v['auroc'] for v in vals]
        ap = [v['auprc'] for v in vals]
        out.append({
            'train_dataset': k[0], 'test_dataset': k[1], 'embedding': k[2], 'clf': k[3],
            'mean_auroc': mean(au), 'std_auroc': pstdev(au) if len(au) > 1 else 0.0,
            'mean_auprc': mean(ap), 'std_auprc': pstdev(ap) if len(ap) > 1 else 0.0,
        })
    return out


def bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=0):
    if len(values) == 0:
        return '', ''
    rng = np.random.default_rng(seed)
    vals = np.array(values, dtype=float)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(vals), size=len(vals))
        boots.append(float(np.mean(vals[idx])))
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return lo, hi


def paired_mlp_deltas(rows):
    """Return paired mean differences for MLP and optional bootstrap CI."""
    # key: (train,test,seed,embedding)->(auroc,auprc)
    rec = {}
    for r in rows:
        if r['clf'] != 'mlp':
            continue
        rec[(r['train_dataset'], r['test_dataset'], r['seed'], r['embedding'])] = (r['auroc'], r['auprc'])

    out = []
    for train_ds, test_ds in [('hESC500', 'mESC500'), ('mESC500', 'hESC500')]:
        for ref in ['baseline', 'scGPT_human']:
            d_auroc, d_auprc = [], []
            for s in SEEDS:
                a = rec.get((train_ds, test_ds, s, 'difference_v3'))
                b = rec.get((train_ds, test_ds, s, ref))
                if a is None or b is None:
                    continue
                d_auroc.append(a[0] - b[0])
                d_auprc.append(a[1] - b[1])
            if not d_auroc:
                continue
            au_lo, au_hi = bootstrap_ci(d_auroc, seed=42)
            ap_lo, ap_hi = bootstrap_ci(d_auprc, seed=42)
            out.append({
                'train_dataset': train_ds,
                'test_dataset': test_ds,
                'ref_embedding': ref,
                'paired_mean_diff_auroc': float(np.mean(d_auroc)),
                'paired_mean_diff_auprc': float(np.mean(d_auprc)),
                'boot95_ci_auroc': f'[{au_lo:.4f}, {au_hi:.4f}]',
                'boot95_ci_auprc': f'[{ap_lo:.4f}, {ap_hi:.4f}]',
            })
    return out


def add_pairwise_delta(summary_rows):
    by = {(r['train_dataset'], r['test_dataset'], r['clf'], r['embedding']): r for r in summary_rows}
    for r in summary_rows:
        b = by.get((r['train_dataset'], r['test_dataset'], r['clf'], 'baseline'))
        s = by.get((r['train_dataset'], r['test_dataset'], r['clf'], 'scGPT_human'))
        r['delta_vs_baseline_auroc'] = r['mean_auroc'] - b['mean_auroc'] if b else ''
        r['delta_vs_baseline_auprc'] = r['mean_auprc'] - b['mean_auprc'] if b else ''
        r['delta_vs_scgpt_auroc'] = r['mean_auroc'] - s['mean_auroc'] if s else ''
        r['delta_vs_scgpt_auprc'] = r['mean_auprc'] - s['mean_auprc'] if s else ''


def strict_common_gene_lr(vocab, emb_map, dataset_data):
    # strict common genes among three embeddings and dataset genes
    out = []
    for train_ds, test_ds in [('hESC500', 'mESC500'), ('mESC500', 'hESC500'), ('hESC500', 'hESC500'), ('mESC500', 'mESC500')]:
        train_genes = dataset_data[train_ds]['genes']
        test_genes = dataset_data[test_ds]['genes']
        shared = set(train_genes) & set(test_genes)
        for emb_name in ['difference_v3', 'baseline', 'scGPT_human']:
            # embedding coverage shared constraint
            shared2 = sorted([g for g in shared if g in vocab])
            if len(shared2) < 10:
                continue

            # original dataset index -> compact shared-gene index (0..len(shared2)-1)
            train_old_to_local = {}
            test_old_to_local = {}
            train_gene_to_old = {g: i for i, g in enumerate(train_genes)}
            test_gene_to_old = {g: i for i, g in enumerate(test_genes)}
            for local_i, g in enumerate(shared2):
                if g in train_gene_to_old:
                    train_old_to_local[train_gene_to_old[g]] = local_i
                if g in test_gene_to_old:
                    test_old_to_local[test_gene_to_old[g]] = local_i

            # remap pairs to compact indices
            def filt_pairs(tf, tgt, y, old_to_local):
                keep_tf, keep_tg, keep_y = [], [], []
                for a, b, l in zip(tf, tgt, y):
                    if a in old_to_local and b in old_to_local:
                        keep_tf.append(old_to_local[a])
                        keep_tg.append(old_to_local[b])
                        keep_y.append(l)
                return np.array(keep_tf), np.array(keep_tg), np.array(keep_y)

            ttf, ttg, ty = filt_pairs(
                dataset_data[train_ds]['train_tf'],
                dataset_data[train_ds]['train_tgt'],
                dataset_data[train_ds]['train_y'],
                train_old_to_local,
            )
            ef, eg, ey = filt_pairs(
                dataset_data[test_ds]['test_tf'],
                dataset_data[test_ds]['test_tgt'],
                dataset_data[test_ds]['test_y'],
                test_old_to_local,
            )
            if len(ty) < 20 or len(ey) < 20:
                continue

            emb = emb_map[emb_name]
            # both train/test use same compact shared-gene order
            lookup_train = np.stack([emb[vocab[g]] for g in shared2])
            lookup_test = np.stack([emb[vocab[g]] for g in shared2])
            Xtr = pair_features(lookup_train, ttf, ttg)
            Xte = pair_features(lookup_test, ef, eg)
            au, ap, _ = fit_eval(Xtr, ty, Xte, ey, 'lr', seed=0)
            out.append({'scope':'strict_common_gene','train_dataset':train_ds,'test_dataset':test_ds,'embedding':emb_name,'clf':'lr','mean_auroc':au,'std_auroc':0.0,'mean_auprc':ap,'std_auprc':0.0,'delta_vs_baseline_auroc':'','delta_vs_baseline_auprc':'','delta_vs_scgpt_auroc':'','delta_vs_scgpt_auprc':''})
    return out


def write_report(summary_rows, strict_rows, mlp_delta_rows):
    # compact markdown
    with open(REPORT_MD, 'w') as f:
        f.write('# GRN Transferability Report\n\n')
        f.write('## 实验目的\n\n')
        f.write('验证 difference_v3 的 cross-dataset 优势是否稳定，并区分 transferability 与 in-domain fitting。\n\n')
        f.write('## 实验设置\n\n')
        f.write('- 数据集：hESC500, mESC500\n- 嵌入：difference_v3 / baseline / scGPT_human\n- 分类器：LR, MLP\n- seeds：0..4\n\n')

        f.write('## 主表A：MLP mean±std\n\n')
        f.write('| train->test | emb | clf | AUROC | AUPRC | ΔAUROC vs baseline | ΔAUROC vs scGPT |\n')
        f.write('|---|---|---:|---:|---:|---:|---:|\n')
        for r in summary_rows:
            if r['train_dataset'] == r['test_dataset'] or r['clf'] != 'mlp':
                continue
            au = f"{r['mean_auroc']:.4f}±{r['std_auroc']:.4f}"
            ap = f"{r['mean_auprc']:.4f}±{r['std_auprc']:.4f}"
            db = '' if r['delta_vs_baseline_auroc']=='' else f"{r['delta_vs_baseline_auroc']:.4f}"
            ds = '' if r['delta_vs_scgpt_auroc']=='' else f"{r['delta_vs_scgpt_auroc']:.4f}"
            f.write(f"| {r['train_dataset']}->{r['test_dataset']} | {r['embedding']} | {r['clf']} | {au} | {ap} | {db} | {ds} |\n")

        f.write('\n## 主表B：LR corrected repeated runs\n\n')
        f.write('| train->test | emb | AUROC | AUPRC |\n')
        f.write('|---|---|---:|---:|\n')
        for r in summary_rows:
            if r['train_dataset'] == r['test_dataset'] or r['clf'] != 'lr':
                continue
            au = f"{r['mean_auroc']:.4f}±{r['std_auroc']:.4f}"
            ap = f"{r['mean_auprc']:.4f}±{r['std_auprc']:.4f}"
            f.write(f"| {r['train_dataset']}->{r['test_dataset']} | {r['embedding']} | {au} | {ap} |\n")

        f.write('\n## 辅助表：strict common-gene evaluation (LR)\n\n')
        f.write('| train->test | emb | AUROC | AUPRC |\n')
        f.write('|---|---|---:|---:|\n')
        for r in strict_rows:
            f.write(f"| {r['train_dataset']}->{r['test_dataset']} | {r['embedding']} | {r['mean_auroc']:.4f} | {r['mean_auprc']:.4f} |\n")

        f.write('\n## MLP paired differences (difference_v3 vs refs)\n\n')
        f.write('| train->test | ref | ΔAUROC | 95% CI | ΔAUPRC | 95% CI |\n')
        f.write('|---|---|---:|---:|---:|---:|\n')
        for r in mlp_delta_rows:
            f.write(f"| {r['train_dataset']}->{r['test_dataset']} | {r['ref_embedding']} | {r['paired_mean_diff_auroc']:.4f} | {r['boot95_ci_auroc']} | {r['paired_mean_diff_auprc']:.4f} | {r['boot95_ci_auprc']} |\n")

        f.write('\n## 结论\n\n')
        # controlled wording requested by user
        f.write('- mESC500->hESC500：若 LR+MLP 与 paired differences均为正，判定为 stronger transfer advantage。\n')
        f.write('- hESC500->mESC500：若对 baseline 的增益较小/跨0，但对 scGPT_human 稳定为正，判定为 weaker over baseline, stronger over scGPT_human。\n')
        f.write('- 不输出 overall superiority，仅在结果支持范围内陈述。\n\n')
        f.write('## Limitations\n\n')
        f.write('- 目前仅 2 个数据集，跨域证据有限。\n- 未做更系统的不确定性估计（例如更大规模 bootstrap）。\n')
        f.write('\n## 代码变更文件\n\n')
        f.write('- `analyze_grn_transferability.py`：生成原始 seed 结果、summary、strict common-gene 结果与 markdown 报告。\n')


def main():
    if not _exists_required():
        write_empty_outputs('/root/autodl-tmp/scGREAT or /root/autodl-tmp/scbenchmark missing')
        return
    try:
        global np, torch, LogisticRegression, MLPClassifier, StandardScaler, roc_auc_score, average_precision_score
        import numpy as np
        import torch
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import roc_auc_score, average_precision_score
    except Exception as e:
        write_empty_outputs(f'missing python deps: {e}')
        return

    vocab = load_vocab()
    emb_map = {k: load_embedding(v['path'], v['key']) for k, v in EMBEDDINGS.items()}

    ds = {}
    for d in DATASETS:
        genes = read_target_genes(d)
        tr_tf, tr_tg, tr_y = read_split(d, 'Train_set')
        va_tf, va_tg, va_y = read_split(d, 'Validation_set')
        te_tf, te_tg, te_y = read_split(d, 'Test_set')
        ds[d] = {
            'genes': genes,
            'train_tf': np.concatenate([tr_tf, va_tf]),
            'train_tgt': np.concatenate([tr_tg, va_tg]),
            'train_y': np.concatenate([tr_y, va_y]),
            'test_tf': te_tf,
            'test_tgt': te_tg,
            'test_y': te_y,
        }

    rows = []
    for train_ds, test_ds in [('hESC500', 'mESC500'), ('mESC500', 'hESC500')]:
        for emb_name in ['difference_v3', 'baseline', 'scGPT_human']:
            emb = emb_map[emb_name]
            lk_tr, mtr = build_lookup(emb, vocab, ds[train_ds]['genes'])
            lk_te, mte = build_lookup(emb, vocab, ds[test_ds]['genes'])
            cov_tr = f'{mtr}/{len(ds[train_ds]["genes"])}'
            cov_te = f'{mte}/{len(ds[test_ds]["genes"])}'
            Xtr = pair_features(lk_tr, ds[train_ds]['train_tf'], ds[train_ds]['train_tgt'])
            Xte = pair_features(lk_te, ds[test_ds]['test_tf'], ds[test_ds]['test_tgt'])

            for clf in ['lr', 'mlp']:
                for seed in SEEDS:
                    Xtr_seed, ytr_seed = Xtr, ds[train_ds]['train_y']
                    # IMPORTANT: introduce real seed variability for LR branch.
                    if clf == 'lr':
                        Xtr_seed, ytr_seed = bootstrap_train_resample(Xtr, ds[train_ds]['train_y'], seed=seed)
                    au, ap, s = fit_eval(Xtr_seed, ytr_seed, Xte, ds[test_ds]['test_y'], clf, seed)
                    rows.append({
                        'train_dataset': train_ds,
                        'test_dataset': test_ds,
                        'embedding': emb_name,
                        'clf': clf,
                        'seed': seed,
                        'coverage_train': cov_tr,
                        'coverage_test': cov_te,
                        'auroc': float(au),
                        'auprc': float(ap),
                    })

    # raw csv
    with open(RAW_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['train_dataset','test_dataset','embedding','clf','seed','coverage_train','coverage_test','auroc','auprc'])
        w.writeheader()
        w.writerows(rows)

    summary = aggregate(rows)
    add_pairwise_delta(summary)
    mlp_delta_rows = paired_mlp_deltas(rows)

    strict_rows = strict_common_gene_lr(vocab, emb_map, ds)
    combined = []
    for r in summary:
        x = {'scope':'transfer_seed', **r}
        combined.append(x)
    combined.extend(strict_rows)

    with open(SUMMARY_CSV, 'w', newline='') as f:
        fields = ['scope','train_dataset','test_dataset','embedding','clf','mean_auroc','std_auroc','mean_auprc','std_auprc','delta_vs_baseline_auroc','delta_vs_baseline_auprc','delta_vs_scgpt_auroc','delta_vs_scgpt_auprc']
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in combined:
            w.writerow(r)

    write_report(summary, strict_rows, mlp_delta_rows)


if __name__ == '__main__':
    main()
