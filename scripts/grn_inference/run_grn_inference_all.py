#!/usr/bin/env python3
"""Run fixed-registry GeneLink-style supervised GRN link prediction."""
from __future__ import annotations

import argparse
from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("grn_inference_benchmark", HERE / "grn_inference_benchmark.py")
bm = importlib.util.module_from_spec(spec); spec.loader.exec_module(bm)


def parse_list(s: str):
    return [x.strip() for x in s.split(',') if x.strip()]


def cap_edges_df(df: pd.DataFrame, max_edges: int, seed: int) -> pd.DataFrame:
    if not max_edges or len(df) <= max_edges:
        return df
    if 'label' not in df or df['label'].nunique() < 2:
        return df.sample(max_edges, random_state=seed)
    pos = df[df.label == 1]
    neg = df[df.label == 0]
    n_pos = min(len(pos), max(1, int(round(max_edges * len(pos) / len(df)))))
    n_neg = min(len(neg), max_edges - n_pos)
    out = pd.concat([pos.sample(n_pos, random_state=seed), neg.sample(n_neg, random_state=seed + 1)], ignore_index=True)
    return out.sample(frac=1.0, random_state=seed + 2).reset_index(drop=True)


def write_report(out: Path, missing: list[dict], datasets: pd.DataFrame):
    lines = [
        "# GRN Inference Report", "",
        "## Implementation plan", "",
        "- Reuse: fixed checkpoint embedding loading and pair-feature idea from `grn_embedding_only.py`; BEELINE/scGREAT raw edge construction and hard-negative cautions from `grn_beeline_full.py`; split/transfer diagnostics ideas from `transfer_v2`.",
        "- Created standalone scripts under `scripts/grn_inference/`; no existing GRN scripts are modified.",
        "- Data assumption: each dataset has `Target.csv` plus `Train_set.csv`, `Validation_set.csv`, `Test_set.csv`, or is reported missing. Splits may contain integer gene indices or gene-symbol TF/target columns.",
        "- Edge schema saved by this benchmark: `tf,target,label,split,dataset,negative_sampling,negative_ratio,seed`.",
        "- Metrics: AUROC, AUPRC (primary), Precision@K, Recall@K, F1@threshold, positive/negative score mean/median.",
        "- Leakage controls: negatives exclude every known positive edge across train/val/test; train/val/test negatives are non-overlapping; directed TF->target edges are preserved.", "",
        "## How this differs from old scripts", "",
        "- `grn_embedding_only.py`: embedding-only classifier over existing scGREAT labels; this script regenerates leak-free negatives and saves exact sampled edge tables.",
        "- `grn_beeline_full.py`: builds BEELINE-like datasets and hard negatives; this script standardizes multiple negative protocols and split modes as supervised link prediction.",
        "- `transfer_v2`: strong transfer diagnostics and topology controls; this script exposes GRN-specific edge sampling tables and AUPRC-first reporting.", "",
        "## Missing assets", "",
    ]
    if missing:
        for m in missing[:50]:
            lines.append(f"- {m.get('asset_name')}: {m.get('status')} ({m.get('notes')})")
    else:
        lines.append("- None detected.")
    lines += ["", "## Dataset discovery", "", datasets.to_markdown(index=False) if len(datasets) else "No runnable datasets discovered.", "", "## GeneLink/GNN note", "", "A GeneLink-like GAT/GNN would require graph construction over genes, node features, train-only adjacency, message passing without test-edge leakage, and edge decoders. This implementation intentionally starts with clean non-GNN link predictors."]
    (out / "grn_inference_report.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-dir', default='.')
    ap.add_argument('--out-dir', default='results/grn_inference')
    ap.add_argument('--datasets', default='auto')
    ap.add_argument('--embeddings', default='fixed')
    ap.add_argument('--models', default='lr,mlp')
    ap.add_argument('--split-modes', default='edge_holdout,cross_dataset_transfer,topology_matched_transfer')
    ap.add_argument('--negative-sampling', default='random_negative,degree_matched_negative')
    ap.add_argument('--negative-ratios', default='1,5')
    ap.add_argument('--seeds', default='0,1,2,3,4')
    ap.add_argument('--max-edges', type=int, default=0)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--feature-modes', default='embedding_pair')
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    edge_dir = out / 'sampled_edges'; edge_dir.mkdir(exist_ok=True)

    datasets, missing = bm.discover_grn_datasets(args.base_dir)
    ds_df = pd.DataFrame(datasets); miss_df = pd.DataFrame(missing)
    if args.datasets != 'auto' and len(ds_df):
        wanted = set(parse_list(args.datasets)); ds_df = ds_df[ds_df.dataset.isin(wanted) | ds_df.dataset_dir.isin(wanted)]
    emb_names = list(bm.EMBEDDINGS) if args.embeddings == 'fixed' else parse_list(args.embeddings)
    emb_names = [e for e in emb_names if e in bm.EMBEDDINGS]

    plan = []
    split_modes, negs, ratios, seeds, models, fms = map(parse_list, [args.split_modes, args.negative_sampling, args.negative_ratios, args.seeds, args.models, args.feature_modes])
    for _, d in ds_df.iterrows():
        for emb in emb_names:
            for model in models:
                for sm in split_modes:
                    if sm in {'cross_dataset_transfer', 'topology_matched_transfer'}:
                        for _, td in ds_df.iterrows():
                            if td.dataset == d.dataset: continue
                            for ns in negs:
                                for nr in ratios:
                                    for seed in seeds:
                                        for fm in fms:
                                            plan.append({"train_dataset": d.dataset, "test_dataset": td.dataset, "train_dir": d.dataset_dir, "test_dir": td.dataset_dir, "embedding": emb, "model": model, "split_mode": sm, "negative_sampling": ns, "negative_ratio": float(nr), "seed": int(seed), "feature_mode": fm})
                    else:
                        for ns in negs:
                            for nr in ratios:
                                for seed in seeds:
                                    for fm in fms:
                                        plan.append({"train_dataset": d.dataset, "test_dataset": d.dataset, "train_dir": d.dataset_dir, "test_dir": d.dataset_dir, "embedding": emb, "model": model, "split_mode": sm, "negative_sampling": ns, "negative_ratio": float(nr), "seed": int(seed), "feature_mode": fm})
    pd.DataFrame(plan).to_csv(out / 'run_plan.csv', index=False)
    ds_df.to_csv(out / 'discovered_datasets.csv', index=False)
    pd.DataFrame([{"embedding": e, **bm.EMBEDDINGS[e]} for e in emb_names]).to_csv(out / 'discovered_embeddings.csv', index=False)
    miss_df.to_csv(out / 'missing_assets.csv', index=False)
    write_report(out, missing, ds_df)
    if args.dry_run:
        return

    vocab = bm.load_vocab()
    emb_cache = {}
    ds_cache = {}
    rows, edge_diag, split_diag, cov_diag = [], [], [], []
    for job in plan:
        try:
            for role in ['train', 'test']:
                key = job[f'{role}_dataset']
                if key not in ds_cache:
                    ds_cache[key] = bm.load_dataset_edges(job[f'{role}_dir'])
            tr_name, tr_genes, tr_splits = ds_cache[job['train_dataset']]
            te_name, te_genes, te_splits = ds_cache[job['test_dataset']]
            if job['embedding'] not in emb_cache:
                emb_cache[job['embedding']] = bm.load_embedding(job['embedding'])
            emb = emb_cache[job['embedding']]

            if job['split_mode'] in {'cross_dataset_transfer', 'topology_matched_transfer'}:
                tr_pos = pd.concat([tr_splits['Train_set'], tr_splits['Validation_set']], ignore_index=True).query('label == 1')[['tf','target']]
                va_pos = tr_splits['Validation_set'].query('label == 1')[['tf','target']]
                te_pos = te_splits['Test_set'].query('label == 1')[['tf','target']]
                genes = sorted(set(tr_genes).intersection(te_genes))
            else:
                all_pos = pd.concat(tr_splits.values(), ignore_index=True).query('label == 1')[['tf','target']]
                pos_parts = bm.split_positive_edges(all_pos, job['split_mode'], job['seed'])
                tr_pos, va_pos, te_pos = pos_parts['train'], pos_parts['val'], pos_parts['test']; genes = tr_genes
            all_known = pd.concat([tr_pos, va_pos, te_pos], ignore_index=True).drop_duplicates()
            tfs, targets, all_pos_set = bm.candidate_space(genes, all_known, vocab)
            pos_split = {'train': tr_pos, 'val': va_pos, 'test': te_pos}
            pos_split = {k: v[v.tf.isin(tfs) & v.target.isin(targets)].drop_duplicates() for k, v in pos_split.items()}
            neg_split, diag = bm.sample_negatives(pos_split, tfs, targets, all_pos_set, job['negative_ratio'], 'degree_matched_negative' if job['split_mode']=='topology_matched_transfer' else job['negative_sampling'], job['seed'])
            for drow in diag:
                edge_diag.append({**job, **drow})
            train_edges = bm.make_labeled(pos_split['train'], neg_split['train'])
            test_edges = bm.make_labeled(pos_split['test'], neg_split['test'])
            train_edges = cap_edges_df(train_edges, args.max_edges, job['seed'])
            test_edges = cap_edges_df(test_edges, args.max_edges, job['seed'] + 1009)
            for split, frame in [('train', train_edges), ('val', bm.make_labeled(pos_split['val'], neg_split['val'])), ('test', test_edges)]:
                pth = edge_dir / f"{job['train_dataset']}__{job['test_dataset']}__{job['split_mode']}__{job['negative_sampling']}__r{job['negative_ratio']}__s{job['seed']}__{split}.csv"
                frame.assign(split=split, dataset=job['test_dataset'], negative_sampling=job['negative_sampling'], negative_ratio=job['negative_ratio'], seed=job['seed']).to_csv(pth, index=False)
            split_diag.append({**job, "n_train": len(train_edges), "n_test": len(test_edges), "train_pos": int(train_edges.label.sum()) if len(train_edges) else 0, "test_pos": int(test_edges.label.sum()) if len(test_edges) else 0, "train_test_edge_overlap": len(set(zip(train_edges.tf, train_edges.target)).intersection(set(zip(test_edges.tf, test_edges.target))))})
            cov_diag.append({**job, "n_dataset_genes": len(genes), "n_vocab_overlap": len([g for g in genes if g in vocab]), "n_candidate_tfs": len(tfs), "n_candidate_targets": len(targets), "n_known_positive_edges": len(all_pos_set)})
            if len(train_edges) == 0 or len(test_edges) == 0 or train_edges.label.nunique()<2 or test_edges.label.nunique()<2:
                raise ValueError('empty or one-class train/test edges after filtering')
            scores = bm.fit_predict(train_edges, test_edges, emb, vocab, job['model'], job['seed'], job['feature_mode'])
            metrics = bm.score_metrics(test_edges.label.to_numpy(), scores)
            rows.append({**job, **metrics, "n_train_edges": len(train_edges), "n_test_edges": len(test_edges), "status": "OK", "error_message": ""})
        except Exception as e:
            rows.append({**job, "status": "FAILED", "error_message": str(e)})
    res = pd.DataFrame(rows); res.to_csv(out / 'grn_inference_all_results.csv', index=False)
    pd.DataFrame(edge_diag).to_csv(out / 'edge_sampling_diagnostics.csv', index=False)
    pd.DataFrame(split_diag).to_csv(out / 'split_diagnostics.csv', index=False)
    pd.DataFrame(cov_diag).to_csv(out / 'gene_coverage_diagnostics.csv', index=False)
    ok = res[res.status == 'OK'] if 'status' in res else pd.DataFrame()
    if len(ok):
        ok.groupby(['embedding','model','split_mode','negative_sampling','negative_ratio'])[['auroc','auprc','precision_at_k','recall_at_k','f1_at_threshold']].agg(['mean','std']).to_csv(out / 'grn_inference_summary.csv')
    else:
        pd.DataFrame().to_csv(out / 'grn_inference_summary.csv', index=False)

if __name__ == '__main__':
    main()
