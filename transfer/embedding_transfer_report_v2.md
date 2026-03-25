# embedding_transfer_report_v2

## 实验设置（h5ad-only, v1-aligned edge-level）

- h5ad_root: processed/native
- pair_manifest: transfer_v2/pair_manifest.csv
- embeddings: ['minus', 'baseline', 'scGPT_human']
- classifiers: ['lr', 'mlp']
- seeds: [0, 1, 2, 3, 4]
- resample_lr: True

## 输出

- `transfer/embedding_transfer_seed_results_v2.csv`
- `transfer/embedding_transfer_summary_v2.csv`
