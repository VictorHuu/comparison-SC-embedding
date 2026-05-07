# Batch-correction / scRNA integration benchmark

Evaluates frozen gene embeddings by cell-level pooled representations, then checks both biology conservation and batch mixing.

## Why both matter
Good integration must preserve cell type (AvgBIO) while mitigating batch signal (AvgBATCH). UMAP is visualization only.

## Auto-discovery
`discover_project_assets(base_dir)` scans common embedding and dataset locations, and auto-detects `batch_key`/`label_key` from common `obs` names.

## One-command usage
```bash
python scripts/batch-correction/run_batch_correction_all.py --base-dir . --out-dir results/batch-correction --datasets auto --embeddings auto
```

Use `--datasets` / `--embeddings` to restrict scope.
If `Immune_Human_openproblems` is present, it is prioritized first by default (`--priority-datasets`).

## Outputs
- `batch_correction_all_results.csv`
- `batch_correction_per_dataset_summary.csv`
- `batch_correction_rankings.csv`
- `batch_correction_report.md`
- `plots/*`

## Caveats
Do not claim success from batch mixing alone; check conservative score `min(AvgBIO, AvgBATCH)` and stability across seeds.

## Required scGPT-style datasets
Coverage claims require at least one of: **PBMC 10K**, **Immune Human** (prefer both). If missing, they are written to `missing_assets.csv` and report warns not to claim full scGPT-style coverage.
