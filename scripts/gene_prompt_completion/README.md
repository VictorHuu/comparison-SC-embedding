# Gene-Prompt Expression Completion

This benchmark now follows the fixed project configuration you provided (no recursive auto-discovery):
- Base data root: `/bigdata2/hyt/projects/scbenchmark`
- Datasets: `adamson`, `dixit`, `norman`
- Embeddings: `minus`, `baseline`, `scGPT_human`, `v4_bias_rec_best`, `v4_plain_best`, `v4_type_pe_best`

## Usage
```bash
python scripts/gene_prompt_completion/run_gene_prompt_completion_all.py \
  --out-dir results/gene_prompt_completion \
  --models mean,knn_prompt,ridge_pair,mlp_pair
```

You can restrict with `--datasets` and `--embeddings` (comma-separated names from the fixed list).

## Outputs
- `run_plan.csv`
- `gene_prompt_completion_all_results.csv`
- `gene_prompt_completion_gene_metrics.csv`
- `prompt_target_manifest.csv`
- `gene_prompt_completion_report.md`

## Conference-style summary tables
After a run, the runner also writes:
- `gene_prompt_completion_conference_tables.md`: one compact markdown file comparing the six fixed embeddings side by side with best/baseline highlighting.
- Conservative ridge_pair-vs-baseline diagnostics are included inside the markdown when the required model rows are present.

You can rebuild these tables from an existing results CSV without rerunning the benchmark:
```bash
python scripts/gene_prompt_completion/gene_prompt_conference_tables.py \
  --results results/gene_prompt_completion/gene_prompt_completion_all_results.csv \
  --out-dir results/gene_prompt_completion
```
