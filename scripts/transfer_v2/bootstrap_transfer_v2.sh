EMB_BASE_DIR=/bigdata2/hyt/projects/scbenchmark_xjq/comparison-SC-embedding
SCBENCHMARK_DIR=/bigdata2/hyt/projects/scbenchmark

python convert_scrnaseq_to_h5ad.py \
  --input-root "$EMB_BASE_DIR/scRNA-Seq" \
  --output-root "$EMB_BASE_DIR/processed/native" \
  --qc-csv "$EMB_BASE_DIR/processed/native/conversion_qc.csv"

python transfer_v2_prepare.py \
  --h5ad-root "$EMB_BASE_DIR/processed/native" \
  --out-dir "$EMB_BASE_DIR/transfer_v2" \
  --strict-mode auto \
  --auto-global-min-ratio 0.2 \
  --case-mode upper

python analyze_grn_transferability_v2.py \
  --base-dir "$SCBENCHMARK_DIR" \
  --h5ad-root ""$EMB_BASE_DIR/processed/native" \
  --pair-manifest ""$EMB_BASE_DIR/transfer_v2/pair_manifest.csv" \
  --out-dir ""$EMB_BASE_DIR/results/transfer_v2"

python build_three_tables_v2.py \
  --seed-results "$EMB_BASE_DIR/results/transfer_v2/embedding_transfer_seed_results_v2.csv" \
  --quality ""$EMB_BASE_DIR/transfer_v2/pair_diagnostics.csv" \
  --out-dir ""$EMB_BASE_DIR/results/transfer_v2" \
  --close-margin-ratio 0.20