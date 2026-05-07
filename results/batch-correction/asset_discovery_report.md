# Discovery

Embedding discovery is disabled.
Embeddings are loaded from fixed registry.

SCBENCH_BASE: /bigdata2/hyt/projects/scbenchmark
VOCAB_PATH: /bigdata2/hyt/projects/scbenchmark/vocab.json
Embeddings: 6
Datasets: 2
Missing/skipped/errors: 0

Rules:
- Do not scan scRNA-Seq/ExpressionData.csv as embedding.
- Do not use GeneOrdering.csv as embedding vocabulary.
- Do not use pseudotime as biological label.
- Do not use dataset as default batch key.
- Prefer PBMC 10K and Immune Human for scGPT-style batch integration.
