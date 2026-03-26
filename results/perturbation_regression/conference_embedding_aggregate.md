# Conference-style Aggregated Embedding Comparison

抹除数据集差异后，对多个 embedding 做聚合对比（主指标：rank 与 pearson_r paired effect）。

## Table A. Aggregated average rank across datasets (lower is better)

| Embedding | Frozen Linear Rank | Backbone+Head Rank | Overall Rank |
|---|---:|---:|---:|
| baseline | 2.667 | 2.500 | 2.583 |
| scGPT_human | 3.333 | 5.000 | 4.167 |
| minus | 4.000 | 4.000 | 4.000 |
| **v4_bias_rec_best** | **2.000** | **1.500** | **1.750** |
| v4_plain_best | 4.000 | 4.500 | 4.250 |
| v4_type_pe_best | 5.000 | 3.500 | 4.250 |

## Table B. Aggregated paired effect vs baseline (pearson_r; higher embedding_advantage is better)

| Embedding | Embedding Advantage | Baseline Effect | Baseline Win Rate | N Paired Folds |
|---|---:|---:|---:|---:|
| baseline | 0.0000 | 0.0000 | 0.500 | 39 |
| scGPT_human | -0.0267 | 0.0267 | 0.487 | 39 |
| minus | -0.0490 | 0.0490 | 0.590 | 39 |
| **v4_bias_rec_best** | **0.0496** | -0.0496 | 0.410 | 39 |
| v4_plain_best | -0.0211 | 0.0211 | 0.487 | 39 |
| v4_type_pe_best | -0.0459 | 0.0459 | 0.590 | 39 |

注：Baseline Effect > 0 表示 baseline 更好；Embedding Advantage = - Baseline Effect。
