# Batch-Correction Conference Tables

Style: **bold** marks the best embedding within a row. Values are mean±std across seeds; higher is better for all displayed metrics.

## Data included

- Input rows: 180
- Successful rows: 120
- Datasets: Immune_Human, PBMC_10K
- Correction methods: linear_residual, none
- Metrics shown: Overall, AvgBIO, AvgBATCH, NMI_label, ARI_label, ASW_label, ASW_batch, GraphConn

## Overall embedding ranking

| Embedding | rank | Overall | AvgBIO | AvgBATCH | conservative_score |
| --- | --- | --- | --- | --- | --- |
| minus | 1 | 0.6091 | 0.3681 | 0.9707 | 0.3681 |
| v4_plain_best | 2 | 0.6043 | 0.3647 | 0.9637 | 0.3647 |
| scGPT_human | 3 | 0.6024 | 0.3558 | 0.9722 | 0.3558 |
| v4_type_pe_best | 4 | 0.5959 | 0.3525 | 0.9610 | 0.3525 |
| baseline | 5 | 0.5902 | 0.3421 | 0.9625 | 0.3421 |
| v4_bias_rec_best | 6 | 0.5781 | 0.3223 | 0.9619 | 0.3223 |

## Overall by dataset and correction method

| dataset / correction_method | baseline | minus | scGPT_human | v4_bias_rec_best | v4_plain_best | v4_type_pe_best |
| --- | --- | --- | --- | --- | --- | --- |
| Immune_Human / linear_residual | 0.561±0.002 | **0.583±0.004** | 0.573±0.003 | 0.544±0.002 | 0.571±0.005 | 0.567±0.004 |
| Immune_Human / none | 0.535±0.002 | **0.568±0.002** | 0.554±0.007 | 0.521±0.003 | 0.552±0.006 | 0.541±0.007 |
| PBMC_10K / linear_residual | 0.633±0.000 | 0.644±0.000 | 0.642±0.000 | 0.623±0.000 | **0.647±0.001** | 0.638±0.000 |
| PBMC_10K / none | 0.633±0.001 | 0.642±0.000 | 0.641±0.000 | 0.624±0.000 | **0.647±0.001** | 0.637±0.000 |

## AvgBIO by dataset and correction method

| dataset / correction_method | baseline | minus | scGPT_human | v4_bias_rec_best | v4_plain_best | v4_type_pe_best |
| --- | --- | --- | --- | --- | --- | --- |
| Immune_Human / linear_residual | 0.282±0.004 | **0.309±0.006** | 0.293±0.005 | 0.255±0.003 | 0.294±0.008 | 0.288±0.006 |
| Immune_Human / none | 0.257±0.003 | **0.301±0.003** | 0.280±0.012 | 0.233±0.004 | 0.288±0.010 | 0.276±0.011 |
| PBMC_10K / linear_residual | 0.414±0.000 | 0.432±0.001 | 0.426±0.000 | 0.400±0.000 | **0.439±0.002** | 0.424±0.000 |
| PBMC_10K / none | 0.415±0.001 | 0.430±0.000 | 0.424±0.001 | 0.401±0.000 | **0.438±0.001** | 0.422±0.000 |

## AvgBATCH by dataset and correction method

| dataset / correction_method | baseline | minus | scGPT_human | v4_bias_rec_best | v4_plain_best | v4_type_pe_best |
| --- | --- | --- | --- | --- | --- | --- |
| Immune_Human / linear_residual | 0.979±0.000 | **0.994±0.000** | 0.992±0.000 | 0.978±0.000 | 0.986±0.000 | 0.987±0.000 |
| Immune_Human / none | 0.951±0.000 | **0.967±0.000** | 0.964±0.000 | 0.954±0.000 | 0.949±0.000 | 0.938±0.000 |
| PBMC_10K / linear_residual | 0.960±0.000 | 0.962±0.000 | **0.966±0.000** | 0.958±0.000 | 0.960±0.000 | 0.960±0.000 |
| PBMC_10K / none | 0.960±0.000 | 0.960±0.000 | **0.967±0.000** | 0.958±0.000 | 0.960±0.000 | 0.960±0.000 |

## Best correction method per embedding

| Embedding | best_correction_method | Overall | n |
| --- | --- | --- | --- |
| baseline | linear_residual | 0.597±0.038 | 10 |
| minus | linear_residual | 0.613±0.032 | 10 |
| scGPT_human | linear_residual | 0.607±0.037 | 10 |
| v4_bias_rec_best | linear_residual | 0.584±0.042 | 10 |
| v4_plain_best | linear_residual | 0.609±0.040 | 10 |
| v4_type_pe_best | linear_residual | 0.603±0.038 | 10 |

## Auxiliary metric rankings

Mean metric values across all successful datasets, correction methods, and seeds.

| Embedding | NMI_label | ARI_label | ASW_label | ASW_batch | GraphConn |
| --- | --- | --- | --- | --- | --- |
| baseline | 0.5452 | 0.3531 | 0.1279 | 0.9897 | 0.9353 |
| minus | 0.5797 | 0.3793 | 0.1454 | 0.9976 | 0.9438 |
| scGPT_human | 0.5652 | 0.3715 | 0.1308 | 0.9929 | 0.9515 |
| v4_bias_rec_best | 0.5185 | 0.3366 | 0.1118 | 0.9939 | 0.9299 |
| v4_plain_best | 0.5713 | 0.3773 | 0.1453 | 0.9913 | 0.9360 |
| v4_type_pe_best | 0.5574 | 0.3666 | 0.1337 | 0.9905 | 0.9315 |

## Interpretation rules

- Prefer embeddings that jointly improve Overall, AvgBIO, and AvgBATCH rather than a single metric.
- Compare embeddings within the same dataset and correction method row to avoid mixing correction effects with embedding effects.
- Treat the best-correction table as a workflow-selection summary, not as evidence that one correction method is universally optimal.

