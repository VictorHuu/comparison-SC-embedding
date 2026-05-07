# Gene-Prompt Completion Conference Tables

Style: **bold** = best embedding within a row; *italic* = better than baseline embedding in the same row. Lower is better for MSE/MAE; higher is better for correlation/R2/ranking metrics.

## Data included

- Input rows: 90
- Successful rows: 90
- Metrics shown: mse, pearson_all, spearman_all, r2

## Main embedding comparison tables

Columns are the six fixed embeddings. **Bold** marks the best embedding in that row; *italic* marks better than the baseline embedding in that row. These tables are intentionally all in this one markdown file for direct paper-style inspection.

### Secondary: cell_holdout + ridge_pair

#### mse

| dataset / prompt_ratio | baseline | minus | scGPT_human | v4_bias_rec_best | v4_plain_best | v4_type_pe_best |
|---|---:|---:|---:|---:|---:|---:|
| ('test_data', 0.05) | 0.04631 | **0.04629** | 0.05444 | 0.04651 | 0.04744 | 0.04738 |
| ('test_data', 0.1) | 0.04348 | *0.0428* | 0.05017 | **0.04265** | 0.04357 | *0.04343* |
| ('test_data', 0.2) | 0.0446 | **0.04455** | 0.0528 | 0.04485 | 0.04491 | 0.04485 |

#### pearson_all

| dataset / prompt_ratio | baseline | minus | scGPT_human | v4_bias_rec_best | v4_plain_best | v4_type_pe_best |
|---|---:|---:|---:|---:|---:|---:|
| ('test_data', 0.05) | 0.5282 | **0.5313** | 0.4750 | *0.5302* | 0.5165 | 0.5190 |
| ('test_data', 0.1) | 0.5066 | *0.5134* | 0.4549 | **0.5163** | 0.5023 | 0.5047 |
| ('test_data', 0.2) | 0.5250 | **0.5295** | 0.4550 | *0.5271* | 0.5219 | 0.5248 |

#### spearman_all

| dataset / prompt_ratio | baseline | minus | scGPT_human | v4_bias_rec_best | v4_plain_best | v4_type_pe_best |
|---|---:|---:|---:|---:|---:|---:|
| ('test_data', 0.05) | 0.2778 | **0.2785** | 0.2613 | 0.2776 | 0.2754 | 0.2750 |
| ('test_data', 0.1) | 0.2622 | 0.2595 | 0.2456 | **0.2630** | 0.2607 | 0.2612 |
| ('test_data', 0.2) | **0.2787** | 0.2778 | 0.2584 | 0.2773 | 0.2778 | 0.2774 |

#### r2

| dataset / prompt_ratio | baseline | minus | scGPT_human | v4_bias_rec_best | v4_plain_best | v4_type_pe_best |
|---|---:|---:|---:|---:|---:|---:|
| ('test_data', 0.05) | 0.2739 | **0.2755** | 0.1533 | *0.2743* | 0.2587 | 0.2594 |
| ('test_data', 0.1) | 0.2538 | *0.2631* | 0.1489 | **0.2658** | 0.2500 | 0.2520 |
| ('test_data', 0.2) | 0.2773 | **0.2818** | 0.1383 | *0.2791* | 0.2731 | 0.2764 |

## Conservative win/loss vs baselines

An embedding is counted as a conservative win only when ridge_pair beats both mean and knn_prompt on MSE for the same dataset/prompt/split.

| Embedding | wins | comparisons | win_rate |
|---|---:|---:|---:|
| baseline | 0.0 | 3.0 | 0.0 |
| minus | 0.0 | 3.0 | 0.0 |
| scGPT_human | 0.0 | 3.0 | 0.0 |
| v4_bias_rec_best | 0.0 | 3.0 | 0.0 |
| v4_plain_best | 0.0 | 3.0 | 0.0 |
| v4_type_pe_best | 0.0 | 3.0 | 0.0 |

## Interpretation rules

- The most direct embedding comparison is ridge_pair, because mean and knn_prompt do not use gene embeddings.
- Treat gains that appear only for MSE but not Pearson/Spearman as calibration-only improvements.
- Do not claim broad superiority from one dataset, one prompt ratio, or cell_holdout alone.
