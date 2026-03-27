# Conference-style Aggregated Embedding Comparison

抹除数据集差异后，对多个 embedding 做聚合对比（常用指标：Pearson r、MSE）。

## Table A. Aggregated average rank across datasets (lower is better)

| Embedding | Frozen Linear Rank | Backbone+Head Rank | Overall Rank |
|---|---:|---:|---:|
| baseline | 2.667 | 2.500 | 2.583 |
| scGPT_human | 3.333 | 5.000 | 4.167 |
| minus | 4.000 | 4.000 | 4.000 |
| **v4_bias_rec_best** | **2.000** | **1.500** | **1.750** |
| v4_plain_best | 4.000 | 4.500 | 4.250 |
| v4_type_pe_best | 5.000 | 3.500 | 4.250 |

## Table B. Dataset-wise regression metrics

| Embedding | adamson Pearson r | adamson MSE | dixit Pearson r | dixit MSE | norman Pearson r | norman MSE | Mean Pearson r | Mean MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline(frozen_linear) | 0.1096 | 1.4094 | 0.4898 | 0.9619 | 0.0989 | 1.5986 | 0.2328 | 1.3233 |
| baseline(frozen_head) | 0.1697 | 1.1714 | - | - | **0.1281** | 1.2054 | 0.1489 | 1.1884 |
| scGPT_human(frozen_linear) | 0.1703 | 1.1517 | 0.3574 | 0.9146 | 0.0975 | 1.2648 | 0.2084 | 1.1104 |
| scGPT_human(frozen_head) | 0.1650 | 1.1371 | - | - | 0.0722 | 1.1855 | 0.1186 | 1.1613 |
| minus(frozen_linear) | 0.1291 | 1.3427 | 0.3691 | 0.9052 | 0.0464 | 1.7808 | 0.1815 | 1.3429 |
| minus(frozen_head) | 0.1752 | 1.1047 | - | - | 0.0314 | 1.3190 | 0.1033 | 1.2118 |
| v4_bias_rec_best(frozen_linear) | 0.1852 | **1.6062** | **0.6705** | 0.6748 | 0.0554 | **1.9956** | **0.3037** | 1.4256 |
| v4_bias_rec_best(frozen_head) | **0.2246** | 1.1887 | - | - | 0.1084 | 1.2902 | 0.1665 | 1.2395 |
| v4_plain_best(frozen_linear) | 0.1088 | 1.5070 | 0.4674 | 0.9339 | 0.0817 | 1.6249 | 0.2193 | 1.3553 |
| v4_plain_best(frozen_head) | 0.1365 | 1.2117 | - | - | 0.0964 | 1.2582 | 0.1165 | 1.2350 |
| v4_type_pe_best(frozen_linear) | 0.1181 | 1.4307 | 0.3584 | **1.0788** | 0.0394 | 1.7985 | 0.1720 | **1.4360** |
| v4_type_pe_best(frozen_head) | 0.1711 | 1.1706 | - | - | 0.0798 | 1.3079 | 0.1254 | 1.2393 |

注：当同一张表内同时出现多个 method 时，embedding 名称后会添加括号用于区分 latent variable。
