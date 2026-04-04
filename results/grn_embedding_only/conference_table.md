# GRN Embedding Only (Conference-style Tables)

说明：`-`表示该组合无结果；按列（同一dataset）比较：**加粗**表示优于baseline；<span style="color:red"><strong>红色加粗</strong></span>表示该列最优。
仅将`dataset`与`embedding`作为显式变量；其余设置作为表上方 latent variables 展示；`A->B`/`A->C`汇总为`A`。

## AUROC | Classifier=lr

Latent variables: metric=AUROC, classifier=lr, aggregation=mean, dataset_split=1/1

| Embedding | hESC500 | hHep500 | mESC500 | mHSC-E500 | mHSC-GM500 | mHSC-L500 |
|---|---:|---:|---:|---:|---:|---:|
| minus | **0.6646** | 0.4348 | **0.4648** | <span style='color:red'><strong>0.6533</strong></span> | 0.5232 | 0.5357 |
| baseline | 0.5121 | 0.5822 | 0.4624 | 0.6475 | 0.6929 | <span style='color:red'><strong>0.7086</strong></span> |
| scGPT_human | **0.5280** | **0.6082** | **0.6044** | 0.5034 | 0.6811 | 0.5351 |
| v4_bias_rec_best | **0.5381** | **0.5861** | **0.6056** | 0.6306 | 0.6905 | 0.7023 |
| v4_plain_best | **0.5161** | 0.5725 | **0.5983** | 0.4821 | <span style='color:red'><strong>0.6931</strong></span> | 0.5254 |
| v4_type_pe_best | **0.6666** | 0.4252 | **0.6293** | 0.4813 | 0.5186 | 0.6953 |
| random_256 | <span style='color:red'><strong>0.8533</strong></span> | <span style='color:red'><strong>1.0000</strong></span> | <span style='color:red'><strong>0.8738</strong></span> | 0.5597 | 0.6790 | 0.6978 |

## AUROC | Classifier=mlp

Latent variables: metric=AUROC, classifier=mlp, aggregation=mean, dataset_split=1/1

| Embedding | hESC500 | mESC500 | mHSC-E500 | mHSC-GM500 | mHSC-L500 |
|---|---:|---:|---:|---:|---:|
| minus | **0.6798** | **0.4711** | 0.6544 | 0.5429 | 0.7343 |
| baseline | 0.5295 | 0.4592 | <span style='color:red'><strong>0.6960</strong></span> | 0.7290 | <span style='color:red'><strong>0.7471</strong></span> |
| scGPT_human | **0.6725** | **0.6261** | 0.5268 | 0.7287 | 0.5722 |
| v4_bias_rec_best | 0.5269 | 0.4539 | 0.6755 | **0.7314** | 0.7340 |
| v4_plain_best | 0.5136 | 0.4519 | 0.4945 | 0.7126 | 0.7396 |
| v4_type_pe_best | **0.6811** | **0.6253** | 0.6904 | <span style='color:red'><strong>0.7342</strong></span> | 0.7387 |
| random_256 | <span style='color:red'><strong>0.8534</strong></span> | <span style='color:red'><strong>0.8987</strong></span> | 0.6146 | 0.6516 | 0.7115 |

## AUPRC | Classifier=lr

Latent variables: metric=AUPRC, classifier=lr, aggregation=mean, dataset_split=1/1

| Embedding | hESC500 | hHep500 | mESC500 | mHSC-E500 | mHSC-GM500 | mHSC-L500 |
|---|---:|---:|---:|---:|---:|---:|
| minus | <span style='color:red'><strong>0.6455</strong></span> | 0.4811 | **0.5521** | 0.6025 | 0.5517 | 0.5612 |
| baseline | 0.5625 | 0.5649 | 0.5441 | 0.6061 | 0.6363 | 0.6523 |
| scGPT_human | **0.5832** | **0.5851** | **0.6187** | 0.5400 | 0.6205 | 0.5613 |
| v4_bias_rec_best | **0.5890** | 0.5579 | **0.6183** | 0.5885 | 0.6350 | 0.6414 |
| v4_plain_best | **0.5808** | 0.5494 | **0.6190** | 0.5154 | 0.6335 | 0.5496 |
| v4_type_pe_best | **0.6364** | 0.4882 | **0.6244** | 0.5177 | 0.5432 | 0.6401 |
| random_256 | 0.5255 | <span style='color:red'><strong>1.0000</strong></span> | <span style='color:red'><strong>0.7514</strong></span> | <span style='color:red'><strong>0.6929</strong></span> | <span style='color:red'><strong>0.7619</strong></span> | <span style='color:red'><strong>0.7965</strong></span> |

## AUPRC | Classifier=mlp

Latent variables: metric=AUPRC, classifier=mlp, aggregation=mean, dataset_split=1/1

| Embedding | hESC500 | mESC500 | mHSC-E500 | mHSC-GM500 | mHSC-L500 |
|---|---:|---:|---:|---:|---:|
| minus | **0.6509** | **0.5565** | 0.6204 | 0.5756 | 0.6712 |
| baseline | 0.5890 | 0.5475 | 0.6519 | 0.6658 | 0.6784 |
| scGPT_human | **0.6467** | **0.6339** | 0.5653 | 0.6636 | 0.5921 |
| v4_bias_rec_best | 0.5888 | **0.5525** | 0.6397 | **0.6708** | 0.6695 |
| v4_plain_best | 0.5754 | 0.5435 | 0.5455 | 0.6542 | 0.6750 |
| v4_type_pe_best | <span style='color:red'><strong>0.6527</strong></span> | **0.6318** | 0.6466 | **0.6751** | 0.6764 |
| random_256 | 0.5301 | <span style='color:red'><strong>0.8088</strong></span> | <span style='color:red'><strong>0.7526</strong></span> | <span style='color:red'><strong>0.7622</strong></span> | <span style='color:red'><strong>0.8031</strong></span> |

