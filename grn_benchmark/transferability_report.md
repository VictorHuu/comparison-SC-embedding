# GRN Transferability Report

## 实验目的

验证 difference_v3 的 cross-dataset 优势是否稳定，并区分 transferability 与 in-domain fitting。

## 实验设置

- 数据集：hESC500, mESC500
- 嵌入：difference_v3 / baseline / scGPT_human
- 分类器：LR, MLP
- seeds：0..4

## 主表A：MLP mean±std

| train->test | emb | clf | AUROC | AUPRC | ΔAUROC vs baseline | ΔAUROC vs scGPT |
|---|---|---:|---:|---:|---:|---:|
| hESC500->mESC500 | difference_v3 | mlp | 0.5474±0.0104 | 0.3483±0.0121 | 0.0116 | 0.0345 |
| hESC500->mESC500 | baseline | mlp | 0.5359±0.0194 | 0.3444±0.0160 | 0.0000 | 0.0230 |
| hESC500->mESC500 | scGPT_human | mlp | 0.5129±0.0126 | 0.3303±0.0076 | -0.0230 | 0.0000 |
| mESC500->hESC500 | difference_v3 | mlp | 0.6101±0.0018 | 0.2029±0.0044 | 0.0202 | 0.0424 |
| mESC500->hESC500 | baseline | mlp | 0.5900±0.0155 | 0.1859±0.0053 | 0.0000 | 0.0222 |
| mESC500->hESC500 | scGPT_human | mlp | 0.5678±0.0112 | 0.1857±0.0058 | -0.0222 | 0.0000 |

## 主表B：LR corrected repeated runs

| train->test | emb | AUROC | AUPRC |
|---|---|---:|---:|
| hESC500->mESC500 | difference_v3 | 0.6128±0.0046 | 0.3958±0.0042 |
| hESC500->mESC500 | baseline | 0.6004±0.0049 | 0.3786±0.0057 |
| hESC500->mESC500 | scGPT_human | 0.5753±0.0041 | 0.3636±0.0050 |
| mESC500->hESC500 | difference_v3 | 0.5736±0.0068 | 0.1914±0.0061 |
| mESC500->hESC500 | baseline | 0.5023±0.0075 | 0.1587±0.0012 |
| mESC500->hESC500 | scGPT_human | 0.5136±0.0040 | 0.1548±0.0021 |

## 辅助表：strict common-gene evaluation (LR)

| train->test | emb | AUROC | AUPRC |
|---|---|---:|---:|
| hESC500->mESC500 | difference_v3 | 0.5697 | 0.4051 |
| hESC500->mESC500 | baseline | 0.5721 | 0.4009 |
| hESC500->mESC500 | scGPT_human | 0.5659 | 0.3980 |
| mESC500->hESC500 | difference_v3 | 0.5653 | 0.2651 |
| mESC500->hESC500 | baseline | 0.5676 | 0.2666 |
| mESC500->hESC500 | scGPT_human | 0.5235 | 0.2408 |
| hESC500->hESC500 | difference_v3 | 0.8538 | 0.5651 |
| hESC500->hESC500 | baseline | 0.8631 | 0.5897 |
| hESC500->hESC500 | scGPT_human | 0.8602 | 0.5850 |
| mESC500->mESC500 | difference_v3 | 0.8949 | 0.8255 |
| mESC500->mESC500 | baseline | 0.8942 | 0.8219 |
| mESC500->mESC500 | scGPT_human | 0.8997 | 0.8345 |

## MLP paired differences (difference_v3 vs refs)

| train->test | ref | ΔAUROC | 95% CI | ΔAUPRC | 95% CI |
|---|---|---:|---:|---:|---:|
| hESC500->mESC500 | baseline | 0.0116 | [-0.0032, 0.0244] | 0.0039 | [-0.0063, 0.0131] |
| hESC500->mESC500 | scGPT_human | 0.0345 | [0.0238, 0.0484] | 0.0180 | [0.0059, 0.0301] |
| mESC500->hESC500 | baseline | 0.0202 | [0.0054, 0.0325] | 0.0170 | [0.0133, 0.0207] |
| mESC500->hESC500 | scGPT_human | 0.0424 | [0.0332, 0.0529] | 0.0172 | [0.0113, 0.0235] |

## 结论

- mESC500->hESC500：若 LR+MLP 与 paired differences均为正，判定为 stronger transfer advantage。
- hESC500->mESC500：若对 baseline 的增益较小/跨0，但对 scGPT_human 稳定为正，判定为 weaker over baseline, stronger over scGPT_human。
- 不输出 overall superiority，仅在结果支持范围内陈述。

## Limitations

- 目前仅 2 个数据集，跨域证据有限。
- 未做更系统的不确定性估计（例如更大规模 bootstrap）。

## 代码变更文件

- `analyze_grn_transferability.py`：生成原始 seed 结果、summary、strict common-gene 结果与 markdown 报告。
