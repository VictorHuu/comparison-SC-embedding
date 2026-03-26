# report_transfer_control

## 实验目的
验证 difference_v3 native transfer gain 是否主要由 coverage confound 导致。

## 实验设置
strict common repeated(5 seeds, LR/MLP, LR含bootstrap重采样)、coverage-matched native(20 subsamples)、gap decomposition。

## 主表1：strict common-gene repeated results
见 `strict_common_gene_seed_results.csv`。

## 主表2：coverage-matched native results
见 `coverage_matched_native_results.csv`。

## 主表3：native vs strict vs coverage-matched gap decomposition
见 `transfer_gap_summary.csv`。

## 核心结论
请基于上述三表优先判断：native优势是否在strict/covmatch后消失，以及是否仅在对scGPT_human上稳定。

## limitations
- 若 strict/covmatch 子图样本过少，部分组合可能为空；此时结论应标记为证据不足。
