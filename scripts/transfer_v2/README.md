# Transfer-v2 README

本目录包含 **transfer-v2** 的完整流程脚本（数据准备、主实验、控制诊断与三表汇总），用于比较不同 embedding 在跨数据集 GRN 边级别迁移任务中的表现。

## 1. 实验设计

### 1.1 任务定义（edge-level transfer）

在每个 `train_dataset -> test_dataset` 方向上：

- 训练集：来源数据集的 `Train_set + Validation_set` 边；
- 测试集：目标数据集的 `Test_set` 边；
- 特征构造：由基因 embedding 生成
  - `a`, `b`, `a*b`, `cosine(a,b)`, `||a-b||_2`；
- 指标：`AUROC`、`AUPRC`；
- 分类器：`LR`、`MLP`；
- 随机种子：`0..4`（每组 5 次重复）。

> 具体实现见 `analyze_grn_transferability_v2.py` 顶部文档与 `fit_eval` 逻辑。

### 1.2 比较对象

- `baseline`
- `minus`
- `scGPT_human`

默认路径和权重 key 可通过 `--embeddings-config` 覆盖（JSON）。

### 1.3 协议（protocol）

transfer-v2 同时评估 3 种协议：

- `native`：不做基因集约束，使用任务中原始可用基因；
- `strict`：只使用 train/test 共享基因（可全局或 pairwise）；
- `coverage_matched`：在共享基因中按检测率排序，取与 strict 相同规模（或指定 `coverage_k`）的基因集。

`transfer_v2_prepare.py` 会输出：

- `transfer_v2/pair_manifest.csv`
- `transfer_v2/pair_diagnostics.csv`
- 各协议用到的 gene set 文件。

#### 协议细化说明（与脚本一致）

- `strict_mode=global`：strict 使用所有数据集的全局交集；
- `strict_mode=pairwise`：strict 使用每个 `(train_dataset, test_dataset)` 方向的两两交集；
- `strict_mode=auto`：当全局交集占比足够大时用 global，否则自动回退 pairwise。

### 1.4 统计口径（当前结果）

基于当前 `transfer/embedding_transfer_summary_v2.csv` 与 `transfer/data_description_table.csv`：

- 数据集：`hESC, hHep, mDC, mHSC-E, mHSC-GM, mHSC-L`（共 6 个）；
- 有向迁移对数量：`N*(N-1)=6*5=30`；
- 每个迁移对有 18 条聚合记录（3 protocol × 2 clf × 3 embedding）；
- 每条聚合记录对应 5 个 seeds。

> 若数据集是 7 个，则有向迁移对应为 `7*6=42`。当前仓库这批 v2 结果文件实际只包含 6 个数据集，因此是 30 对。

## 2. 实验流程（复现）

### 2.1 数据准备

```bash
python scripts/transfer_v2/transfer_v2_prepare.py \
  --h5ad-root processed/native \
  --out-dir transfer_v2 \
  --strict-mode auto \
  --case-mode upper
```

### 2.2 主实验

```bash
python scripts/transfer_v2/analyze_grn_transferability_v2.py \
  --h5ad-root processed/native \
  --pair-manifest transfer_v2/pair_manifest.csv \
  --out-dir transfer
```

默认输出：

- `transfer/embedding_transfer_seed_results_v2.csv`
- `transfer/embedding_transfer_summary_v2.csv`
- `transfer/embedding_transfer_report_v2.md`

### 2.3 控制诊断 + 三表汇总

```bash
python scripts/transfer_v2/run_transfer_control_v2.py
python scripts/transfer_v2/build_three_tables_v2.py \
  --seed-results transfer/embedding_transfer_seed_results_v2.csv \
  --quality transfer_v2/pair_diagnostics.csv \
  --out-dir transfer
```

## 3. 实验结果（当前仓库结果）

### 3.1 总体 winner（30 个迁移对）

`transfer/winner_table.csv` 统计：

- `baseline` 胜出：11 对
- `scgpt_human` 胜出：10 对
- `minus` 胜出：7 对
- `mixed`：2 对

结论：当前结果不存在单一 embedding 在全部迁移方向上占优，表现具有明显方向性与任务依赖性。

### 3.2 按协议分层 winner

来自 `transfer/winner_subtables/winner_by_protocol_*.csv`：

- `native`：baseline 11 / minus 3 / scgpt_human 8 / mixed 8
- `strict`：baseline 9 / minus 7 / scgpt_human 9 / mixed 5
- `coverage_matched`：baseline 10 / minus 8 / scgpt_human 7 / mixed 5

结论：协议变化会显著影响 winner 判定，说明结果具有 protocol sensitivity。

### 3.3 数据质量与基因名规范化

`transfer/data_description_table.csv` 显示 `canonical_over_raw_ratio` 接近 1（约 `1.0001`），说明本批数据中大小写归一化主要用于稳健性保障，而非造成大规模基因集偏移。

## 4. 关键输出文件

- 主结果：
  - `transfer/embedding_transfer_seed_results_v2.csv`
  - `transfer/embedding_transfer_summary_v2.csv`
  - `transfer/embedding_transfer_report_v2.md`
- 三张主表：
  - `transfer/data_description_table.csv`
  - `transfer/winner_table.csv`
  - `transfer/embedding_transfer_seed_results_v2.csv`（seed-level 主表）
- 诊断：
  - `transfer_v2/pair_manifest.csv`
  - `transfer_v2/pair_diagnostics.csv`
  - `transfer/report_transfer_control_v2.md`

## 5. 术语与缩写（避免歧义）

- **LR**：Logistic Regression（逻辑回归），这里用于边分类任务（正负边判别）。
- **MLP**：Multi-Layer Perceptron（多层感知机），这里指前馈神经网络分类器。
- **AUROC**：Area Under ROC Curve，分类阈值扫描下的 ROC 曲线面积。
- **AUPRC**：Area Under Precision-Recall Curve，类别不平衡时常更敏感。
- **seed**：随机种子。`0..4` 表示同一配置重复 5 次，用于估计稳定性。
