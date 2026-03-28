# Perturbation Regression README

本目录包含 `perturbation_regression_benchmark.py`，用于评估不同基因 embedding 在扰动效应回归任务中的表现。

> 说明：本实验强调 **保守、无泄漏（leak-free）** 评估；主结论以线性 probe 结果为准。

---

## 1. 任务定义

### 1.1 样本与标签

- 每个样本对应一个 perturbation gene（被扰动基因）。
- 标签是该基因扰动的表达变化向量（delta）：
  - `delta = mean(perturbed) - mean(control)`。
- 默认按 cell type 构建 context；若没有有效 cell type context，则回退到 `dataset::all`。

### 1.2 输入特征

- 输入特征是 perturbation gene 在 embedding 矩阵中的向量。
- 支持的 embedding（当前脚本配置）：
  - `minus`
  - `baseline`
  - `scGPT_human`
  - `v4_bias_rec_best`
  - `v4_plain_best`
  - `v4_type_pe_best`

---

## 2. 评估设置（Conservative + Leak-Free）

### 2.1 方法分层

- **PRIMARY（主结论）**：`frozen_linear`
  - 冻结 embedding，仅训练线性回归头（Ridge）。
- **SECONDARY（次要）**：`frozen_backbone_trainable_head`
  - 冻结 embedding backbone，训练非线性 MLP head。
- **EXPLORATORY（探索性）**：`full_finetune_embedding_head`
  - embedding 与 head 一起微调，默认关闭。

### 2.2 交叉验证策略

- `n_pert_genes < 25`：Leave-One-Out (LOO)
- `n_pert_genes >= 25`：5-fold KFold（随机种子固定）

### 2.3 防泄漏机制

- 每个 fold 内只用训练折标签做 `top-k` 目标基因筛选。
- 标准化（StandardScaler）严格按训练折拟合，再作用到测试折。

### 2.4 小样本保护

- `frozen_mlp` 在 `n_pert_genes < 25` 时直接跳过。
- 神经网络设置要求最小训练样本阈值（默认 `<20` 会跳过）。

---

## 3. 指标解释

- `pearson_r`：预测向量与真实向量的 Pearson 相关（越高越好）。
- `mse`：均方误差（越低越好）。
- `sign_acc`：符号一致率，即预测上/下调方向是否与真实一致（越高越好）。

---

## 4. 术语与缩写（避免歧义）

- **LR**：Logistic Regression（逻辑回归），用于二分类任务（如 transfer_v2 的边分类）。
- **MLP**：Multi-Layer Perceptron（多层感知机）。
- **Ridge**：L2 正则线性回归；本脚本中 `frozen_linear` 使用的是 Ridge，而不是 LR。
- **LOO**：Leave-One-Out 交叉验证。
- **Leakage（数据泄漏）**：测试信息在训练阶段被使用，导致结果偏乐观。

---

## 5. 运行示例

```bash
python scripts/perturbation_regression/perturbation_regression_benchmark.py \
  --top_k 256 \
  --enable_full_finetune false
```

常见可调参数：

- `--top_k`：每 fold 选择的目标基因维度上限。
- `--hidden_dim`：神经 head 隐层维度。
- `--finetune_epochs` / `--finetune_lr` / `--finetune_weight_decay`：神经头训练超参数。
- `--enable_sign_reg` / `--sign_reg_weight`：是否启用方向一致性正则。

---

## 6. 输出文件

默认写入 `results/perturbation_regression/`：

- `perturbation_regression_results.csv`（summary 级）
- `perturbation_regression_fold_results.csv`（fold 级）
- `perturbation_regression_ranking_summary.csv`（描述性排名）
- `perturbation_regression.log`（日志）

---

## 7. 结果解读建议

- 论文主文建议优先使用 `frozen_linear`（probe）结果。
- `frozen_backbone_trainable_head` 作为补充，避免与主结论混写。
- 小样本/高方差数据集（如 Dixit）应使用保守措辞，不做强 superiority 结论。
