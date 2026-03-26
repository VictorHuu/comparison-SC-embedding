# Paper-ready Summary: Perturbation Regression

## Headline interpretation

- Primary setting: `frozen_linear`.
- Secondary setting: `frozen_backbone_trainable_head`.
- Exploratory settings are not used for headline conclusions.
- Across datasets, `baseline` did **not** show consistent superiority.

## Descriptive observations by dataset

- Adamson (primary/frozen_linear): best embedding = `v4_bias_rec_best`.
- Norman (primary/frozen_linear): best embedding = `baseline`.
- Dixit (primary/frozen_linear): best embedding = `v4_bias_rec_best`, but this dataset has small sample size and high variance; treat conclusions as unstable.

## Paired fold-level comparison notes

- Fold-level paired differences are reported with bootstrap 95% CI (descriptive uncertainty quantification).
- Positive mean_diff is defined as better for `baseline` across all metrics (including MSE via sign flip).
- Avoid interpreting bootstrap intervals as formal proof by themselves.
- Wilcoxon p-values are provided when valid; if missing/NaN, no formal nonparametric inference was possible for that row.

### frozen_linear

- adamson: baseline was often lower than comparators in paired folds.
- dixit: baseline appeared competitive in some paired comparisons.
- norman: baseline appeared competitive in some paired comparisons.

### frozen_backbone_trainable_head

- adamson: baseline was often lower than comparators in paired folds.
- norman: baseline appeared competitive in some paired comparisons.

## Conservative conclusion

- Main claims should be based on `frozen_linear` only.
- `frozen_backbone_trainable_head` is secondary and should be discussed separately from the primary probe result.
- Current evidence supports mixed performance across datasets rather than a universal winner.
- For small/high-variance settings (e.g., Dixit), use cautious language and avoid strong superiority claims.
