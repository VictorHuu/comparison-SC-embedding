# GRN Benchmark from run_all.sh (Conference-style Table)

说明：数值格式为`mean±std`；`-`表示缺失/失败；按列（同一dataset）比较：**加粗**表示优于baseline；<span style="color:red"><strong>红色加粗</strong></span>表示该列最优。
仅将`dataset`与`embedding`作为显式变量；其余设置作为表上方 latent variables 展示；`A->B`/`A->C`汇总为`A`。

## AUROC

Latent variables: metric=AUROC, aggregation=mean, dataset_split=1/1

| Embedding | hESC500 | mESC500 | mHSC-E500 | mHSC-GM500 | mHSC-L500 |
|---|---:|---:|---:|---:|---:|
| minus | 0.8859±0.0015 | **0.9412±0.0004** | **0.7088±0.0095** | **0.8184±0.0043** | **0.8121±0.0040** |
| baseline | 0.8885±0.0007 | 0.9408±0.0002 | 0.6922±0.0144 | 0.8168±0.0035 | 0.8094±0.0052 |
| scGPT_human | 0.8869±0.0033 | **0.9411±0.0007** | **0.6928±0.0202** | 0.8150±0.0022 | **0.8118±0.0022** |
| v4_bias_rec_best | <span style='color:red'><strong>0.8895±0.0019</strong></span> | **0.9411±0.0005** | **0.7060±0.0110** | <span style='color:red'><strong>0.8198±0.0082</strong></span> | **0.8117±0.0038** |
| v4_plain_best | 0.8869±0.0014 | **0.9411±0.0004** | <span style='color:red'><strong>0.7119±0.0067</strong></span> | **0.8177±0.0020** | **0.8134±0.0043** |
| v4_type_pe_best | 0.8861±0.0036 | <span style='color:red'><strong>0.9414±0.0002</strong></span> | **0.7011±0.0148** | 0.8159±0.0040 | <span style='color:red'><strong>0.8142±0.0030</strong></span> |
| difference_v3 | 0.8846±0.0030 | **0.9410±0.0004** | - | - | - |
| BioBERT_original | 0.8881±0.0024 | **0.9409±0.0010** | - | - | - |

## AUPRC

Latent variables: metric=AUPRC, aggregation=mean, dataset_split=1/1

| Embedding | hESC500 | mESC500 | mHSC-E500 | mHSC-GM500 | mHSC-L500 |
|---|---:|---:|---:|---:|---:|
| minus | 0.6259±0.0029 | **0.8861±0.0008** | **0.8267±0.0111** | **0.8712±0.0030** | **0.8714±0.0028** |
| baseline | 0.6276±0.0045 | 0.8853±0.0005 | 0.8066±0.0205 | 0.8696±0.0036 | 0.8699±0.0043 |
| scGPT_human | **0.6291±0.0072** | **0.8859±0.0013** | **0.8204±0.0124** | 0.8680±0.0038 | 0.8694±0.0023 |
| v4_bias_rec_best | **0.6282±0.0041** | <span style='color:red'><strong>0.8866±0.0007</strong></span> | **0.8114±0.0206** | <span style='color:red'><strong>0.8722±0.0068</strong></span> | 0.8692±0.0034 |
| v4_plain_best | 0.6262±0.0025 | **0.8861±0.0008** | <span style='color:red'><strong>0.8292±0.0127</strong></span> | **0.8718±0.0008** | **0.8703±0.0027** |
| v4_type_pe_best | 0.6270±0.0068 | **0.8863±0.0011** | **0.8140±0.0261** | 0.8677±0.0027 | <span style='color:red'><strong>0.8732±0.0033</strong></span> |
| difference_v3 | 0.6229±0.0028 | **0.8861±0.0009** | - | - | - |
| BioBERT_original | <span style='color:red'><strong>0.6293±0.0033</strong></span> | **0.8854±0.0016** | - | - | - |

