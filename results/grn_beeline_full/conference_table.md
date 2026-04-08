# GRN BEELINE Full (Conference-style Tables)

说明：`-`表示该组合无结果；按列（同一dataset）比较：**加粗**表示优于baseline；<span style="color:red"><strong>红色加粗</strong></span>表示该列最优。
仅将`dataset`与`embedding`作为显式变量；其余设置作为表上方 latent variables 展示；`dataset_split`与`classifier`已聚合，不再展示拆分明细。

## AUROC

Latent variables: metric=AUROC, classifier=aggregated(lr,mlp), aggregation=mean

### Specific

| Embedding | hESC_Specific_1000 | hESC_Specific_500 | hHep_Specific_1000 | mDC_Specific_1000 | mDC_Specific_500 | mHSC-E_Specific_1000 | mHSC-E_Specific_500 | mHSC-GM_Specific_1000 | mHSC-GM_Specific_500 | mHSC-L_Specific_1000 | mHSC-L_Specific_500 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| minus | 0.8466 | 0.7985 | 0.8886 | 0.8112 | 0.7063 | **0.8957** | 0.8188 | **0.8699** | <span style='color:red'><strong>0.7788</strong></span> | <span style='color:red'><strong>0.8697</strong></span> | <span style='color:red'><strong>0.7912</strong></span> |
| baseline | <span style='color:red'><strong>0.8551</strong></span> | <span style='color:red'><strong>0.8210</strong></span> | 0.8953 | 0.8342 | 0.8136 | 0.8903 | 0.8317 | 0.8680 | 0.7605 | 0.8568 | 0.7786 |
| scGPT_human | 0.8383 | 0.7982 | 0.8794 | <span style='color:red'><strong>0.8375</strong></span> | 0.7464 | **0.8959** | 0.8074 | **0.8695** | **0.7610** | **0.8592** | 0.7702 |
| v4_bias_rec_best | 0.8452 | 0.8097 | 0.8930 | 0.8099 | 0.6085 | <span style='color:red'><strong>0.8966</strong></span> | <span style='color:red'><strong>0.8364</strong></span> | **0.8687** | **0.7764** | 0.8561 | **0.7815** |
| v4_plain_best | 0.8525 | 0.8127 | **0.8956** | 0.8302 | <span style='color:red'><strong>0.8237</strong></span> | **0.8927** | 0.8245 | <span style='color:red'><strong>0.8717</strong></span> | 0.7560 | **0.8629** | **0.7826** |
| v4_type_pe_best | 0.8549 | 0.8142 | <span style='color:red'><strong>0.8973</strong></span> | 0.8263 | 0.7386 | **0.8941** | **0.8322** | **0.8706** | **0.7618** | **0.8626** | **0.7879** |

### Non-Specific

| Embedding | hESC_Non-Specific_1000 | hESC_Non-Specific_500 | hHep_Non-Specific_1000 | hHep_Non-Specific_500 | mDC_Non-Specific_1000 | mDC_Non-Specific_500 | mHSC-E_Non-Specific_1000 | mHSC-E_Non-Specific_500 | mHSC-GM_Non-Specific_1000 | mHSC-GM_Non-Specific_500 | mHSC-L_Non-Specific_1000 | mHSC-L_Non-Specific_500 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| minus | **0.8847** | 0.8421 | <span style='color:red'><strong>0.8637</strong></span> | <span style='color:red'><strong>0.8369</strong></span> | **0.8695** | **0.6466** | 0.8679 | **0.8356** | 0.8355 | <span style='color:red'><strong>0.8085</strong></span> | 0.8258 | **0.8111** |
| baseline | 0.8817 | 0.8675 | 0.8461 | 0.7431 | 0.8594 | 0.6291 | <span style='color:red'><strong>0.8722</strong></span> | 0.8295 | 0.8423 | 0.7696 | 0.8334 | 0.7628 |
| scGPT_human | 0.8523 | 0.8578 | 0.8389 | **0.8039** | 0.8349 | **0.6659** | 0.8534 | **0.8534** | 0.8342 | **0.7976** | **0.8354** | <span style='color:red'><strong>0.8185</strong></span> |
| v4_bias_rec_best | 0.8701 | 0.8590 | 0.8284 | **0.8088** | **0.8635** | **0.6444** | 0.8588 | <span style='color:red'><strong>0.8609</strong></span> | <span style='color:red'><strong>0.8590</strong></span> | **0.7863** | 0.8163 | **0.8054** |
| v4_plain_best | <span style='color:red'><strong>0.8941</strong></span> | <span style='color:red'><strong>0.8770</strong></span> | **0.8511** | **0.7837** | **0.8763** | <span style='color:red'><strong>0.6758</strong></span> | 0.8602 | **0.8453** | 0.8345 | 0.7685 | **0.8343** | **0.7942** |
| v4_type_pe_best | **0.8930** | 0.8402 | **0.8543** | **0.7725** | <span style='color:red'><strong>0.8767</strong></span> | **0.6468** | 0.8671 | 0.8254 | **0.8496** | 0.7621 | <span style='color:red'><strong>0.8431</strong></span> | **0.7822** |

### STRING

| Embedding | hESC_STRING_1000 | hESC_STRING_500 | hHep_STRING_1000 | hHep_STRING_500 | mDC_STRING_1000 | mDC_STRING_500 | mHSC-E_STRING_1000 | mHSC-E_STRING_500 | mHSC-GM_STRING_1000 | mHSC-GM_STRING_500 | mHSC-L_STRING_1000 | mHSC-L_STRING_500 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| minus | 0.8246 | **0.8167** | **0.8865** | 0.9041 | 0.8969 | 0.8192 | **0.8925** | <span style='color:red'><strong>0.9281</strong></span> | **0.8740** | <span style='color:red'><strong>0.8612</strong></span> | **0.8768** | **0.8642** |
| baseline | 0.8308 | 0.7884 | 0.8806 | <span style='color:red'><strong>0.9180</strong></span> | 0.8974 | 0.8303 | 0.8896 | 0.8984 | 0.8709 | 0.7924 | 0.8597 | 0.8416 |
| scGPT_human | <span style='color:red'><strong>0.8468</strong></span> | **0.8337** | **0.8867** | 0.8955 | 0.8920 | **0.8691** | 0.8894 | **0.9091** | 0.8697 | **0.8133** | 0.8594 | **0.8664** |
| v4_bias_rec_best | **0.8423** | **0.8126** | **0.8842** | 0.9055 | 0.8953 | **0.8449** | 0.8844 | 0.8829 | 0.8543 | **0.8271** | **0.8650** | <span style='color:red'><strong>0.8820</strong></span> |
| v4_plain_best | 0.7983 | **0.8083** | **0.8832** | 0.8947 | **0.8997** | <span style='color:red'><strong>0.8795</strong></span> | 0.8840 | **0.9053** | 0.8594 | **0.8194** | **0.8649** | **0.8703** |
| v4_type_pe_best | **0.8342** | <span style='color:red'><strong>0.8501</strong></span> | <span style='color:red'><strong>0.8912</strong></span> | 0.9028 | <span style='color:red'><strong>0.9070</strong></span> | **0.8606** | <span style='color:red'><strong>0.9046</strong></span> | 0.8835 | <span style='color:red'><strong>0.8773</strong></span> | **0.8492** | <span style='color:red'><strong>0.8787</strong></span> | **0.8444** |

## AUPRC

Latent variables: metric=AUPRC, classifier=aggregated(lr,mlp), aggregation=mean

### Specific

| Embedding | hESC_Specific_1000 | hESC_Specific_500 | hHep_Specific_1000 | mDC_Specific_1000 | mDC_Specific_500 | mHSC-E_Specific_1000 | mHSC-E_Specific_500 | mHSC-GM_Specific_1000 | mHSC-GM_Specific_500 | mHSC-L_Specific_1000 | mHSC-L_Specific_500 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| minus | 0.4534 | 0.3972 | 0.7571 | 0.3613 | 0.1186 | **0.8900** | 0.7527 | <span style='color:red'><strong>0.8571</strong></span> | <span style='color:red'><strong>0.8143</strong></span> | <span style='color:red'><strong>0.8935</strong></span> | <span style='color:red'><strong>0.8320</strong></span> |
| baseline | 0.4733 | 0.4196 | 0.7608 | <span style='color:red'><strong>0.3838</strong></span> | 0.2020 | 0.8788 | 0.7750 | 0.8570 | 0.7902 | 0.8816 | 0.8240 |
| scGPT_human | 0.4370 | **0.4323** | 0.7277 | 0.3664 | <span style='color:red'><strong>0.2153</strong></span> | <span style='color:red'><strong>0.8905</strong></span> | 0.7511 | 0.8564 | **0.7945** | **0.8830** | **0.8245** |
| v4_bias_rec_best | 0.4502 | 0.4097 | 0.7551 | 0.3521 | 0.0930 | **0.8872** | **0.7817** | 0.8539 | **0.7990** | 0.8813 | **0.8292** |
| v4_plain_best | **0.4817** | <span style='color:red'><strong>0.4376</strong></span> | <span style='color:red'><strong>0.7791</strong></span> | 0.3708 | 0.1687 | **0.8841** | 0.7682 | 0.8543 | 0.7868 | **0.8832** | **0.8257** |
| v4_type_pe_best | <span style='color:red'><strong>0.4840</strong></span> | 0.4083 | **0.7785** | 0.3572 | 0.1253 | **0.8864** | <span style='color:red'><strong>0.7880</strong></span> | 0.8530 | **0.7997** | **0.8842** | **0.8284** |

### Non-Specific

| Embedding | hESC_Non-Specific_1000 | hESC_Non-Specific_500 | hHep_Non-Specific_1000 | hHep_Non-Specific_500 | mDC_Non-Specific_1000 | mDC_Non-Specific_500 | mHSC-E_Non-Specific_1000 | mHSC-E_Non-Specific_500 | mHSC-GM_Non-Specific_1000 | mHSC-GM_Non-Specific_500 | mHSC-L_Non-Specific_1000 | mHSC-L_Non-Specific_500 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| minus | 0.1561 | 0.1911 | **0.1500** | **0.1197** | **0.2040** | **0.1220** | <span style='color:red'><strong>0.2695</strong></span> | **0.2480** | <span style='color:red'><strong>0.2546</strong></span> | **0.1888** | **0.2284** | **0.1902** |
| baseline | 0.1623 | <span style='color:red'><strong>0.2095</strong></span> | 0.1484 | 0.0739 | 0.1601 | 0.0909 | 0.2604 | 0.2025 | 0.2303 | 0.1491 | 0.1676 | 0.1400 |
| scGPT_human | 0.1291 | 0.1784 | <span style='color:red'><strong>0.1828</strong></span> | **0.1266** | **0.2073** | 0.0727 | 0.2553 | <span style='color:red'><strong>0.2602</strong></span> | **0.2407** | <span style='color:red'><strong>0.2699</strong></span> | **0.2160** | <span style='color:red'><strong>0.3192</strong></span> |
| v4_bias_rec_best | 0.1500 | 0.1649 | 0.1070 | <span style='color:red'><strong>0.1383</strong></span> | **0.1965** | **0.1129** | 0.2377 | **0.2367** | **0.2365** | 0.1367 | **0.1777** | **0.2150** |
| v4_plain_best | **0.1639** | 0.1930 | 0.1291 | **0.0964** | <span style='color:red'><strong>0.2240</strong></span> | **0.1074** | 0.2157 | **0.2044** | **0.2488** | 0.1381 | <span style='color:red'><strong>0.2341</strong></span> | **0.2197** |
| v4_type_pe_best | <span style='color:red'><strong>0.1729</strong></span> | 0.1714 | **0.1668** | **0.1046** | **0.2130** | <span style='color:red'><strong>0.1433</strong></span> | **0.2610** | **0.2417** | **0.2436** | 0.1467 | **0.2091** | **0.1990** |

### STRING

| Embedding | hESC_STRING_1000 | hESC_STRING_500 | hHep_STRING_1000 | hHep_STRING_500 | mDC_STRING_1000 | mDC_STRING_500 | mHSC-E_STRING_1000 | mHSC-E_STRING_500 | mHSC-GM_STRING_1000 | mHSC-GM_STRING_500 | mHSC-L_STRING_1000 | mHSC-L_STRING_500 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| minus | **0.2290** | **0.2445** | 0.4118 | 0.5104 | 0.4419 | 0.2443 | **0.5186** | **0.5508** | **0.3897** | **0.2356** | <span style='color:red'><strong>0.3570</strong></span> | **0.3362** |
| baseline | 0.2227 | 0.1923 | 0.4188 | 0.5468 | 0.4453 | 0.3374 | 0.4910 | 0.5360 | 0.3852 | 0.1410 | 0.3323 | 0.3231 |
| scGPT_human | 0.1940 | 0.1824 | <span style='color:red'><strong>0.4377</strong></span> | **0.5499** | **0.4544** | **0.4142** | **0.4994** | **0.5650** | 0.3446 | <span style='color:red'><strong>0.2386</strong></span> | 0.3186 | <span style='color:red'><strong>0.4338</strong></span> |
| v4_bias_rec_best | 0.2157 | 0.1654 | **0.4204** | <span style='color:red'><strong>0.5513</strong></span> | 0.4194 | **0.3630** | 0.4726 | <span style='color:red'><strong>0.5819</strong></span> | 0.3384 | **0.2222** | 0.2807 | **0.4101** |
| v4_plain_best | <span style='color:red'><strong>0.2411</strong></span> | 0.1516 | 0.4030 | 0.4769 | **0.4496** | <span style='color:red'><strong>0.4462</strong></span> | **0.5163** | **0.5775** | 0.3716 | **0.2089** | 0.3208 | **0.3923** |
| v4_type_pe_best | **0.2366** | <span style='color:red'><strong>0.2898</strong></span> | **0.4295** | 0.5140 | <span style='color:red'><strong>0.4734</strong></span> | **0.3709** | <span style='color:red'><strong>0.5211</strong></span> | 0.5322 | <span style='color:red'><strong>0.4037</strong></span> | **0.1812** | 0.3208 | **0.4059** |

