# GRN BEELINE Full (Conference-style Tables)

说明：`-`表示该组合无结果；按列（同一dataset）比较：**加粗**表示优于baseline；<span style="color:red"><strong>红色加粗</strong></span>表示该列最优。
仅将`dataset`与`embedding`作为显式变量；其余设置作为表上方 latent variables 展示；`dataset_split`与`classifier`已聚合，不再展示拆分明细。

## AUROC (Main)

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

## AUPRC (Main)

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

## PRECISION_AT_K (Supplementary)

Latent variables: metric=PRECISION_AT_K, classifier=aggregated(lr,mlp), aggregation=mean

### Specific

| Embedding | hESC_Specific_1000 | hESC_Specific_500 | hHep_Specific_1000 | mDC_Specific_1000 | mDC_Specific_500 | mHSC-E_Specific_1000 | mHSC-E_Specific_500 | mHSC-GM_Specific_1000 | mHSC-GM_Specific_500 | mHSC-L_Specific_1000 | mHSC-L_Specific_500 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| minus | 0.4542 | 0.3844 | 0.6884 | 0.3811 | 0.1500 | **0.8037** | 0.6786 | **0.7782** | <span style='color:red'><strong>0.7384</strong></span> | <span style='color:red'><strong>0.8141</strong></span> | <span style='color:red'><strong>0.7821</strong></span> |
| baseline | 0.4831 | 0.4191 | <span style='color:red'><strong>0.7071</strong></span> | 0.3846 | 0.2500 | 0.7956 | 0.6891 | 0.7764 | 0.7296 | 0.8031 | 0.7622 |
| scGPT_human | 0.4590 | **0.4306** | 0.6716 | <span style='color:red'><strong>0.4056</strong></span> | <span style='color:red'><strong>0.3500</strong></span> | **0.8040** | 0.6660 | 0.7755 | 0.7219 | **0.8051** | 0.7532 |
| v4_bias_rec_best | 0.4747 | **0.4220** | 0.6847 | 0.3706 | 0.1500 | <span style='color:red'><strong>0.8081</strong></span> | **0.6954** | 0.7729 | **0.7351** | 0.7993 | **0.7658** |
| v4_plain_best | 0.4819 | <span style='color:red'><strong>0.4364</strong></span> | 0.7034 | 0.3776 | 0.1000 | **0.7983** | **0.6912** | <span style='color:red'><strong>0.7788</strong></span> | 0.7219 | **0.8053** | **0.7658** |
| v4_type_pe_best | <span style='color:red'><strong>0.4843</strong></span> | **0.4277** | 0.6996 | 0.3636 | 0.2000 | **0.8013** | <span style='color:red'><strong>0.7038</strong></span> | **0.7773** | **0.7307** | **0.8060** | **0.7676** |

### Non-Specific

| Embedding | hESC_Non-Specific_1000 | hESC_Non-Specific_500 | hHep_Non-Specific_1000 | hHep_Non-Specific_500 | mDC_Non-Specific_1000 | mDC_Non-Specific_500 | mHSC-E_Non-Specific_1000 | mHSC-E_Non-Specific_500 | mHSC-GM_Non-Specific_1000 | mHSC-GM_Non-Specific_500 | mHSC-L_Non-Specific_1000 | mHSC-L_Non-Specific_500 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| minus | <span style='color:red'><strong>0.2337</strong></span> | <span style='color:red'><strong>0.2558</strong></span> | **0.2108** | **0.1774** | **0.2553** | <span style='color:red'><strong>0.1538</strong></span> | <span style='color:red'><strong>0.3280</strong></span> | **0.2746** | **0.2817** | **0.2273** | <span style='color:red'><strong>0.2826</strong></span> | **0.1961** |
| baseline | 0.2308 | 0.2209 | 0.1863 | 0.0968 | 0.2163 | 0.1154 | 0.3069 | 0.2394 | 0.2676 | 0.1818 | 0.2236 | 0.1765 |
| scGPT_human | 0.1864 | 0.1860 | **0.2157** | **0.1290** | <span style='color:red'><strong>0.3121</strong></span> | 0.1154 | 0.2989 | <span style='color:red'><strong>0.2817</strong></span> | <span style='color:red'><strong>0.2923</strong></span> | <span style='color:red'><strong>0.2727</strong></span> | **0.2516** | <span style='color:red'><strong>0.3333</strong></span> |
| v4_bias_rec_best | 0.2071 | 0.1977 | 0.1814 | <span style='color:red'><strong>0.2097</strong></span> | **0.2943** | <span style='color:red'><strong>0.1538</strong></span> | 0.2778 | 0.2324 | 0.2606 | 0.1477 | **0.2391** | **0.2843** |
| v4_plain_best | 0.2249 | **0.2442** | **0.1912** | **0.1452** | **0.2801** | 0.1154 | 0.2804 | **0.2535** | 0.2676 | 0.1705 | **0.2422** | **0.2353** |
| v4_type_pe_best | 0.2308 | 0.2093 | <span style='color:red'><strong>0.2206</strong></span> | 0.0968 | **0.2730** | <span style='color:red'><strong>0.1538</strong></span> | 0.2963 | **0.2676** | 0.2606 | 0.1364 | **0.2391** | **0.2157** |

### STRING

| Embedding | hESC_STRING_1000 | hESC_STRING_500 | hHep_STRING_1000 | hHep_STRING_500 | mDC_STRING_1000 | mDC_STRING_500 | mHSC-E_STRING_1000 | mHSC-E_STRING_500 | mHSC-GM_STRING_1000 | mHSC-GM_STRING_500 | mHSC-L_STRING_1000 | mHSC-L_STRING_500 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| minus | 0.2782 | <span style='color:red'><strong>0.3571</strong></span> | **0.4520** | 0.4865 | **0.4862** | 0.3056 | <span style='color:red'><strong>0.5173</strong></span> | **0.5362** | 0.3723 | <span style='color:red'><strong>0.3462</strong></span> | <span style='color:red'><strong>0.3720</strong></span> | 0.3462 |
| baseline | 0.2857 | 0.1857 | 0.4498 | 0.5338 | 0.4585 | 0.4167 | 0.5074 | 0.5145 | <span style='color:red'><strong>0.4220</strong></span> | 0.1923 | 0.3537 | 0.3718 |
| scGPT_human | 0.2218 | **0.2286** | <span style='color:red'><strong>0.4629</strong></span> | 0.5270 | **0.4677** | <span style='color:red'><strong>0.4861</strong></span> | 0.4876 | **0.5435** | 0.3759 | **0.2308** | **0.3567** | **0.3846** |
| v4_bias_rec_best | 0.2594 | **0.2143** | 0.4476 | <span style='color:red'><strong>0.5405</strong></span> | 0.4562 | 0.4028 | 0.4703 | <span style='color:red'><strong>0.5580</strong></span> | 0.3333 | **0.2821** | 0.3354 | **0.4103** |
| v4_plain_best | 0.2707 | **0.2429** | 0.4214 | 0.4865 | **0.4908** | **0.4722** | <span style='color:red'><strong>0.5173</strong></span> | **0.5507** | 0.3723 | **0.2692** | 0.3476 | **0.4231** |
| v4_type_pe_best | <span style='color:red'><strong>0.2970</strong></span> | <span style='color:red'><strong>0.3571</strong></span> | 0.4323 | 0.5270 | <span style='color:red'><strong>0.5138</strong></span> | **0.4306** | **0.5149** | **0.5290** | 0.4043 | **0.2564** | 0.3415 | <span style='color:red'><strong>0.4359</strong></span> |

## RECALL_AT_K (Supplementary)

Latent variables: metric=RECALL_AT_K, classifier=aggregated(lr,mlp), aggregation=mean

### Specific

| Embedding | hESC_Specific_1000 | hESC_Specific_500 | hHep_Specific_1000 | mDC_Specific_1000 | mDC_Specific_500 | mHSC-E_Specific_1000 | mHSC-E_Specific_500 | mHSC-GM_Specific_1000 | mHSC-GM_Specific_500 | mHSC-L_Specific_1000 | mHSC-L_Specific_500 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| minus | 0.4542 | 0.3844 | 0.6884 | 0.3811 | 0.1500 | **0.8037** | 0.6786 | **0.7782** | <span style='color:red'><strong>0.7384</strong></span> | <span style='color:red'><strong>0.8141</strong></span> | <span style='color:red'><strong>0.7821</strong></span> |
| baseline | 0.4831 | 0.4191 | <span style='color:red'><strong>0.7071</strong></span> | 0.3846 | 0.2500 | 0.7956 | 0.6891 | 0.7764 | 0.7296 | 0.8031 | 0.7622 |
| scGPT_human | 0.4590 | **0.4306** | 0.6716 | <span style='color:red'><strong>0.4056</strong></span> | <span style='color:red'><strong>0.3500</strong></span> | **0.8040** | 0.6660 | 0.7755 | 0.7219 | **0.8051** | 0.7532 |
| v4_bias_rec_best | 0.4747 | **0.4220** | 0.6847 | 0.3706 | 0.1500 | <span style='color:red'><strong>0.8081</strong></span> | **0.6954** | 0.7729 | **0.7351** | 0.7993 | **0.7658** |
| v4_plain_best | 0.4819 | <span style='color:red'><strong>0.4364</strong></span> | 0.7034 | 0.3776 | 0.1000 | **0.7983** | **0.6912** | <span style='color:red'><strong>0.7788</strong></span> | 0.7219 | **0.8053** | **0.7658** |
| v4_type_pe_best | <span style='color:red'><strong>0.4843</strong></span> | **0.4277** | 0.6996 | 0.3636 | 0.2000 | **0.8013** | <span style='color:red'><strong>0.7038</strong></span> | **0.7773** | **0.7307** | **0.8060** | **0.7676** |

### Non-Specific

| Embedding | hESC_Non-Specific_1000 | hESC_Non-Specific_500 | hHep_Non-Specific_1000 | hHep_Non-Specific_500 | mDC_Non-Specific_1000 | mDC_Non-Specific_500 | mHSC-E_Non-Specific_1000 | mHSC-E_Non-Specific_500 | mHSC-GM_Non-Specific_1000 | mHSC-GM_Non-Specific_500 | mHSC-L_Non-Specific_1000 | mHSC-L_Non-Specific_500 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| minus | <span style='color:red'><strong>0.2337</strong></span> | <span style='color:red'><strong>0.2558</strong></span> | **0.2108** | **0.1774** | **0.2553** | <span style='color:red'><strong>0.1538</strong></span> | <span style='color:red'><strong>0.3280</strong></span> | **0.2746** | **0.2817** | **0.2273** | <span style='color:red'><strong>0.2826</strong></span> | **0.1961** |
| baseline | 0.2308 | 0.2209 | 0.1863 | 0.0968 | 0.2163 | 0.1154 | 0.3069 | 0.2394 | 0.2676 | 0.1818 | 0.2236 | 0.1765 |
| scGPT_human | 0.1864 | 0.1860 | **0.2157** | **0.1290** | <span style='color:red'><strong>0.3121</strong></span> | 0.1154 | 0.2989 | <span style='color:red'><strong>0.2817</strong></span> | <span style='color:red'><strong>0.2923</strong></span> | <span style='color:red'><strong>0.2727</strong></span> | **0.2516** | <span style='color:red'><strong>0.3333</strong></span> |
| v4_bias_rec_best | 0.2071 | 0.1977 | 0.1814 | <span style='color:red'><strong>0.2097</strong></span> | **0.2943** | <span style='color:red'><strong>0.1538</strong></span> | 0.2778 | 0.2324 | 0.2606 | 0.1477 | **0.2391** | **0.2843** |
| v4_plain_best | 0.2249 | **0.2442** | **0.1912** | **0.1452** | **0.2801** | 0.1154 | 0.2804 | **0.2535** | 0.2676 | 0.1705 | **0.2422** | **0.2353** |
| v4_type_pe_best | 0.2308 | 0.2093 | <span style='color:red'><strong>0.2206</strong></span> | 0.0968 | **0.2730** | <span style='color:red'><strong>0.1538</strong></span> | 0.2963 | **0.2676** | 0.2606 | 0.1364 | **0.2391** | **0.2157** |

### STRING

| Embedding | hESC_STRING_1000 | hESC_STRING_500 | hHep_STRING_1000 | hHep_STRING_500 | mDC_STRING_1000 | mDC_STRING_500 | mHSC-E_STRING_1000 | mHSC-E_STRING_500 | mHSC-GM_STRING_1000 | mHSC-GM_STRING_500 | mHSC-L_STRING_1000 | mHSC-L_STRING_500 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| minus | 0.2782 | <span style='color:red'><strong>0.3571</strong></span> | **0.4520** | 0.4865 | **0.4862** | 0.3056 | <span style='color:red'><strong>0.5173</strong></span> | **0.5362** | 0.3723 | <span style='color:red'><strong>0.3462</strong></span> | <span style='color:red'><strong>0.3720</strong></span> | 0.3462 |
| baseline | 0.2857 | 0.1857 | 0.4498 | 0.5338 | 0.4585 | 0.4167 | 0.5074 | 0.5145 | <span style='color:red'><strong>0.4220</strong></span> | 0.1923 | 0.3537 | 0.3718 |
| scGPT_human | 0.2218 | **0.2286** | <span style='color:red'><strong>0.4629</strong></span> | 0.5270 | **0.4677** | <span style='color:red'><strong>0.4861</strong></span> | 0.4876 | **0.5435** | 0.3759 | **0.2308** | **0.3567** | **0.3846** |
| v4_bias_rec_best | 0.2594 | **0.2143** | 0.4476 | <span style='color:red'><strong>0.5405</strong></span> | 0.4562 | 0.4028 | 0.4703 | <span style='color:red'><strong>0.5580</strong></span> | 0.3333 | **0.2821** | 0.3354 | **0.4103** |
| v4_plain_best | 0.2707 | **0.2429** | 0.4214 | 0.4865 | **0.4908** | **0.4722** | <span style='color:red'><strong>0.5173</strong></span> | **0.5507** | 0.3723 | **0.2692** | 0.3476 | **0.4231** |
| v4_type_pe_best | <span style='color:red'><strong>0.2970</strong></span> | <span style='color:red'><strong>0.3571</strong></span> | 0.4323 | 0.5270 | <span style='color:red'><strong>0.5138</strong></span> | **0.4306** | **0.5149** | **0.5290** | 0.4043 | **0.2564** | 0.3415 | <span style='color:red'><strong>0.4359</strong></span> |

## F1 (Supplementary)

Latent variables: metric=F1, classifier=aggregated(lr,mlp), aggregation=mean

### Specific

| Embedding | hESC_Specific_1000 | hESC_Specific_500 | hHep_Specific_1000 | mDC_Specific_1000 | mDC_Specific_500 | mHSC-E_Specific_1000 | mHSC-E_Specific_500 | mHSC-GM_Specific_1000 | mHSC-GM_Specific_500 | mHSC-L_Specific_1000 | mHSC-L_Specific_500 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| minus | 0.3784 | **0.3613** | 0.6857 | **0.3031** | 0.0000 | **0.8082** | 0.6943 | **0.7782** | **0.7540** | <span style='color:red'><strong>0.8149</strong></span> | <span style='color:red'><strong>0.7927</strong></span> |
| baseline | 0.4397 | 0.3552 | 0.6999 | 0.2911 | 0.0769 | 0.8006 | 0.6979 | 0.7762 | 0.7332 | 0.8067 | 0.7643 |
| scGPT_human | 0.3708 | <span style='color:red'><strong>0.3852</strong></span> | 0.6811 | <span style='color:red'><strong>0.3658</strong></span> | <span style='color:red'><strong>0.0909</strong></span> | **0.8070** | 0.6735 | **0.7769** | 0.7299 | **0.8091** | 0.7601 |
| v4_bias_rec_best | 0.4377 | **0.3727** | 0.6836 | **0.3457** | 0.0625 | <span style='color:red'><strong>0.8128</strong></span> | <span style='color:red'><strong>0.7016</strong></span> | **0.7849** | <span style='color:red'><strong>0.7562</strong></span> | **0.8095** | **0.7709** |
| v4_plain_best | 0.4153 | **0.3736** | <span style='color:red'><strong>0.7109</strong></span> | **0.3070** | 0.0714 | **0.8014** | 0.6852 | **0.7888** | 0.7273 | **0.8077** | **0.7714** |
| v4_type_pe_best | <span style='color:red'><strong>0.4409</strong></span> | **0.3784** | 0.6938 | **0.3238** | 0.0000 | **0.8074** | **0.6995** | <span style='color:red'><strong>0.7901</strong></span> | **0.7499** | **0.8129** | **0.7791** |

### Non-Specific

| Embedding | hESC_Non-Specific_1000 | hESC_Non-Specific_500 | hHep_Non-Specific_1000 | hHep_Non-Specific_500 | mDC_Non-Specific_1000 | mDC_Non-Specific_500 | mHSC-E_Non-Specific_1000 | mHSC-E_Non-Specific_500 | mHSC-GM_Non-Specific_1000 | mHSC-GM_Non-Specific_500 | mHSC-L_Non-Specific_1000 | mHSC-L_Non-Specific_500 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| minus | 0.0733 | 0.1500 | 0.0667 | **0.0932** | **0.1542** | 0.0625 | <span style='color:red'><strong>0.2837</strong></span> | **0.2258** | **0.2644** | 0.1039 | <span style='color:red'><strong>0.2462</strong></span> | **0.1455** |
| baseline | 0.0746 | <span style='color:red'><strong>0.1682</strong></span> | 0.0840 | 0.0588 | 0.1024 | <span style='color:red'><strong>0.0714</strong></span> | 0.2739 | 0.1738 | 0.2227 | 0.1235 | 0.1479 | 0.1321 |
| scGPT_human | **0.0924** | 0.1194 | <span style='color:red'><strong>0.2075</strong></span> | 0.0250 | <span style='color:red'><strong>0.2622</strong></span> | 0.0000 | 0.2685 | <span style='color:red'><strong>0.2529</strong></span> | <span style='color:red'><strong>0.2728</strong></span> | <span style='color:red'><strong>0.2332</strong></span> | **0.2002** | <span style='color:red'><strong>0.2500</strong></span> |
| v4_bias_rec_best | 0.0521 | 0.0831 | 0.0538 | <span style='color:red'><strong>0.1222</strong></span> | **0.1606** | 0.0625 | 0.2678 | **0.1886** | **0.2625** | 0.1139 | 0.1444 | **0.2051** |
| v4_plain_best | <span style='color:red'><strong>0.1011</strong></span> | 0.1272 | 0.0667 | **0.0800** | **0.2059** | <span style='color:red'><strong>0.0714</strong></span> | 0.1584 | 0.1662 | **0.2568** | **0.1294** | **0.2040** | **0.1892** |
| v4_type_pe_best | **0.1004** | 0.0886 | **0.1008** | **0.0652** | **0.2075** | 0.0667 | 0.2696 | **0.2510** | **0.2320** | 0.0957 | **0.1919** | **0.1815** |

### STRING

| Embedding | hESC_STRING_1000 | hESC_STRING_500 | hHep_STRING_1000 | hHep_STRING_500 | mDC_STRING_1000 | mDC_STRING_500 | mHSC-E_STRING_1000 | mHSC-E_STRING_500 | mHSC-GM_STRING_1000 | mHSC-GM_STRING_500 | mHSC-L_STRING_1000 | mHSC-L_STRING_500 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| minus | <span style='color:red'><strong>0.2856</strong></span> | 0.1346 | **0.4086** | 0.4606 | **0.3851** | 0.2053 | 0.5003 | **0.5244** | **0.4193** | <span style='color:red'><strong>0.1647</strong></span> | <span style='color:red'><strong>0.3609</strong></span> | 0.2714 |
| baseline | 0.2667 | 0.1600 | 0.3995 | 0.5443 | 0.3529 | 0.3089 | 0.5020 | 0.4649 | 0.3828 | 0.1000 | 0.3603 | 0.3200 |
| scGPT_human | 0.2313 | **0.1636** | <span style='color:red'><strong>0.4639</strong></span> | <span style='color:red'><strong>0.5575</strong></span> | <span style='color:red'><strong>0.4757</strong></span> | **0.3880** | **0.5156** | **0.5616** | 0.3692 | **0.1575** | 0.3193 | <span style='color:red'><strong>0.4375</strong></span> |
| v4_bias_rec_best | 0.2472 | 0.1429 | **0.4426** | **0.5531** | **0.3988** | **0.3508** | 0.4718 | <span style='color:red'><strong>0.5735</strong></span> | 0.3592 | **0.1607** | 0.2670 | **0.3577** |
| v4_plain_best | **0.2844** | <span style='color:red'><strong>0.1930</strong></span> | 0.3933 | 0.4429 | **0.4075** | <span style='color:red'><strong>0.4686</strong></span> | **0.5237** | **0.5104** | 0.3460 | **0.1540** | 0.3099 | 0.2824 |
| v4_type_pe_best | 0.2502 | 0.1429 | **0.4224** | 0.5330 | **0.4257** | **0.3895** | <span style='color:red'><strong>0.5274</strong></span> | **0.5410** | <span style='color:red'><strong>0.4258</strong></span> | 0.0810 | 0.3515 | 0.2414 |

## SPECIFICITY (Supplementary)

Latent variables: metric=SPECIFICITY, classifier=aggregated(lr,mlp), aggregation=mean

### Specific

| Embedding | hESC_Specific_1000 | hESC_Specific_500 | hHep_Specific_1000 | mDC_Specific_1000 | mDC_Specific_500 | mHSC-E_Specific_1000 | mHSC-E_Specific_500 | mHSC-GM_Specific_1000 | mHSC-GM_Specific_500 | mHSC-L_Specific_1000 | mHSC-L_Specific_500 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| minus | **0.9354** | 0.9043 | <span style='color:red'><strong>0.9361</strong></span> | <span style='color:red'><strong>0.9772</strong></span> | 0.9955 | **0.7820** | **0.7645** | <span style='color:red'><strong>0.7809</strong></span> | 0.6274 | <span style='color:red'><strong>0.7466</strong></span> | 0.6129 |
| baseline | 0.9291 | 0.9220 | 0.9339 | 0.9763 | 0.9955 | 0.7777 | 0.7370 | 0.7798 | <span style='color:red'><strong>0.6479</strong></span> | 0.7099 | <span style='color:red'><strong>0.6562</strong></span> |
| scGPT_human | **0.9338** | 0.9151 | 0.9254 | 0.9655 | <span style='color:red'><strong>1.0000</strong></span> | <span style='color:red'><strong>0.7945</strong></span> | **0.8049** | 0.7761 | 0.6123 | **0.7264** | 0.6142 |
| v4_bias_rec_best | 0.9169 | <span style='color:red'><strong>0.9248</strong></span> | 0.9288 | 0.9754 | 0.9888 | **0.7787** | **0.7543** | 0.7497 | 0.6192 | 0.6978 | 0.5892 |
| v4_plain_best | <span style='color:red'><strong>0.9389</strong></span> | <span style='color:red'><strong>0.9248</strong></span> | 0.9316 | 0.9734 | 0.9933 | 0.7764 | <span style='color:red'><strong>0.8150</strong></span> | 0.7635 | 0.6301 | **0.7255** | 0.6063 |
| v4_type_pe_best | 0.9286 | <span style='color:red'><strong>0.9248</strong></span> | 0.9322 | 0.9760 | **0.9978** | **0.7791** | **0.8136** | 0.7539 | 0.6014 | 0.7075 | 0.5866 |

### Non-Specific

| Embedding | hESC_Non-Specific_1000 | hESC_Non-Specific_500 | hHep_Non-Specific_1000 | hHep_Non-Specific_500 | mDC_Non-Specific_1000 | mDC_Non-Specific_500 | mHSC-E_Non-Specific_1000 | mHSC-E_Non-Specific_500 | mHSC-GM_Non-Specific_1000 | mHSC-GM_Non-Specific_500 | mHSC-L_Non-Specific_1000 | mHSC-L_Non-Specific_500 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| minus | 0.9963 | **0.9931** | 0.9986 | **0.9956** | 0.9985 | 0.9988 | 0.9970 | 0.9893 | 0.9986 | **0.9924** | 0.9977 | **0.9931** |
| baseline | 0.9967 | 0.9920 | <span style='color:red'><strong>0.9990</strong></span> | 0.9951 | <span style='color:red'><strong>0.9988</strong></span> | <span style='color:red'><strong>1.0000</strong></span> | 0.9971 | 0.9895 | 0.9987 | 0.9921 | 0.9986 | 0.9901 |
| scGPT_human | 0.9930 | <span style='color:red'><strong>0.9956</strong></span> | 0.9935 | <span style='color:red'><strong>0.9974</strong></span> | 0.9933 | 0.9988 | 0.9952 | <span style='color:red'><strong>0.9922</strong></span> | 0.9959 | <span style='color:red'><strong>0.9936</strong></span> | 0.9961 | <span style='color:red'><strong>0.9945</strong></span> |
| v4_bias_rec_best | **0.9974** | 0.9918 | 0.9987 | <span style='color:red'><strong>0.9974</strong></span> | 0.9979 | 0.9988 | 0.9967 | 0.9880 | 0.9985 | **0.9924** | **0.9986** | **0.9920** |
| v4_plain_best | **0.9971** | **0.9953** | 0.9978 | **0.9956** | 0.9977 | <span style='color:red'><strong>1.0000</strong></span> | <span style='color:red'><strong>0.9978</strong></span> | 0.9887 | **0.9990** | 0.9912 | <span style='color:red'><strong>0.9991</strong></span> | **0.9923** |
| v4_type_pe_best | <span style='color:red'><strong>0.9974</strong></span> | 0.9918 | 0.9984 | **0.9965** | 0.9984 | 0.9994 | 0.9971 | 0.9885 | <span style='color:red'><strong>0.9991</strong></span> | 0.9880 | 0.9985 | **0.9926** |

### STRING

| Embedding | hESC_STRING_1000 | hESC_STRING_500 | hHep_STRING_1000 | hHep_STRING_500 | mDC_STRING_1000 | mDC_STRING_500 | mHSC-E_STRING_1000 | mHSC-E_STRING_500 | mHSC-GM_STRING_1000 | mHSC-GM_STRING_500 | mHSC-L_STRING_1000 | mHSC-L_STRING_500 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| minus | **0.9985** | 0.9982 | 0.9968 | **0.9928** | 0.9978 | **0.9932** | **0.9976** | **0.9956** | **0.9989** | **0.9978** | <span style='color:red'><strong>0.9991</strong></span> | **0.9981** |
| baseline | 0.9984 | <span style='color:red'><strong>0.9987</strong></span> | 0.9969 | 0.9895 | <span style='color:red'><strong>0.9980</strong></span> | 0.9928 | 0.9973 | 0.9947 | 0.9987 | 0.9970 | 0.9990 | 0.9977 |
| scGPT_human | 0.9954 | 0.9980 | 0.9955 | <span style='color:red'><strong>0.9941</strong></span> | 0.9943 | <span style='color:red'><strong>0.9972</strong></span> | 0.9949 | **0.9963** | 0.9961 | <span style='color:red'><strong>0.9996</strong></span> | 0.9976 | **0.9985** |
| v4_bias_rec_best | **0.9984** | 0.9976 | 0.9963 | **0.9914** | 0.9968 | 0.9916 | 0.9971 | <span style='color:red'><strong>0.9967</strong></span> | 0.9987 | **0.9976** | 0.9990 | 0.9973 |
| v4_plain_best | <span style='color:red'><strong>0.9988</strong></span> | 0.9980 | **0.9972** | 0.9857 | 0.9972 | 0.9920 | **0.9978** | **0.9958** | <span style='color:red'><strong>0.9990</strong></span> | **0.9982** | **0.9991** | 0.9973 |
| v4_type_pe_best | **0.9986** | <span style='color:red'><strong>0.9987</strong></span> | <span style='color:red'><strong>0.9976</strong></span> | 0.9892 | 0.9973 | **0.9936** | <span style='color:red'><strong>0.9979</strong></span> | **0.9961** | **0.9989** | **0.9976** | 0.9990 | <span style='color:red'><strong>0.9990</strong></span> |

