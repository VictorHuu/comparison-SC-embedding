# GRN BEELINE Full (Conference-style Tables)

说明：`-`表示该组合无结果；按列（同一dataset）比较：**加粗**表示优于baseline；<span style="color:red"><strong>红色加粗</strong></span>表示该列最优。
仅将`dataset`与`embedding`作为显式变量；其余设置作为表上方 latent variables 展示；`A->B`/`A->C`汇总为`A`。

## AUROC | Classifier=lr

Latent variables: metric=AUROC, classifier=lr, aggregation=mean, dataset_split=1/4

| Embedding | hESC500 [scGREAT] | hESC_Non-Specific_1000 | hESC_Non-Specific_500 | hESC_STRING_1000 | hESC_STRING_500 | hESC_Specific_1000 | hESC_Specific_500 | hHep_Non-Specific_1000 | hHep_Non-Specific_500 | hHep_STRING_1000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| minus | 0.8569 | **0.8780** | **0.8705** | 0.8242 | **0.8300** | 0.8434 | 0.7773 | <span style='color:red'><strong>0.8540</strong></span> | **0.8201** | **0.8802** |
| baseline | <span style='color:red'><strong>0.8670</strong></span> | 0.8678 | 0.8614 | 0.8370 | 0.8115 | <span style='color:red'><strong>0.8524</strong></span> | <span style='color:red'><strong>0.8161</strong></span> | 0.8335 | 0.7582 | 0.8737 |
| scGPT_human | 0.8636 | 0.8256 | 0.8608 | **0.8379** | <span style='color:red'><strong>0.8734</strong></span> | 0.8283 | 0.7596 | **0.8343** | **0.7999** | **0.8823** |
| v4_bias_rec_best | 0.8638 | 0.8563 | 0.8426 | 0.8364 | **0.8156** | 0.8511 | 0.8161 | 0.8123 | <span style='color:red'><strong>0.8275</strong></span> | **0.8757** |
| v4_plain_best | 0.8635 | <span style='color:red'><strong>0.8833</strong></span> | <span style='color:red'><strong>0.8741</strong></span> | 0.7966 | **0.8126** | 0.8501 | 0.7905 | 0.8259 | **0.7973** | **0.8800** |
| v4_type_pe_best | 0.8600 | **0.8832** | 0.8469 | <span style='color:red'><strong>0.8444</strong></span> | **0.8493** | 0.8499 | 0.8005 | **0.8367** | **0.8232** | <span style='color:red'><strong>0.8823</strong></span> |

Latent variables: metric=AUROC, classifier=lr, aggregation=mean, dataset_split=2/4

| Embedding | hHep_STRING_500 | hHep_Specific_1000 | mDC_Non-Specific_1000 | mDC_Non-Specific_500 | mDC_STRING_1000 | mDC_STRING_500 | mDC_Specific_1000 | mDC_Specific_500 | mESC500 [scGREAT] | mHSC-E_Non-Specific_1000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| minus | 0.8768 | 0.8780 | <span style='color:red'><strong>0.8697</strong></span> | 0.7265 | **0.8978** | **0.8385** | <span style='color:red'><strong>0.8228</strong></span> | 0.7603 | **0.8938** | <span style='color:red'><strong>0.8663</strong></span> |
| baseline | 0.9074 | 0.8833 | 0.8530 | 0.7431 | 0.8854 | 0.7955 | 0.8060 | 0.7991 | 0.8932 | 0.8652 |
| scGPT_human | 0.8876 | 0.8497 | 0.8216 | 0.6665 | 0.8764 | **0.8476** | **0.8126** | 0.7750 | <span style='color:red'><strong>0.8976</strong></span> | 0.8288 |
| v4_bias_rec_best | <span style='color:red'><strong>0.9121</strong></span> | 0.8793 | 0.8496 | 0.7063 | **0.8932** | **0.8273** | 0.7980 | 0.7562 | **0.8935** | 0.8526 |
| v4_plain_best | 0.8608 | <span style='color:red'><strong>0.8870</strong></span> | **0.8633** | <span style='color:red'><strong>0.7639</strong></span> | **0.8910** | <span style='color:red'><strong>0.8618</strong></span> | 0.7974 | <span style='color:red'><strong>0.8701</strong></span> | **0.8932** | 0.8497 |
| v4_type_pe_best | 0.8970 | **0.8868** | **0.8622** | 0.7042 | <span style='color:red'><strong>0.9004</strong></span> | **0.8352** | **0.8180** | 0.7714 | **0.8946** | 0.8622 |

Latent variables: metric=AUROC, classifier=lr, aggregation=mean, dataset_split=3/4

| Embedding | mHSC-E_Non-Specific_500 | mHSC-E_STRING_1000 | mHSC-E_STRING_500 | mHSC-E_Specific_1000 | mHSC-E_Specific_500 | mHSC-GM_Non-Specific_1000 | mHSC-GM_Non-Specific_500 | mHSC-GM_STRING_1000 | mHSC-GM_STRING_500 | mHSC-GM_Specific_1000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| minus | **0.8057** | **0.8872** | <span style='color:red'><strong>0.9206</strong></span> | **0.8943** | 0.7857 | 0.8189 | 0.7840 | 0.8538 | 0.8634 | **0.8692** |
| baseline | 0.7961 | 0.8848 | 0.8832 | 0.8916 | <span style='color:red'><strong>0.8080</strong></span> | 0.8386 | 0.7902 | 0.8647 | <span style='color:red'><strong>0.8667</strong></span> | 0.8684 |
| scGPT_human | **0.8396** | 0.8822 | **0.8956** | 0.8915 | 0.7573 | 0.8174 | **0.7992** | <span style='color:red'><strong>0.8651</strong></span> | 0.8533 | 0.8643 |
| v4_bias_rec_best | <span style='color:red'><strong>0.8493</strong></span> | 0.8795 | **0.8843** | **0.8941** | 0.8065 | <span style='color:red'><strong>0.8526</strong></span> | <span style='color:red'><strong>0.8115</strong></span> | 0.8443 | 0.8104 | 0.8628 |
| v4_plain_best | **0.8272** | 0.8828 | **0.8905** | <span style='color:red'><strong>0.8944</strong></span> | 0.7894 | 0.8292 | **0.7979** | 0.8604 | 0.8445 | <span style='color:red'><strong>0.8692</strong></span> |
| v4_type_pe_best | **0.8174** | <span style='color:red'><strong>0.9003</strong></span> | 0.8647 | **0.8922** | 0.8071 | **0.8513** | 0.7736 | 0.8624 | 0.8496 | 0.8663 |

Latent variables: metric=AUROC, classifier=lr, aggregation=mean, dataset_split=4/4

| Embedding | mHSC-GM_Specific_500 | mHSC-L_Non-Specific_1000 | mHSC-L_Non-Specific_500 | mHSC-L_STRING_1000 | mHSC-L_STRING_500 | mHSC-L_Specific_1000 | mHSC-L_Specific_500 |
|---|---:|---:|---:|---:|---:|---:|---:|
| minus | **0.7661** | 0.8195 | **0.7910** | **0.8640** | <span style='color:red'><strong>0.8912</strong></span> | <span style='color:red'><strong>0.8677</strong></span> | <span style='color:red'><strong>0.7864</strong></span> |
| baseline | 0.7525 | 0.8288 | 0.7786 | 0.8566 | 0.8559 | 0.8571 | 0.7777 |
| scGPT_human | 0.7337 | 0.8130 | <span style='color:red'><strong>0.8118</strong></span> | **0.8573** | **0.8570** | 0.8543 | 0.7370 |
| v4_bias_rec_best | <span style='color:red'><strong>0.7706</strong></span> | 0.8143 | **0.8113** | **0.8571** | **0.8707** | 0.8551 | 0.7603 |
| v4_plain_best | 0.7461 | 0.8201 | **0.8084** | **0.8584** | **0.8751** | **0.8604** | 0.7724 |
| v4_type_pe_best | 0.7510 | <span style='color:red'><strong>0.8394</strong></span> | **0.8083** | <span style='color:red'><strong>0.8773</strong></span> | 0.8493 | **0.8618** | **0.7801** |

## AUROC | Classifier=mlp

Latent variables: metric=AUROC, classifier=mlp, aggregation=mean, dataset_split=1/4

| Embedding | hESC500 [scGREAT] | hESC_Non-Specific_1000 | hESC_Non-Specific_500 | hESC_STRING_1000 | hESC_STRING_500 | hESC_Specific_1000 | hESC_Specific_500 | hHep_Non-Specific_1000 | hHep_Non-Specific_500 | hHep_STRING_1000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| minus | **0.8563** | 0.8914 | 0.8137 | **0.8250** | **0.8033** | 0.8499 | 0.8198 | **0.8733** | <span style='color:red'><strong>0.8538</strong></span> | **0.8929** |
| baseline | 0.8562 | 0.8957 | 0.8737 | 0.8246 | 0.7654 | 0.8578 | 0.8258 | 0.8587 | 0.7279 | 0.8875 |
| scGPT_human | 0.8481 | 0.8790 | 0.8548 | <span style='color:red'><strong>0.8556</strong></span> | **0.7940** | 0.8483 | <span style='color:red'><strong>0.8367</strong></span> | 0.8434 | **0.8079** | **0.8911** |
| v4_bias_rec_best | <span style='color:red'><strong>0.8612</strong></span> | 0.8839 | **0.8754** | **0.8482** | **0.8097** | 0.8393 | 0.8033 | 0.8445 | **0.7902** | **0.8928** |
| v4_plain_best | 0.8475 | <span style='color:red'><strong>0.9050</strong></span> | <span style='color:red'><strong>0.8798</strong></span> | 0.8000 | **0.8039** | 0.8549 | **0.8349** | <span style='color:red'><strong>0.8763</strong></span> | **0.7701** | 0.8865 |
| v4_type_pe_best | 0.8559 | **0.9028** | 0.8335 | 0.8240 | <span style='color:red'><strong>0.8509</strong></span> | <span style='color:red'><strong>0.8600</strong></span> | **0.8279** | **0.8720** | 0.7219 | <span style='color:red'><strong>0.9001</strong></span> |

Latent variables: metric=AUROC, classifier=mlp, aggregation=mean, dataset_split=2/4

| Embedding | hHep_STRING_500 | hHep_Specific_1000 | mDC_Non-Specific_1000 | mDC_Non-Specific_500 | mDC_STRING_1000 | mDC_STRING_500 | mDC_Specific_1000 | mDC_Specific_500 | mESC500 [scGREAT] | mHSC-E_Non-Specific_1000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| minus | <span style='color:red'><strong>0.9314</strong></span> | 0.8992 | **0.8693** | **0.5668** | 0.8960 | 0.7999 | 0.7995 | 0.6522 | 0.8980 | 0.8694 |
| baseline | 0.9286 | 0.9074 | 0.8658 | 0.5152 | 0.9094 | 0.8651 | 0.8624 | <span style='color:red'><strong>0.8281</strong></span> | 0.8988 | <span style='color:red'><strong>0.8793</strong></span> |
| scGPT_human | 0.9035 | <span style='color:red'><strong>0.9092</strong></span> | 0.8482 | <span style='color:red'><strong>0.6653</strong></span> | 0.9077 | **0.8906** | **0.8625** | 0.7179 | <span style='color:red'><strong>0.9000</strong></span> | 0.8780 |
| v4_bias_rec_best | 0.8989 | 0.9067 | **0.8774** | **0.5826** | 0.8973 | 0.8625 | 0.8218 | 0.4607 | 0.8983 | 0.8650 |
| v4_plain_best | 0.9285 | 0.9043 | **0.8892** | **0.5878** | 0.9085 | <span style='color:red'><strong>0.8972</strong></span> | <span style='color:red'><strong>0.8629</strong></span> | 0.7772 | 0.8965 | 0.8708 |
| v4_type_pe_best | 0.9085 | **0.9077** | <span style='color:red'><strong>0.8913</strong></span> | **0.5894** | <span style='color:red'><strong>0.9137</strong></span> | **0.8860** | 0.8346 | 0.7058 | **0.9000** | 0.8721 |

Latent variables: metric=AUROC, classifier=mlp, aggregation=mean, dataset_split=3/4

| Embedding | mHSC-E_Non-Specific_500 | mHSC-E_STRING_1000 | mHSC-E_STRING_500 | mHSC-E_Specific_1000 | mHSC-E_Specific_500 | mHSC-GM_Non-Specific_1000 | mHSC-GM_Non-Specific_500 | mHSC-GM_STRING_1000 | mHSC-GM_STRING_500 | mHSC-GM_Specific_1000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| minus | **0.8655** | **0.8978** | <span style='color:red'><strong>0.9355</strong></span> | **0.8971** | 0.8518 | **0.8522** | <span style='color:red'><strong>0.8330</strong></span> | <span style='color:red'><strong>0.8942</strong></span> | <span style='color:red'><strong>0.8590</strong></span> | **0.8706** |
| baseline | 0.8628 | 0.8944 | 0.9137 | 0.8890 | 0.8554 | 0.8460 | 0.7491 | 0.8771 | 0.7180 | 0.8677 |
| scGPT_human | **0.8671** | **0.8966** | **0.9227** | <span style='color:red'><strong>0.9002</strong></span> | **0.8575** | **0.8510** | **0.7961** | 0.8743 | **0.7733** | **0.8747** |
| v4_bias_rec_best | <span style='color:red'><strong>0.8725</strong></span> | 0.8893 | 0.8816 | **0.8990** | <span style='color:red'><strong>0.8664</strong></span> | <span style='color:red'><strong>0.8653</strong></span> | **0.7611** | 0.8643 | **0.8439** | **0.8746** |
| v4_plain_best | **0.8634** | 0.8853 | **0.9200** | **0.8910** | **0.8597** | 0.8398 | 0.7391 | 0.8585 | **0.7944** | **0.8741** |
| v4_type_pe_best | 0.8333 | <span style='color:red'><strong>0.9089</strong></span> | 0.9023 | **0.8961** | **0.8573** | **0.8479** | **0.7507** | **0.8923** | **0.8488** | <span style='color:red'><strong>0.8750</strong></span> |

Latent variables: metric=AUROC, classifier=mlp, aggregation=mean, dataset_split=4/4

| Embedding | mHSC-GM_Specific_500 | mHSC-L_Non-Specific_1000 | mHSC-L_Non-Specific_500 | mHSC-L_STRING_1000 | mHSC-L_STRING_500 | mHSC-L_Specific_1000 | mHSC-L_Specific_500 |
|---|---:|---:|---:|---:|---:|---:|---:|
| minus | <span style='color:red'><strong>0.7916</strong></span> | 0.8321 | <span style='color:red'><strong>0.8313</strong></span> | <span style='color:red'><strong>0.8897</strong></span> | **0.8372** | <span style='color:red'><strong>0.8717</strong></span> | **0.7959** |
| baseline | 0.7684 | 0.8380 | 0.7469 | 0.8628 | 0.8273 | 0.8564 | 0.7795 |
| scGPT_human | **0.7883** | <span style='color:red'><strong>0.8579</strong></span> | **0.8252** | 0.8615 | **0.8757** | **0.8640** | <span style='color:red'><strong>0.8033</strong></span> |
| v4_bias_rec_best | **0.7821** | 0.8183 | **0.7995** | **0.8729** | <span style='color:red'><strong>0.8933</strong></span> | **0.8570** | **0.8027** |
| v4_plain_best | 0.7659 | **0.8485** | **0.7799** | **0.8713** | **0.8655** | **0.8654** | **0.7927** |
| v4_type_pe_best | **0.7725** | **0.8467** | **0.7560** | **0.8802** | **0.8395** | **0.8635** | **0.7957** |

## AUPRC | Classifier=lr

Latent variables: metric=AUPRC, classifier=lr, aggregation=mean, dataset_split=1/4

| Embedding | hESC500 [scGREAT] | hESC_Non-Specific_1000 | hESC_Non-Specific_500 | hESC_STRING_1000 | hESC_STRING_500 | hESC_Specific_1000 | hESC_Specific_500 | hHep_Non-Specific_1000 | hHep_Non-Specific_500 | hHep_STRING_1000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| minus | 0.5577 | 0.1362 | 0.2321 | <span style='color:red'><strong>0.2252</strong></span> | 0.2805 | 0.4338 | 0.3769 | 0.1270 | **0.1351** | 0.3888 |
| baseline | <span style='color:red'><strong>0.5829</strong></span> | 0.1438 | <span style='color:red'><strong>0.2378</strong></span> | 0.2110 | 0.2919 | 0.4636 | <span style='color:red'><strong>0.4470</strong></span> | 0.1440 | 0.0836 | <span style='color:red'><strong>0.4056</strong></span> |
| scGPT_human | 0.5790 | 0.1183 | 0.2016 | 0.1676 | **0.2924** | 0.4294 | 0.3792 | <span style='color:red'><strong>0.1462</strong></span> | **0.1100** | 0.3842 |
| v4_bias_rec_best | 0.5668 | 0.1407 | 0.1312 | 0.1947 | 0.2326 | 0.4542 | 0.4341 | 0.1103 | **0.1567** | 0.3927 |
| v4_plain_best | 0.5767 | **0.1648** | 0.1799 | **0.2183** | 0.2513 | <span style='color:red'><strong>0.4654</strong></span> | 0.4069 | 0.1097 | **0.1219** | 0.3509 |
| v4_type_pe_best | 0.5554 | <span style='color:red'><strong>0.1820</strong></span> | 0.1805 | 0.2105 | <span style='color:red'><strong>0.2978</strong></span> | 0.4586 | 0.3959 | 0.1396 | <span style='color:red'><strong>0.1577</strong></span> | 0.3901 |

Latent variables: metric=AUPRC, classifier=lr, aggregation=mean, dataset_split=2/4

| Embedding | hHep_STRING_500 | hHep_Specific_1000 | mDC_Non-Specific_1000 | mDC_Non-Specific_500 | mDC_STRING_1000 | mDC_STRING_500 | mDC_Specific_1000 | mDC_Specific_500 | mESC500 [scGREAT] | mHSC-E_Non-Specific_1000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| minus | 0.4320 | 0.7286 | <span style='color:red'><strong>0.2471</strong></span> | **0.2227** | **0.3925** | **0.2730** | <span style='color:red'><strong>0.3674</strong></span> | 0.1196 | **0.8094** | 0.2260 |
| baseline | 0.5086 | 0.7321 | 0.1696 | 0.1627 | 0.3854 | 0.2000 | 0.3363 | 0.2275 | 0.8068 | <span style='color:red'><strong>0.2280</strong></span> |
| scGPT_human | **0.5185** | 0.6707 | **0.1971** | 0.0966 | 0.3570 | **0.3275** | 0.3052 | <span style='color:red'><strong>0.2778</strong></span> | <span style='color:red'><strong>0.8174</strong></span> | 0.2008 |
| v4_bias_rec_best | <span style='color:red'><strong>0.5366</strong></span> | 0.7214 | **0.1712** | **0.2053** | 0.3733 | **0.3269** | 0.3071 | 0.1449 | **0.8083** | 0.1976 |
| v4_plain_best | 0.4102 | <span style='color:red'><strong>0.7685</strong></span> | **0.1863** | **0.1954** | 0.3789 | <span style='color:red'><strong>0.3437</strong></span> | 0.3040 | 0.2150 | **0.8079** | 0.1852 |
| v4_type_pe_best | 0.4404 | **0.7542** | **0.1839** | <span style='color:red'><strong>0.2377</strong></span> | <span style='color:red'><strong>0.4297</strong></span> | **0.2786** | 0.3217 | 0.1715 | **0.8112** | 0.2130 |

Latent variables: metric=AUPRC, classifier=lr, aggregation=mean, dataset_split=3/4

| Embedding | mHSC-E_Non-Specific_500 | mHSC-E_STRING_1000 | mHSC-E_STRING_500 | mHSC-E_Specific_1000 | mHSC-E_Specific_500 | mHSC-GM_Non-Specific_1000 | mHSC-GM_Non-Specific_500 | mHSC-GM_STRING_1000 | mHSC-GM_STRING_500 | mHSC-GM_Specific_1000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| minus | **0.1758** | **0.4798** | **0.4640** | **0.8876** | 0.6991 | **0.2270** | 0.1914 | 0.3462 | **0.2026** | 0.8525 |
| baseline | 0.1520 | 0.4496 | 0.4512 | 0.8841 | 0.7446 | 0.2054 | 0.2040 | 0.3492 | 0.1291 | <span style='color:red'><strong>0.8544</strong></span> |
| scGPT_human | <span style='color:red'><strong>0.2405</strong></span> | 0.4288 | **0.5261** | <span style='color:red'><strong>0.8877</strong></span> | 0.6831 | 0.1803 | <span style='color:red'><strong>0.2566</strong></span> | 0.3189 | <span style='color:red'><strong>0.2859</strong></span> | 0.8460 |
| v4_bias_rec_best | **0.1729** | 0.4371 | <span style='color:red'><strong>0.5368</strong></span> | **0.8842** | **0.7449** | **0.2351** | 0.1633 | 0.2938 | **0.1583** | 0.8465 |
| v4_plain_best | 0.1404 | **0.4754** | **0.5310** | **0.8845** | 0.7255 | **0.2114** | **0.2092** | **0.3659** | **0.2409** | 0.8495 |
| v4_type_pe_best | **0.2318** | <span style='color:red'><strong>0.4814</strong></span> | **0.4664** | 0.8828 | <span style='color:red'><strong>0.7594</strong></span> | <span style='color:red'><strong>0.2353</strong></span> | 0.1937 | <span style='color:red'><strong>0.3716</strong></span> | **0.1440** | 0.8483 |

Latent variables: metric=AUPRC, classifier=lr, aggregation=mean, dataset_split=4/4

| Embedding | mHSC-GM_Specific_500 | mHSC-L_Non-Specific_1000 | mHSC-L_Non-Specific_500 | mHSC-L_STRING_1000 | mHSC-L_STRING_500 | mHSC-L_Specific_1000 | mHSC-L_Specific_500 |
|---|---:|---:|---:|---:|---:|---:|---:|
| minus | <span style='color:red'><strong>0.8009</strong></span> | **0.1891** | 0.1360 | <span style='color:red'><strong>0.3232</strong></span> | **0.3322** | <span style='color:red'><strong>0.8905</strong></span> | <span style='color:red'><strong>0.8220</strong></span> |
| baseline | 0.7821 | 0.1556 | 0.1455 | 0.2975 | 0.3068 | 0.8815 | 0.8205 |
| scGPT_human | 0.7580 | **0.1799** | <span style='color:red'><strong>0.3058</strong></span> | **0.3225** | **0.4180** | 0.8752 | 0.7957 |
| v4_bias_rec_best | **0.7835** | **0.1734** | **0.2417** | 0.2617 | **0.3316** | 0.8794 | 0.8110 |
| v4_plain_best | 0.7727 | <span style='color:red'><strong>0.2107</strong></span> | **0.2958** | 0.2860 | **0.3660** | 0.8804 | 0.8148 |
| v4_type_pe_best | **0.7886** | **0.1945** | **0.2363** | **0.3054** | <span style='color:red'><strong>0.4878</strong></span> | **0.8826** | 0.8198 |

## AUPRC | Classifier=mlp

Latent variables: metric=AUPRC, classifier=mlp, aggregation=mean, dataset_split=1/4

| Embedding | hESC500 [scGREAT] | hESC_Non-Specific_1000 | hESC_Non-Specific_500 | hESC_STRING_1000 | hESC_STRING_500 | hESC_Specific_1000 | hESC_Specific_500 | hHep_Non-Specific_1000 | hHep_Non-Specific_500 | hHep_STRING_1000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| minus | **0.5574** | 0.1761 | 0.1500 | 0.2328 | **0.2084** | 0.4730 | **0.4175** | **0.1731** | **0.1042** | **0.4348** |
| baseline | 0.5565 | <span style='color:red'><strong>0.1807</strong></span> | 0.1812 | 0.2343 | 0.0928 | 0.4829 | 0.3922 | 0.1528 | 0.0643 | 0.4320 |
| scGPT_human | 0.5367 | 0.1399 | 0.1551 | 0.2203 | 0.0723 | 0.4446 | <span style='color:red'><strong>0.4853</strong></span> | <span style='color:red'><strong>0.2194</strong></span> | <span style='color:red'><strong>0.1432</strong></span> | <span style='color:red'><strong>0.4913</strong></span> |
| v4_bias_rec_best | <span style='color:red'><strong>0.5591</strong></span> | 0.1593 | **0.1987** | **0.2367** | **0.0982** | 0.4462 | 0.3853 | 0.1037 | **0.1198** | **0.4481** |
| v4_plain_best | 0.5491 | 0.1630 | <span style='color:red'><strong>0.2060</strong></span> | <span style='color:red'><strong>0.2638</strong></span> | 0.0518 | **0.4980** | **0.4684** | 0.1486 | **0.0709** | **0.4551** |
| v4_type_pe_best | **0.5581** | 0.1638 | 0.1623 | **0.2626** | <span style='color:red'><strong>0.2819</strong></span> | <span style='color:red'><strong>0.5093</strong></span> | **0.4207** | **0.1940** | 0.0515 | **0.4690** |

Latent variables: metric=AUPRC, classifier=mlp, aggregation=mean, dataset_split=2/4

| Embedding | hHep_STRING_500 | hHep_Specific_1000 | mDC_Non-Specific_1000 | mDC_Non-Specific_500 | mDC_STRING_1000 | mDC_STRING_500 | mDC_Specific_1000 | mDC_Specific_500 | mESC500 [scGREAT] | mHSC-E_Non-Specific_1000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| minus | <span style='color:red'><strong>0.5887</strong></span> | 0.7855 | **0.1609** | **0.0213** | 0.4912 | 0.2157 | 0.3551 | 0.1175 | 0.8189 | <span style='color:red'><strong>0.3129</strong></span> |
| baseline | 0.5849 | 0.7896 | 0.1506 | 0.0190 | 0.5051 | 0.4749 | 0.4313 | <span style='color:red'><strong>0.1765</strong></span> | 0.8197 | 0.2927 |
| scGPT_human | 0.5813 | 0.7848 | **0.2176** | **0.0488** | <span style='color:red'><strong>0.5518</strong></span> | **0.5008** | 0.4275 | 0.1527 | **0.8224** | **0.3099** |
| v4_bias_rec_best | 0.5659 | 0.7888 | **0.2218** | **0.0205** | 0.4655 | 0.3990 | 0.3971 | 0.0412 | 0.8195 | 0.2777 |
| v4_plain_best | 0.5436 | **0.7897** | <span style='color:red'><strong>0.2616</strong></span> | **0.0195** | **0.5202** | <span style='color:red'><strong>0.5486</strong></span> | <span style='color:red'><strong>0.4376</strong></span> | 0.1224 | 0.8170 | 0.2461 |
| v4_type_pe_best | **0.5875** | <span style='color:red'><strong>0.8027</strong></span> | **0.2421** | <span style='color:red'><strong>0.0489</strong></span> | **0.5172** | 0.4632 | 0.3927 | 0.0792 | <span style='color:red'><strong>0.8251</strong></span> | **0.3091** |

Latent variables: metric=AUPRC, classifier=mlp, aggregation=mean, dataset_split=3/4

| Embedding | mHSC-E_Non-Specific_500 | mHSC-E_STRING_1000 | mHSC-E_STRING_500 | mHSC-E_Specific_1000 | mHSC-E_Specific_500 | mHSC-GM_Non-Specific_1000 | mHSC-GM_Non-Specific_500 | mHSC-GM_STRING_1000 | mHSC-GM_STRING_500 | mHSC-GM_Specific_1000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| minus | <span style='color:red'><strong>0.3202</strong></span> | **0.5575** | <span style='color:red'><strong>0.6377</strong></span> | **0.8925** | **0.8064** | **0.2823** | **0.1863** | **0.4332** | **0.2686** | **0.8616** |
| baseline | 0.2531 | 0.5325 | 0.6209 | 0.8735 | 0.8053 | 0.2552 | 0.0942 | 0.4211 | 0.1530 | 0.8595 |
| scGPT_human | **0.2800** | <span style='color:red'><strong>0.5699</strong></span> | 0.6039 | <span style='color:red'><strong>0.8934</strong></span> | <span style='color:red'><strong>0.8191</strong></span> | <span style='color:red'><strong>0.3011</strong></span> | <span style='color:red'><strong>0.2832</strong></span> | 0.3703 | **0.1913** | <span style='color:red'><strong>0.8669</strong></span> |
| v4_bias_rec_best | **0.3005** | 0.5080 | **0.6269** | **0.8903** | **0.8185** | 0.2378 | **0.1101** | 0.3830 | <span style='color:red'><strong>0.2862</strong></span> | **0.8613** |
| v4_plain_best | **0.2683** | **0.5571** | **0.6239** | **0.8836** | **0.8108** | **0.2862** | 0.0670 | 0.3774 | **0.1768** | 0.8592 |
| v4_type_pe_best | 0.2516 | **0.5609** | 0.5980 | **0.8900** | **0.8165** | 0.2520 | **0.0998** | <span style='color:red'><strong>0.4358</strong></span> | **0.2183** | 0.8577 |

Latent variables: metric=AUPRC, classifier=mlp, aggregation=mean, dataset_split=4/4

| Embedding | mHSC-GM_Specific_500 | mHSC-L_Non-Specific_1000 | mHSC-L_Non-Specific_500 | mHSC-L_STRING_1000 | mHSC-L_STRING_500 | mHSC-L_Specific_1000 | mHSC-L_Specific_500 |
|---|---:|---:|---:|---:|---:|---:|---:|
| minus | **0.8276** | <span style='color:red'><strong>0.2677</strong></span> | **0.2444** | <span style='color:red'><strong>0.3908</strong></span> | **0.3402** | <span style='color:red'><strong>0.8965</strong></span> | **0.8419** |
| baseline | 0.7984 | 0.1796 | 0.1345 | 0.3672 | 0.3393 | 0.8817 | 0.8275 |
| scGPT_human | <span style='color:red'><strong>0.8310</strong></span> | **0.2521** | <span style='color:red'><strong>0.3325</strong></span> | 0.3147 | **0.4495** | **0.8907** | <span style='color:red'><strong>0.8534</strong></span> |
| v4_bias_rec_best | **0.8146** | **0.1820** | **0.1883** | 0.2998 | <span style='color:red'><strong>0.4886</strong></span> | **0.8832** | **0.8474** |
| v4_plain_best | **0.8009** | **0.2574** | **0.1437** | 0.3555 | **0.4186** | **0.8861** | **0.8366** |
| v4_type_pe_best | **0.8108** | **0.2236** | **0.1617** | 0.3361 | 0.3240 | **0.8859** | **0.8370** |

