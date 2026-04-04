# GRN BEELINE Full (Conference-style Tables)

说明：`-`表示该组合无结果；**加粗**表示优于同一行 baseline；<span style="color:red"><strong>红色加粗</strong></span>表示该行最优。

## AUROC

### Classifier = lr

| Dataset | minus | baseline | scGPT_human | v4_bias_rec_best | v4_plain_best | v4_type_pe_best |
|---|---:|---:|---:|---:|---:|---:|
| hESC500 [scGREAT] | 0.8569 | <span style='color:red'><strong>0.8670</strong></span> | 0.8636 | 0.8638 | 0.8635 | 0.8600 |
| hESC_Non-Specific_1000 | **0.8780** | 0.8678 | 0.8256 | 0.8563 | <span style='color:red'><strong>0.8833</strong></span> | **0.8832** |
| hESC_Non-Specific_500 | **0.8705** | 0.8614 | 0.8608 | 0.8426 | <span style='color:red'><strong>0.8741</strong></span> | 0.8469 |
| hESC_STRING_1000 | 0.8242 | 0.8370 | **0.8379** | 0.8364 | 0.7966 | <span style='color:red'><strong>0.8444</strong></span> |
| hESC_STRING_500 | **0.8300** | 0.8115 | <span style='color:red'><strong>0.8734</strong></span> | **0.8156** | **0.8126** | **0.8493** |
| hESC_Specific_1000 | 0.8434 | <span style='color:red'><strong>0.8524</strong></span> | 0.8283 | 0.8511 | 0.8501 | 0.8499 |
| hESC_Specific_500 | 0.7773 | <span style='color:red'><strong>0.8161</strong></span> | 0.7596 | 0.8161 | 0.7905 | 0.8005 |
| hHep_Non-Specific_1000 | <span style='color:red'><strong>0.8540</strong></span> | 0.8335 | **0.8343** | 0.8123 | 0.8259 | **0.8367** |
| hHep_Non-Specific_500 | **0.8201** | 0.7582 | **0.7999** | <span style='color:red'><strong>0.8275</strong></span> | **0.7973** | **0.8232** |
| hHep_STRING_1000 | **0.8802** | 0.8737 | **0.8823** | **0.8757** | **0.8800** | <span style='color:red'><strong>0.8823</strong></span> |
| hHep_STRING_500 | 0.8768 | 0.9074 | 0.8876 | <span style='color:red'><strong>0.9121</strong></span> | 0.8608 | 0.8970 |
| hHep_Specific_1000 | 0.8780 | 0.8833 | 0.8497 | 0.8793 | <span style='color:red'><strong>0.8870</strong></span> | **0.8868** |
| mDC_Non-Specific_1000 | <span style='color:red'><strong>0.8697</strong></span> | 0.8530 | 0.8216 | 0.8496 | **0.8633** | **0.8622** |
| mDC_Non-Specific_500 | 0.7265 | 0.7431 | 0.6665 | 0.7063 | <span style='color:red'><strong>0.7639</strong></span> | 0.7042 |
| mDC_STRING_1000 | **0.8978** | 0.8854 | 0.8764 | **0.8932** | **0.8910** | <span style='color:red'><strong>0.9004</strong></span> |
| mDC_STRING_500 | **0.8385** | 0.7955 | **0.8476** | **0.8273** | <span style='color:red'><strong>0.8618</strong></span> | **0.8352** |
| mDC_Specific_1000 | <span style='color:red'><strong>0.8228</strong></span> | 0.8060 | **0.8126** | 0.7980 | 0.7974 | **0.8180** |
| mDC_Specific_500 | 0.7603 | 0.7991 | 0.7750 | 0.7562 | <span style='color:red'><strong>0.8701</strong></span> | 0.7714 |
| mESC500 [scGREAT] | **0.8938** | 0.8932 | <span style='color:red'><strong>0.8976</strong></span> | **0.8935** | **0.8932** | **0.8946** |
| mHSC-E_Non-Specific_1000 | <span style='color:red'><strong>0.8663</strong></span> | 0.8652 | 0.8288 | 0.8526 | 0.8497 | 0.8622 |
| mHSC-E_Non-Specific_500 | **0.8057** | 0.7961 | **0.8396** | <span style='color:red'><strong>0.8493</strong></span> | **0.8272** | **0.8174** |
| mHSC-E_STRING_1000 | **0.8872** | 0.8848 | 0.8822 | 0.8795 | 0.8828 | <span style='color:red'><strong>0.9003</strong></span> |
| mHSC-E_STRING_500 | <span style='color:red'><strong>0.9206</strong></span> | 0.8832 | **0.8956** | **0.8843** | **0.8905** | 0.8647 |
| mHSC-E_Specific_1000 | **0.8943** | 0.8916 | 0.8915 | **0.8941** | <span style='color:red'><strong>0.8944</strong></span> | **0.8922** |
| mHSC-E_Specific_500 | 0.7857 | <span style='color:red'><strong>0.8080</strong></span> | 0.7573 | 0.8065 | 0.7894 | 0.8071 |
| mHSC-GM_Non-Specific_1000 | 0.8189 | 0.8386 | 0.8174 | <span style='color:red'><strong>0.8526</strong></span> | 0.8292 | **0.8513** |
| mHSC-GM_Non-Specific_500 | 0.7840 | 0.7902 | **0.7992** | <span style='color:red'><strong>0.8115</strong></span> | **0.7979** | 0.7736 |
| mHSC-GM_STRING_1000 | 0.8538 | 0.8647 | <span style='color:red'><strong>0.8651</strong></span> | 0.8443 | 0.8604 | 0.8624 |
| mHSC-GM_STRING_500 | 0.8634 | <span style='color:red'><strong>0.8667</strong></span> | 0.8533 | 0.8104 | 0.8445 | 0.8496 |
| mHSC-GM_Specific_1000 | **0.8692** | 0.8684 | 0.8643 | 0.8628 | <span style='color:red'><strong>0.8692</strong></span> | 0.8663 |
| mHSC-GM_Specific_500 | **0.7661** | 0.7525 | 0.7337 | <span style='color:red'><strong>0.7706</strong></span> | 0.7461 | 0.7510 |
| mHSC-L_Non-Specific_1000 | 0.8195 | 0.8288 | 0.8130 | 0.8143 | 0.8201 | <span style='color:red'><strong>0.8394</strong></span> |
| mHSC-L_Non-Specific_500 | **0.7910** | 0.7786 | <span style='color:red'><strong>0.8118</strong></span> | **0.8113** | **0.8084** | **0.8083** |
| mHSC-L_STRING_1000 | **0.8640** | 0.8566 | **0.8573** | **0.8571** | **0.8584** | <span style='color:red'><strong>0.8773</strong></span> |
| mHSC-L_STRING_500 | <span style='color:red'><strong>0.8912</strong></span> | 0.8559 | **0.8570** | **0.8707** | **0.8751** | 0.8493 |
| mHSC-L_Specific_1000 | <span style='color:red'><strong>0.8677</strong></span> | 0.8571 | 0.8543 | 0.8551 | **0.8604** | **0.8618** |
| mHSC-L_Specific_500 | <span style='color:red'><strong>0.7864</strong></span> | 0.7777 | 0.7370 | 0.7603 | 0.7724 | **0.7801** |

### Classifier = mlp

| Dataset | minus | baseline | scGPT_human | v4_bias_rec_best | v4_plain_best | v4_type_pe_best |
|---|---:|---:|---:|---:|---:|---:|
| hESC500 [scGREAT] | **0.8563** | 0.8562 | 0.8481 | <span style='color:red'><strong>0.8612</strong></span> | 0.8475 | 0.8559 |
| hESC_Non-Specific_1000 | 0.8914 | 0.8957 | 0.8790 | 0.8839 | <span style='color:red'><strong>0.9050</strong></span> | **0.9028** |
| hESC_Non-Specific_500 | 0.8137 | 0.8737 | 0.8548 | **0.8754** | <span style='color:red'><strong>0.8798</strong></span> | 0.8335 |
| hESC_STRING_1000 | **0.8250** | 0.8246 | <span style='color:red'><strong>0.8556</strong></span> | **0.8482** | 0.8000 | 0.8240 |
| hESC_STRING_500 | **0.8033** | 0.7654 | **0.7940** | **0.8097** | **0.8039** | <span style='color:red'><strong>0.8509</strong></span> |
| hESC_Specific_1000 | 0.8499 | 0.8578 | 0.8483 | 0.8393 | 0.8549 | <span style='color:red'><strong>0.8600</strong></span> |
| hESC_Specific_500 | 0.8198 | 0.8258 | <span style='color:red'><strong>0.8367</strong></span> | 0.8033 | **0.8349** | **0.8279** |
| hHep_Non-Specific_1000 | **0.8733** | 0.8587 | 0.8434 | 0.8445 | <span style='color:red'><strong>0.8763</strong></span> | **0.8720** |
| hHep_Non-Specific_500 | <span style='color:red'><strong>0.8538</strong></span> | 0.7279 | **0.8079** | **0.7902** | **0.7701** | 0.7219 |
| hHep_STRING_1000 | **0.8929** | 0.8875 | **0.8911** | **0.8928** | 0.8865 | <span style='color:red'><strong>0.9001</strong></span> |
| hHep_STRING_500 | <span style='color:red'><strong>0.9314</strong></span> | 0.9286 | 0.9035 | 0.8989 | 0.9285 | 0.9085 |
| hHep_Specific_1000 | 0.8992 | 0.9074 | <span style='color:red'><strong>0.9092</strong></span> | 0.9067 | 0.9043 | **0.9077** |
| mDC_Non-Specific_1000 | **0.8693** | 0.8658 | 0.8482 | **0.8774** | **0.8892** | <span style='color:red'><strong>0.8913</strong></span> |
| mDC_Non-Specific_500 | **0.5668** | 0.5152 | <span style='color:red'><strong>0.6653</strong></span> | **0.5826** | **0.5878** | **0.5894** |
| mDC_STRING_1000 | 0.8960 | 0.9094 | 0.9077 | 0.8973 | 0.9085 | <span style='color:red'><strong>0.9137</strong></span> |
| mDC_STRING_500 | 0.7999 | 0.8651 | **0.8906** | 0.8625 | <span style='color:red'><strong>0.8972</strong></span> | **0.8860** |
| mDC_Specific_1000 | 0.7995 | 0.8624 | **0.8625** | 0.8218 | <span style='color:red'><strong>0.8629</strong></span> | 0.8346 |
| mDC_Specific_500 | 0.6522 | <span style='color:red'><strong>0.8281</strong></span> | 0.7179 | 0.4607 | 0.7772 | 0.7058 |
| mESC500 [scGREAT] | 0.8980 | 0.8988 | <span style='color:red'><strong>0.9000</strong></span> | 0.8983 | 0.8965 | **0.9000** |
| mHSC-E_Non-Specific_1000 | 0.8694 | <span style='color:red'><strong>0.8793</strong></span> | 0.8780 | 0.8650 | 0.8708 | 0.8721 |
| mHSC-E_Non-Specific_500 | **0.8655** | 0.8628 | **0.8671** | <span style='color:red'><strong>0.8725</strong></span> | **0.8634** | 0.8333 |
| mHSC-E_STRING_1000 | **0.8978** | 0.8944 | **0.8966** | 0.8893 | 0.8853 | <span style='color:red'><strong>0.9089</strong></span> |
| mHSC-E_STRING_500 | <span style='color:red'><strong>0.9355</strong></span> | 0.9137 | **0.9227** | 0.8816 | **0.9200** | 0.9023 |
| mHSC-E_Specific_1000 | **0.8971** | 0.8890 | <span style='color:red'><strong>0.9002</strong></span> | **0.8990** | **0.8910** | **0.8961** |
| mHSC-E_Specific_500 | 0.8518 | 0.8554 | **0.8575** | <span style='color:red'><strong>0.8664</strong></span> | **0.8597** | **0.8573** |
| mHSC-GM_Non-Specific_1000 | **0.8522** | 0.8460 | **0.8510** | <span style='color:red'><strong>0.8653</strong></span> | 0.8398 | **0.8479** |
| mHSC-GM_Non-Specific_500 | <span style='color:red'><strong>0.8330</strong></span> | 0.7491 | **0.7961** | **0.7611** | 0.7391 | **0.7507** |
| mHSC-GM_STRING_1000 | <span style='color:red'><strong>0.8942</strong></span> | 0.8771 | 0.8743 | 0.8643 | 0.8585 | **0.8923** |
| mHSC-GM_STRING_500 | <span style='color:red'><strong>0.8590</strong></span> | 0.7180 | **0.7733** | **0.8439** | **0.7944** | **0.8488** |
| mHSC-GM_Specific_1000 | **0.8706** | 0.8677 | **0.8747** | **0.8746** | **0.8741** | <span style='color:red'><strong>0.8750</strong></span> |
| mHSC-GM_Specific_500 | <span style='color:red'><strong>0.7916</strong></span> | 0.7684 | **0.7883** | **0.7821** | 0.7659 | **0.7725** |
| mHSC-L_Non-Specific_1000 | 0.8321 | 0.8380 | <span style='color:red'><strong>0.8579</strong></span> | 0.8183 | **0.8485** | **0.8467** |
| mHSC-L_Non-Specific_500 | <span style='color:red'><strong>0.8313</strong></span> | 0.7469 | **0.8252** | **0.7995** | **0.7799** | **0.7560** |
| mHSC-L_STRING_1000 | <span style='color:red'><strong>0.8897</strong></span> | 0.8628 | 0.8615 | **0.8729** | **0.8713** | **0.8802** |
| mHSC-L_STRING_500 | **0.8372** | 0.8273 | **0.8757** | <span style='color:red'><strong>0.8933</strong></span> | **0.8655** | **0.8395** |
| mHSC-L_Specific_1000 | <span style='color:red'><strong>0.8717</strong></span> | 0.8564 | **0.8640** | **0.8570** | **0.8654** | **0.8635** |
| mHSC-L_Specific_500 | **0.7959** | 0.7795 | <span style='color:red'><strong>0.8033</strong></span> | **0.8027** | **0.7927** | **0.7957** |

## AUPRC

### Classifier = lr

| Dataset | minus | baseline | scGPT_human | v4_bias_rec_best | v4_plain_best | v4_type_pe_best |
|---|---:|---:|---:|---:|---:|---:|
| hESC500 [scGREAT] | 0.5577 | <span style='color:red'><strong>0.5829</strong></span> | 0.5790 | 0.5668 | 0.5767 | 0.5554 |
| hESC_Non-Specific_1000 | 0.1362 | 0.1438 | 0.1183 | 0.1407 | **0.1648** | <span style='color:red'><strong>0.1820</strong></span> |
| hESC_Non-Specific_500 | 0.2321 | <span style='color:red'><strong>0.2378</strong></span> | 0.2016 | 0.1312 | 0.1799 | 0.1805 |
| hESC_STRING_1000 | <span style='color:red'><strong>0.2252</strong></span> | 0.2110 | 0.1676 | 0.1947 | **0.2183** | 0.2105 |
| hESC_STRING_500 | 0.2805 | 0.2919 | **0.2924** | 0.2326 | 0.2513 | <span style='color:red'><strong>0.2978</strong></span> |
| hESC_Specific_1000 | 0.4338 | 0.4636 | 0.4294 | 0.4542 | <span style='color:red'><strong>0.4654</strong></span> | 0.4586 |
| hESC_Specific_500 | 0.3769 | <span style='color:red'><strong>0.4470</strong></span> | 0.3792 | 0.4341 | 0.4069 | 0.3959 |
| hHep_Non-Specific_1000 | 0.1270 | 0.1440 | <span style='color:red'><strong>0.1462</strong></span> | 0.1103 | 0.1097 | 0.1396 |
| hHep_Non-Specific_500 | **0.1351** | 0.0836 | **0.1100** | **0.1567** | **0.1219** | <span style='color:red'><strong>0.1577</strong></span> |
| hHep_STRING_1000 | 0.3888 | <span style='color:red'><strong>0.4056</strong></span> | 0.3842 | 0.3927 | 0.3509 | 0.3901 |
| hHep_STRING_500 | 0.4320 | 0.5086 | **0.5185** | <span style='color:red'><strong>0.5366</strong></span> | 0.4102 | 0.4404 |
| hHep_Specific_1000 | 0.7286 | 0.7321 | 0.6707 | 0.7214 | <span style='color:red'><strong>0.7685</strong></span> | **0.7542** |
| mDC_Non-Specific_1000 | <span style='color:red'><strong>0.2471</strong></span> | 0.1696 | **0.1971** | **0.1712** | **0.1863** | **0.1839** |
| mDC_Non-Specific_500 | **0.2227** | 0.1627 | 0.0966 | **0.2053** | **0.1954** | <span style='color:red'><strong>0.2377</strong></span> |
| mDC_STRING_1000 | **0.3925** | 0.3854 | 0.3570 | 0.3733 | 0.3789 | <span style='color:red'><strong>0.4297</strong></span> |
| mDC_STRING_500 | **0.2730** | 0.2000 | **0.3275** | **0.3269** | <span style='color:red'><strong>0.3437</strong></span> | **0.2786** |
| mDC_Specific_1000 | <span style='color:red'><strong>0.3674</strong></span> | 0.3363 | 0.3052 | 0.3071 | 0.3040 | 0.3217 |
| mDC_Specific_500 | 0.1196 | 0.2275 | <span style='color:red'><strong>0.2778</strong></span> | 0.1449 | 0.2150 | 0.1715 |
| mESC500 [scGREAT] | **0.8094** | 0.8068 | <span style='color:red'><strong>0.8174</strong></span> | **0.8083** | **0.8079** | **0.8112** |
| mHSC-E_Non-Specific_1000 | 0.2260 | <span style='color:red'><strong>0.2280</strong></span> | 0.2008 | 0.1976 | 0.1852 | 0.2130 |
| mHSC-E_Non-Specific_500 | **0.1758** | 0.1520 | <span style='color:red'><strong>0.2405</strong></span> | **0.1729** | 0.1404 | **0.2318** |
| mHSC-E_STRING_1000 | **0.4798** | 0.4496 | 0.4288 | 0.4371 | **0.4754** | <span style='color:red'><strong>0.4814</strong></span> |
| mHSC-E_STRING_500 | **0.4640** | 0.4512 | **0.5261** | <span style='color:red'><strong>0.5368</strong></span> | **0.5310** | **0.4664** |
| mHSC-E_Specific_1000 | **0.8876** | 0.8841 | <span style='color:red'><strong>0.8877</strong></span> | **0.8842** | **0.8845** | 0.8828 |
| mHSC-E_Specific_500 | 0.6991 | 0.7446 | 0.6831 | **0.7449** | 0.7255 | <span style='color:red'><strong>0.7594</strong></span> |
| mHSC-GM_Non-Specific_1000 | **0.2270** | 0.2054 | 0.1803 | **0.2351** | **0.2114** | <span style='color:red'><strong>0.2353</strong></span> |
| mHSC-GM_Non-Specific_500 | 0.1914 | 0.2040 | <span style='color:red'><strong>0.2566</strong></span> | 0.1633 | **0.2092** | 0.1937 |
| mHSC-GM_STRING_1000 | 0.3462 | 0.3492 | 0.3189 | 0.2938 | **0.3659** | <span style='color:red'><strong>0.3716</strong></span> |
| mHSC-GM_STRING_500 | **0.2026** | 0.1291 | <span style='color:red'><strong>0.2859</strong></span> | **0.1583** | **0.2409** | **0.1440** |
| mHSC-GM_Specific_1000 | 0.8525 | <span style='color:red'><strong>0.8544</strong></span> | 0.8460 | 0.8465 | 0.8495 | 0.8483 |
| mHSC-GM_Specific_500 | <span style='color:red'><strong>0.8009</strong></span> | 0.7821 | 0.7580 | **0.7835** | 0.7727 | **0.7886** |
| mHSC-L_Non-Specific_1000 | **0.1891** | 0.1556 | **0.1799** | **0.1734** | <span style='color:red'><strong>0.2107</strong></span> | **0.1945** |
| mHSC-L_Non-Specific_500 | 0.1360 | 0.1455 | <span style='color:red'><strong>0.3058</strong></span> | **0.2417** | **0.2958** | **0.2363** |
| mHSC-L_STRING_1000 | <span style='color:red'><strong>0.3232</strong></span> | 0.2975 | **0.3225** | 0.2617 | 0.2860 | **0.3054** |
| mHSC-L_STRING_500 | **0.3322** | 0.3068 | **0.4180** | **0.3316** | **0.3660** | <span style='color:red'><strong>0.4878</strong></span> |
| mHSC-L_Specific_1000 | <span style='color:red'><strong>0.8905</strong></span> | 0.8815 | 0.8752 | 0.8794 | 0.8804 | **0.8826** |
| mHSC-L_Specific_500 | <span style='color:red'><strong>0.8220</strong></span> | 0.8205 | 0.7957 | 0.8110 | 0.8148 | 0.8198 |

### Classifier = mlp

| Dataset | minus | baseline | scGPT_human | v4_bias_rec_best | v4_plain_best | v4_type_pe_best |
|---|---:|---:|---:|---:|---:|---:|
| hESC500 [scGREAT] | **0.5574** | 0.5565 | 0.5367 | <span style='color:red'><strong>0.5591</strong></span> | 0.5491 | **0.5581** |
| hESC_Non-Specific_1000 | 0.1761 | <span style='color:red'><strong>0.1807</strong></span> | 0.1399 | 0.1593 | 0.1630 | 0.1638 |
| hESC_Non-Specific_500 | 0.1500 | 0.1812 | 0.1551 | **0.1987** | <span style='color:red'><strong>0.2060</strong></span> | 0.1623 |
| hESC_STRING_1000 | 0.2328 | 0.2343 | 0.2203 | **0.2367** | <span style='color:red'><strong>0.2638</strong></span> | **0.2626** |
| hESC_STRING_500 | **0.2084** | 0.0928 | 0.0723 | **0.0982** | 0.0518 | <span style='color:red'><strong>0.2819</strong></span> |
| hESC_Specific_1000 | 0.4730 | 0.4829 | 0.4446 | 0.4462 | **0.4980** | <span style='color:red'><strong>0.5093</strong></span> |
| hESC_Specific_500 | **0.4175** | 0.3922 | <span style='color:red'><strong>0.4853</strong></span> | 0.3853 | **0.4684** | **0.4207** |
| hHep_Non-Specific_1000 | **0.1731** | 0.1528 | <span style='color:red'><strong>0.2194</strong></span> | 0.1037 | 0.1486 | **0.1940** |
| hHep_Non-Specific_500 | **0.1042** | 0.0643 | <span style='color:red'><strong>0.1432</strong></span> | **0.1198** | **0.0709** | 0.0515 |
| hHep_STRING_1000 | **0.4348** | 0.4320 | <span style='color:red'><strong>0.4913</strong></span> | **0.4481** | **0.4551** | **0.4690** |
| hHep_STRING_500 | <span style='color:red'><strong>0.5887</strong></span> | 0.5849 | 0.5813 | 0.5659 | 0.5436 | **0.5875** |
| hHep_Specific_1000 | 0.7855 | 0.7896 | 0.7848 | 0.7888 | **0.7897** | <span style='color:red'><strong>0.8027</strong></span> |
| mDC_Non-Specific_1000 | **0.1609** | 0.1506 | **0.2176** | **0.2218** | <span style='color:red'><strong>0.2616</strong></span> | **0.2421** |
| mDC_Non-Specific_500 | **0.0213** | 0.0190 | **0.0488** | **0.0205** | **0.0195** | <span style='color:red'><strong>0.0489</strong></span> |
| mDC_STRING_1000 | 0.4912 | 0.5051 | <span style='color:red'><strong>0.5518</strong></span> | 0.4655 | **0.5202** | **0.5172** |
| mDC_STRING_500 | 0.2157 | 0.4749 | **0.5008** | 0.3990 | <span style='color:red'><strong>0.5486</strong></span> | 0.4632 |
| mDC_Specific_1000 | 0.3551 | 0.4313 | 0.4275 | 0.3971 | <span style='color:red'><strong>0.4376</strong></span> | 0.3927 |
| mDC_Specific_500 | 0.1175 | <span style='color:red'><strong>0.1765</strong></span> | 0.1527 | 0.0412 | 0.1224 | 0.0792 |
| mESC500 [scGREAT] | 0.8189 | 0.8197 | **0.8224** | 0.8195 | 0.8170 | <span style='color:red'><strong>0.8251</strong></span> |
| mHSC-E_Non-Specific_1000 | <span style='color:red'><strong>0.3129</strong></span> | 0.2927 | **0.3099** | 0.2777 | 0.2461 | **0.3091** |
| mHSC-E_Non-Specific_500 | <span style='color:red'><strong>0.3202</strong></span> | 0.2531 | **0.2800** | **0.3005** | **0.2683** | 0.2516 |
| mHSC-E_STRING_1000 | **0.5575** | 0.5325 | <span style='color:red'><strong>0.5699</strong></span> | 0.5080 | **0.5571** | **0.5609** |
| mHSC-E_STRING_500 | <span style='color:red'><strong>0.6377</strong></span> | 0.6209 | 0.6039 | **0.6269** | **0.6239** | 0.5980 |
| mHSC-E_Specific_1000 | **0.8925** | 0.8735 | <span style='color:red'><strong>0.8934</strong></span> | **0.8903** | **0.8836** | **0.8900** |
| mHSC-E_Specific_500 | **0.8064** | 0.8053 | <span style='color:red'><strong>0.8191</strong></span> | **0.8185** | **0.8108** | **0.8165** |
| mHSC-GM_Non-Specific_1000 | **0.2823** | 0.2552 | <span style='color:red'><strong>0.3011</strong></span> | 0.2378 | **0.2862** | 0.2520 |
| mHSC-GM_Non-Specific_500 | **0.1863** | 0.0942 | <span style='color:red'><strong>0.2832</strong></span> | **0.1101** | 0.0670 | **0.0998** |
| mHSC-GM_STRING_1000 | **0.4332** | 0.4211 | 0.3703 | 0.3830 | 0.3774 | <span style='color:red'><strong>0.4358</strong></span> |
| mHSC-GM_STRING_500 | **0.2686** | 0.1530 | **0.1913** | <span style='color:red'><strong>0.2862</strong></span> | **0.1768** | **0.2183** |
| mHSC-GM_Specific_1000 | **0.8616** | 0.8595 | <span style='color:red'><strong>0.8669</strong></span> | **0.8613** | 0.8592 | 0.8577 |
| mHSC-GM_Specific_500 | **0.8276** | 0.7984 | <span style='color:red'><strong>0.8310</strong></span> | **0.8146** | **0.8009** | **0.8108** |
| mHSC-L_Non-Specific_1000 | <span style='color:red'><strong>0.2677</strong></span> | 0.1796 | **0.2521** | **0.1820** | **0.2574** | **0.2236** |
| mHSC-L_Non-Specific_500 | **0.2444** | 0.1345 | <span style='color:red'><strong>0.3325</strong></span> | **0.1883** | **0.1437** | **0.1617** |
| mHSC-L_STRING_1000 | <span style='color:red'><strong>0.3908</strong></span> | 0.3672 | 0.3147 | 0.2998 | 0.3555 | 0.3361 |
| mHSC-L_STRING_500 | **0.3402** | 0.3393 | **0.4495** | <span style='color:red'><strong>0.4886</strong></span> | **0.4186** | 0.3240 |
| mHSC-L_Specific_1000 | <span style='color:red'><strong>0.8965</strong></span> | 0.8817 | **0.8907** | **0.8832** | **0.8861** | **0.8859** |
| mHSC-L_Specific_500 | **0.8419** | 0.8275 | <span style='color:red'><strong>0.8534</strong></span> | **0.8474** | **0.8366** | **0.8370** |

