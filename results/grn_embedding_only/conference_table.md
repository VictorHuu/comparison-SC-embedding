# GRN Embedding Only (Conference-style Tables)

说明：`-`表示该组合无结果；**加粗**表示优于同一行 baseline；<span style="color:red"><strong>红色加粗</strong></span>表示该行最优。

## AUROC

### Classifier = lr

| Dataset | minus | baseline | scGPT_human | v4_bias_rec_best | v4_plain_best | v4_type_pe_best | random_256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| hESC500 | 0.8569 | <span style='color:red'><strong>0.8670</strong></span> | 0.8636 | 0.8638 | 0.8635 | 0.8600 | 0.8533 |
| hESC500->hHep500 | <span style='color:red'><strong>1.0000</strong></span> | 0.0000 | 0.0000 | 0.0000 | 0.0000 | <span style='color:red'><strong>1.0000</strong></span> | - |
| hESC500->mESC500 | <span style='color:red'><strong>0.6148</strong></span> | 0.6049 | 0.5839 | 0.6001 | 0.5803 | 0.5720 | - |
| hESC500->mHSC-E500 | **0.5554** | 0.5423 | **0.5957** | <span style='color:red'><strong>0.6047</strong></span> | 0.5394 | **0.5452** | - |
| hESC500->mHSC-GM500 | 0.4947 | 0.5371 | **0.5714** | <span style='color:red'><strong>0.5716</strong></span> | **0.5575** | 0.5229 | - |
| hESC500->mHSC-L500 | 0.4657 | 0.5212 | **0.5534** | <span style='color:red'><strong>0.5887</strong></span> | **0.5560** | 0.4993 | - |
| hHep500 | 0.0000 | <span style='color:red'><strong>1.0000</strong></span> | <span style='color:red'><strong>1.0000</strong></span> | <span style='color:red'><strong>1.0000</strong></span> | <span style='color:red'><strong>1.0000</strong></span> | 0.0000 | <span style='color:red'><strong>1.0000</strong></span> |
| hHep500->hESC500 | **0.5024** | 0.4915 | **0.5038** | **0.4974** | <span style='color:red'><strong>0.5125</strong></span> | **0.4969** | - |
| hHep500->mESC500 | <span style='color:red'><strong>0.5117</strong></span> | 0.5020 | **0.5034** | **0.5060** | 0.4983 | **0.5048** | - |
| hHep500->mHSC-E500 | <span style='color:red'><strong>0.5506</strong></span> | 0.5438 | 0.5333 | 0.4976 | 0.4505 | **0.5505** | - |
| hHep500->mHSC-GM500 | **0.5402** | 0.4845 | <span style='color:red'><strong>0.5873</strong></span> | **0.5295** | 0.4632 | **0.5060** | - |
| hHep500->mHSC-L500 | **0.5037** | 0.4712 | <span style='color:red'><strong>0.5215</strong></span> | **0.4859** | **0.5105** | **0.4932** | - |
| mESC500 | **0.8938** | 0.8932 | <span style='color:red'><strong>0.8976</strong></span> | **0.8935** | **0.8932** | **0.8946** | 0.8738 |
| mESC500->hESC500 | **0.5682** | 0.5013 | **0.5097** | 0.4549 | 0.4867 | <span style='color:red'><strong>0.6196</strong></span> | - |
| mESC500->hHep500 | 0.0000 | 0.0000 | <span style='color:red'><strong>1.0000</strong></span> | <span style='color:red'><strong>1.0000</strong></span> | <span style='color:red'><strong>1.0000</strong></span> | <span style='color:red'><strong>1.0000</strong></span> | - |
| mESC500->mHSC-E500 | <span style='color:red'><strong>0.4301</strong></span> | 0.4149 | 0.3963 | 0.3873 | 0.3768 | **0.4195** | - |
| mESC500->mHSC-GM500 | <span style='color:red'><strong>0.4621</strong></span> | 0.4517 | 0.4040 | 0.4408 | 0.4052 | 0.4312 | - |
| mESC500->mHSC-L500 | 0.4349 | <span style='color:red'><strong>0.5131</strong></span> | 0.4190 | 0.4573 | 0.4279 | 0.4112 | - |
| mHSC-E500 | **0.6626** | 0.6292 | <span style='color:red'><strong>0.6939</strong></span> | 0.6168 | **0.6380** | **0.6632** | 0.5597 |
| mHSC-E500->hESC500 | <span style='color:red'><strong>0.5260</strong></span> | 0.5196 | 0.5150 | 0.5141 | 0.5014 | 0.4988 | - |
| mHSC-E500->hHep500 | <span style='color:red'><strong>1.0000</strong></span> | <span style='color:red'><strong>1.0000</strong></span> | 0.0000 | <span style='color:red'><strong>1.0000</strong></span> | 0.0000 | 0.0000 | - |
| mHSC-E500->mESC500 | **0.5156** | 0.4991 | **0.5194** | <span style='color:red'><strong>0.5212</strong></span> | **0.5050** | **0.5027** | - |
| mHSC-E500->mHSC-GM500 | 0.6229 | <span style='color:red'><strong>0.6585</strong></span> | 0.6386 | 0.5646 | 0.6379 | 0.6153 | - |
| mHSC-E500->mHSC-L500 | **0.5924** | 0.5784 | <span style='color:red'><strong>0.6535</strong></span> | 0.5671 | **0.6104** | **0.6078** | - |
| mHSC-GM500 | **0.7136** | 0.7082 | 0.6959 | **0.7293** | **0.7279** | <span style='color:red'><strong>0.7420</strong></span> | 0.6790 |
| mHSC-GM500->hESC500 | **0.5225** | 0.5165 | 0.5108 | <span style='color:red'><strong>0.5467</strong></span> | 0.4988 | **0.5259** | - |
| mHSC-GM500->hHep500 | 0.0000 | <span style='color:red'><strong>1.0000</strong></span> | <span style='color:red'><strong>1.0000</strong></span> | <span style='color:red'><strong>1.0000</strong></span> | <span style='color:red'><strong>1.0000</strong></span> | 0.0000 | - |
| mHSC-GM500->mESC500 | 0.5005 | <span style='color:red'><strong>0.5223</strong></span> | 0.5101 | 0.5127 | 0.5209 | 0.5190 | - |
| mHSC-GM500->mHSC-E500 | <span style='color:red'><strong>0.6894</strong></span> | 0.6739 | 0.6164 | 0.6553 | 0.6639 | 0.5951 | - |
| mHSC-GM500->mHSC-L500 | 0.7135 | 0.7363 | <span style='color:red'><strong>0.7534</strong></span> | 0.6988 | **0.7470** | 0.7295 | - |
| mHSC-L500 | <span style='color:red'><strong>0.7487</strong></span> | 0.7397 | 0.7336 | 0.7289 | **0.7467** | 0.7355 | 0.6978 |
| mHSC-L500->hESC500 | 0.5304 | 0.5320 | 0.5214 | <span style='color:red'><strong>0.5408</strong></span> | 0.5098 | 0.5157 | - |
| mHSC-L500->hHep500 | 0.0000 | <span style='color:red'><strong>1.0000</strong></span> | 0.0000 | <span style='color:red'><strong>1.0000</strong></span> | 0.0000 | <span style='color:red'><strong>1.0000</strong></span> | - |
| mHSC-L500->mESC500 | 0.5065 | 0.5354 | 0.5031 | 0.5256 | 0.5236 | <span style='color:red'><strong>0.5362</strong></span> | - |
| mHSC-L500->mHSC-E500 | 0.6689 | <span style='color:red'><strong>0.6791</strong></span> | 0.6313 | 0.6566 | 0.6058 | 0.6045 | - |
| mHSC-L500->mHSC-GM500 | 0.7600 | 0.7655 | <span style='color:red'><strong>0.8212</strong></span> | 0.7618 | **0.7667** | **0.7797** | - |

### Classifier = mlp

| Dataset | minus | baseline | scGPT_human | v4_bias_rec_best | v4_plain_best | v4_type_pe_best | random_256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| hESC500 | **0.8563** | 0.8562 | 0.8481 | <span style='color:red'><strong>0.8612</strong></span> | 0.8475 | 0.8559 | 0.8534 |
| hESC500->hHep500 | <span style='color:red'><strong>1.0000</strong></span> | 0.0000 | <span style='color:red'><strong>1.0000</strong></span> | 0.0000 | 0.0000 | <span style='color:red'><strong>1.0000</strong></span> | - |
| hESC500->mESC500 | <span style='color:red'><strong>0.5716</strong></span> | 0.5451 | 0.4861 | 0.5413 | 0.5005 | 0.5261 | - |
| hESC500->mHSC-E500 | <span style='color:red'><strong>0.6207</strong></span> | 0.6159 | 0.5777 | 0.6008 | 0.5703 | 0.5949 | - |
| hESC500->mHSC-GM500 | 0.5230 | 0.5859 | 0.5584 | **0.5874** | <span style='color:red'><strong>0.6047</strong></span> | 0.5637 | - |
| hESC500->mHSC-L500 | 0.5074 | <span style='color:red'><strong>0.5741</strong></span> | 0.5645 | 0.5707 | 0.5583 | 0.5460 | - |
| mESC500 | 0.8980 | 0.8988 | <span style='color:red'><strong>0.9000</strong></span> | 0.8983 | 0.8965 | **0.9000** | 0.8987 |
| mESC500->hESC500 | <span style='color:red'><strong>0.6201</strong></span> | 0.5909 | 0.5803 | 0.5592 | 0.5077 | **0.6111** | - |
| mESC500->hHep500 | 0.0000 | 0.0000 | <span style='color:red'><strong>1.0000</strong></span> | 0.0000 | 0.0000 | <span style='color:red'><strong>1.0000</strong></span> | - |
| mESC500->mHSC-E500 | <span style='color:red'><strong>0.4280</strong></span> | 0.3765 | **0.4207** | 0.3609 | **0.4018** | 0.3726 | - |
| mESC500->mHSC-GM500 | **0.4477** | 0.4328 | 0.4179 | <span style='color:red'><strong>0.4554</strong></span> | **0.4479** | 0.4101 | - |
| mESC500->mHSC-L500 | 0.4325 | 0.4560 | 0.4379 | 0.4495 | **0.4577** | <span style='color:red'><strong>0.4580</strong></span> | - |
| mHSC-E500 | 0.6563 | 0.7152 | <span style='color:red'><strong>0.7187</strong></span> | 0.6989 | 0.6825 | 0.7001 | 0.6146 |
| mHSC-E500->hESC500 | 0.4811 | 0.5330 | **0.5448** | <span style='color:red'><strong>0.5502</strong></span> | 0.4917 | **0.5333** | - |
| mHSC-E500->hHep500 | <span style='color:red'><strong>1.0000</strong></span> | <span style='color:red'><strong>1.0000</strong></span> | 0.0000 | <span style='color:red'><strong>1.0000</strong></span> | 0.0000 | <span style='color:red'><strong>1.0000</strong></span> | - |
| mHSC-E500->mESC500 | 0.5170 | <span style='color:red'><strong>0.5451</strong></span> | 0.5152 | 0.4998 | 0.5073 | 0.5314 | - |
| mHSC-E500->mHSC-GM500 | 0.6446 | <span style='color:red'><strong>0.7016</strong></span> | 0.6940 | 0.6564 | 0.6363 | 0.6910 | - |
| mHSC-E500->mHSC-L500 | 0.6271 | 0.6814 | <span style='color:red'><strong>0.6884</strong></span> | 0.6477 | 0.6491 | **0.6868** | - |
| mHSC-GM500 | 0.7289 | 0.7438 | 0.7418 | **0.7454** | 0.7271 | <span style='color:red'><strong>0.7510</strong></span> | 0.6516 |
| mHSC-GM500->hESC500 | 0.4679 | 0.5606 | 0.5533 | 0.5455 | 0.5328 | <span style='color:red'><strong>0.6058</strong></span> | - |
| mHSC-GM500->hHep500 | 0.0000 | <span style='color:red'><strong>1.0000</strong></span> | <span style='color:red'><strong>1.0000</strong></span> | <span style='color:red'><strong>1.0000</strong></span> | <span style='color:red'><strong>1.0000</strong></span> | <span style='color:red'><strong>1.0000</strong></span> | - |
| mHSC-GM500->mESC500 | 0.5045 | <span style='color:red'><strong>0.5454</strong></span> | 0.5165 | 0.5298 | 0.5042 | 0.5281 | - |
| mHSC-GM500->mHSC-E500 | 0.7373 | 0.7495 | 0.7367 | <span style='color:red'><strong>0.7616</strong></span> | 0.7082 | 0.7259 | - |
| mHSC-GM500->mHSC-L500 | **0.8188** | 0.7749 | <span style='color:red'><strong>0.8240</strong></span> | **0.8060** | **0.8034** | **0.7941** | - |
| mHSC-L500 | **0.7716** | 0.7434 | **0.7698** | **0.7530** | <span style='color:red'><strong>0.7793</strong></span> | **0.7661** | 0.7115 |
| mHSC-L500->hESC500 | 0.5243 | <span style='color:red'><strong>0.5719</strong></span> | 0.5293 | 0.5264 | 0.5416 | 0.5282 | - |
| mHSC-L500->hHep500 | <span style='color:red'><strong>1.0000</strong></span> | <span style='color:red'><strong>1.0000</strong></span> | 0.0000 | <span style='color:red'><strong>1.0000</strong></span> | <span style='color:red'><strong>1.0000</strong></span> | <span style='color:red'><strong>1.0000</strong></span> | - |
| mHSC-L500->mESC500 | 0.5294 | <span style='color:red'><strong>0.5497</strong></span> | 0.5095 | 0.5447 | 0.5104 | 0.5355 | - |
| mHSC-L500->mHSC-E500 | 0.7345 | <span style='color:red'><strong>0.7475</strong></span> | 0.7292 | 0.6980 | 0.7203 | 0.7450 | - |
| mHSC-L500->mHSC-GM500 | 0.8460 | 0.8698 | <span style='color:red'><strong>0.8955</strong></span> | **0.8822** | **0.8862** | 0.8577 | - |

## AUPRC

### Classifier = lr

| Dataset | minus | baseline | scGPT_human | v4_bias_rec_best | v4_plain_best | v4_type_pe_best | random_256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| hESC500 | 0.5577 | <span style='color:red'><strong>0.5829</strong></span> | 0.5790 | 0.5668 | 0.5767 | 0.5554 | 0.5255 |
| hESC500->hHep500 | <span style='color:red'><strong>1.0000</strong></span> | 0.5000 | 0.5000 | 0.5000 | 0.5000 | <span style='color:red'><strong>1.0000</strong></span> | - |
| hESC500->mESC500 | <span style='color:red'><strong>0.3971</strong></span> | 0.3823 | 0.3701 | 0.3619 | 0.3580 | 0.3572 | - |
| hESC500->mHSC-E500 | **0.6907** | 0.6576 | **0.7139** | <span style='color:red'><strong>0.7190</strong></span> | **0.6845** | **0.6738** | - |
| hESC500->mHSC-GM500 | 0.6183 | 0.6269 | **0.6755** | <span style='color:red'><strong>0.6920</strong></span> | **0.6725** | 0.6265 | - |
| hESC500->mHSC-L500 | 0.6091 | 0.6252 | **0.6608** | <span style='color:red'><strong>0.6941</strong></span> | **0.6929** | 0.6058 | - |
| hHep500 | 0.5000 | <span style='color:red'><strong>1.0000</strong></span> | <span style='color:red'><strong>1.0000</strong></span> | <span style='color:red'><strong>1.0000</strong></span> | <span style='color:red'><strong>1.0000</strong></span> | 0.5000 | <span style='color:red'><strong>1.0000</strong></span> |
| hHep500->hESC500 | **0.1483** | 0.1443 | **0.1534** | **0.1501** | <span style='color:red'><strong>0.1801</strong></span> | **0.1536** | - |
| hHep500->mESC500 | 0.3178 | <span style='color:red'><strong>0.3214</strong></span> | 0.3183 | 0.3145 | 0.3107 | 0.3104 | - |
| hHep500->mHSC-E500 | 0.6846 | 0.6973 | 0.6851 | 0.6321 | 0.5982 | <span style='color:red'><strong>0.7117</strong></span> | - |
| hHep500->mHSC-GM500 | **0.6243** | 0.6098 | <span style='color:red'><strong>0.7006</strong></span> | **0.6466** | 0.5818 | **0.6291** | - |
| hHep500->mHSC-L500 | 0.6115 | 0.6167 | <span style='color:red'><strong>0.6531</strong></span> | 0.6041 | **0.6257** | **0.6242** | - |
| mESC500 | **0.8094** | 0.8068 | <span style='color:red'><strong>0.8174</strong></span> | **0.8083** | **0.8079** | **0.8112** | 0.7514 |
| mESC500->hESC500 | **0.1839** | 0.1592 | 0.1537 | 0.1271 | **0.1629** | <span style='color:red'><strong>0.2059</strong></span> | - |
| mESC500->hHep500 | 0.5000 | 0.5000 | <span style='color:red'><strong>1.0000</strong></span> | <span style='color:red'><strong>1.0000</strong></span> | <span style='color:red'><strong>1.0000</strong></span> | <span style='color:red'><strong>1.0000</strong></span> | - |
| mESC500->mHSC-E500 | <span style='color:red'><strong>0.6096</strong></span> | 0.5773 | **0.5924** | **0.5790** | **0.5880** | **0.5890** | - |
| mESC500->mHSC-GM500 | <span style='color:red'><strong>0.6136</strong></span> | 0.5760 | 0.5644 | **0.5851** | 0.5515 | 0.5642 | - |
| mESC500->mHSC-L500 | 0.5962 | <span style='color:red'><strong>0.6453</strong></span> | 0.5841 | 0.6103 | 0.6034 | 0.5760 | - |
| mHSC-E500 | 0.7534 | 0.7545 | <span style='color:red'><strong>0.7926</strong></span> | 0.7370 | 0.7353 | **0.7583** | 0.6929 |
| mHSC-E500->hESC500 | <span style='color:red'><strong>0.1640</strong></span> | 0.1610 | **0.1612** | 0.1552 | 0.1490 | 0.1525 | - |
| mHSC-E500->hHep500 | <span style='color:red'><strong>1.0000</strong></span> | <span style='color:red'><strong>1.0000</strong></span> | 0.5000 | <span style='color:red'><strong>1.0000</strong></span> | 0.5000 | 0.5000 | - |
| mHSC-E500->mESC500 | <span style='color:red'><strong>0.3310</strong></span> | 0.3136 | **0.3210** | **0.3243** | 0.3099 | **0.3191** | - |
| mHSC-E500->mHSC-GM500 | 0.6946 | <span style='color:red'><strong>0.7516</strong></span> | 0.7273 | 0.6543 | 0.7150 | 0.6998 | - |
| mHSC-E500->mHSC-L500 | **0.6721** | 0.6558 | <span style='color:red'><strong>0.7380</strong></span> | **0.6604** | **0.6829** | **0.6764** | - |
| mHSC-GM500 | 0.7691 | 0.7764 | 0.7387 | **0.7977** | **0.7833** | <span style='color:red'><strong>0.7996</strong></span> | 0.7619 |
| mHSC-GM500->hESC500 | **0.1638** | 0.1543 | **0.1597** | <span style='color:red'><strong>0.1755</strong></span> | 0.1475 | **0.1637** | - |
| mHSC-GM500->hHep500 | 0.5000 | <span style='color:red'><strong>1.0000</strong></span> | <span style='color:red'><strong>1.0000</strong></span> | <span style='color:red'><strong>1.0000</strong></span> | <span style='color:red'><strong>1.0000</strong></span> | 0.5000 | - |
| mHSC-GM500->mESC500 | 0.3128 | <span style='color:red'><strong>0.3311</strong></span> | 0.3214 | 0.3207 | 0.3296 | 0.3297 | - |
| mHSC-GM500->mHSC-E500 | <span style='color:red'><strong>0.7805</strong></span> | 0.7693 | 0.7136 | 0.7534 | 0.7382 | 0.6775 | - |
| mHSC-GM500->mHSC-L500 | 0.7837 | 0.7869 | **0.7894** | 0.7629 | <span style='color:red'><strong>0.8024</strong></span> | **0.7888** | - |
| mHSC-L500 | **0.8116** | 0.8072 | **0.8091** | 0.7922 | **0.8122** | <span style='color:red'><strong>0.8162</strong></span> | 0.7965 |
| mHSC-L500->hESC500 | 0.1674 | <span style='color:red'><strong>0.1674</strong></span> | 0.1620 | 0.1621 | 0.1576 | 0.1559 | - |
| mHSC-L500->hHep500 | 0.5000 | <span style='color:red'><strong>1.0000</strong></span> | 0.5000 | <span style='color:red'><strong>1.0000</strong></span> | 0.5000 | <span style='color:red'><strong>1.0000</strong></span> | - |
| mHSC-L500->mESC500 | 0.3172 | 0.3390 | 0.3178 | 0.3339 | 0.3293 | <span style='color:red'><strong>0.3447</strong></span> | - |
| mHSC-L500->mHSC-E500 | 0.7603 | <span style='color:red'><strong>0.7850</strong></span> | 0.7273 | 0.7495 | 0.6974 | 0.7088 | - |
| mHSC-L500->mHSC-GM500 | 0.8106 | 0.8150 | <span style='color:red'><strong>0.8515</strong></span> | 0.8105 | 0.8014 | **0.8150** | - |

### Classifier = mlp

| Dataset | minus | baseline | scGPT_human | v4_bias_rec_best | v4_plain_best | v4_type_pe_best | random_256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| hESC500 | **0.5574** | 0.5565 | 0.5367 | <span style='color:red'><strong>0.5591</strong></span> | 0.5491 | **0.5581** | 0.5301 |
| hESC500->hHep500 | <span style='color:red'><strong>1.0000</strong></span> | 0.5000 | <span style='color:red'><strong>1.0000</strong></span> | 0.5000 | 0.5000 | <span style='color:red'><strong>1.0000</strong></span> | - |
| hESC500->mESC500 | <span style='color:red'><strong>0.3637</strong></span> | 0.3447 | 0.3091 | **0.3537** | 0.3141 | 0.3404 | - |
| hESC500->mHSC-E500 | 0.7331 | <span style='color:red'><strong>0.7506</strong></span> | 0.6973 | 0.7327 | 0.7021 | 0.7029 | - |
| hESC500->mHSC-GM500 | 0.6231 | <span style='color:red'><strong>0.7015</strong></span> | 0.6609 | 0.7011 | 0.7001 | 0.6689 | - |
| hESC500->mHSC-L500 | 0.6280 | 0.6806 | 0.6762 | **0.6860** | <span style='color:red'><strong>0.6869</strong></span> | 0.6459 | - |
| mESC500 | 0.8189 | 0.8197 | **0.8224** | 0.8195 | 0.8170 | <span style='color:red'><strong>0.8251</strong></span> | 0.8088 |
| mESC500->hESC500 | **0.2131** | 0.1913 | **0.1933** | 0.1892 | 0.1789 | <span style='color:red'><strong>0.2167</strong></span> | - |
| mESC500->hHep500 | 0.5000 | 0.5000 | <span style='color:red'><strong>1.0000</strong></span> | 0.5000 | 0.5000 | <span style='color:red'><strong>1.0000</strong></span> | - |
| mESC500->mHSC-E500 | **0.6043** | 0.5889 | <span style='color:red'><strong>0.6101</strong></span> | 0.5607 | **0.5908** | 0.5665 | - |
| mESC500->mHSC-GM500 | **0.6183** | 0.5815 | 0.5753 | <span style='color:red'><strong>0.6278</strong></span> | 0.5803 | 0.5728 | - |
| mESC500->mHSC-L500 | 0.5845 | 0.6034 | 0.6024 | <span style='color:red'><strong>0.6181</strong></span> | 0.5936 | **0.6094** | - |
| mHSC-E500 | 0.7852 | <span style='color:red'><strong>0.8333</strong></span> | 0.8163 | 0.8175 | 0.7943 | 0.8122 | 0.7526 |
| mHSC-E500->hESC500 | 0.1384 | 0.1639 | <span style='color:red'><strong>0.1817</strong></span> | **0.1775** | 0.1452 | **0.1641** | - |
| mHSC-E500->hHep500 | <span style='color:red'><strong>1.0000</strong></span> | <span style='color:red'><strong>1.0000</strong></span> | 0.5000 | <span style='color:red'><strong>1.0000</strong></span> | 0.5000 | <span style='color:red'><strong>1.0000</strong></span> | - |
| mHSC-E500->mESC500 | 0.3124 | 0.3280 | 0.3141 | 0.2982 | 0.3048 | <span style='color:red'><strong>0.3282</strong></span> | - |
| mHSC-E500->mHSC-GM500 | 0.7481 | <span style='color:red'><strong>0.8069</strong></span> | 0.7957 | 0.7838 | 0.7619 | 0.7898 | - |
| mHSC-E500->mHSC-L500 | 0.7386 | 0.7791 | **0.7842** | 0.7613 | 0.7670 | <span style='color:red'><strong>0.7853</strong></span> | - |
| mHSC-GM500 | 0.8102 | <span style='color:red'><strong>0.8232</strong></span> | 0.7957 | 0.8161 | 0.8093 | 0.8220 | 0.7622 |
| mHSC-GM500->hESC500 | 0.1389 | 0.1769 | 0.1702 | **0.1797** | 0.1533 | <span style='color:red'><strong>0.2112</strong></span> | - |
| mHSC-GM500->hHep500 | 0.5000 | <span style='color:red'><strong>1.0000</strong></span> | <span style='color:red'><strong>1.0000</strong></span> | <span style='color:red'><strong>1.0000</strong></span> | <span style='color:red'><strong>1.0000</strong></span> | <span style='color:red'><strong>1.0000</strong></span> | - |
| mHSC-GM500->mESC500 | 0.3045 | <span style='color:red'><strong>0.3237</strong></span> | 0.3189 | 0.3226 | 0.3033 | 0.3212 | - |
| mHSC-GM500->mHSC-E500 | 0.8259 | 0.8310 | 0.8287 | <span style='color:red'><strong>0.8430</strong></span> | 0.7965 | **0.8324** | - |
| mHSC-GM500->mHSC-L500 | <span style='color:red'><strong>0.8742</strong></span> | 0.8398 | **0.8682** | **0.8634** | **0.8632** | **0.8640** | - |
| mHSC-L500 | **0.8420** | 0.8200 | **0.8286** | 0.8150 | <span style='color:red'><strong>0.8456</strong></span> | **0.8410** | 0.8031 |
| mHSC-L500->hESC500 | 0.1609 | <span style='color:red'><strong>0.1813</strong></span> | 0.1622 | 0.1629 | 0.1673 | 0.1612 | - |
| mHSC-L500->hHep500 | <span style='color:red'><strong>1.0000</strong></span> | <span style='color:red'><strong>1.0000</strong></span> | 0.5000 | <span style='color:red'><strong>1.0000</strong></span> | <span style='color:red'><strong>1.0000</strong></span> | <span style='color:red'><strong>1.0000</strong></span> | - |
| mHSC-L500->mESC500 | 0.3300 | 0.3355 | 0.3130 | <span style='color:red'><strong>0.3417</strong></span> | 0.3158 | **0.3365** | - |
| mHSC-L500->mHSC-E500 | 0.8188 | <span style='color:red'><strong>0.8292</strong></span> | 0.8188 | 0.7816 | 0.8125 | 0.8247 | - |
| mHSC-L500->mHSC-GM500 | 0.8753 | 0.9041 | <span style='color:red'><strong>0.9303</strong></span> | **0.9154** | **0.9086** | 0.8948 | - |

