# AUROC matrices by setting (embedding × train_dataset)

## coverage_matched + lr | AUROC matrix (embedding × train_dataset)

| Embedding | hESC | hHep | mDC | mHSC-E | mHSC-GM | mHSC-L |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.553340 ± 0.124807 | 0.499402 ± 0.111660 | 0.416033 ± 0.138085 | 0.635447 ± 0.054419 | **0.767839 ± 0.040389** | **0.773093 ± 0.028095** |
| minus | <span style="color:red">0.564712 ± 0.076704</span> | <span style="color:red">0.591100 ± 0.120288</span> | **<span style="color:red">0.546040 ± 0.051816</span>** | <span style="color:red">0.648680 ± 0.069086</span> | 0.659162 ± 0.030558 | 0.633138 ± 0.054041 |
| scGPT_human | <span style="color:red">0.613403 ± 0.040084</span> | <span style="color:red">0.620721 ± 0.131478</span> | <span style="color:red">0.462725 ± 0.100585</span> | 0.586907 ± 0.084375 | 0.758417 ± 0.087719 | 0.601712 ± 0.218244 |
| v4_bias_rec_best | **<span style="color:red">0.672298 ± 0.105960</span>** | <span style="color:red">0.659456 ± 0.085983</span> | 0.242932 ± 0.123854 | <span style="color:red">0.656159 ± 0.052191</span> | 0.730575 ± 0.088260 | 0.641482 ± 0.033981 |
| v4_plain_best | <span style="color:red">0.572940 ± 0.111621</span> | <span style="color:red">0.599921 ± 0.140749</span> | <span style="color:red">0.464396 ± 0.080310</span> | **<span style="color:red">0.694123 ± 0.061243</span>** | 0.746566 ± 0.002237 | 0.701002 ± 0.027717 |
| v4_type_pe_best | <span style="color:red">0.577888 ± 0.137256</span> | **<span style="color:red">0.714920 ± 0.144843</span>** | 0.339868 ± 0.073666 | 0.626405 ± 0.014425 | 0.713265 ± 0.084610 | 0.630327 ± 0.186221 |

## coverage_matched + mlp | AUROC matrix (embedding × train_dataset)

| Embedding | hESC | hHep | mDC | mESC | mHSC-E | mHSC-GM | mHSC-L |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.506603 ± 0.134891 | 0.569067 ± 0.137257 | 0.473429 ± 0.183481 | **0.581248 ± 0.063990** | 0.678287 ± 0.054648 | **0.832197 ± 0.055692** | **0.807135 ± 0.051806** |
| minus | 0.494521 ± 0.072035 | <span style="color:red">0.609284 ± 0.080892</span> | <span style="color:red">0.508730 ± 0.035349</span> | 0.527818 ± 0.099580 | 0.638081 ± 0.084424 | 0.712294 ± 0.002739 | 0.713893 ± 0.105144 |
| scGPT_human | **<span style="color:red">0.561933 ± 0.100959</span>** | <span style="color:red">0.598148 ± 0.067424</span> | **<span style="color:red">0.516552 ± 0.010368</span>** | 0.465094 ± 0.035183 | 0.629616 ± 0.086923 | 0.802713 ± 0.073719 | 0.648520 ± 0.196579 |
| v4_bias_rec_best | <span style="color:red">0.543690 ± 0.153596</span> | <span style="color:red">0.651217 ± 0.130126</span> | 0.308657 ± 0.082062 | 0.512021 ± 0.042111 | 0.678037 ± 0.054536 | 0.750030 ± 0.055542 | 0.785074 ± 0.043467 |
| v4_plain_best | 0.489443 ± 0.061999 | <span style="color:red">0.571204 ± 0.144643</span> | <span style="color:red">0.505313 ± 0.059466</span> | 0.483724 ± 0.047419 | **<span style="color:red">0.699375 ± 0.071831</span>** | 0.784074 ± 0.014157 | 0.787824 ± 0.028808 |
| v4_type_pe_best | <span style="color:red">0.553039 ± 0.143729</span> | **<span style="color:red">0.679792 ± 0.169383</span>** | 0.394497 ± 0.112262 | 0.503860 ± 0.075354 | 0.668946 ± 0.064191 | 0.732942 ± 0.157748 | 0.662852 ± 0.222779 |

## native + lr | AUROC matrix (embedding × train_dataset)

| Embedding | hESC | hHep | mDC | mHSC-E | mHSC-GM | mHSC-L |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.588572 ± 0.098392 | 0.643520 ± 0.131841 | 0.371128 ± 0.114806 | 0.651820 ± 0.119345 | 0.663105 ± 0.155297 | 0.651372 ± 0.133514 |
| minus | 0.550907 ± 0.064736 | 0.626166 ± 0.092723 | <span style="color:red">0.455201 ± 0.082311</span> | 0.616385 ± 0.114167 | <span style="color:red">0.663731 ± 0.115128</span> | 0.592019 ± 0.120809 |
| scGPT_human | **<span style="color:red">0.618202 ± 0.094000</span>** | 0.640041 ± 0.089260 | **<span style="color:red">0.469674 ± 0.077823</span>** | 0.573736 ± 0.122387 | 0.640809 ± 0.171993 | 0.602470 ± 0.112625 |
| v4_bias_rec_best | 0.538437 ± 0.135132 | **<span style="color:red">0.665575 ± 0.106414</span>** | 0.312124 ± 0.127372 | 0.640704 ± 0.100669 | <span style="color:red">0.670306 ± 0.122162</span> | 0.593457 ± 0.080450 |
| v4_plain_best | 0.586881 ± 0.061607 | 0.637061 ± 0.095001 | 0.359866 ± 0.088096 | **<span style="color:red">0.674132 ± 0.122816</span>** | <span style="color:red">0.677429 ± 0.091827</span> | 0.612785 ± 0.087253 |
| v4_type_pe_best | <span style="color:red">0.588823 ± 0.119861</span> | 0.591774 ± 0.159593 | 0.365038 ± 0.089333 | 0.617133 ± 0.098357 | **<span style="color:red">0.696392 ± 0.101044</span>** | **<span style="color:red">0.658631 ± 0.104280</span>** |

## native + mlp | AUROC matrix (embedding × train_dataset)

| Embedding | hESC | hHep | mDC | mESC | mHSC-E | mHSC-GM | mHSC-L |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.574821 ± 0.107696 | 0.713162 ± 0.142827 | 0.384023 ± 0.096351 | **0.551514 ± 0.061029** | 0.671738 ± 0.119599 | **0.738900 ± 0.152084** | 0.703303 ± 0.149981 |
| minus | <span style="color:red">0.582036 ± 0.112034</span> | 0.692463 ± 0.116045 | <span style="color:red">0.429945 ± 0.048062</span> | 0.526030 ± 0.071298 | 0.620382 ± 0.129063 | 0.686738 ± 0.137031 | 0.678780 ± 0.143330 |
| scGPT_human | **<span style="color:red">0.643199 ± 0.100633</span>** | 0.689992 ± 0.118414 | **<span style="color:red">0.455415 ± 0.052359</span>** | 0.475363 ± 0.046115 | 0.605435 ± 0.142228 | 0.680618 ± 0.178377 | 0.676449 ± 0.116438 |
| v4_bias_rec_best | 0.542131 ± 0.148778 | **<span style="color:red">0.748468 ± 0.136372</span>** | 0.310293 ± 0.104364 | 0.500645 ± 0.038279 | 0.651992 ± 0.113829 | 0.714460 ± 0.127777 | 0.673891 ± 0.151257 |
| v4_plain_best | <span style="color:red">0.588933 ± 0.092029</span> | 0.675269 ± 0.123895 | <span style="color:red">0.387388 ± 0.072903</span> | 0.482893 ± 0.036344 | **<span style="color:red">0.673618 ± 0.134891</span>** | 0.718849 ± 0.124904 | 0.686780 ± 0.128604 |
| v4_type_pe_best | <span style="color:red">0.597921 ± 0.117467</span> | 0.675767 ± 0.152840 | <span style="color:red">0.384779 ± 0.060796</span> | 0.529878 ± 0.042975 | 0.655031 ± 0.136823 | 0.734740 ± 0.134719 | **<span style="color:red">0.715669 ± 0.129057</span>** |

## strict + lr | AUROC matrix (embedding × train_dataset)

| Embedding | hESC | hHep | mDC | mHSC-E | mHSC-GM | mHSC-L |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.554442 ± 0.081932 | 0.717081 ± 0.180002 | 0.329655 ± 0.125227 | 0.642819 ± 0.045640 | 0.705870 ± 0.093636 | 0.689759 ± 0.140419 |
| minus | 0.528118 ± 0.121544 | 0.652357 ± 0.161280 | <span style="color:red">0.454336 ± 0.163177</span> | 0.640193 ± 0.070134 | 0.560030 ± 0.142397 | 0.608607 ± 0.086238 |
| scGPT_human | **<span style="color:red">0.585041 ± 0.143157</span>** | 0.659017 ± 0.161361 | **<span style="color:red">0.508456 ± 0.138230</span>** | 0.578526 ± 0.069904 | 0.651970 ± 0.166709 | 0.687215 ± 0.098426 |
| v4_bias_rec_best | 0.447700 ± 0.080520 | **<span style="color:red">0.722434 ± 0.144301</span>** | <span style="color:red">0.398047 ± 0.083609</span> | **<span style="color:red">0.645142 ± 0.045372</span>** | **<span style="color:red">0.712835 ± 0.097055</span>** | 0.617053 ± 0.066850 |
| v4_plain_best | 0.551503 ± 0.129629 | 0.657407 ± 0.149840 | <span style="color:red">0.373199 ± 0.157597</span> | 0.637959 ± 0.093858 | 0.701772 ± 0.063374 | 0.656271 ± 0.067187 |
| v4_type_pe_best | <span style="color:red">0.558718 ± 0.176334</span> | 0.689304 ± 0.152737 | <span style="color:red">0.406361 ± 0.141764</span> | 0.595335 ± 0.045492 | 0.668919 ± 0.122458 | **<span style="color:red">0.708269 ± 0.078166</span>** |

## strict + mlp | AUROC matrix (embedding × train_dataset)

| Embedding | hESC | hHep | mDC | mESC | mHSC-E | mHSC-GM | mHSC-L |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | **0.579307 ± 0.051729** | 0.688533 ± 0.178258 | 0.458259 ± 0.118173 | **0.526748 ± 0.055267** | 0.669418 ± 0.046349 | 0.708440 ± 0.180830 | 0.691241 ± 0.213008 |
| minus | 0.523966 ± 0.065599 | 0.674500 ± 0.151368 | **<span style="color:red">0.544078 ± 0.095756</span>** | 0.501852 ± 0.096176 | 0.657773 ± 0.082102 | 0.602174 ± 0.155750 | 0.677597 ± 0.156378 |
| scGPT_human | 0.559739 ± 0.130185 | 0.664009 ± 0.137417 | <span style="color:red">0.513809 ± 0.050029</span> | 0.480836 ± 0.061504 | 0.629688 ± 0.070972 | 0.676738 ± 0.188048 | <span style="color:red">0.700941 ± 0.122605</span> |
| v4_bias_rec_best | 0.484950 ± 0.128475 | **<span style="color:red">0.703952 ± 0.126376</span>** | <span style="color:red">0.493007 ± 0.085203</span> | 0.473489 ± 0.040474 | <span style="color:red">0.681296 ± 0.044767</span> | <span style="color:red">0.708946 ± 0.103764</span> | 0.678314 ± 0.109790 |
| v4_plain_best | 0.529198 ± 0.124906 | 0.634353 ± 0.154702 | 0.440315 ± 0.121016 | 0.459744 ± 0.071905 | **<span style="color:red">0.699163 ± 0.058650</span>** | **<span style="color:red">0.723416 ± 0.086558</span>** | <span style="color:red">0.713871 ± 0.107197</span> |
| v4_type_pe_best | 0.547399 ± 0.138259 | 0.656197 ± 0.161903 | <span style="color:red">0.491450 ± 0.071121</span> | 0.489892 ± 0.079373 | 0.602127 ± 0.108059 | 0.669519 ± 0.199459 | **<span style="color:red">0.754659 ± 0.093668</span>** |

## topology_matched + lr | AUROC matrix (embedding × train_dataset)

| Embedding | hESC | hHep | mDC | mHSC-E | mHSC-GM | mHSC-L |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.579025 ± 0.053021 | 0.712459 ± 0.148931 | 0.305416 ± 0.131411 | 0.683014 ± 0.033825 | 0.677332 ± 0.162882 | 0.678700 ± 0.119398 |
| minus | <span style="color:red">0.637274 ± 0.084813</span> | 0.619334 ± 0.175419 | <span style="color:red">0.425448 ± 0.130401</span> | 0.663917 ± 0.073515 | 0.575378 ± 0.122274 | 0.613698 ± 0.022079 |
| scGPT_human | <span style="color:red">0.634020 ± 0.070238</span> | 0.594873 ± 0.142308 | **<span style="color:red">0.692882 ± 0.003993</span>** | 0.599415 ± 0.076281 | 0.664149 ± 0.171648 | 0.663114 ± 0.057866 |
| v4_bias_rec_best | 0.452493 ± 0.115644 | <span style="color:red">0.721246 ± 0.163198</span> | <span style="color:red">0.443258 ± 0.187456</span> | 0.629226 ± 0.057913 | 0.667866 ± 0.131707 | 0.609028 ± 0.039387 |
| v4_plain_best | 0.490403 ± 0.224162 | 0.653925 ± 0.195670 | <span style="color:red">0.417852 ± 0.181616</span> | **<span style="color:red">0.690728 ± 0.058170</span>** | **<span style="color:red">0.692312 ± 0.114467</span>** | 0.650969 ± 0.126063 |
| v4_type_pe_best | **<span style="color:red">0.673012 ± 0.122610</span>** | **<span style="color:red">0.757826 ± 0.156291</span>** | <span style="color:red">0.428537 ± 0.185785</span> | 0.638540 ± 0.024697 | 0.626008 ± 0.197095 | **<span style="color:red">0.683246 ± 0.063258</span>** |

## topology_matched + mlp | AUROC matrix (embedding × train_dataset)

| Embedding | hESC | hHep | mDC | mESC | mHSC-E | mHSC-GM | mHSC-L |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.564423 ± 0.090203 | 0.697200 ± 0.129165 | 0.484981 ± 0.117876 | **0.563885 ± 0.057493** | 0.692982 ± 0.029176 | 0.710704 ± 0.177127 | 0.702190 ± 0.234295 |
| minus | <span style="color:red">0.596853 ± 0.060827</span> | 0.638149 ± 0.134675 | <span style="color:red">0.496014 ± 0.117949</span> | 0.475752 ± 0.043296 | 0.632046 ± 0.085598 | 0.604451 ± 0.140749 | 0.681239 ± 0.155010 |
| scGPT_human | <span style="color:red">0.607258 ± 0.043889</span> | 0.601347 ± 0.133120 | <span style="color:red">0.497708 ± 0.006042</span> | 0.486538 ± 0.016662 | 0.612663 ± 0.067702 | 0.698612 ± 0.135504 | 0.682555 ± 0.166253 |
| v4_bias_rec_best | 0.441392 ± 0.141450 | <span style="color:red">0.701441 ± 0.167396</span> | <span style="color:red">0.510276 ± 0.150812</span> | 0.539105 ± 0.029382 | 0.665795 ± 0.056395 | 0.671587 ± 0.162111 | 0.688268 ± 0.083276 |
| v4_plain_best | 0.524698 ± 0.118588 | 0.643480 ± 0.194807 | 0.463525 ± 0.105857 | 0.488909 ± 0.052034 | **<span style="color:red">0.696538 ± 0.061450</span>** | **<span style="color:red">0.713313 ± 0.112518</span>** | <span style="color:red">0.705556 ± 0.115585</span> |
| v4_type_pe_best | **<span style="color:red">0.651547 ± 0.108946</span>** | **<span style="color:red">0.718296 ± 0.131968</span>** | **<span style="color:red">0.515023 ± 0.123746</span>** | 0.539223 ± 0.068912 | 0.622649 ± 0.095070 | 0.649720 ± 0.237676 | **<span style="color:red">0.717117 ± 0.124910</span>** |
