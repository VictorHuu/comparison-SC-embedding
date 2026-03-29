# AUROC matrices by setting (embedding × train_dataset)

## coverage_matched + lr | AUROC matrix (embedding × train_dataset)

| Embedding | hESC | hHep | mDC | mHSC-E | mHSC-GM | mHSC-L |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.609102 ± 0.117206 | 0.640728 ± 0.095895 | 0.453349 ± 0.107422 | 0.628073 ± 0.136378 | **0.752238 ± 0.063198** | 0.663748 ± 0.079827 |
| minus | 0.586287 ± 0.131554 | <span style="color:red">0.681547 ± 0.057586</span> | **<span style="color:red">0.532344 ± 0.053719</span>** | <span style="color:red">0.651291 ± 0.042923</span> | 0.686650 ± 0.093401 | 0.485732 ± 0.158822 |
| scGPT_human | **<span style="color:red">0.637610 ± 0.050558</span>** | <span style="color:red">0.664423 ± 0.039936</span> | <span style="color:red">0.474922 ± 0.046741</span> | 0.591954 ± 0.152781 | 0.703199 ± 0.171993 | 0.607281 ± 0.130677 |
| v4_bias_rec_best | <span style="color:red">0.621177 ± 0.132887</span> | **<span style="color:red">0.717938 ± 0.068201</span>** | 0.369735 ± 0.081914 | <span style="color:red">0.632658 ± 0.098449</span> | 0.678091 ± 0.097538 | 0.655729 ± 0.147516 |
| v4_plain_best | <span style="color:red">0.634213 ± 0.099010</span> | <span style="color:red">0.702991 ± 0.068602</span> | <span style="color:red">0.467723 ± 0.050634</span> | 0.577043 ± 0.053411 | 0.673779 ± 0.088833 | 0.621104 ± 0.115858 |
| v4_type_pe_best | <span style="color:red">0.612777 ± 0.123296</span> | <span style="color:red">0.657651 ± 0.128189</span> | <span style="color:red">0.491810 ± 0.094671</span> | **<span style="color:red">0.661296 ± 0.092844</span>** | 0.650450 ± 0.126525 | **<span style="color:red">0.676880 ± 0.116588</span>** |

## coverage_matched + mlp | AUROC matrix (embedding × train_dataset)

| Embedding | hESC | hHep | mDC | mHSC-E | mHSC-GM | mHSC-L |
|---|---:|---:|---:|---:|---:|---:|
| baseline | **0.633401 ± 0.095369** | 0.669233 ± 0.058311 | 0.497337 ± 0.087694 | 0.651082 ± 0.139098 | **0.770730 ± 0.060528** | 0.672491 ± 0.109189 |
| minus | 0.577439 ± 0.127514 | <span style="color:red">0.695050 ± 0.074533</span> | **<span style="color:red">0.530186 ± 0.046546</span>** | **<span style="color:red">0.674493 ± 0.052704</span>** | 0.716461 ± 0.117196 | 0.566571 ± 0.221102 |
| scGPT_human | 0.617573 ± 0.038531 | <span style="color:red">0.682561 ± 0.038649</span> | 0.486187 ± 0.048304 | <span style="color:red">0.657217 ± 0.120919</span> | 0.755197 ± 0.110802 | <span style="color:red">0.682348 ± 0.106474</span> |
| v4_bias_rec_best | 0.613431 ± 0.132640 | **<span style="color:red">0.741939 ± 0.046025</span>** | 0.420249 ± 0.065586 | <span style="color:red">0.673726 ± 0.090227</span> | 0.699907 ± 0.129561 | **<span style="color:red">0.742794 ± 0.173779</span>** |
| v4_plain_best | 0.595505 ± 0.106572 | <span style="color:red">0.692057 ± 0.061960</span> | 0.493180 ± 0.038904 | 0.620726 ± 0.067778 | 0.719863 ± 0.098274 | 0.648270 ± 0.109265 |
| v4_type_pe_best | 0.602907 ± 0.123655 | <span style="color:red">0.687898 ± 0.078459</span> | <span style="color:red">0.506977 ± 0.074827</span> | <span style="color:red">0.667859 ± 0.057749</span> | 0.677941 ± 0.122204 | <span style="color:red">0.728336 ± 0.087534</span> |

## native + lr | AUROC matrix (embedding × train_dataset)

| Embedding | hESC | hHep | mDC | mHSC-E | mHSC-GM | mHSC-L |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.616599 ± 0.054730 | 0.627728 ± 0.028859 | **0.488042 ± 0.041428** | 0.651871 ± 0.065228 | 0.708894 ± 0.105948 | 0.665477 ± 0.078645 |
| minus | 0.562170 ± 0.149756 | <span style="color:red">0.630042 ± 0.072932</span> | 0.487621 ± 0.055540 | 0.636022 ± 0.072128 | 0.674551 ± 0.114428 | 0.565329 ± 0.100159 |
| scGPT_human | <span style="color:red">0.627832 ± 0.073324</span> | 0.624472 ± 0.062392 | 0.483946 ± 0.038206 | 0.628360 ± 0.050925 | **<span style="color:red">0.710270 ± 0.125151</span>** | **<span style="color:red">0.666687 ± 0.082242</span>** |
| v4_bias_rec_best | 0.574280 ± 0.098367 | <span style="color:red">0.636566 ± 0.054804</span> | 0.367530 ± 0.030812 | **<span style="color:red">0.653971 ± 0.070858</span>** | 0.694233 ± 0.078813 | 0.557496 ± 0.114470 |
| v4_plain_best | 0.612095 ± 0.104795 | <span style="color:red">0.647224 ± 0.019111</span> | 0.427315 ± 0.040664 | 0.643223 ± 0.047036 | 0.694140 ± 0.073002 | 0.604300 ± 0.110199 |
| v4_type_pe_best | **<span style="color:red">0.642109 ± 0.076736</span>** | **<span style="color:red">0.653566 ± 0.031942</span>** | 0.417542 ± 0.030353 | 0.621870 ± 0.064718 | 0.687618 ± 0.104707 | 0.631017 ± 0.094143 |

## native + mlp | AUROC matrix (embedding × train_dataset)

| Embedding | hESC | hHep | mDC | mHSC-E | mHSC-GM | mHSC-L |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.633631 ± 0.076905 | 0.698212 ± 0.044220 | 0.495671 ± 0.051301 | 0.681026 ± 0.072909 | **0.744806 ± 0.115428** | 0.698648 ± 0.111534 |
| minus | 0.581894 ± 0.157554 | 0.694005 ± 0.056927 | <span style="color:red">0.496989 ± 0.055460</span> | 0.667961 ± 0.079217 | 0.714095 ± 0.115221 | 0.646127 ± 0.124444 |
| scGPT_human | <span style="color:red">0.655537 ± 0.069314</span> | 0.693386 ± 0.057590 | **<span style="color:red">0.510915 ± 0.042645</span>** | 0.665070 ± 0.073815 | 0.737004 ± 0.110623 | **<span style="color:red">0.725978 ± 0.103819</span>** |
| v4_bias_rec_best | 0.574413 ± 0.138528 | **<span style="color:red">0.719074 ± 0.039732</span>** | 0.401905 ± 0.038148 | **<span style="color:red">0.687236 ± 0.076595</span>** | 0.735113 ± 0.084414 | 0.679068 ± 0.148952 |
| v4_plain_best | 0.620100 ± 0.119882 | <span style="color:red">0.711636 ± 0.046743</span> | 0.452491 ± 0.025817 | 0.660965 ± 0.055409 | 0.726137 ± 0.077288 | 0.674905 ± 0.114309 |
| v4_type_pe_best | **<span style="color:red">0.662118 ± 0.099245</span>** | <span style="color:red">0.706910 ± 0.055026</span> | 0.471013 ± 0.039173 | 0.662613 ± 0.070202 | 0.723142 ± 0.112310 | 0.696667 ± 0.104707 |

## strict + lr | AUROC matrix (embedding × train_dataset)

| Embedding | hESC | hHep | mDC | mHSC-E | mHSC-GM | mHSC-L |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.602813 ± 0.055150 | 0.707724 ± 0.076867 | 0.550072 ± 0.058218 | 0.687281 ± 0.127526 | 0.715497 ± 0.090776 | 0.644999 ± 0.091067 |
| minus | 0.555838 ± 0.063024 | **<span style="color:red">0.734140 ± 0.110737</span>** | 0.546129 ± 0.108464 | <span style="color:red">0.690292 ± 0.145964</span> | 0.706202 ± 0.144245 | 0.552150 ± 0.116561 |
| scGPT_human | <span style="color:red">0.616281 ± 0.072370</span> | 0.698005 ± 0.093277 | **<span style="color:red">0.554244 ± 0.063544</span>** | <span style="color:red">0.693869 ± 0.167177</span> | **<span style="color:red">0.729068 ± 0.067913</span>** | **<span style="color:red">0.667512 ± 0.070529</span>** |
| v4_bias_rec_best | 0.577036 ± 0.030139 | 0.682130 ± 0.068463 | 0.463118 ± 0.130221 | 0.656861 ± 0.132511 | 0.628158 ± 0.116956 | 0.570317 ± 0.113371 |
| v4_plain_best | <span style="color:red">0.618165 ± 0.057492</span> | <span style="color:red">0.727548 ± 0.091620</span> | 0.532874 ± 0.110294 | **<span style="color:red">0.706848 ± 0.112725</span>** | 0.711336 ± 0.088270 | 0.631090 ± 0.113776 |
| v4_type_pe_best | **<span style="color:red">0.624613 ± 0.074658</span>** | <span style="color:red">0.733144 ± 0.044840</span> | 0.464948 ± 0.093463 | 0.680130 ± 0.135436 | 0.687404 ± 0.092762 | <span style="color:red">0.653859 ± 0.090314</span> |

## strict + mlp | AUROC matrix (embedding × train_dataset)

| Embedding | hESC | hHep | mDC | mHSC-E | mHSC-GM | mHSC-L |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.581482 ± 0.072568 | 0.717636 ± 0.070329 | 0.561099 ± 0.100147 | 0.700422 ± 0.124291 | 0.726827 ± 0.067965 | 0.669229 ± 0.107777 |
| minus | 0.535727 ± 0.085284 | <span style="color:red">0.724634 ± 0.096458</span> | 0.549053 ± 0.103195 | 0.676420 ± 0.160874 | 0.718581 ± 0.117547 | 0.641409 ± 0.131312 |
| scGPT_human | 0.569863 ± 0.062988 | <span style="color:red">0.719717 ± 0.103152</span> | **<span style="color:red">0.562471 ± 0.049479</span>** | 0.699721 ± 0.158801 | <span style="color:red">0.735523 ± 0.060015</span> | <span style="color:red">0.694819 ± 0.101125</span> |
| v4_bias_rec_best | 0.546425 ± 0.108032 | <span style="color:red">0.734930 ± 0.051094</span> | 0.512348 ± 0.086063 | 0.690776 ± 0.131013 | 0.679538 ± 0.113736 | <span style="color:red">0.703104 ± 0.099656</span> |
| v4_plain_best | 0.578771 ± 0.096522 | <span style="color:red">0.717690 ± 0.091375</span> | 0.537749 ± 0.089884 | **<span style="color:red">0.700945 ± 0.129061</span>** | **<span style="color:red">0.736534 ± 0.066209</span>** | <span style="color:red">0.697962 ± 0.093839</span> |
| v4_type_pe_best | **<span style="color:red">0.581486 ± 0.089513</span>** | **<span style="color:red">0.739468 ± 0.062326</span>** | 0.517964 ± 0.127281 | 0.683468 ± 0.125043 | <span style="color:red">0.726955 ± 0.089883</span> | **<span style="color:red">0.712993 ± 0.080278</span>** |
