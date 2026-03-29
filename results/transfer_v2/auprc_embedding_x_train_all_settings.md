# AUPRC matrices by setting (embedding × train_dataset)

## coverage_matched + lr | AUPRC matrix (embedding × train_dataset)

| Embedding | hESC | hHep | mDC | mHSC-E | mHSC-GM | mHSC-L |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.603357 ± 0.114826 | 0.647507 ± 0.072269 | 0.498878 ± 0.092525 | **0.694198 ± 0.130688** | **0.754954 ± 0.038931** | 0.691468 ± 0.076828 |
| minus | <span style="color:red">0.605287 ± 0.119138</span> | <span style="color:red">0.696072 ± 0.062405</span> | 0.493514 ± 0.030192 | 0.623435 ± 0.050928 | 0.713686 ± 0.071405 | 0.562896 ± 0.120932 |
| scGPT_human | 0.599991 ± 0.083707 | <span style="color:red">0.664026 ± 0.066011</span> | 0.493611 ± 0.109292 | 0.584239 ± 0.153029 | 0.672585 ± 0.170787 | 0.621820 ± 0.150265 |
| v4_bias_rec_best | **<span style="color:red">0.657003 ± 0.076060</span>** | **<span style="color:red">0.712792 ± 0.070566</span>** | 0.455635 ± 0.065434 | 0.655190 ± 0.087922 | 0.644134 ± 0.070305 | 0.671325 ± 0.108586 |
| v4_plain_best | 0.596444 ± 0.087424 | <span style="color:red">0.683466 ± 0.068871</span> | 0.470018 ± 0.080226 | 0.634437 ± 0.057212 | 0.735386 ± 0.069005 | **<span style="color:red">0.691715 ± 0.130728</span>** |
| v4_type_pe_best | <span style="color:red">0.610733 ± 0.084276</span> | <span style="color:red">0.648637 ± 0.082572</span> | **<span style="color:red">0.513523 ± 0.130095</span>** | 0.678469 ± 0.116192 | 0.651576 ± 0.119623 | 0.684928 ± 0.137572 |

## coverage_matched + mlp | AUPRC matrix (embedding × train_dataset)

| Embedding | hESC | hHep | mDC | mHSC-E | mHSC-GM | mHSC-L |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.630504 ± 0.095362 | 0.668914 ± 0.051844 | **0.529131 ± 0.084422** | **0.716306 ± 0.126579** | **0.779172 ± 0.038553** | 0.696016 ± 0.113142 |
| minus | 0.599471 ± 0.115149 | <span style="color:red">0.700748 ± 0.079190</span> | 0.488929 ± 0.038979 | 0.644233 ± 0.054358 | 0.743611 ± 0.086803 | 0.614446 ± 0.123375 |
| scGPT_human | 0.582291 ± 0.080495 | <span style="color:red">0.678489 ± 0.069099</span> | 0.501238 ± 0.108688 | 0.629768 ± 0.173479 | 0.708379 ± 0.109943 | 0.672285 ± 0.126193 |
| v4_bias_rec_best | **<span style="color:red">0.647944 ± 0.071609</span>** | **<span style="color:red">0.732081 ± 0.071851</span>** | 0.488025 ± 0.053454 | 0.685527 ± 0.089520 | 0.679869 ± 0.085045 | **<span style="color:red">0.737208 ± 0.124588</span>** |
| v4_plain_best | 0.568940 ± 0.086966 | 0.663051 ± 0.050424 | 0.493713 ± 0.091011 | 0.673567 ± 0.056075 | 0.758123 ± 0.069422 | <span style="color:red">0.702630 ± 0.115615</span> |
| v4_type_pe_best | 0.600684 ± 0.078913 | <span style="color:red">0.671951 ± 0.057631</span> | 0.512167 ± 0.125030 | 0.687655 ± 0.074003 | 0.682338 ± 0.133703 | <span style="color:red">0.720597 ± 0.126875</span> |

## native + lr | AUPRC matrix (embedding × train_dataset)

| Embedding | hESC | hHep | mDC | mHSC-E | mHSC-GM | mHSC-L |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.625459 ± 0.054109 | 0.628708 ± 0.038227 | **0.503946 ± 0.039746** | **0.658987 ± 0.069444** | 0.710653 ± 0.069605 | **0.681479 ± 0.074031** |
| minus | 0.588317 ± 0.090588 | <span style="color:red">0.630817 ± 0.046666</span> | 0.498659 ± 0.046236 | 0.638538 ± 0.052287 | 0.693709 ± 0.082964 | 0.592588 ± 0.090333 |
| scGPT_human | **<span style="color:red">0.640655 ± 0.035688</span>** | <span style="color:red">0.636392 ± 0.041623</span> | 0.493464 ± 0.027730 | 0.631838 ± 0.060938 | **<span style="color:red">0.728875 ± 0.084320</span>** | 0.669185 ± 0.063840 |
| v4_bias_rec_best | 0.584930 ± 0.054940 | <span style="color:red">0.643485 ± 0.041572</span> | 0.408062 ± 0.018250 | 0.657599 ± 0.079808 | 0.689014 ± 0.060390 | 0.581489 ± 0.103460 |
| v4_plain_best | 0.615609 ± 0.075897 | <span style="color:red">0.640732 ± 0.027978</span> | 0.459526 ± 0.029662 | 0.658592 ± 0.057769 | 0.700599 ± 0.057509 | 0.630335 ± 0.098947 |
| v4_type_pe_best | <span style="color:red">0.637603 ± 0.045549</span> | **<span style="color:red">0.657538 ± 0.027036</span>** | 0.440806 ± 0.018945 | 0.627122 ± 0.074797 | 0.698298 ± 0.065507 | 0.642190 ± 0.088146 |

## native + mlp | AUPRC matrix (embedding × train_dataset)

| Embedding | hESC | hHep | mDC | mHSC-E | mHSC-GM | mHSC-L |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.635728 ± 0.057889 | 0.681611 ± 0.049191 | 0.508811 ± 0.053061 | 0.688625 ± 0.082955 | **0.750647 ± 0.074546** | 0.699882 ± 0.112195 |
| minus | 0.605444 ± 0.092389 | <span style="color:red">0.687537 ± 0.040180</span> | <span style="color:red">0.511175 ± 0.047994</span> | 0.669040 ± 0.067531 | 0.727847 ± 0.077785 | 0.647390 ± 0.111426 |
| scGPT_human | **<span style="color:red">0.662806 ± 0.050429</span>** | <span style="color:red">0.688172 ± 0.052012</span> | **<span style="color:red">0.519893 ± 0.035938</span>** | 0.670169 ± 0.078808 | 0.744785 ± 0.079163 | **<span style="color:red">0.716344 ± 0.091689</span>** |
| v4_bias_rec_best | 0.591929 ± 0.081440 | **<span style="color:red">0.713809 ± 0.040963</span>** | 0.430114 ± 0.026445 | **<span style="color:red">0.688672 ± 0.089736</span>** | 0.729327 ± 0.067787 | 0.681735 ± 0.134070 |
| v4_plain_best | 0.625130 ± 0.084918 | <span style="color:red">0.695696 ± 0.033655</span> | 0.477360 ± 0.030538 | 0.672307 ± 0.061914 | 0.728329 ± 0.057911 | 0.675695 ± 0.109996 |
| v4_type_pe_best | <span style="color:red">0.655817 ± 0.067303</span> | <span style="color:red">0.701469 ± 0.049640</span> | 0.481006 ± 0.036423 | 0.669043 ± 0.075993 | 0.728913 ± 0.074031 | 0.690552 ± 0.104782 |

## strict + lr | AUPRC matrix (embedding × train_dataset)

| Embedding | hESC | hHep | mDC | mHSC-E | mHSC-GM | mHSC-L |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.620595 ± 0.049604 | 0.631647 ± 0.174833 | 0.375841 ± 0.070368 | 0.661600 ± 0.179144 | 0.682895 ± 0.112975 | 0.615886 ± 0.125809 |
| minus | 0.559145 ± 0.030946 | **<span style="color:red">0.644457 ± 0.231233</span>** | <span style="color:red">0.389697 ± 0.093138</span> | 0.656762 ± 0.182677 | **<span style="color:red">0.709632 ± 0.130278</span>** | 0.547825 ± 0.114394 |
| scGPT_human | <span style="color:red">0.621659 ± 0.073090</span> | 0.607190 ± 0.230609 | <span style="color:red">0.383491 ± 0.067988</span> | <span style="color:red">0.666710 ± 0.208047</span> | <span style="color:red">0.702483 ± 0.100452</span> | **<span style="color:red">0.642923 ± 0.087694</span>** |
| v4_bias_rec_best | 0.583885 ± 0.023677 | 0.613849 ± 0.184174 | 0.328992 ± 0.139206 | 0.613506 ± 0.183087 | 0.595486 ± 0.124715 | 0.554955 ± 0.139553 |
| v4_plain_best | <span style="color:red">0.624740 ± 0.039485</span> | 0.626397 ± 0.230311 | **<span style="color:red">0.397595 ± 0.142022</span>** | **<span style="color:red">0.670486 ± 0.145029</span>** | <span style="color:red">0.699721 ± 0.100969</span> | <span style="color:red">0.619330 ± 0.108842</span> |
| v4_type_pe_best | **<span style="color:red">0.625084 ± 0.048865</span>** | <span style="color:red">0.642132 ± 0.190966</span> | 0.314945 ± 0.083418 | 0.647079 ± 0.179102 | 0.664413 ± 0.117301 | 0.606122 ± 0.117403 |

## strict + mlp | AUPRC matrix (embedding × train_dataset)

| Embedding | hESC | hHep | mDC | mHSC-E | mHSC-GM | mHSC-L |
|---|---:|---:|---:|---:|---:|---:|
| baseline | **0.604100 ± 0.033593** | 0.625632 ± 0.201862 | 0.384385 ± 0.103322 | 0.667402 ± 0.180519 | 0.692898 ± 0.100901 | 0.625972 ± 0.135587 |
| minus | 0.559418 ± 0.027919 | <span style="color:red">0.638518 ± 0.217574</span> | **<span style="color:red">0.396673 ± 0.091425</span>** | 0.650453 ± 0.191115 | **<span style="color:red">0.711261 ± 0.104477</span>** | 0.608818 ± 0.134698 |
| scGPT_human | 0.583988 ± 0.055148 | 0.615991 ± 0.237814 | <span style="color:red">0.385015 ± 0.048537</span> | **<span style="color:red">0.668576 ± 0.201342</span>** | <span style="color:red">0.702510 ± 0.087219</span> | <span style="color:red">0.654236 ± 0.121774</span> |
| v4_bias_rec_best | 0.574372 ± 0.033681 | **<span style="color:red">0.650784 ± 0.165581</span>** | 0.345171 ± 0.096481 | 0.647884 ± 0.188774 | 0.644313 ± 0.142497 | <span style="color:red">0.653475 ± 0.143787</span> |
| v4_plain_best | 0.598858 ± 0.051712 | 0.607031 ± 0.237029 | <span style="color:red">0.384674 ± 0.108276</span> | 0.667104 ± 0.181051 | <span style="color:red">0.710973 ± 0.080888</span> | **<span style="color:red">0.654802 ± 0.119069</span>** |
| v4_type_pe_best | 0.592560 ± 0.045320 | <span style="color:red">0.634961 ± 0.203057</span> | 0.368144 ± 0.113412 | 0.651028 ± 0.174233 | <span style="color:red">0.694193 ± 0.119662</span> | <span style="color:red">0.651925 ± 0.131078</span> |
