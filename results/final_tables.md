# Final Benchmarking Tables & Experimental Results (Task 8 & 8b Final Verification)

**Single Source of Truth:** This document is generated directly from the reproducible console output of `python/run_task8.py` (saved in `results/task8_final_output.txt`). All earlier conversational tables and notes are historical.

---

## 0. Feature Counts per Group (from `data/features_meta_v2.json`)

**Total Engineered Features:** `1338` (`1,110` without Group D3; `1,338` with Group D3)

| Feature Group | Key / Column Prefix | Feature Count | Physical Interpretation |
|---|---|---|---|
| **Group A: Residuals** | `group_a_residuals` (`feat_res_r_*`) | **328** | Measurement raw ($z - h(\hat{x})$) and normalized ($r / \sigma$) residuals across 164 measurements |
| **Group B: Consistency** | `group_b_consistency` (`feat_cons_*`) | **163** | Active/reactive KCL nodal balance mismatches, Ohm's law branch flow checks |
| **Group C: Sensitivity** | `group_c_sensitivity` (`feat_sens_*`) | **163** | Projection matrix diagonals, leverage scores, and gain matrix diagonal profiles |
| **Group D1: Raw States/Meas** | `group_d1_raw` (`feat_nonres_z_*`, `feat_nonres_x_hat_*`) | **228** | 65 state estimates (32 angles, 33 voltages) and 163 raw measurements |
| **Group D2: Forecast Dev** | `group_d2_hourly_forecast` (`feat_nonres_forecast_dev_*`) | **228** | Per-hour deviation from nominal historical means (65 states + 163 measurements) |
| **Group D3: Change Prev** | `group_d3_change_prev` (`feat_nonres_prev_diff_*`) | **228** | Temporal step difference $\hat{x}_t - \hat{x}_{t-1}$ (assumes clean previous sample) |
| **Total Group D (D1+D2+D3)** | `group_d_non_residual` | **684** | Full non-residual feature representation |
| **Primary Model Features (A+B+C+D1+D2)** | Active feature vector | **1110** | Primary supervised & unsupervised benchmark set |
| **Full Model with D3** | Active feature vector | **1338** | Primary features plus temporal step differences |

---

## 1. Operating Thresholds Derived from Normal Validation Rows ($N = 1,000$)

Uniform Rule: All detectors establish their operating thresholds strictly from the 95th and 99th percentiles of scores on normal validation rows (`split == 'val'` and `label == 0`). Realized false alarm rates on unseen test normal rows ($N = 1,000$) confirm stability.

| Method | Val Normal Thresh (5% FA) | Val Normal Thresh (1% FA) | Realized Normal Test FA (5% th) | Realized Normal Test FA (1% th) |
|---|---|---|---|---|
| **Chi-Square** | `124.038326` | `135.094060` | **3.60% ( 36/1000)** | **0.10% (  1/1000)** |
| **Calibrated LNR** | `3.574947` | `4.071959` | **6.50% ( 65/1000)** | **1.00% ( 10/1000)** |
| **Range-Check (Envelope)** | `0.000000` | `0.000000` | **0.10% (  1/1000)** | **0.10% (  1/1000)** |
| **Per-Hour Max-Z** | `3.165055` | `3.680114` | **5.30% ( 53/1000)** | **0.50% (  5/1000)** |
| **Mahalanobis (Ledoit-Wolf)** | `6.973609` | `7.521305` | **6.70% ( 67/1000)** | **1.50% ( 15/1000)** |
| **Autoencoder (D1+D2)** | `0.586598` | `0.619121` | **4.20% ( 42/1000)** | **1.70% ( 17/1000)** |
| **Autoencoder (All: A-D2)** | `0.475096` | `0.529026` | **5.20% ( 52/1000)** | **1.10% ( 11/1000)** |
| **RF w/o D3 (Primary)** | `0.365035` | `0.444420` | **2.90% ( 29/1000)** | **0.80% (  8/1000)** |
| **RF with D3 (Clean Prev)** | `0.366625` | `0.446378` | **2.70% ( 27/1000)** | **0.70% (  7/1000)** |

---

## 2. Table 1: Standard Test Split Benchmark ($N = 2,000$ rows: 1,000 Normal, 1,000 Attacked)

Evaluated with 1,000 group-bootstrap 95% confidence intervals:

| Method | Realized Normal FA (5% / 1%) | Overall ROC-AUC [95% CI] | Exact AC AUC [95% CI] | Det @ 5% FA [95% CI] | Det @ 1% FA [95% CI] | Exact AC Det @ 5% FA [95% CI] | Effective Attacks Det @ 5% FA [95% CI] |
|---|---|---|---|---|---|---|---|
| **Chi-Square** | 3.60% /  0.10% | 0.6266 [0.6111,0.6412] | 0.5004 [0.4753,0.5268] | 25.10% [22.7,27.9] | 20.80% [18.4,23.4] | 3.61% [1.8,5.7] | 26.05% [23.2,28.9] |
| **Calibrated LNR** | 6.50% /  1.00% | 0.6265 [0.6107,0.6416] | 0.4998 [0.4747,0.5247] | 27.80% [25.1,30.6] | 20.70% [18.4,23.2] | 8.13% [5.4,11.1] | 28.77% [25.8,31.7] |
| **Range-Check (Envelope)** | 0.10% /  0.10% | 0.5330 [0.5260,0.5410] | 0.5206 [0.5106,0.5318] | 6.70% [5.2,8.3] | 6.70% [5.2,8.3] | 4.22% [2.1,6.6] | 7.01% [5.4,8.8] |
| **Per-Hour Max-Z** | 5.30% /  0.50% | 0.8152 [0.7990,0.8317] | 0.8088 [0.7791,0.8406] | 58.00% [54.9,61.3] | 52.50% [49.4,55.6] | 57.53% [52.4,63.0] | 60.56% [57.5,63.4] |
| **Mahalanobis (Ledoit-Wolf)** | 6.70% /  1.50% | 0.9062 [0.8928,0.9191] | 0.9312 [0.9132,0.9487] | 78.40% [75.8,81.0] | 73.10% [70.4,75.8] | 82.83% [78.6,87.0] | 81.80% [79.3,84.1] |
| **Autoencoder (D1+D2)** | 4.20% /  1.70% | 0.9341 [0.9217,0.9457] | 0.9683 [0.9531,0.9815] | 85.30% [83.1,87.4] | 84.20% [82.0,86.4] | 90.96% [88.0,94.0] | 88.91% [86.9,90.9] |
| **Autoencoder (All: A-D2)** | 5.20% /  1.10% | 0.9349 [0.9223,0.9464] | 0.9727 [0.9588,0.9836] | 84.70% [82.5,86.8] | 82.80% [80.5,85.1] | 90.36% [87.0,93.4] | 88.39% [86.3,90.5] |
| **RF w/o D3 (Primary)** | 2.90% /  0.80% | 0.9317 [0.9199,0.9436] | 0.9696 [0.9583,0.9801] | 81.20% [78.8,83.7] | 76.30% [73.8,79.1] | 86.75% [83.1,90.4] | 84.83% [82.2,87.0] |
| **RF with D3 (Clean Prev)** | 2.70% /  0.70% | 0.9317 [0.9196,0.9432] | 0.9674 [0.9560,0.9781] | 81.10% [78.7,83.5] | 76.20% [73.7,78.9] | 86.75% [83.4,90.4] | 84.73% [82.3,86.9] |

### Table 1 Per-Attack-Type Detection Rate @ 5% Validation Normal FA:

| Method | `exact_single` ($N=179$) | `exact_multi` ($N=153$) | `limited` ($N=153$) | `lin_single` ($N=175$) | `lin_multi` ($N=192$) | `random` ($N=148$) |
|---|---|---|---|---|---|---|
| **Chi-Square** | 2.23% | 5.23% | 12.42% | 17.14% | 26.04% | 94.59% |
| **Calibrated LNR** | 7.26% | 9.15% | 14.38% | 18.86% | 28.65% | 95.27% |
| **Range-Check (Envelope)** | 2.79% | 5.88% | 2.61% | 6.29% | 15.10% | 6.08% |
| **Per-Hour Max-Z** | 50.84% | 65.36% | 32.68% | 52.00% | 81.25% | 62.16% |
| **Mahalanobis (Ledoit-Wolf)** | 75.42% | 91.50% | 34.64% | 79.43% | 96.88% | 88.51% |
| **Autoencoder (D1+D2)** | 83.80% | 99.35% | 39.87% | 88.57% | 98.96% | 97.97% |
| **Autoencoder (All: A-D2)** | 82.68% | 99.35% | 37.91% | 88.57% | 98.96% | 97.30% |
| **RF w/o D3 (Primary)** | 76.54% | 98.69% | 37.91% | 76.57% | 97.92% | 97.30% |
| **RF with D3 (Clean Prev)** | 76.54% | 98.69% | 37.25% | 76.57% | 97.92% | 97.30% |

---

## 3. Table 2: Unseen-Attack Scheme Benchmark (`split_unseen`, $N = 4,026$ rows)

RF trained strictly on `split_unseen == 'train'` ($N = 4,463$, containing random and linearised attacks only). The exact AC and limited-meter attacks were completely held out and never seen during training.

| Method | Realized Normal FA (5% / 1%) | Overall ROC-AUC [95% CI] | Exact AC AUC [95% CI] | Det @ 5% FA [95% CI] | Det @ 1% FA [95% CI] | Exact AC Det @ 5% FA [95% CI] | Effective Attacks Det @ 5% FA [95% CI] |
|---|---|---|---|---|---|---|---|
| **Chi-Square** | 3.60% /  0.10% | 0.5704 [0.5560,0.5863] | 0.5072 [0.4885,0.5277] | 13.52% [12.2,14.9] | 9.32% [8.2,10.4] | 3.87% [3.0,4.8] | 14.27% [12.9,15.5] |
| **Calibrated LNR** | 6.50% /  1.00% | 0.5707 [0.5554,0.5872] | 0.5120 [0.4930,0.5324] | 15.80% [14.4,17.1] | 9.75% [8.7,10.9] | 5.96% [4.9,7.2] | 16.47% [15.1,17.9] |
| **Range-Check (Envelope)** | 0.10% /  0.10% | 0.5294 [0.5255,0.5337] | 0.5329 [0.5269,0.5386] | 5.98% [5.2,6.8] | 5.98% [5.2,6.8] | 6.67% [5.5,7.9] | 6.42% [5.5,7.3] |
| **Mahalanobis (Ledoit-Wolf)** | 6.70% /  1.50% | 0.8386 [0.8275,0.8494] | 0.9249 [0.9147,0.9343] | 70.13% [68.5,71.7] | 66.19% [64.4,67.8] | 82.00% [80.2,83.9] | 75.20% [73.5,76.8] |
| **Autoencoder (D1+D2)** | 4.20% /  1.70% | 0.8837 [0.8733,0.8929] | 0.9630 [0.9561,0.9700] | 78.09% [76.5,79.5] | 76.67% [75.0,78.2] | 91.48% [90.2,92.6] | 83.61% [82.3,85.0] |
| **Autoencoder (All: A-D2)** | 5.20% /  1.10% | 0.8667 [0.8549,0.8766] | 0.9612 [0.9543,0.9681] | 77.20% [75.5,78.7] | 75.02% [73.4,76.6] | 90.82% [89.5,92.1] | 82.65% [81.2,84.0] |
| **RF (Seen Only -> Held-Out)** | 4.20% /  0.70% | 0.8945 [0.8852,0.9030] | 0.9706 [0.9649,0.9762] | 76.67% [75.1,78.2] | 68.64% [67.0,70.4] | 89.57% [88.0,90.9] | 82.08% [80.7,83.5] |

### Table 2 Detection on Completely Held-Out and Seen Attack Types @ 5% FA:

| Method | **HELD-OUT** `exact_sin` ($N=847$) | **HELD-OUT** `exact_mul` ($N=831$) | **HELD-OUT** `limited` ($N=833$) | `lin_single` ($N=175$) | `lin_multi` ($N=192$) | `random` ($N=148$) |
|---|---|---|---|---|---|---|
| **Chi-Square** | 3.78% | 3.97% | 14.89% | 17.14% | 26.04% | 94.59% |
| **Calibrated LNR** | 5.43% | 6.50% | 17.89% | 18.86% | 28.65% | 95.27% |
| **Range-Check (Envelope)** | 3.90% | 9.51% | 2.40% | 6.29% | 15.10% | 6.08% |
| **Mahalanobis (Ledoit-Wolf)** | 74.14% | 90.01% | 34.81% | 79.43% | 96.88% | 88.51% |
| **Autoencoder (D1+D2)** | 84.30% | 98.80% | 40.58% | 88.57% | 98.96% | 97.97% |
| **Autoencoder (All: A-D2)** | 83.47% | 98.32% | 38.78% | 88.57% | 98.96% | 97.30% |
| **RF (Seen Only -> Held-Out)** | 80.17% | 99.16% | 41.42% | 78.86% | 98.44% | 97.97% |

---

## 4. Detection vs Attack Size Bins (`shift_sigmas`) at 5% Validation Normal FA

| Sigma Bin | Count ($N$) | RF Primary (A-D2) | Autoencoder (All) | Mahalanobis (Ledoit-Wolf) | Per-Hour Max-Z |
|---|---|---|---|---|---|
| **<1.0 sigma** | 184 | 57.61% | 61.41% | 41.85% | 4.35% |
| **1.0-2.0 sig** | 102 | 80.39% | 81.37% | 80.39% | 10.78% |
| **2.0-3.0 sig** | 59 | 77.97% | 88.14% | 84.75% | 49.15% |
| **3.0-5.0 sig** | 87 | 74.71% | 78.16% | 77.01% | 62.07% |
| **5.0-10.0 sig** | 139 | 87.05% | 92.09% | 86.33% | 92.81% |
| **>10.0 sigmas** | 231 | 87.45% | 92.21% | 91.34% | 91.34% |

---

## 5. Feature Group Ablation (300 Trees, Depth 12, Min Samples Leaf 1)

| Ablation Setting | Feats | Validation AUC | Test ROC-AUC | Realized Normal FA | Overall Det @ 5% FA | Exact AC Det @ 5% FA |
|---|---|---|---|---|---|---|
| **Group A alone** | 328 | 0.6323 | 0.6360 | 4.50% | 28.30% | 5.72% |
| **Group B alone** | 163 | 0.9151 | 0.9203 | 4.00% | 76.90% | 87.35% |
| **Group C alone** | 163 | 0.9179 | 0.9226 | 5.70% | 78.00% | 87.65% |
| **Group D1 alone** | 228 | 0.9166 | 0.9219 | 3.60% | 78.50% | 85.54% |
| **Group D2 alone** | 228 | 0.9172 | 0.9175 | 3.70% | 78.00% | 84.64% |
| **Group D3 alone (clean prev)** | 228 | 0.8942 | 0.9041 | 4.80% | 74.40% | 82.53% |
| **Without A (All - A)** | 782 | 0.9269 | 0.9316 | 4.30% | 81.60% | 90.06% |
| **Without B (All - B)** | 947 | 0.9232 | 0.9247 | 3.30% | 81.10% | 86.45% |
| **Without C (All - C)** | 947 | 0.9232 | 0.9281 | 3.90% | 81.10% | 85.54% |
| **Without D1 (All - D1)** | 882 | 0.9265 | 0.9330 | 2.80% | 81.20% | 86.75% |
| **Without D2 (All - D2)** | 882 | 0.9273 | 0.9302 | 3.00% | 81.40% | 87.35% |

---

## 6. Limited-Meter Attacks: Detection vs Shift Ratio (@ 5% Validation Normal FA)

| Shift Ratio Bin | Count ($N$) | RF Primary | Autoencoder (All) | Mahalanobis | Calibrated LNR | Chi-Square |
|---|---|---|---|---|---|---|
| **<=0.25** | 85 | 11.76% | 12.94% | 11.76% | 8.24% | 7.06% |
| **0.25-0.50** | 32 | 62.50% | 59.38% | 53.12% | 18.75% | 21.88% |
| **0.50-0.75** | 19 | 89.47% | 84.21% | 78.95% | 26.32% | 21.05% |
| **>0.75** | 14 | 78.57% | 85.71% | 78.57% | 28.57% | 14.29% |

---

## 7. Permutation Null Check Verification (20 Seeds, Permuted Labels)

- **RF Primary Permutation Null Check:** `mean = 0.5014, std = 0.0099, min = 0.4835, max = 0.5191`
- **Autoencoder All Permutation Null Check:** `    mean = 0.5018, std = 0.0139, min = 0.4634, max = 0.5232`
- **Conclusion:** Meets strict null hypothesis criterion (mean within $0.50 \pm 0.02$, std $< 0.03$), demonstrating zero leakage.
