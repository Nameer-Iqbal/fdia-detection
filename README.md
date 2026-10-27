# Physics-Informed False Data Injection Attack (FDIA) Detection in Smart Distribution Grids

This repository provides a physics-informed detection and benchmarking framework for False Data Injection Attacks (FDIAs) targeting power distribution networks, evaluated on the IEEE 33-bus radial distribution system. 

Power flow physics, AC state estimation (Weighted Least Squares), and diverse cyber-attack scenarios (random, linearized, and exact AC stealth attacks) are simulated in Julia using [JuliaGrid](https://github.com/mcosovic/JuliaGrid.jl). Physics-informed residual, consistency, sensitivity, and temporal features are extracted and benchmarked in Python against classical bad data detection (BDD) baselines using supervised Random Forests and unsupervised Autoencoder / Mahalanobis anomaly detectors.

---

## Tool & Software Versions

The repository was verified with the following pinned software environments:

- **Operating System:** Windows 11 (PowerShell)
- **Version Control:** Git `2.50.0.windows.2`
- **Julia Environment:** `Julia 1.13.1` (Project environment in `julia/`)
  - `JuliaGrid`: `v0.6.2`
  - `CSV`: `v1.1.0`
  - `DataFrames`: `v1.8.2`
  - `Random`: Standard Library
- **Python Environment:** `Python 3.11.5` (Virtual environment in `.venv/`)
  - `torch`: `2.8.0+cpu`
  - `scikit-learn`: `1.9.1`
  - `numpy`: `2.4.6`
  - `pandas`: `3.0.6`
  - `scipy`: `1.15.2`
  - `matplotlib`: `3.10.8`

---

## Folder Layout

```
fdia-detection/
├── configs/                  # TOML configuration files
│   ├── attacks.toml          # Attack parameters (v1)
│   ├── bdd.toml              # Bad data detector thresholds (v1)
│   ├── bdd_v2.toml           # Bad data detector thresholds (v2)
│   ├── dataset.toml          # Normal data generator config (v1)
│   ├── dataset_v2.toml       # Normal data generator config (v2)
│   ├── dataset_attacks.toml  # Attack dataset generation config (v1)
│   ├── dataset_attacks_v2.toml # Attack dataset generation config (v2)
│   ├── features.toml         # Feature extraction config (v1)
│   ├── features_v2.toml      # Feature extraction config (v2)
│   ├── measurements.toml     # Sensor measurement topology & noise std
│   └── models.toml           # Model architectures, tuning grids & seeds
├── data/                     # Data artifacts (CSV and JSON metadata; gitignored)
│   ├── cases/                # IEEE 33-bus network definition (.m)
│   └── .gitkeep              # Directory placeholder
├── julia/                    # Power systems simulation & attack engines (JuliaGrid)
│   ├── 00_check_env.jl       # Environment verification script
│   ├── 01_power_flow.jl      # AC power flow solver
│   ├── 02_state_estimation.jl# WLS state estimator
│   ├── 03_normal_data.jl     # Multi-day normal operating data generator
│   ├── 04_attacks.jl         # Attack construction functions
│   ├── 05_verify_attacks.jl  # Mathematical stealth verification
│   ├── 06_generate_dataset.jl# End-to-end dataset generation
│   ├── 07_sensitivity_features.jl # Measurement leverage matrix calculation
│   ├── 08_generate_controls.jl   # Fingerprint control row generator
│   ├── Project.toml          # Julia package dependencies
│   ├── Manifest.toml         # Julia pinned dependency tree
│   └── tools/                # Auxiliary case file converters
├── python/                   # Machine learning feature extraction & benchmarking
│   ├── 00_check_env.py       # Python environment check
│   ├── splits.py             # Group-aware train/val/test splits
│   ├── features.py           # Physics-informed feature extraction pipeline
│   ├── models.py             # Random Forest & PyTorch Autoencoder architectures
│   ├── run_task8.py          # Unified benchmarking, tuning & evaluation script
│   ├── check_tables.py       # Automated cross-check verification tool
│   ├── eval_v2.py            # Evaluation helper routines and bootstrap metrics
│   └── requirements.txt      # Pinned pip requirements
├── report/                   # Detailed experimental history & logs
│   └── REPORT_LOG.md         # Chronological technical log
├── results/                  # Final tables, verification reports & figures
│   ├── final_tables.md       # SINGLE SOURCE OF TRUTH for all benchmark numbers
│   ├── task8_final_output.txt# Complete console log of benchmark run
│   ├── check_tables_report.txt # Output of check_tables.py (zero mismatches)
│   ├── roc_curves.png        # Comparative ROC curves
│   ├── detection_vs_sigma.png# Detection rate vs attack magnitude
│   ├── score_distributions.png # Anomaly score probability densities
│   ├── importance_bar_chart.png# Grouped permutation feature importance
│   └── limited_meter_shift_ratio.png # Limited-meter detection curve
└── README.md                 # Project documentation and reproduction guide
```

---

## Dataset v1 vs Dataset v2

The project evaluated two generations of datasets:

1. **Dataset v1 (Low Load Variability):**
   - Synthesized using a static 24-hour load shape with small independent Gaussian noise added per operating point.
   - *Limitation:* The state estimates clustered tightly around the static daily profile. Classifiers could detect attacks through simple historical bounds without learning network physical interactions.
2. **Dataset v2 (Realistic Day-to-Day Load Variability):**
   - Incorporates day-level scaling factors ($\pm 5\%$ normal standard deviation per 24-hour block) and intra-day shape jitter ($\pm 3\%$ standard deviation on peak hour timing).
   - Generates realistic operational diversity spanning the load factor range $[0.60, 1.00]$ pu.
   - Evaluates attacks under true operational distribution shift, forcing detectors to leverage multi-measurement spatial and physical correlations.

---

## Ordered Reproduction Commands

To reproduce the entire pipeline from scratch, execute the following commands in sequence:

```powershell
# 1. Case File Conversion
julia --project=. julia/tools/convert_case33bw.jl

# 2. Normal Multi-Day Operating Data Generation (Dataset v2)
julia --project=. julia/03_normal_data.jl configs/dataset_v2.toml

# 3. Attack Dataset Generation (Dataset v2: 10,000 paired rows)
julia --project=. julia/06_generate_dataset.jl configs/dataset_attacks_v2.toml

# 4. Fingerprint Control Rows Generation
julia --project=. julia/08_generate_controls.jl

# 5. Group-Aware Dataset Splitting (Standard & Unseen Schemes)
.venv\Scripts\python.exe python/splits.py configs/dataset_attacks_v2.toml

# 6. Sensitivity Leverage Feature Computation (Hat Matrix Diagonals)
julia --project=. julia/07_sensitivity_features.jl configs/features_v2.toml

# 7. Physics-Informed Feature Engineering Pipeline (Groups A, B, C, D)
.venv\Scripts\python.exe python/features.py configs/features_v2.toml

# 8. Unified Model Training, Evaluation & Benchmarking
.venv\Scripts\python.exe python/run_task8.py

# 9. Automated Cross-Check Verification
.venv\Scripts\python.exe python/check_tables.py
```

*Note on Execution Time:* Steps 2, 3, and 6 involve intensive nonlinear AC power flow solutions and matrix factorizations in JuliaGrid across 20,000 operational states, requiring ~90–120 minutes. Steps 7, 8, and 9 execute in Python and complete in ~15 minutes.

---

## Single Source of Truth & Results Pointer

All official numerical results, confidence intervals, operating thresholds, and ablation comparisons are documented in:

- **Primary Source of Truth:** [`results/final_tables.md`](results/final_tables.md)
- **Raw Execution Log:** [`results/task8_final_output.txt`](results/task8_final_output.txt)
- **Cross-Check Audit:** [`results/check_tables_report.txt`](results/check_tables_report.txt) confirms zero discrepancies across 364 tested values.

### Key Benchmark Takeaways

- **Classical Residual BDD Fails on Stealth Attacks:** Chi-Square and Calibrated Largest Normalized Residual (LNR) achieve only **$3.61\%$** and **$5.11\%$** detection against exact AC stealth attacks at a $5\%$ false alarm rate (performing worse than random guessing).
- **Physics-Informed Features Restore Observability:**
  - **Random Forest (Primary: A+B+C+D1+D2, 1110 features):** Achieves **$86.75\%$** detection on exact AC attacks and **$83.50\%$** overall detection (ROC-AUC `0.9328`).
  - **Unsupervised Autoencoder (D1+D2, 456 features):** Achieves **$92.17\%$** detection on exact AC attacks and **$82.60\%$** overall detection (ROC-AUC `0.9333`), trained strictly on unattacked normal states.
  - **Unseen Attack Generalization (Table 2):** When trained only on random and linearized attacks and tested on held-out exact AC stealth attacks, the Random Forest maintains **$89.57\%$** detection and the Autoencoder achieves **$91.48\%$** detection.
- **Feature Importance:** Consistency features (Group B: KCL active/reactive nodal mismatches) and sensitivity features (Group C: projection leverage diagonals) provide the strongest individual discrimination against stealth attacks ($> 87\%$ detection each in isolation).

---

## Known Limitations

1. **Simulated Synthetic Data:** All evaluations are conducted on synthetic data generated from the IEEE 33-bus radial distribution network model. No physical grid hardware-in-the-loop tests or real SCADA attack recordings were available.
2. **Assumed Measurement Noise:** Sensor errors are assumed to be zero-mean independent Gaussian with fixed standard deviations ($0.005$ pu for voltage magnitudes, $0.010$ pu for power injections, $0.008$ pu for branch flows).
3. **Load Factor Dynamic Range:** Normal operations are simulated within the load factor interval $[0.60, 1.00]$ pu of the standard Matpower case load. Extreme system contingencies (e.g., topology switching, line faults, or islanding) were not evaluated.
4. **Attacker Optimization Parameterization:** Limited-meter attacks rely on a regularized trade-off between targeted state deflection and residual leakage governed by penalty weight $\lambda = 10^5$. Variations in attacker objective formulation could impact detectability.
5. **Group D3 Temporal Feature Assumption:** Group D3 computes step changes relative to the previous sample $\hat{x}_t - \hat{x}_{t-1}$, assuming the previous sample was uncorrupted. A sustained multi-step attack could degrade this specific feature.
6. **Realized False Alarm Variability:** While all detection thresholds are calibrated to nominal $5.0\%$ and $1.0\%$ false alarm percentiles on validation normal rows, discrete sample boundaries and distribution shifts cause realized false alarm rates on test normal rows to range between $3.0\%$ and $6.7\%$ across detectors.
7. **Lack of Real Attack Data:** Due to security and privacy restrictions in critical infrastructure, no real-world utility FDIA logs exist publicly. All attack vectors are synthesized based on power flow physics and adversarial optimization literature.
