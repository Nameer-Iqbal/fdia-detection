# Physics-Informed False Data Injection Attack (FDIA) Detection in Smart Distribution Grids

This repository simulates and benchmarks the detection of False Data Injection Attacks (FDIAs) against AC state estimation on the IEEE 33-bus radial distribution system.

Power flow, Weighted Least Squares AC state estimation and the attacks (random, linearised stealth, exact AC stealth, limited-meter and distribution-aware "mimic" stealth attacks) are simulated in Julia with [JuliaGrid](https://github.com/mcosovic/JuliaGrid.jl). Residual, consistency, sensitivity and non-residual features are extracted in Python and used to compare classical Bad Data Detection (chi-square test, largest normalised residual) with a supervised Random Forest, unsupervised Autoencoders and a per-hour Mahalanobis baseline.

The central result is a qualified one. Data-driven detectors catch the exact AC stealth attacks that classical BDD cannot, but they do so by recognising that the falsified state is unusual for that hour, not by detecting a physics violation. An attacker who moves the state to a normal-looking operating point defeats every detector (see the headline findings below and `results/final_tables.md`, Section 4).

---

## Tool & Software Versions

Committed results (`results/`) were produced with:

- **Julia** `1.13.1` with the project environment in `julia/` (`Project.toml` + `Manifest.toml`): `JuliaGrid v0.6.2`, `CSV v1.1.0`, `DataFrames v1.8.2`
- **Python** `3.13.18` in a virtual environment `.venv/`: `torch 2.15.0` (CPU), `scikit-learn 1.9.1`, `numpy 2.5.4`, `pandas 3.1.1`, `scipy 1.18.1`, `matplotlib 3.11.2`
- **OS:** Ubuntu 24.04. The pipeline was originally developed on Windows 11 (Python 3.11.5, torch 2.8.0); every Random Forest, BDD and Mahalanobis number is identical across the two environments, and the Autoencoder rows differ by at most about one percentage point because of the PyTorch version.

---

## Folder Layout

```
fdia-detection/
├── configs/                          # Every parameter lives here (seeds, noise, paths, model grids)
│   ├── measurements.toml             # Meter set and noise for julia/02_state_estimation.jl
│   ├── dataset.toml / dataset_v2.toml            # Normal data generators (v1 / v2) incl. [noise] (single source)
│   ├── attacks.toml                  # Attack verification suite (v1, julia/05)
│   ├── dataset_attacks.toml / dataset_attacks_v2.toml   # Labelled attack datasets (v1 / v2)
│   ├── bdd.toml / bdd_v2.toml        # BDD thresholds and voltage envelope
│   ├── features.toml / features_v2.toml          # Splits, sensitivity and feature outputs
│   ├── mimic_attacks_v2.toml         # Distribution-aware (mimic) stealth attacks
│   ├── models.toml                   # RF grid, Autoencoder settings, evaluation bins
│   └── *_lownoise.toml               # Low-noise sensitivity variant (same seeds, ~10x lower meter noise)
├── data/                             # Generated CSV/JSON (gitignored) + data/cases/case33bw.m
├── julia/
│   ├── 00_check_env.jl               # Environment check
│   ├── 01_power_flow.jl              # AC power flow (Baran & Wu benchmark check)
│   ├── 02_state_estimation.jl        # WLS state estimation, observability, chi-square Monte Carlo
│   ├── 03_normal_data.jl             # Multi-day normal operating points + state estimates
│   ├── 04_attacks.jl                 # Attack library + shared helpers (noise config, leverage)
│   ├── 05_verify_attacks.jl          # Stealth / leakage verification suite (dataset v1)
│   ├── 06_generate_dataset.jl        # 10,000-row paired normal/attacked dataset
│   ├── 07_sensitivity_features.jl    # Leverage and residual-covariance diagonals per row
│   ├── 08_generate_controls.jl       # c = 0 control rows (state-estimator idempotence check)
│   ├── 09_mimic_attacks.jl           # Distribution-aware (mimic) exact AC stealth attacks
│   └── tools/convert_case33bw.jl     # Case file conversion
├── python/
│   ├── splits.py                     # Group-aware and leak-free unseen-attack splits
│   ├── features.py                   # Feature groups A, B, C, D1, D2, D3 (+ mimic rows)
│   ├── models.py                     # Random Forest tuning, PyTorch Autoencoder
│   ├── run_task8.py                  # All benchmarks, extra experiments and figures
│   ├── eval_v2.py                    # Group-bootstrap confidence intervals (+ earlier shortcut tests)
│   ├── table_utils.py                # Parses result tables from run logs / markdown
│   ├── make_final_tables.py          # Builds results/final_tables.md and the README headline
│   └── check_tables.py               # Verifies final_tables.md cell-by-cell against the run logs
├── report/REPORT_LOG.md              # Chronological technical log
└── results/
    ├── final_tables.md               # SINGLE SOURCE OF TRUTH for all reported numbers (generated)
    ├── task8_final_output.txt        # Raw log of the main benchmark run
    ├── check_tables_report.txt       # Output of check_tables.py
    ├── *.png                         # ROC curves, detection vs attack size, score distributions,
    │                                 # permutation importance, limited-meter curve, mimic_vs_exact.png
    └── lownoise/                     # Raw log and figures of the low-noise sensitivity run
```

---

## Dataset v1 vs Dataset v2

1. **Dataset v1 (low load variability):** a fixed 24-hour load shape with small independent per-bus noise. State estimates clustered tightly around the daily profile, so simple historical bounds were enough to flag attacks.
2. **Dataset v2 (day-to-day variability, used for all reported results):** adds a per-day load scale factor (Gaussian, 5% standard deviation) and a per-day peak-hour shift (±1 h), giving load factors over [0.60, 1.00] and about 2.4x wider per-hour state variability.

## Threat Models

| Attack | How the state shift c is chosen | Visible to BDD? |
|---|---|---|
| Random | no state shift; 5-20 meters offset by 3-10 sigma | yes |
| Linearised stealth (single / multi-bus) | random c on target buses, a = H c | partly (linearisation error leaks into residuals) |
| Exact AC stealth (single / multi-bus) | random c on target buses, a = h(x + c) - h(x) | no |
| Limited-meter | least-squares attack using only a random subset of meters | partly |
| **Mimic exact AC stealth** (`julia/09_mimic_attacks.jl`) | c moves the estimate onto a genuine normal state from another day at the same hour | no, and the falsified state is typical for that hour |

---

## Ordered Reproduction Commands

Run from the repository root. Julia uses the project environment in `julia/` (`--project=julia`). On Windows replace `.venv/bin/python` with `.venv\Scripts\python.exe`.

One-time setup:

```bash
julia --project=julia -e "using Pkg; Pkg.instantiate()"
```

```bash
python -m venv .venv
.venv/bin/python -m pip install -r python/requirements.txt
```

Optional checks of the power-flow benchmark, state estimator and attack stealth (the last one needs the v1 normal data):

```bash
julia --project=julia julia/01_power_flow.jl
julia --project=julia julia/02_state_estimation.jl
julia --project=julia julia/03_normal_data.jl configs/dataset.toml
julia --project=julia julia/05_verify_attacks.jl
```

Main study (dataset v2), in order:

```bash
julia --project=julia julia/03_normal_data.jl configs/dataset_v2.toml
julia --project=julia julia/06_generate_dataset.jl configs/dataset_attacks_v2.toml
julia --project=julia julia/08_generate_controls.jl configs/dataset_attacks_v2.toml
.venv/bin/python python/splits.py configs/features_v2.toml
julia --project=julia julia/09_mimic_attacks.jl configs/mimic_attacks_v2.toml
julia --project=julia julia/07_sensitivity_features.jl configs/features_v2.toml
.venv/bin/python python/features.py configs/features_v2.toml
.venv/bin/python python/run_task8.py configs/models.toml results/task8_final_output.txt
```

Low-noise sensitivity check (same steps, `_lownoise` configs; step 08 is not needed):

```bash
julia --project=julia julia/03_normal_data.jl configs/dataset_v2_lownoise.toml
julia --project=julia julia/06_generate_dataset.jl configs/dataset_attacks_v2_lownoise.toml
.venv/bin/python python/splits.py configs/features_v2_lownoise.toml
julia --project=julia julia/09_mimic_attacks.jl configs/mimic_attacks_v2_lownoise.toml
julia --project=julia julia/07_sensitivity_features.jl configs/features_v2_lownoise.toml
.venv/bin/python python/features.py configs/features_v2_lownoise.toml
.venv/bin/python python/run_task8.py configs/models_lownoise.toml results/lownoise/task8_output.txt
```

Build and verify the results tables (also refreshes the headline block below):

```bash
.venv/bin/python python/make_final_tables.py
.venv/bin/python python/check_tables.py
```

*Execution time:* each Julia step takes under a minute. Feature engineering takes about 3 minutes per dataset and each `run_task8.py` run about 15 minutes on 4 CPU cores.

---

## Headline Findings

All numbers below are generated from the run logs by `python/make_final_tables.py`; full tables with confidence intervals are in [`results/final_tables.md`](results/final_tables.md).

<!-- BEGIN GENERATED HEADLINE (python/make_final_tables.py) -->
1. **Classical BDD cannot see exact AC stealth attacks.** At the 5% validation false-alarm threshold the chi-square test detects 3.61% and the calibrated largest normalised residual 8.13% of exact AC attacks, i.e. chance level, as theory predicts (residuals are unchanged by a = h(x+c) - h(x)).
2. **Data-driven detectors catch the project's exact AC attacks** (Random Forest 86.75%, Autoencoder 90.36%, per-hour Mahalanobis distance on the state estimate 82.83%), and the RF trained only on random and linearised attacks still detects 87.95% of held-out exact AC attacks (Table 2, leak-free split).
3. **But they do so by recognising an atypical state, not a physics violation.** Against a distribution-aware ('mimic') exact AC attacker that moves the estimate onto a genuine normal state from another day at the same hour (donors never seen in training), detection at the 5% threshold collapses to the false-alarm level for every detector: RF 3.40% (false alarms 2.90%), Autoencoder 4.60% (4.70%), Mahalanobis 6.10% (6.70%), chi-square 3.60% (3.60%). The best ranking score is the primary Random Forest's AUC of 0.578, a weak signal with no practical detection value. The mimic attacks are not smaller than the project's own exact AC attacks (median maximum voltage shift 0.0055 pu vs 0.0040 pu; Section 4).
4. **The 'physics-informed' feature groups act as state proxies.** Group C (leverage) is predicted from the state estimate alone with median R^2 = 1.0000. In Group B the KCL power-balance mismatches are dominated by meter noise (median R^2 = -0.0274) and separate exact AC attacks only weakly (AUC 0.588); Group B's discriminative power comes from the neighbour-voltage block (exact AC AUC 0.973), itself a function of the state (Section 5).
5. **Conclusions do not depend on the conservative meter noise.** With ~10x lower power-meter noise, chi-square detection of linearised single-bus attacks rises from 17.14% to 36.00%, yet exact AC attacks remain invisible to BDD (3.61%) and mimic attacks remain invisible to the RF (5.30%) and Autoencoder (4.60%) even though the Autoencoder now detects 100.00% of the random-shift exact AC attacks. The only residual signal comes from the D3 'change from previous hour' features, which assume a clean previous sample: RF with D3 reaches mimic AUC 0.657 but detects only 7.50% at a 4.70% false-alarm rate (Section 8).
<!-- END GENERATED HEADLINE -->

---

## Known Limitations

1. **Synthetic data only.** All results come from simulations of the IEEE 33-bus feeder; no field measurements or recorded attacks were available.
2. **Meter noise.** Noise is zero-mean independent Gaussian with a constant standard deviation per meter type. The main study uses 0.005 / 0.010 / 0.008 pu (voltage / injection / flow), so an injection meter's noise (100 kW) is comparable to a typical bus load. This weakens every residual-based test. The low-noise run (0.001 / 0.001 / 0.0008 pu) shows which conclusions depend on this choice (`results/final_tables.md`, Section 8).
3. **Full measurement set.** Every bus has voltage and injection meters and every branch has a from-end flow meter (163 measurements, redundancy 2.5), which is richer than a typical distribution feeder.
4. **Operating range.** Load factors span [0.60, 1.00] of the case load; topology changes, faults and distributed generation are not modelled.
5. **Attacker knowledge.** The random-shift attacks ignore the joint behaviour of the bus states, which is what lets data-driven detectors flag them. The mimic attacker assumes full write access to all 163 meters plus historical data. A mimic attacker restricted to a meter subset was not evaluated.
6. **Group D3 assumption.** D3 features compare against the previous hour's sample and assume it was not attacked; a sustained multi-step attack would invalidate this.
7. **Realized false-alarm variability.** Thresholds are set on 1,000 validation normal rows; realized test false-alarm rates range from 0.1% to 6.7% (5% nominal) across detectors, and all detection rates must be read next to them.
8. **No attack localisation.** The optional localisation output in the project brief is not implemented.
