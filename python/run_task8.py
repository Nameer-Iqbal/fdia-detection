"""
Task 8 Full Experiment Runner (Updated for Task 8b):
  Part 1: Supervised Random Forest (tuning on validation split, test evaluation, grouped permutation importance)
  Part 2: Unsupervised PyTorch Autoencoders (D1+D2 and All, validation bottleneck selection)
  Part 3: Comprehensive Comparison Tables with UNIFORM VALIDATION-DERIVED THRESHOLDING:
          Threshold = percentile of scores on VALIDATION normal rows (5% and 1%).
          Realized false alarm rates on test normal rows reported next to each method.
          Corrected range-check envelope logic (fixing the 100% detection bug).
          Clarification of Chi-Square ROC-AUC (0.6266 true objective_J AUC vs 0.8147 Max-Z).
  Part 4: Extra Experiments:
          Detection vs Sigma Bins (@ 5% validation-derived FA).
          Feature Group Ablations using the identical tuned main RF hyperparameters.
          Limited-Meter vs Shift Ratio.
  Part 5: Honesty Rules (Permutation Null Checks, Visualization Generation to results/).
  Part 6: Report Generation.
"""

import os
import sys
import tomllib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.covariance import LedoitWolf

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("python"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse group_bootstrap_ci from eval_v2.py as required
from eval_v2 import group_bootstrap_ci
from models import (
    get_feature_columns_by_group,
    select_feature_columns,
    tune_random_forest,
    grouped_permutation_importance,
    select_best_autoencoder,
)


def load_config(config_path="configs/models.toml"):
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def compute_flags(scores, threshold, is_range_check=False):
    """
    Uniform thresholding rule:
    If range-check or threshold == 0.0 (where nominal envelope excursion is 0.0),
    anomaly requires strictly positive excursion (scores > 0.0).
    For standard continuous scores, anomaly requires scores >= threshold.
    """
    if is_range_check or threshold == 0.0:
        return scores > threshold
    return scores >= threshold


class _Tee:
    """Writes everything printed to the console and to a log file at the same time."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for st in self.streams:
            st.write(text)

    def flush(self):
        for st in self.streams:
            st.flush()


def classification_metrics(flags, y):
    """Accuracy, precision, recall (= detection rate), F1 and false-alarm rate, all in %."""
    flags = np.asarray(flags, bool); y = np.asarray(y)
    tp = int(np.sum(flags & (y == 1))); fp = int(np.sum(flags & (y == 0)))
    fn = int(np.sum(~flags & (y == 1))); tn = int(np.sum(~flags & (y == 0)))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"acc": 100.0 * (tp + tn) / len(y), "prec": 100.0 * prec, "rec": 100.0 * rec,
            "f1": 100.0 * f1, "fa": 100.0 * fp / max(fp + tn, 1)}


def print_classification_table(title, methods_dict, y):
    """
    Prints the brief's classification metrics (accuracy, precision, recall, F1, false
    alarm) for every method at its 5% and 1% validation-normal thresholds.
    Note: accuracy and precision depend on the class balance of the evaluated set.
    """
    n_pos, n_neg = int(np.sum(y == 1)), int(np.sum(y == 0))
    print(f"\n--- {title} ---")
    print(f"(normal N={n_neg}, attacked N={n_pos}; thresholds = 95th / 99th percentile of validation normal scores)")
    hdr = (f"{'Method':<26} | {'Acc @5%':<8} | {'Prec @5%':<8} | {'Recall @5%':<10} | {'F1 @5%':<8} | {'FA @5%':<8} | "
           f"{'Acc @1%':<8} | {'Prec @1%':<8} | {'Recall @1%':<10} | {'F1 @1%':<8} | {'FA @1%':<8}")
    print(hdr)
    print("-" * len(hdr))
    for m_name, (scores, th5, th1, is_rc) in methods_dict.items():
        a = classification_metrics(compute_flags(scores, th5, is_rc), y)
        b = classification_metrics(compute_flags(scores, th1, is_rc), y)
        print(f"{m_name:<26} | {a['acc']:6.2f}%  | {a['prec']:6.2f}%  | {a['rec']:6.2f}%    | {a['f1']:6.2f}%  | {a['fa']:6.2f}%  | "
              f"{b['acc']:6.2f}%  | {b['prec']:6.2f}%  | {b['rec']:6.2f}%    | {b['f1']:6.2f}%  | {b['fa']:6.2f}%")


def main():
    print("=" * 80)
    print("TASK 8b: BENCHMARKING WITH UNIFORM VALIDATION THRESHOLDING & BUG FIXES")
    print("=" * 80)

    # Usage: python python/run_task8.py [config.toml] [log.txt]
    # With a log path, everything printed is also written (UTF-8) to that file,
    # which is what python/make_final_tables.py and python/check_tables.py read.
    if len(sys.argv) > 2:
        os.makedirs(os.path.dirname(sys.argv[2]) or ".", exist_ok=True)
        sys.stdout = _Tee(sys.__stdout__, open(sys.argv[2], "w", encoding="utf-8", buffering=1))
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/models.toml"
    cfg = load_config(cfg_path)
    print(f"Config: {cfg_path}")
    seed = cfg["general"]["seed"]
    np.random.seed(seed)
    os.makedirs(cfg["general"]["results_dir"], exist_ok=True)

    # -------------------------------------------------------------------------
    # 0. Load Data and Metadata
    # -------------------------------------------------------------------------
    print("\n[0] Loading Data and Aligning Feature Matrices...")
    df_attacks = pd.read_csv(cfg["general"]["attacks_file"])
    df_splits = pd.read_csv(cfg["general"]["splits_file"])

    df_train_feat = pd.read_csv(cfg["general"]["train_features"])
    df_val_feat = pd.read_csv(cfg["general"]["val_features"])
    df_test_feat = pd.read_csv(cfg["general"]["test_features"])

    # Combine to index by split and split_unseen easily
    df_all_feat = pd.concat([df_train_feat, df_val_feat, df_test_feat], ignore_index=True)
    df_all_feat = df_all_feat.sort_values("row_id").reset_index(drop=True)
    df_attacks = df_attacks.sort_values("row_id").reset_index(drop=True)
    df_splits = df_splits.sort_values("row_id").reset_index(drop=True)

    groups_dict = get_feature_columns_by_group(cfg["general"]["features_meta"])
    print(f"Feature Groups available: { {k: len(v) for k, v in groups_dict.items()} }")

    # Feature sets
    cols_d12 = select_feature_columns(groups_dict, {"d1": True, "d2": True})
    cols_primary = select_feature_columns(groups_dict, {
        "a": cfg["features"]["use_group_a"],
        "b": cfg["features"]["use_group_b"],
        "c": cfg["features"]["use_group_c"],
        "d1": cfg["features"]["use_group_d1"],
        "d2": cfg["features"]["use_group_d2"],
        "d3": False,
    })
    cols_all_with_d3 = select_feature_columns(groups_dict, {
        "a": True, "b": True, "c": True, "d1": True, "d2": True, "d3": True
    })

    print(f"Active feature counts: D1+D2={len(cols_d12)}, Primary (A+B+C+D1+D2)={len(cols_primary)}, Full with D3={len(cols_all_with_d3)}")

    # Standard split masks
    train_mask = (df_splits["split"] == "train").values
    val_mask = (df_splits["split"] == "val").values
    test_mask = (df_splits["split"] == "test").values

    y_all = df_all_feat["label"].values
    group_all = df_all_feat["group_id"].values

    y_train = y_all[train_mask]
    y_val = y_all[val_mask]
    y_test = y_all[test_mask]

    test_meta = df_attacks[test_mask].reset_index(drop=True)
    val_meta = df_attacks[val_mask].reset_index(drop=True)
    train_meta = df_attacks[train_mask].reset_index(drop=True)

    test_groups = group_all[test_mask]
    val_groups = group_all[val_mask]

    val_norm_mask = (val_meta["label"] == 0).values
    norm_mask_test = (test_meta["label"] == 0).values
    att_mask_test = (test_meta["label"] == 1).values
    exact_mask_test = test_meta["attack_type"].isin(["exact_ac_single_bus", "exact_ac_multi_bus"]).values
    eff_mask_test = (test_meta["label"] == 1) & (test_meta["effective_attack"] == True).values

    # -------------------------------------------------------------------------
    # PART 1: Random Forest (Supervised)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART 1: RANDOM FOREST (SUPERVISED) TUNING & VALIDATION")
    print("=" * 80)

    X_train_prim = df_all_feat[cols_primary].values[train_mask]
    X_val_prim = df_all_feat[cols_primary].values[val_mask]
    X_test_prim = df_all_feat[cols_primary].values[test_mask]

    print("Tuning Random Forest on VALIDATION split only...")
    best_rf_prim, best_rf_params, tune_history = tune_random_forest(
        X_train_prim, y_train, X_val_prim, y_val, cfg["random_forest"], seed=cfg["random_forest"]["seed"]
    )
    print(f"Chosen Hyperparameters (Validation Best): {best_rf_params}")
    for h in tune_history:
        print(f"  n_est={h['n_estimators']}, max_depth={h['max_depth']}, min_leaf={h['min_samples_leaf']} -> Val AUC: {h['val_auc']:.4f}")

    # Predict validation normal scores to derive thresholds
    rf_prim_val_scores = best_rf_prim.predict_proba(X_val_prim)[:, 1]
    rf_prim_val_norm = rf_prim_val_scores[val_norm_mask]
    rf_prim_th_5fa = float(np.percentile(rf_prim_val_norm, 95.0))
    rf_prim_th_1fa = float(np.percentile(rf_prim_val_norm, 99.0))
    print(f"RF Primary Validation Normal Thresholds: 5% FA = {rf_prim_th_5fa:.4f}, 1% FA = {rf_prim_th_1fa:.4f}")

    # Predict test
    rf_prim_test_scores = best_rf_prim.predict_proba(X_test_prim)[:, 1]
    rf_prim_flags_5 = compute_flags(rf_prim_test_scores, rf_prim_th_5fa)
    rf_prim_flags_1 = compute_flags(rf_prim_test_scores, rf_prim_th_1fa)

    realized_fa_5_rf = np.mean(rf_prim_flags_5[norm_mask_test]) * 100.0
    realized_fa_1_rf = np.mean(rf_prim_flags_1[norm_mask_test]) * 100.0
    print(f"Realized Test Normal False Alarm: @5% th = {realized_fa_5_rf:.2f}%, @1% th = {realized_fa_1_rf:.2f}%")

    auc_prim_all = roc_auc_score(y_test, rf_prim_test_scores)
    ci_auc_prim_all = group_bootstrap_ci(y_test, rf_prim_test_scores, test_groups)

    exact_sub_mask = exact_mask_test | norm_mask_test
    auc_prim_exact = roc_auc_score(y_test[exact_sub_mask], rf_prim_test_scores[exact_sub_mask])
    ci_auc_prim_exact = group_bootstrap_ci(y_test[exact_sub_mask], rf_prim_test_scores[exact_sub_mask], test_groups[exact_sub_mask])

    det_prim_5fa_all = np.mean(rf_prim_flags_5[att_mask_test]) * 100.0
    ci_det_prim_5fa_all = group_bootstrap_ci(None, rf_prim_flags_5[att_mask_test], test_groups[att_mask_test], is_binary_flag=True)

    det_prim_1fa_all = np.mean(rf_prim_flags_1[att_mask_test]) * 100.0
    ci_det_prim_1fa_all = group_bootstrap_ci(None, rf_prim_flags_1[att_mask_test], test_groups[att_mask_test], is_binary_flag=True)

    print(f"\nPrimary RF (A+B+C+D1+D2) Test Performance:")
    print(f"  Overall ROC-AUC:   {auc_prim_all:.4f} [{ci_auc_prim_all[0]:.4f}, {ci_auc_prim_all[1]:.4f}]")
    print(f"  Exact AC ROC-AUC:  {auc_prim_exact:.4f} [{ci_auc_prim_exact[0]:.4f}, {ci_auc_prim_exact[1]:.4f}]")
    print(f"  Detection @ 5% FA: {det_prim_5fa_all:.2f}% [{ci_det_prim_5fa_all[0]:.2f}%, {ci_det_prim_5fa_all[1]:.2f}%]")
    print(f"  Detection @ 1% FA: {det_prim_1fa_all:.2f}% [{ci_det_prim_1fa_all[0]:.2f}%, {ci_det_prim_1fa_all[1]:.2f}%]")

    # Grouped Permutation Importance on VALIDATION
    print("\nComputing Grouped Permutation Importance on VALIDATION split (5 repeats)...")
    val_base_auc, group_importances = grouped_permutation_importance(
        best_rf_prim, X_val_prim, y_val, cols_primary, groups_dict,
        active_groups=["a", "b", "c", "d1", "d2"], n_repeats=5, seed=seed
    )
    print(f"Validation Baseline AUC: {val_base_auc:.4f}")
    for g, res in group_importances.items():
        print(f"  Group {g.upper():<3} ({res['n_features']:3d} feats): drop = {res['mean_drop']:+.4f} +/- {res['std_drop']:.4f}")

    # Also train RF WITH D3 (labelled: assumes verified clean previous sample)
    print("\nTraining RF WITH D3 (assumes verified clean previous sample)...")
    X_train_d3 = df_all_feat[cols_all_with_d3].values[train_mask]
    X_val_d3 = df_all_feat[cols_all_with_d3].values[val_mask]
    X_test_d3 = df_all_feat[cols_all_with_d3].values[test_mask]
    rf_d3 = RandomForestClassifier(
        n_estimators=best_rf_params["n_estimators"],
        max_depth=best_rf_params["max_depth"],
        min_samples_leaf=best_rf_params["min_samples_leaf"],
        random_state=seed,
        n_jobs=-1
    )
    rf_d3.fit(X_train_d3, y_train)
    rf_d3_val_scores = rf_d3.predict_proba(X_val_d3)[:, 1]
    rf_d3_th_5fa = float(np.percentile(rf_d3_val_scores[val_norm_mask], 95.0))
    rf_d3_th_1fa = float(np.percentile(rf_d3_val_scores[val_norm_mask], 99.0))
    rf_d3_test_scores = rf_d3.predict_proba(X_test_d3)[:, 1]

    # -------------------------------------------------------------------------
    # PART 2: Autoencoder (Unsupervised, Normal Rows Only)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART 2: AUTOENCODER (UNSUPERVISED) NORMAL-ONLY TRAINING")
    print("=" * 80)

    train_norm_mask = (train_meta["label"] == 0).values

    # Model (i): D1+D2
    print("\n--- Training Autoencoder (i) on D1+D2 (456 features) ---")
    X_train_d12_norm = df_all_feat[cols_d12].values[train_mask][train_norm_mask]
    X_val_d12_norm = df_all_feat[cols_d12].values[val_mask][val_norm_mask]
    X_val_d12_all = df_all_feat[cols_d12].values[val_mask]
    X_test_d12_all = df_all_feat[cols_d12].values[test_mask]

    best_ae_d12, cand_d12_res = select_best_autoencoder(
        X_train_d12_norm, X_val_d12_norm,
        input_dim=len(cols_d12),
        hidden_dims=[128, 64],
        bottleneck_candidates=cfg["autoencoder"]["bottleneck_candidates_d12"],
        seed=cfg["autoencoder"]["seed"],
        lr=cfg["autoencoder"]["learning_rate"],
        max_epochs=cfg["autoencoder"]["max_epochs"],
        batch_size=cfg["autoencoder"]["batch_size"],
        patience=cfg["autoencoder"]["patience"],
    )
    print(f"Candidate Bottlenecks for D1+D2: {cand_d12_res}")
    print(f"Chosen Bottleneck for D1+D2: {best_ae_d12.bottleneck_dim} (val loss: {best_ae_d12.best_val_loss:.6f})")

    val_norm_err_d12 = best_ae_d12.score_samples(X_val_d12_norm)
    ae_d12_th_5fa = float(np.percentile(val_norm_err_d12, 95.0))
    ae_d12_th_1fa = float(np.percentile(val_norm_err_d12, 99.0))
    print(f"Autoencoder D1+D2 Thresholds (Validation Normal): 5% FA = {ae_d12_th_5fa:.6f}, 1% FA = {ae_d12_th_1fa:.6f}")
    ae_d12_test_scores = best_ae_d12.score_samples(X_test_d12_all)

    # Model (ii): All Features (A+B+C+D1+D2)
    print("\n--- Training Autoencoder (ii) on All Features (1110 features) ---")
    X_train_prim_norm = X_train_prim[train_norm_mask]
    X_val_prim_norm = X_val_prim[val_norm_mask]

    best_ae_prim, cand_prim_res = select_best_autoencoder(
        X_train_prim_norm, X_val_prim_norm,
        input_dim=len(cols_primary),
        hidden_dims=[256, 128],
        bottleneck_candidates=cfg["autoencoder"]["bottleneck_candidates_all"],
        seed=cfg["autoencoder"]["seed"],
        lr=cfg["autoencoder"]["learning_rate"],
        max_epochs=cfg["autoencoder"]["max_epochs"],
        batch_size=cfg["autoencoder"]["batch_size"],
        patience=cfg["autoencoder"]["patience"],
    )
    print(f"Candidate Bottlenecks for All Features: {cand_prim_res}")
    print(f"Chosen Bottleneck for All: {best_ae_prim.bottleneck_dim} (val loss: {best_ae_prim.best_val_loss:.6f})")

    val_norm_err_prim = best_ae_prim.score_samples(X_val_prim_norm)
    ae_prim_th_5fa = float(np.percentile(val_norm_err_prim, 95.0))
    ae_prim_th_1fa = float(np.percentile(val_norm_err_prim, 99.0))
    print(f"Autoencoder All Thresholds (Validation Normal): 5% FA = {ae_prim_th_5fa:.6f}, 1% FA = {ae_prim_th_1fa:.6f}")
    ae_prim_test_scores = best_ae_prim.score_samples(X_test_prim)

    # -------------------------------------------------------------------------
    # PART 3: BASELINE METHODS & COMPARISON TABLES WITH UNIFORM THRESHOLDING
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART 3: BASELINES & COMPREHENSIVE COMPARISON TABLES (UNIFORM RULE)")
    print("=" * 80)

    # 1. Chi-Square (objective_J)
    chi_val_scores = val_meta["objective_J"].values
    chi_test_scores = test_meta["objective_J"].values
    chi_th_5fa = float(np.percentile(chi_val_scores[val_norm_mask], 95.0))
    chi_th_1fa = float(np.percentile(chi_val_scores[val_norm_mask], 99.0))

    # 2. Calibrated LNR (max_norm_residual)
    lnr_val_scores = val_meta["max_norm_residual"].values
    lnr_test_scores = test_meta["max_norm_residual"].values
    lnr_th_5fa = float(np.percentile(lnr_val_scores[val_norm_mask], 95.0))
    lnr_th_1fa = float(np.percentile(lnr_val_scores[val_norm_mask], 99.0))

    # 3. Range-Check (Operating Envelope)
    train_norm = df_attacks[(df_splits["split"] == "train") & (df_attacks["label"] == 0)]
    v_cols = [f"x_hat_{i}" for i in range(33, 66)]
    v_min_norm = float(train_norm[v_cols].min().min())
    v_max_norm = float(train_norm[v_cols].max().max())

    val_v = val_meta[v_cols].values
    rc_val_viol = np.maximum(0.0, np.maximum(v_min_norm - val_v, val_v - v_max_norm)).max(axis=1)
    rc_th_5fa = float(np.percentile(rc_val_viol[val_norm_mask], 95.0))  # 0.0
    rc_th_1fa = float(np.percentile(rc_val_viol[val_norm_mask], 99.0))  # 0.0

    test_v = test_meta[v_cols].values
    rc_test_viol = np.maximum(0.0, np.maximum(v_min_norm - test_v, test_v - v_max_norm)).max(axis=1)

    # 4. Per-Hour Max-Z on x_hat
    x_cols = [f"x_hat_{i}" for i in range(1, 66)]
    hourly_stats = {}
    for h in range(24):
        sub = train_norm[train_norm["hour_of_day"] == h][x_cols]
        mu = sub.mean(axis=0).values
        std = sub.std(axis=0).values.copy()
        std[std < 1e-9] = 1.0
        hourly_stats[h] = (mu, std)

    max_abs_z_val = np.zeros(len(val_meta))
    for idx, row in val_meta.iterrows():
        h = int(row["hour_of_day"])
        mu, std = hourly_stats[h]
        x_val = row[x_cols].values.astype(float)
        max_abs_z_val[idx] = np.max(np.abs((x_val - mu) / std))
    z_th_5fa = float(np.percentile(max_abs_z_val[val_norm_mask], 95.0))
    z_th_1fa = float(np.percentile(max_abs_z_val[val_norm_mask], 99.0))

    max_abs_z_test = np.zeros(len(test_meta))
    for idx, row in test_meta.iterrows():
        h = int(row["hour_of_day"])
        mu, std = hourly_stats[h]
        x_val = row[x_cols].values.astype(float)
        max_abs_z_test[idx] = np.max(np.abs((x_val - mu) / std))

    # 5. Mahalanobis (Ledoit-Wolf)
    hourly_lw = {}
    for h in range(24):
        sub = train_norm[train_norm["hour_of_day"] == h][x_cols].values
        hourly_lw[h] = LedoitWolf().fit(sub)

    mahal_val = np.zeros(len(val_meta))
    val_x = val_meta[x_cols].values
    val_hours = val_meta["hour_of_day"].values
    for idx in range(len(val_meta)):
        h = int(val_hours[idx])
        lw = hourly_lw[h]
        diff = val_x[idx:idx+1] - lw.location_
        dist_sq = diff @ lw.precision_ @ diff.T
        mahal_val[idx] = np.sqrt(dist_sq[0, 0])
    mahal_th_5fa = float(np.percentile(mahal_val[val_norm_mask], 95.0))
    mahal_th_1fa = float(np.percentile(mahal_val[val_norm_mask], 99.0))

    mahal_test = np.zeros(len(test_meta))
    test_x = test_meta[x_cols].values
    test_hours = test_meta["hour_of_day"].values
    for idx in range(len(test_meta)):
        h = int(test_hours[idx])
        lw = hourly_lw[h]
        diff = test_x[idx:idx+1] - lw.location_
        dist_sq = diff @ lw.precision_ @ diff.T
        mahal_test[idx] = np.sqrt(dist_sq[0, 0])

    # Method definitions with validation-derived thresholds: (scores, th5, th1, is_range_check)
    methods = {
        "Chi-Square": (chi_test_scores, chi_th_5fa, chi_th_1fa, False),
        "Calibrated LNR": (lnr_test_scores, lnr_th_5fa, lnr_th_1fa, False),
        "Range-Check (Envelope)": (rc_test_viol, rc_th_5fa, rc_th_1fa, True),
        "Per-Hour Max-Z": (max_abs_z_test, z_th_5fa, z_th_1fa, False),
        "Mahalanobis (Ledoit-Wolf)": (mahal_test, mahal_th_5fa, mahal_th_1fa, False),
        "Autoencoder (D1+D2)": (ae_d12_test_scores, ae_d12_th_5fa, ae_d12_th_1fa, False),
        "Autoencoder (All: A-D2)": (ae_prim_test_scores, ae_prim_th_5fa, ae_prim_th_1fa, False),
        "RF w/o D3 (Primary)": (rf_prim_test_scores, rf_prim_th_5fa, rf_prim_th_1fa, False),
        "RF with D3 (Clean Prev)": (rf_d3_test_scores, rf_d3_th_5fa, rf_d3_th_1fa, False),
    }

    print("\n--- Validation Normal Thresholds and Realized Normal Test False Alarm Rates ---")
    print(f"{'Method':<25} | {'Val Thresh 5%':<14} | {'Val Thresh 1%':<14} | {'Realized FA @ 5%':<18} | {'Realized FA @ 1%':<18}")
    print("-" * 100)
    for m_name, (scores, th5, th1, is_rc) in methods.items():
        f5 = compute_flags(scores, th5, is_rc)
        f1 = compute_flags(scores, th1, is_rc)
        r_fa5 = np.mean(f5[norm_mask_test]) * 100.0
        r_fa1 = np.mean(f1[norm_mask_test]) * 100.0
        n_nt = int(np.sum(norm_mask_test))
        print(f"{m_name:<25} | {th5:14.6f} | {th1:14.6f} | {r_fa5:6.2f}% ({np.sum(f5[norm_mask_test]):3d}/{n_nt})   | {r_fa1:6.2f}% ({np.sum(f1[norm_mask_test]):3d}/{n_nt})")

    # Chi-Square vs Per-Hour Max-Z AUC (computed, not hard-coded): earlier summaries
    # confused the two, so both are printed side by side.
    exact_sub_tmp = exact_mask_test | norm_mask_test
    print("\n--- Chi-Square ROC-AUC Clarification ---")
    print(f"  Chi-Square (objective J) ROC-AUC: overall {roc_auc_score(y_test, chi_test_scores):.4f}, "
          f"exact AC {roc_auc_score(y_test[exact_sub_tmp], chi_test_scores[exact_sub_tmp]):.4f}")
    print(f"  Per-Hour Max-Z ROC-AUC (a different detector): overall {roc_auc_score(y_test, max_abs_z_test):.4f}")

    # Evaluate table on standard test split
    attack_types = ["exact_ac_single_bus", "exact_ac_multi_bus", "limited_meter", "linearised_single_bus", "linearised_multi_bus", "random"]
    std_table_rows = []

    for m_name, (scores, th5, th1, is_rc) in methods.items():
        flags5 = compute_flags(scores, th5, is_rc)
        flags1 = compute_flags(scores, th1, is_rc)

        r_fa5 = np.mean(flags5[norm_mask_test]) * 100.0
        r_fa1 = np.mean(flags1[norm_mask_test]) * 100.0

        auc_all = roc_auc_score(y_test, scores)
        ci_auc_all = group_bootstrap_ci(y_test, scores, test_groups)

        auc_exact = roc_auc_score(y_test[exact_sub_mask], scores[exact_sub_mask])
        ci_auc_exact = group_bootstrap_ci(y_test[exact_sub_mask], scores[exact_sub_mask], test_groups[exact_sub_mask])

        # Overall Attacked
        det5_all = np.mean(flags5[att_mask_test]) * 100.0
        ci_det5_all = group_bootstrap_ci(None, flags5[att_mask_test], test_groups[att_mask_test], is_binary_flag=True)
        det1_all = np.mean(flags1[att_mask_test]) * 100.0
        ci_det1_all = group_bootstrap_ci(None, flags1[att_mask_test], test_groups[att_mask_test], is_binary_flag=True)

        # Exact AC
        det5_exact = np.mean(flags5[exact_mask_test]) * 100.0
        ci_det5_exact = group_bootstrap_ci(None, flags5[exact_mask_test], test_groups[exact_mask_test], is_binary_flag=True)
        det1_exact = np.mean(flags1[exact_mask_test]) * 100.0
        ci_det1_exact = group_bootstrap_ci(None, flags1[exact_mask_test], test_groups[exact_mask_test], is_binary_flag=True)

        # Effective Only
        det5_eff = np.mean(flags5[eff_mask_test]) * 100.0
        ci_det5_eff = group_bootstrap_ci(None, flags5[eff_mask_test], test_groups[eff_mask_test], is_binary_flag=True)

        # Per type
        per_type_det5 = {}
        for at in attack_types:
            mask_at = (test_meta["attack_type"] == at).values
            per_type_det5[at] = np.mean(flags5[mask_at]) * 100.0

        std_table_rows.append({
            "Method": m_name,
            "Realized_FA": (r_fa5, r_fa1),
            "AUC_All": (auc_all, ci_auc_all),
            "AUC_Exact": (auc_exact, ci_auc_exact),
            "Det5_All": (det5_all, ci_det5_all),
            "Det1_All": (det1_all, ci_det1_all),
            "Det5_Exact": (det5_exact, ci_det5_exact),
            "Det1_Exact": (det1_exact, ci_det1_exact),
            "Det5_Eff": (det5_eff, ci_det5_eff),
            "PerType5": per_type_det5,
        })

    # Print Table 1: Standard Test Split
    print(f"\n--- TABLE 1: STANDARD TEST SPLIT COMPARISON (N={len(y_test):,}, VALIDATION THRESHOLDS) ---")
    hdr = f"{'Method':<25} | {'Realized Normal FA':<20} | {'Overall AUC [95% CI]':<25} | {'Exact AC AUC [95% CI]':<25} | {'Det @ 5% FA':<14} | {'Det @ 1% FA':<14} | {'Exact @ 5% FA':<14} | {'Eff @ 5% FA':<14}"
    print(hdr)
    print("-" * len(hdr))
    for r in std_table_rows:
        fa_s = f"{r['Realized_FA'][0]:5.2f}% / {r['Realized_FA'][1]:5.2f}%"
        auc_s = f"{r['AUC_All'][0]:.4f} [{r['AUC_All'][1][0]:.4f},{r['AUC_All'][1][1]:.4f}]"
        auc_e = f"{r['AUC_Exact'][0]:.4f} [{r['AUC_Exact'][1][0]:.4f},{r['AUC_Exact'][1][1]:.4f}]"
        d5_s = f"{r['Det5_All'][0]:5.2f}% [{r['Det5_All'][1][0]:.1f},{r['Det5_All'][1][1]:.1f}]"
        d1_s = f"{r['Det1_All'][0]:5.2f}% [{r['Det1_All'][1][0]:.1f},{r['Det1_All'][1][1]:.1f}]"
        ex5_s = f"{r['Det5_Exact'][0]:5.2f}% [{r['Det5_Exact'][1][0]:.1f},{r['Det5_Exact'][1][1]:.1f}]"
        eff5_s = f"{r['Det5_Eff'][0]:5.2f}% [{r['Det5_Eff'][1][0]:.1f},{r['Det5_Eff'][1][1]:.1f}]"
        print(f"{r['Method']:<25} | {fa_s:<20} | {auc_s:<25} | {auc_e:<25} | {d5_s:<14} | {d1_s:<14} | {ex5_s:<14} | {eff5_s:<14}")

    # Per Type Breakdown for Table 1
    print("\n--- Table 1 Per Attack Type Detection @ 5% FA ---")
    hdr_pt = f"{'Method':<25} | {'exact_single':<12} | {'exact_multi':<12} | {'limited':<10} | {'lin_single':<10} | {'lin_multi':<10} | {'random':<10}"
    print(hdr_pt)
    print("-" * len(hdr_pt))
    for r in std_table_rows:
        pt = r["PerType5"]
        print(f"{r['Method']:<25} | {pt['exact_ac_single_bus']:6.2f}%     | {pt['exact_ac_multi_bus']:6.2f}%     | {pt['limited_meter']:6.2f}%  | {pt['linearised_single_bus']:6.2f}%  | {pt['linearised_multi_bus']:6.2f}%  | {pt['random']:6.2f}%")

    print_classification_table("Table 1 Classification Metrics (Standard Test Split)", methods, y_test)

    # -------------------------------------------------------------------------
    # UNSEEN-ATTACK SPLIT EVALUATION
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("UNSEEN-ATTACK SCHEME EVALUATION (HELD OUT: exact_ac & limited_meter)")
    print("=" * 80)

    unseen_train_mask = (df_splits["split_unseen"] == "train").values
    unseen_val_mask = (df_splits["split_unseen"] == "val").values
    unseen_test_mask = (df_splits["split_unseen"] == "test").values

    y_unseen_train = y_all[unseen_train_mask]
    y_unseen_val = y_all[unseen_val_mask]
    y_unseen_test = y_all[unseen_test_mask]

    unseen_val_meta = df_attacks[unseen_val_mask].reset_index(drop=True)
    unseen_val_norm_mask = (unseen_val_meta["label"] == 0).values

    unseen_test_meta = df_attacks[unseen_test_mask].reset_index(drop=True)
    unseen_test_groups = group_all[unseen_test_mask]
    norm_mask_unseen_test = (unseen_test_meta["label"] == 0).values
    att_mask_unseen_test = (unseen_test_meta["label"] == 1).values
    exact_mask_unseen_test = unseen_test_meta["attack_type"].isin(["exact_ac_single_bus", "exact_ac_multi_bus"]).values
    eff_mask_unseen_test = (unseen_test_meta["label"] == 1) & (unseen_test_meta["effective_attack"] == True).values

    print(f"Unseen split rows: Train={np.sum(unseen_train_mask)}, Val={np.sum(unseen_val_mask)}, Test={np.sum(unseen_test_mask)}")
    print(f"Unseen Test Breakdown: Normal={np.sum(norm_mask_unseen_test)}, Attacked={np.sum(att_mask_unseen_test)}")

    # Train RF on seen attacks only
    print("Training RF on seen attack types only (random + linearised)...")
    X_unseen_train_prim = df_all_feat[cols_primary].values[unseen_train_mask]
    X_unseen_val_prim = df_all_feat[cols_primary].values[unseen_val_mask]
    X_unseen_test_prim = df_all_feat[cols_primary].values[unseen_test_mask]

    rf_unseen = RandomForestClassifier(
        n_estimators=best_rf_params["n_estimators"],
        max_depth=best_rf_params["max_depth"],
        min_samples_leaf=best_rf_params["min_samples_leaf"],
        random_state=seed,
        n_jobs=-1
    )
    rf_unseen.fit(X_unseen_train_prim, y_unseen_train)
    rf_unseen_val_scores = rf_unseen.predict_proba(X_unseen_val_prim)[:, 1]
    rf_unseen_th_5fa = float(np.percentile(rf_unseen_val_scores[unseen_val_norm_mask], 95.0))
    rf_unseen_th_1fa = float(np.percentile(rf_unseen_val_scores[unseen_val_norm_mask], 99.0))
    rf_unseen_scores = rf_unseen.predict_proba(X_unseen_test_prim)[:, 1]

    # Baselines on Unseen Test Split
    chi_unseen_scores = unseen_test_meta["objective_J"].values
    lnr_unseen_scores = unseen_test_meta["max_norm_residual"].values

    unseen_test_v = unseen_test_meta[v_cols].values
    rc_unseen_viol = np.maximum(0.0, np.maximum(v_min_norm - unseen_test_v, unseen_test_v - v_max_norm)).max(axis=1)

    mahal_unseen = np.zeros(len(unseen_test_meta))
    unseen_test_x = unseen_test_meta[x_cols].values
    unseen_test_hours = unseen_test_meta["hour_of_day"].values
    for idx in range(len(unseen_test_meta)):
        h = int(unseen_test_hours[idx])
        lw = hourly_lw[h]
        diff = unseen_test_x[idx:idx+1] - lw.location_
        dist_sq = diff @ lw.precision_ @ diff.T
        mahal_unseen[idx] = np.sqrt(dist_sq[0, 0])

    ae_d12_unseen_scores = best_ae_d12.score_samples(df_all_feat[cols_d12].values[unseen_test_mask])
    ae_prim_unseen_scores = best_ae_prim.score_samples(X_unseen_test_prim)

    unseen_methods = {
        "Chi-Square": (chi_unseen_scores, chi_th_5fa, chi_th_1fa, False),
        "Calibrated LNR": (lnr_unseen_scores, lnr_th_5fa, lnr_th_1fa, False),
        "Range-Check (Envelope)": (rc_unseen_viol, rc_th_5fa, rc_th_1fa, True),
        "Mahalanobis (Ledoit-Wolf)": (mahal_unseen, mahal_th_5fa, mahal_th_1fa, False),
        "Autoencoder (D1+D2)": (ae_d12_unseen_scores, ae_d12_th_5fa, ae_d12_th_1fa, False),
        "Autoencoder (All: A-D2)": (ae_prim_unseen_scores, ae_prim_th_5fa, ae_prim_th_1fa, False),
        "RF (Seen Only -> Held-Out)": (rf_unseen_scores, rf_unseen_th_5fa, rf_unseen_th_1fa, False),
    }

    unseen_table_rows = []
    unseen_exact_sub = exact_mask_unseen_test | norm_mask_unseen_test

    for m_name, (scores, th5, th1, is_rc) in unseen_methods.items():
        flags5 = compute_flags(scores, th5, is_rc)
        flags1 = compute_flags(scores, th1, is_rc)

        r_fa5 = np.mean(flags5[norm_mask_unseen_test]) * 100.0
        r_fa1 = np.mean(flags1[norm_mask_unseen_test]) * 100.0

        auc_all = roc_auc_score(y_unseen_test, scores)
        ci_auc_all = group_bootstrap_ci(y_unseen_test, scores, unseen_test_groups)

        auc_exact = roc_auc_score(y_unseen_test[unseen_exact_sub], scores[unseen_exact_sub])
        ci_auc_exact = group_bootstrap_ci(y_unseen_test[unseen_exact_sub], scores[unseen_exact_sub], unseen_test_groups[unseen_exact_sub])

        det5_all = np.mean(flags5[att_mask_unseen_test]) * 100.0
        ci_det5_all = group_bootstrap_ci(None, flags5[att_mask_unseen_test], unseen_test_groups[att_mask_unseen_test], is_binary_flag=True)
        det1_all = np.mean(flags1[att_mask_unseen_test]) * 100.0
        ci_det1_all = group_bootstrap_ci(None, flags1[att_mask_unseen_test], unseen_test_groups[att_mask_unseen_test], is_binary_flag=True)

        det5_exact = np.mean(flags5[exact_mask_unseen_test]) * 100.0
        ci_det5_exact = group_bootstrap_ci(None, flags5[exact_mask_unseen_test], unseen_test_groups[exact_mask_unseen_test], is_binary_flag=True)

        det5_eff = np.mean(flags5[eff_mask_unseen_test]) * 100.0
        ci_det5_eff = group_bootstrap_ci(None, flags5[eff_mask_unseen_test], unseen_test_groups[eff_mask_unseen_test], is_binary_flag=True)

        per_type_det5 = {}
        for at in attack_types:
            mask_at = (unseen_test_meta["attack_type"] == at).values
            per_type_det5[at] = np.mean(flags5[mask_at]) * 100.0

        unseen_table_rows.append({
            "Method": m_name,
            "Realized_FA": (r_fa5, r_fa1),
            "AUC_All": (auc_all, ci_auc_all),
            "AUC_Exact": (auc_exact, ci_auc_exact),
            "Det5_All": (det5_all, ci_det5_all),
            "Det1_All": (det1_all, ci_det1_all),
            "Det5_Exact": (det5_exact, ci_det5_exact),
            "Det5_Eff": (det5_eff, ci_det5_eff),
            "PerType5": per_type_det5,
        })

    print(f"\n--- TABLE 2: UNSEEN-ATTACK SPLIT COMPARISON (N={len(y_unseen_test):,}, VALIDATION THRESHOLDS) ---")
    print(hdr)
    print("-" * len(hdr))
    for r in unseen_table_rows:
        fa_s = f"{r['Realized_FA'][0]:5.2f}% / {r['Realized_FA'][1]:5.2f}%"
        auc_s = f"{r['AUC_All'][0]:.4f} [{r['AUC_All'][1][0]:.4f},{r['AUC_All'][1][1]:.4f}]"
        auc_e = f"{r['AUC_Exact'][0]:.4f} [{r['AUC_Exact'][1][0]:.4f},{r['AUC_Exact'][1][1]:.4f}]"
        d5_s = f"{r['Det5_All'][0]:5.2f}% [{r['Det5_All'][1][0]:.1f},{r['Det5_All'][1][1]:.1f}]"
        d1_s = f"{r['Det1_All'][0]:5.2f}% [{r['Det1_All'][1][0]:.1f},{r['Det1_All'][1][1]:.1f}]"
        ex5_s = f"{r['Det5_Exact'][0]:5.2f}% [{r['Det5_Exact'][1][0]:.1f},{r['Det5_Exact'][1][1]:.1f}]"
        eff5_s = f"{r['Det5_Eff'][0]:5.2f}% [{r['Det5_Eff'][1][0]:.1f},{r['Det5_Eff'][1][1]:.1f}]"
        print(f"{r['Method']:<25} | {fa_s:<20} | {auc_s:<25} | {auc_e:<25} | {d5_s:<14} | {d1_s:<14} | {ex5_s:<14} | {eff5_s:<14}")

    print("\n--- Table 2 Held-Out and Seen Attack Detection @ 5% FA ---")
    n_at = {at: int(np.sum((unseen_test_meta["attack_type"] == at).values)) for at in attack_types}
    hdr_pt2 = (f"{'Method':<25} | {'HELD exact_sin (N=' + str(n_at['exact_ac_single_bus']) + ')':<22} | {'HELD exact_mul (N=' + str(n_at['exact_ac_multi_bus']) + ')':<22} | "
               f"{'HELD limited (N=' + str(n_at['limited_meter']) + ')':<22} | {'lin_single (N=' + str(n_at['linearised_single_bus']) + ')':<18} | "
               f"{'lin_multi (N=' + str(n_at['linearised_multi_bus']) + ')':<18} | {'random (N=' + str(n_at['random']) + ')':<14}")
    print(hdr_pt2)
    print("-" * len(hdr_pt2))
    for r in unseen_table_rows:
        pt = r["PerType5"]
        print(f"{r['Method']:<25} | {pt['exact_ac_single_bus']:6.2f}%                 | {pt['exact_ac_multi_bus']:6.2f}%                 | {pt['limited_meter']:6.2f}%               | {pt['linearised_single_bus']:6.2f}%            | {pt['linearised_multi_bus']:6.2f}%            | {pt['random']:6.2f}%")


    print_classification_table("Table 2 Classification Metrics (Unseen-Attack Split)", unseen_methods, y_unseen_test)

    # -------------------------------------------------------------------------
    # PART 4: EXTRA EXPERIMENTS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART 4: EXTRA EXPERIMENTS")
    print("=" * 80)

    # 1. Detection vs Shift Sigmas Bins
    print("\n--- (1) Detection vs Shift Sigmas Bins (@ 5% Validation-Derived FA) ---")
    sig_bins = cfg["evaluation"]["sigma_bins"]
    sig_labels = cfg["evaluation"]["sigma_labels"]
    test_meta = test_meta.copy()
    test_meta["sig_bin"] = pd.cut(test_meta["shift_sigmas"], bins=sig_bins, labels=sig_labels)

    sig_det_results = {"RF_Primary": [], "AE_All": [], "Mahalanobis": [], "Max_Z": [], "Counts": []}
    print(f"{'Sigma Bin':<15} | {'Count':<6} | {'RF (Primary)':<14} | {'Autoencoder':<14} | {'Mahalanobis':<14} | {'Per-Hour Max-Z':<14}")
    print("-" * 85)
    for sl in sig_labels:
        sub_mask = (test_meta["sig_bin"] == sl).values & (test_meta["label"] == 1).values
        cnt = np.sum(sub_mask)
        det_rf = np.mean(rf_prim_flags_5[sub_mask]) * 100.0 if cnt > 0 else 0.0
        det_ae = np.mean(compute_flags(ae_prim_test_scores, ae_prim_th_5fa)[sub_mask]) * 100.0 if cnt > 0 else 0.0
        det_m = np.mean(compute_flags(mahal_test, mahal_th_5fa)[sub_mask]) * 100.0 if cnt > 0 else 0.0
        det_z = np.mean(compute_flags(max_abs_z_test, z_th_5fa)[sub_mask]) * 100.0 if cnt > 0 else 0.0

        sig_det_results["Counts"].append(cnt)
        sig_det_results["RF_Primary"].append(det_rf)
        sig_det_results["AE_All"].append(det_ae)
        sig_det_results["Mahalanobis"].append(det_m)
        sig_det_results["Max_Z"].append(det_z)
        print(f"{sl:<15} | {cnt:<6} | {det_rf:6.2f}%       | {det_ae:6.2f}%       | {det_m:6.2f}%       | {det_z:6.2f}%")

    # 2. Feature Group Ablations with IDENTICAL tuned hyperparams: (n_estimators=300, max_depth=12, min_samples_leaf=1)
    print("\n--- (2) Feature Group Ablations (Identical Tuned Hyperparams: 300 trees, depth 12, min leaf 1) ---")
    all_groups = ["a", "b", "c", "d1", "d2"]
    ablation_results = []

    # Alone
    for g in all_groups + ["d3"]:
        sw = {k: (k == g) for k in (all_groups + ["d3"])}
        cols_g = select_feature_columns(groups_dict, sw)
        rf_g = RandomForestClassifier(
            n_estimators=best_rf_params["n_estimators"],
            max_depth=best_rf_params["max_depth"],
            min_samples_leaf=best_rf_params["min_samples_leaf"],
            random_state=seed,
            n_jobs=-1
        )
        rf_g.fit(df_all_feat[cols_g].values[train_mask], y_train)
        p_val_g = rf_g.predict_proba(df_all_feat[cols_g].values[val_mask])[:, 1]
        p_test_g = rf_g.predict_proba(df_all_feat[cols_g].values[test_mask])[:, 1]

        val_auc_g = roc_auc_score(y_val, p_val_g)
        test_auc_g = roc_auc_score(y_test, p_test_g)

        # Threshold from validation normals
        th5_g = float(np.percentile(p_val_g[val_norm_mask], 95.0))
        flags_g = compute_flags(p_test_g, th5_g)
        real_fa_g = np.mean(flags_g[norm_mask_test]) * 100.0
        test_det5_g = np.mean(flags_g[att_mask_test]) * 100.0
        test_exact5_g = np.mean(flags_g[exact_mask_test]) * 100.0

        lbl = f"Group {g.upper()} alone" + (" (clean prev)" if g == "d3" else "")
        ablation_results.append({
            "Config": lbl,
            "N_Feat": len(cols_g),
            "Val_AUC": val_auc_g,
            "Test_AUC": test_auc_g,
            "Real_FA": real_fa_g,
            "Test_Det5": test_det5_g,
            "Test_Exact5": test_exact5_g,
        })

    # Removed (leave-one-group-out from primary A+B+C+D1+D2)
    for g in all_groups:
        sw = {k: (k != g and k in all_groups) for k in (all_groups + ["d3"])}
        cols_rem = select_feature_columns(groups_dict, sw)
        rf_rem = RandomForestClassifier(
            n_estimators=best_rf_params["n_estimators"],
            max_depth=best_rf_params["max_depth"],
            min_samples_leaf=best_rf_params["min_samples_leaf"],
            random_state=seed,
            n_jobs=-1
        )
        rf_rem.fit(df_all_feat[cols_rem].values[train_mask], y_train)
        p_val_rem = rf_rem.predict_proba(df_all_feat[cols_rem].values[val_mask])[:, 1]
        p_test_rem = rf_rem.predict_proba(df_all_feat[cols_rem].values[test_mask])[:, 1]

        val_auc_rem = roc_auc_score(y_val, p_val_rem)
        test_auc_rem = roc_auc_score(y_test, p_test_rem)

        th5_rem = float(np.percentile(p_val_rem[val_norm_mask], 95.0))
        flags_rem = compute_flags(p_test_rem, th5_rem)
        real_fa_rem = np.mean(flags_rem[norm_mask_test]) * 100.0
        test_det5_rem = np.mean(flags_rem[att_mask_test]) * 100.0
        test_exact5_rem = np.mean(flags_rem[exact_mask_test]) * 100.0

        ablation_results.append({
            "Config": f"Without {g.upper()} (All - {g.upper()})",
            "N_Feat": len(cols_rem),
            "Val_AUC": val_auc_rem,
            "Test_AUC": test_auc_rem,
            "Real_FA": real_fa_rem,
            "Test_Det5": test_det5_rem,
            "Test_Exact5": test_exact5_rem,
        })

    print(f"{'Ablation Setting':<32} | {'Feats':<5} | {'Val AUC':<8} | {'Test AUC':<8} | {'Norm FA':<8} | {'Overall Det@5%':<15} | {'Exact AC Det@5%':<15}")
    print("-" * 105)
    for ar in ablation_results:
        print(f"{ar['Config']:<32} | {ar['N_Feat']:<5} | {ar['Val_AUC']:.4f}   | {ar['Test_AUC']:.4f}   | {ar['Real_FA']:5.2f}%   | {ar['Test_Det5']:6.2f}%         | {ar['Test_Exact5']:6.2f}%")

    # 3. Limited-Meter: Detection vs Shift Ratio
    print("\n--- (3) Limited-Meter: Detection vs Shift Ratio (@ 5% FA) ---")
    lm_mask = (test_meta["attack_type"] == "limited_meter").values
    test_meta["ratio_bin"] = pd.cut(test_meta["shift_ratio"], bins=cfg["evaluation"]["shift_ratio_bins"], labels=cfg["evaluation"]["shift_ratio_labels"])

    lm_ratio_results = {"RF_Primary": [], "AE_All": [], "Mahalanobis": [], "Calibrated_LNR": [], "Chi_Square": [], "Counts": []}
    print(f"{'Shift Ratio Bin':<18} | {'Count':<6} | {'RF Primary':<12} | {'Autoencoder':<12} | {'Mahalanobis':<12} | {'Calibrated LNR':<15} | {'Chi-Square':<12}")
    print("-" * 100)
    for rl in cfg["evaluation"]["shift_ratio_labels"]:
        sub_mask = lm_mask & (test_meta["ratio_bin"] == rl).values
        cnt = np.sum(sub_mask)
        det_rf = np.mean(rf_prim_flags_5[sub_mask]) * 100.0 if cnt > 0 else 0.0
        det_ae = np.mean(compute_flags(ae_prim_test_scores, ae_prim_th_5fa)[sub_mask]) * 100.0 if cnt > 0 else 0.0
        det_m = np.mean(compute_flags(mahal_test, mahal_th_5fa)[sub_mask]) * 100.0 if cnt > 0 else 0.0
        det_lnr = np.mean(compute_flags(lnr_test_scores, lnr_th_5fa)[sub_mask]) * 100.0 if cnt > 0 else 0.0
        det_chi = np.mean(compute_flags(chi_test_scores, chi_th_5fa)[sub_mask]) * 100.0 if cnt > 0 else 0.0

        lm_ratio_results["Counts"].append(cnt)
        lm_ratio_results["RF_Primary"].append(det_rf)
        lm_ratio_results["AE_All"].append(det_ae)
        lm_ratio_results["Mahalanobis"].append(det_m)
        lm_ratio_results["Calibrated_LNR"].append(det_lnr)
        lm_ratio_results["Chi_Square"].append(det_chi)
        print(f"{rl:<18} | {cnt:<6} | {det_rf:6.2f}%     | {det_ae:6.2f}%     | {det_m:6.2f}%     | {det_lnr:6.2f}%         | {det_chi:6.2f}%")

    # 4. What do Groups B and C actually encode? (state-proxy diagnostics)
    # An exact AC stealth attack produces measurements that are perfectly consistent
    # with the falsified state, so any feature that still separates it must be
    # reacting to the falsified STATE being unusual, not to a physics violation.
    # (a) R^2 of predicting each feature column from the state estimate alone
    #     (quadratic ridge, fitted on train normals, scored on test normals).
    # (b) RF trained on each sub-block alone (same tuned hyperparameters).
    print("\n--- (4) State-Proxy Diagnostics: What Groups B and C Encode ---")
    from sklearn.linear_model import Ridge
    xhat_cols = [c for c in groups_dict["d1"] if c.startswith("feat_nonres_x_hat_")]
    tr_norm_all = train_mask & (y_all == 0)
    te_norm_all = test_mask & (y_all == 0)
    X_state = df_all_feat[xhat_cols].values
    X_state_q = np.hstack([X_state, X_state ** 2])

    def r2_from_state(cols):
        Y = df_all_feat[cols].values
        keep = Y[tr_norm_all].std(axis=0) > 1e-12
        reg = Ridge(alpha=1e-8).fit(X_state_q[tr_norm_all], Y[tr_norm_all][:, keep])
        Yt = Y[te_norm_all][:, keep]
        P = reg.predict(X_state_q[te_norm_all])
        r2 = 1.0 - ((Yt - P) ** 2).sum(axis=0) / ((Yt - Yt.mean(axis=0)) ** 2).sum(axis=0)
        return float(np.median(r2)), float(np.min(r2))

    b_cols = groups_dict["b"]
    proxy_blocks = [
        ("B: KCL power-balance mismatch", [c for c in b_cols if "_mismatch_bus_" in c]),
        ("B: voltage vs neighbour mean", [c for c in b_cols if "v_neighbor_dev" in c]),
        ("B: measured vs recomputed flow", [c for c in b_cols if "_flow_br_" in c]),
        ("C: measurement leverage", groups_dict["c"]),
    ]
    print(f"{'Feature Block':<32} | {'Feats':<5} | {'R2 med':<7} | {'R2 min':<7} | {'Test AUC':<8} | {'Exact AUC':<9} | {'Exact Det@5%':<12}")
    print("-" * 100)
    for blk_name, blk_cols in proxy_blocks:
        r2_med, r2_min = r2_from_state(blk_cols)
        rf_blk = RandomForestClassifier(
            n_estimators=best_rf_params["n_estimators"],
            max_depth=best_rf_params["max_depth"],
            min_samples_leaf=best_rf_params["min_samples_leaf"],
            random_state=seed,
            n_jobs=-1,
        )
        rf_blk.fit(df_all_feat[blk_cols].values[train_mask], y_train)
        p_val_b = rf_blk.predict_proba(df_all_feat[blk_cols].values[val_mask])[:, 1]
        p_test_b = rf_blk.predict_proba(df_all_feat[blk_cols].values[test_mask])[:, 1]
        th5_b = float(np.percentile(p_val_b[val_norm_mask], 95.0))
        auc_b = roc_auc_score(y_test, p_test_b)
        auc_b_ex = roc_auc_score(y_test[exact_sub_mask], p_test_b[exact_sub_mask])
        det_b_ex = np.mean(compute_flags(p_test_b, th5_b)[exact_mask_test]) * 100.0
        print(f"{blk_name:<32} | {len(blk_cols):<5} | {r2_med:.4f}  | {r2_min:.4f}  | {auc_b:.4f}   | {auc_b_ex:.4f}    | {det_b_ex:6.2f}%")
    print("R2 ~ 1 means the block is (almost) a deterministic function of the state estimate;")
    print("R2 ~ 0 means it is dominated by measurement noise.")

    # -------------------------------------------------------------------------
    # PART 4b: DISTRIBUTION-AWARE ("MIMIC") STEALTH ATTACKS
    # -------------------------------------------------------------------------
    # The SAME trained detectors and the SAME validation thresholds as Table 1 are
    # applied to the test normal rows plus one mimic attack per test operating point
    # (julia/09_mimic_attacks.jl). Each mimic attack is an exact AC stealth attack that
    # moves the state estimate onto a genuine normal state from another day at the
    # same hour, so it is residual-consistent AND typical for that hour.
    print("\n" + "=" * 80)
    print("PART 4b: DISTRIBUTION-AWARE (MIMIC) EXACT AC STEALTH ATTACKS")
    print("=" * 80)

    def score_all_methods(meta, feats):
        """Scores arbitrary rows with every Table 1 detector (same models, same rules)."""
        V = meta[v_cols].values
        rc = np.maximum(0.0, np.maximum(v_min_norm - V, V - v_max_norm)).max(axis=1)
        X = meta[x_cols].values.astype(float)
        H = meta["hour_of_day"].values.astype(int)
        mz = np.zeros(len(meta)); mh = np.zeros(len(meta))
        for i in range(len(meta)):
            mu, sd = hourly_stats[H[i]]
            mz[i] = np.max(np.abs((X[i] - mu) / sd))
            lw = hourly_lw[H[i]]
            d = X[i:i + 1] - lw.location_
            mh[i] = np.sqrt((d @ lw.precision_ @ d.T)[0, 0])
        return {
            "Chi-Square": meta["objective_J"].values,
            "Calibrated LNR": meta["max_norm_residual"].values,
            "Range-Check (Envelope)": rc,
            "Per-Hour Max-Z": mz,
            "Mahalanobis (Ledoit-Wolf)": mh,
            "Autoencoder (D1+D2)": best_ae_d12.score_samples(feats[cols_d12].values),
            "Autoencoder (All: A-D2)": best_ae_prim.score_samples(feats[cols_primary].values),
            "RF w/o D3 (Primary)": best_rf_prim.predict_proba(feats[cols_primary].values)[:, 1],
            "RF with D3 (Clean Prev)": rf_d3.predict_proba(feats[cols_all_with_d3].values)[:, 1],
        }

    # Self-check: re-scoring the standard test rows must reproduce the Table 1 scores exactly.
    test_rescored = score_all_methods(test_meta, df_all_feat[test_mask].reset_index(drop=True))
    # (tolerance 1e-6 relative: the autoencoders compute in float32)
    max_rel = 0.0
    for m_name, (scores, _, _, _) in methods.items():
        rel = np.max(np.abs(test_rescored[m_name] - scores) / np.maximum(np.abs(scores), 1e-12))
        max_rel = max(max_rel, float(rel))
        assert rel < 1e-6, f"re-scoring mismatch for {m_name}: max relative difference {rel:.2e}"
    print(f"Self-check: re-scoring the standard test rows reproduces every Table 1 score (max relative difference {max_rel:.1e}): PASSED")

    exact5_table1 = {r["Method"]: r["Det5_Exact"][0] for r in std_table_rows}
    mimic_results = {}
    mimic_cfg_path = cfg["general"].get("mimic_config")
    mimic_sets = []
    if mimic_cfg_path and os.path.exists(mimic_cfg_path):
        with open(mimic_cfg_path, "rb") as f_m:
            mimic_sets = tomllib.load(f_m)["sets"]
    n_norm_test = int(np.sum(norm_mask_test))
    for mset in mimic_sets:
        if not os.path.exists(mset["features_path"]):
            print(f"Mimic set '{mset['name']}': features file missing ({mset['features_path']}); skipped.")
            continue
        mim_meta = pd.read_csv(mset["attacks_path"])
        mim_feat = pd.read_csv(mset["features_path"])
        assert (mim_meta["row_id"].values == mim_feat["row_id"].values).all()
        mim_scores = score_all_methods(mim_meta, mim_feat)
        y_comb = np.r_[np.zeros(n_norm_test, dtype=int), np.ones(len(mim_meta), dtype=int)]
        g_comb = np.r_[test_groups[norm_mask_test], mim_meta["group_id"].values]

        print(f"\nMimic set '{mset['name']}' (donors from {mset['donor_split']} normals): N attacks = {len(mim_meta)}, "
              f"SE converged = {int(mim_meta['converged_se'].sum())}, "
              f"median |J_att - J_twin| = {np.median(np.abs(mim_meta['delta_J_vs_twin'])):.2e}, "
              f"median max voltage shift = {np.median(mim_meta['shift_magnitude_pu']):.4f} pu "
              f"(project exact AC attacks in test split: {np.median(test_meta.loc[exact_mask_test, 'shift_magnitude_pu']):.4f} pu)")
        print(f"\n--- (5) MIMIC ATTACK BENCHMARK: {mset['name']} (N={len(y_comb):,}, VALIDATION THRESHOLDS) ---")
        hdr_m = (f"{'Method':<25} | {'Realized Normal FA':<20} | {'Mimic AUC [95% CI]':<25} | {'Det @ 5% FA':<14} | {'Det @ 1% FA':<14} | "
                 f"{'Exact AC Det @ 5% (Table 1)':<27} | {'Prec @5%':<8} | {'F1 @5%':<8} | {'Acc @5%':<8}")
        print(hdr_m)
        print("-" * len(hdr_m))
        set_res = {}
        for m_name, (scores, th5, th1, is_rc) in methods.items():
            s_comb = np.r_[scores[norm_mask_test], mim_scores[m_name]]
            f5 = compute_flags(s_comb, th5, is_rc)
            f1_ = compute_flags(s_comb, th1, is_rc)
            pos = y_comb == 1
            auc_m = roc_auc_score(y_comb, s_comb)
            ci_auc_m = group_bootstrap_ci(y_comb, s_comb, g_comb)
            det5_m = np.mean(f5[pos]) * 100.0
            ci_det5_m = group_bootstrap_ci(None, f5[pos], g_comb[pos], is_binary_flag=True)
            det1_m = np.mean(f1_[pos]) * 100.0
            ci_det1_m = group_bootstrap_ci(None, f1_[pos], g_comb[pos], is_binary_flag=True)
            met = classification_metrics(f5, y_comb)
            fa_s = f"{np.mean(f5[~pos]) * 100:5.2f}% / {np.mean(f1_[~pos]) * 100:5.2f}%"
            auc_s = f"{auc_m:.4f} [{ci_auc_m[0]:.4f},{ci_auc_m[1]:.4f}]"
            d5_s = f"{det5_m:5.2f}% [{ci_det5_m[0]:.1f},{ci_det5_m[1]:.1f}]"
            d1_s = f"{det1_m:5.2f}% [{ci_det1_m[0]:.1f},{ci_det1_m[1]:.1f}]"
            print(f"{m_name:<25} | {fa_s:<20} | {auc_s:<25} | {d5_s:<14} | {d1_s:<14} | {exact5_table1[m_name]:6.2f}%                     | "
                  f"{met['prec']:6.2f}%  | {met['f1']:6.2f}%  | {met['acc']:6.2f}%")
            set_res[m_name] = {"det5": det5_m, "flags5": f5[pos], "auc": auc_m}
        mimic_results[mset["name"]] = {"meta": mim_meta, "res": set_res}

    # Detection vs the size of the mimic shift (primary set = first set listed)
    if mimic_results:
        prim_name = next(iter(mimic_results))
        pm = mimic_results[prim_name]
        shift_bins = cfg["evaluation"]["mimic_shift_bins_pu"]
        shift_labels = cfg["evaluation"]["mimic_shift_labels"]
        sbin = pd.cut(pm["meta"]["shift_magnitude_pu"], bins=shift_bins, labels=shift_labels)
        print(f"\n--- (6) Mimic Attacks ({prim_name}): Detection vs Max Voltage Shift (@ 5% FA) ---")
        print(f"{'Voltage Shift Bin':<18} | {'Count':<6} | {'RF Primary':<12} | {'Autoencoder':<12} | {'Mahalanobis':<12} | {'Calibrated LNR':<15} | {'Chi-Square':<12}")
        print("-" * 100)
        for sl in shift_labels:
            mk = (sbin == sl).values
            cnt = int(mk.sum())
            vals = [np.mean(pm["res"][k]["flags5"][mk]) * 100.0 if cnt > 0 else 0.0 for k in
                    ["RF w/o D3 (Primary)", "Autoencoder (All: A-D2)", "Mahalanobis (Ledoit-Wolf)", "Calibrated LNR", "Chi-Square"]]
            print(f"{sl:<18} | {cnt:<6} | {vals[0]:6.2f}%     | {vals[1]:6.2f}%     | {vals[2]:6.2f}%     | {vals[3]:6.2f}%         | {vals[4]:6.2f}%")

        # Figure: exact AC (random c) vs mimic detection at 5% FA, per detector
        res_dir_m = cfg["general"]["results_dir"]
        os.makedirs(res_dir_m, exist_ok=True)
        fig_methods = ["Chi-Square", "Calibrated LNR", "Per-Hour Max-Z", "Mahalanobis (Ledoit-Wolf)",
                       "Autoencoder (D1+D2)", "Autoencoder (All: A-D2)", "RF w/o D3 (Primary)"]
        xi = np.arange(len(fig_methods))
        n_bars = 1 + len(mimic_results)
        width = 0.8 / n_bars
        plt.figure(figsize=(10, 5))
        plt.bar(xi - 0.4 + width / 2, [exact5_table1[k] for k in fig_methods], width,
                label="Exact AC, random state shift (Table 1)", color="steelblue", edgecolor="black")
        colors = ["crimson", "salmon", "orange"]
        for j, (nm, mr) in enumerate(mimic_results.items()):
            plt.bar(xi - 0.4 + width * (j + 1.5), [mr["res"][k]["det5"] for k in fig_methods], width,
                    label=f"Mimic exact AC ({nm.replace('_', ' ')})", color=colors[j % 3], edgecolor="black")
        plt.axhline(5.0, color="gray", ls=":", label="5% false-alarm level (chance)")
        plt.xticks(xi, [k.replace(" (Ledoit-Wolf)", "").replace("Autoencoder", "AE").replace(" (Primary)", "") for k in fig_methods],
                   rotation=20, ha="right")
        plt.ylabel("Detection rate at 5% validation false-alarm threshold (%)")
        plt.title("Stealth-attack detection: random state shift vs distribution-aware (mimic) shift")
        plt.ylim(0, 100)
        plt.legend(loc="upper left", fontsize=8)
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{res_dir_m}/mimic_vs_exact.png", dpi=300)
        plt.close()
        print(f"  Saved: {res_dir_m}/mimic_vs_exact.png")

    # -------------------------------------------------------------------------
    # PART 5: HONESTY RULES & FIGURES GENERATION
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART 5: PERMUTATION NULL CHECKS & PUBLICATION FIGURES")
    print("=" * 80)

    # Permutation Null Check (20 seeds) on final RF and AE pipelines
    print("Running Permutation Null Check (test labels permuted, 20 seeds)...")
    rf_null_aucs = []
    ae_null_aucs = []
    rng_null = np.random.default_rng(seed)
    for s_idx in range(cfg["evaluation"]["null_check_seeds"]):
        perm_y = rng_null.permutation(y_test)
        rf_null_aucs.append(roc_auc_score(perm_y, rf_prim_test_scores))
        ae_null_aucs.append(roc_auc_score(perm_y, ae_prim_test_scores))

    print(f"RF Primary Permutation Null Check: mean = {np.mean(rf_null_aucs):.4f}, std = {np.std(rf_null_aucs):.4f}, min = {np.min(rf_null_aucs):.4f}, max = {np.max(rf_null_aucs):.4f}")
    print(f"AE All Permutation Null Check:     mean = {np.mean(ae_null_aucs):.4f}, std = {np.std(ae_null_aucs):.4f}, min = {np.min(ae_null_aucs):.4f}, max = {np.max(ae_null_aucs):.4f}")

    # Generate Figures
    print("\nGenerating Figures in results/ ...")
    res_dir = cfg["general"]["results_dir"]

    # 1. ROC Curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for m_name in ["RF w/o D3 (Primary)", "Autoencoder (All: A-D2)", "Mahalanobis (Ledoit-Wolf)", "Per-Hour Max-Z", "Calibrated LNR", "Chi-Square", "Range-Check (Envelope)"]:
        sc, _, _, _ = methods[m_name]
        fpr, tpr, _ = roc_curve(y_test, sc)
        auc_val = roc_auc_score(y_test, sc)
        plt.plot(fpr, tpr, label=f"{m_name} (AUC={auc_val:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel("False Positive Rate (Normal Rows)")
    plt.ylabel("True Positive Rate (All Attacks)")
    plt.title(f"ROC Curves: All Attacks (N={len(y_test):,})")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    for m_name in ["RF w/o D3 (Primary)", "Autoencoder (All: A-D2)", "Mahalanobis (Ledoit-Wolf)", "Per-Hour Max-Z", "Calibrated LNR", "Chi-Square", "Range-Check (Envelope)"]:
        sc, _, _, _ = methods[m_name]
        fpr, tpr, _ = roc_curve(y_test[exact_sub_mask], sc[exact_sub_mask])
        auc_val = roc_auc_score(y_test[exact_sub_mask], sc[exact_sub_mask])
        plt.plot(fpr, tpr, label=f"{m_name} (AUC={auc_val:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel("False Positive Rate (Normal Rows)")
    plt.ylabel("True Positive Rate (Exact AC Attacks)")
    plt.title(f"ROC Curves: Exact AC Attacks Only (N={int(np.sum(exact_sub_mask)):,})")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{res_dir}/roc_curves.png", dpi=300)
    plt.close()
    print(f"  Saved: {res_dir}/roc_curves.png")

    # 2. Detection vs Sigma Bins
    plt.figure(figsize=(8, 5))
    x_indices = np.arange(len(sig_labels))
    plt.plot(x_indices, sig_det_results["RF_Primary"], marker="o", label="RF Primary (A+B+C+D1+D2)", color="royalblue", lw=2)
    plt.plot(x_indices, sig_det_results["AE_All"], marker="s", label="Autoencoder (All: A-D2)", color="darkorange", lw=2)
    plt.plot(x_indices, sig_det_results["Mahalanobis"], marker="^", label="Mahalanobis (Ledoit-Wolf)", color="forestgreen", lw=2)
    plt.plot(x_indices, sig_det_results["Max_Z"], marker="d", label="Per-Hour Max-Z", color="purple", lw=1.5, ls="--")
    plt.axhline(5.0, color="gray", linestyle=":", label="5% False Alarm Floor")
    plt.xticks(x_indices, sig_labels)
    plt.xlabel("Shift Magnitude Bin (Normal Standard Deviations)")
    plt.ylabel("Detection Rate at 5% False Alarm Point (%)")
    plt.title("Detection Rate vs Attack Size (shift_sigmas)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{res_dir}/detection_vs_sigma.png", dpi=300)
    plt.close()
    print(f"  Saved: {res_dir}/detection_vs_sigma.png")

    # 3. Score Distributions
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.hist(rf_prim_test_scores[norm_mask_test], bins=30, density=True, alpha=0.5, label="Normal", color="gray")
    plt.hist(rf_prim_test_scores[exact_mask_test], bins=30, density=True, alpha=0.6, label="Exact AC", color="crimson")
    plt.hist(rf_prim_test_scores[att_mask_test & ~exact_mask_test], bins=30, density=True, alpha=0.5, label="Other Attacks", color="royalblue")
    plt.axvline(rf_prim_th_5fa, color="black", linestyle="--", label=f"Val 5% FA ({rf_prim_th_5fa:.4f})")
    plt.xlabel("RF Probability P(Attack)")
    plt.ylabel("Density")
    plt.title("RF Score Distribution")
    plt.legend(fontsize=7)

    plt.subplot(1, 3, 2)
    min_ae = max(1e-3, float(np.min(ae_prim_test_scores)))
    max_ae = float(np.max(ae_prim_test_scores))
    bins_ae = np.logspace(np.log10(min_ae), np.log10(max_ae), 35)
    plt.hist(ae_prim_test_scores[norm_mask_test], bins=bins_ae, density=True, alpha=0.5, label="Normal", color="gray")
    plt.hist(ae_prim_test_scores[exact_mask_test], bins=bins_ae, density=True, alpha=0.6, label="Exact AC", color="crimson")
    plt.hist(ae_prim_test_scores[att_mask_test & ~exact_mask_test], bins=bins_ae, density=True, alpha=0.5, label="Other Attacks", color="royalblue")
    plt.axvline(ae_prim_th_5fa, color="black", linestyle="--", label=f"Val 5% FA ({ae_prim_th_5fa:.6f})")
    plt.xscale("log")
    plt.xlabel("Autoencoder Reconstruction MSE (log scale)")
    plt.ylabel("Density")
    plt.title("Autoencoder Score Distribution")
    plt.legend(fontsize=7)

    plt.subplot(1, 3, 3)
    min_m = max(1e-2, float(np.min(mahal_test)))
    max_m = float(np.max(mahal_test))
    bins_m = np.logspace(np.log10(min_m), np.log10(max_m), 35)
    plt.hist(mahal_test[norm_mask_test], bins=bins_m, density=True, alpha=0.5, label="Normal", color="gray")
    plt.hist(mahal_test[exact_mask_test], bins=bins_m, density=True, alpha=0.6, label="Exact AC", color="crimson")
    plt.hist(mahal_test[att_mask_test & ~exact_mask_test], bins=bins_m, density=True, alpha=0.5, label="Other Attacks", color="royalblue")
    plt.axvline(mahal_th_5fa, color="black", linestyle="--", label=f"Val 5% FA ({mahal_th_5fa:.6f})")
    plt.xscale("log")
    plt.xlabel("Mahalanobis Distance D_M (log scale)")
    plt.ylabel("Density")
    plt.title("Mahalanobis Score Distribution")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{res_dir}/score_distributions.png", dpi=300)
    plt.close()
    print(f"  Saved: {res_dir}/score_distributions.png")

    # 4. Grouped Importance Bar Chart
    plt.figure(figsize=(7, 4.5))
    g_labels = ["A (Residuals)", "B (Consistency)", "C (Sensitivity)", "D1 (Raw States)", "D2 (Forecast Dev)"]
    g_drops = [group_importances[k]["mean_drop"] for k in ["a", "b", "c", "d1", "d2"]]
    g_errs = [group_importances[k]["std_drop"] for k in ["a", "b", "c", "d1", "d2"]]
    plt.bar(g_labels, g_drops, yerr=g_errs, capsize=4, color="steelblue", edgecolor="black", alpha=0.85)
    plt.ylabel("Drop in validation ROC-AUC")
    plt.xlabel("Feature Group")
    plt.title("Grouped Permutation Importance on Validation Split")
    plt.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(f"{res_dir}/importance_bar_chart.png", dpi=300)
    plt.close()
    print(f"  Saved: {res_dir}/importance_bar_chart.png")

    # 5. Limited-Meter vs Shift Ratio
    plt.figure(figsize=(7, 4.5))
    r_indices = np.arange(len(cfg["evaluation"]["shift_ratio_labels"]))
    plt.plot(r_indices, lm_ratio_results["RF_Primary"], marker="o", label="RF Primary", color="royalblue", lw=2)
    plt.plot(r_indices, lm_ratio_results["AE_All"], marker="s", label="Autoencoder (All)", color="darkorange", lw=2)
    plt.plot(r_indices, lm_ratio_results["Mahalanobis"], marker="^", label="Mahalanobis", color="forestgreen", lw=2)
    plt.plot(r_indices, lm_ratio_results["Calibrated_LNR"], marker="d", label="Calibrated LNR", color="firebrick", lw=1.5, ls="--")
    plt.xticks(r_indices, cfg["evaluation"]["shift_ratio_labels"])
    plt.xlabel("Shift Ratio (Achieved / Targeted)")
    plt.ylabel("Detection Rate @ 5% FA (%)")
    plt.title("Limited-Meter Attacks: Detection vs Shift Ratio")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{res_dir}/limited_meter_shift_ratio.png", dpi=300)
    plt.close()
    print(f"  Saved: {res_dir}/limited_meter_shift_ratio.png")

    print("\nTask 8b execution finished successfully.")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
