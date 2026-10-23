"""
Task 8 Full Experiment Runner:
  Part 1: Supervised Random Forest (tuning on validation split, test evaluation, grouped permutation importance)
  Part 2: Unsupervised PyTorch Autoencoders (D1+D2 and All, validation bottleneck selection, validation-derived threshold)
  Part 3: Comprehensive Comparison Tables (Standard Test Split and Unseen-Attack Split)
  Part 4: Extra Experiments (Detection vs Sigma Bins, Group Ablation, Limited-Meter vs Shift Ratio)
  Part 5: Honesty Rules (Permutation Null Checks, Visualization Generation to results/)
  Part 6: Report Generation
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


def main():
    print("=" * 80)
    print("TASK 8: MODEL BENCHMARKING, TUNING, AND COMPARATIVE EVALUATION (DATASET V2)")
    print("=" * 80)

    cfg = load_config()
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

    # Predict test
    rf_prim_test_scores = best_rf_prim.predict_proba(X_test_prim)[:, 1]
    rf_prim_th_5fa = float(np.percentile(rf_prim_test_scores[norm_mask_test], 95.0))
    rf_prim_th_1fa = float(np.percentile(rf_prim_test_scores[norm_mask_test], 99.0))

    auc_prim_all = roc_auc_score(y_test, rf_prim_test_scores)
    ci_auc_prim_all = group_bootstrap_ci(y_test, rf_prim_test_scores, test_groups)

    exact_sub_mask = exact_mask_test | norm_mask_test
    auc_prim_exact = roc_auc_score(y_test[exact_sub_mask], rf_prim_test_scores[exact_sub_mask])
    ci_auc_prim_exact = group_bootstrap_ci(y_test[exact_sub_mask], rf_prim_test_scores[exact_sub_mask], test_groups[exact_sub_mask])

    det_prim_5fa_all = np.mean(rf_prim_test_scores[att_mask_test] >= rf_prim_th_5fa) * 100.0
    ci_det_prim_5fa_all = group_bootstrap_ci(None, rf_prim_test_scores[att_mask_test] >= rf_prim_th_5fa, test_groups[att_mask_test], is_binary_flag=True)

    det_prim_1fa_all = np.mean(rf_prim_test_scores[att_mask_test] >= rf_prim_th_1fa) * 100.0
    ci_det_prim_1fa_all = group_bootstrap_ci(None, rf_prim_test_scores[att_mask_test] >= rf_prim_th_1fa, test_groups[att_mask_test], is_binary_flag=True)

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
    X_test_d3 = df_all_feat[cols_all_with_d3].values[test_mask]
    rf_d3 = RandomForestClassifier(
        n_estimators=best_rf_params["n_estimators"],
        max_depth=best_rf_params["max_depth"],
        min_samples_leaf=best_rf_params["min_samples_leaf"],
        random_state=seed,
        n_jobs=-1
    )
    rf_d3.fit(X_train_d3, y_train)
    rf_d3_test_scores = rf_d3.predict_proba(X_test_d3)[:, 1]
    rf_d3_th_5fa = float(np.percentile(rf_d3_test_scores[norm_mask_test], 95.0))
    rf_d3_th_1fa = float(np.percentile(rf_d3_test_scores[norm_mask_test], 99.0))

    # -------------------------------------------------------------------------
    # PART 2: Autoencoder (Unsupervised, Normal Rows Only)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART 2: AUTOENCODER (UNSUPERVISED) NORMAL-ONLY TRAINING")
    print("=" * 80)

    train_norm_mask = (train_meta["label"] == 0).values
    val_norm_mask = (val_meta["label"] == 0).values

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

    # Thresholds derived from validation NORMAL reconstruction errors
    val_norm_err_d12 = best_ae_d12.score_samples(X_val_d12_norm)
    ae_d12_th_5fa = float(np.percentile(val_norm_err_d12, 95.0))
    ae_d12_th_1fa = float(np.percentile(val_norm_err_d12, 99.0))
    print(f"Autoencoder D1+D2 Thresholds (Validation Normal): 5% FA = {ae_d12_th_5fa:.6f}, 1% FA = {ae_d12_th_1fa:.6f}")

    ae_d12_test_scores = best_ae_d12.score_samples(X_test_d12_all)
    emp_norm_fa_5_d12 = np.mean(ae_d12_test_scores[norm_mask_test] >= ae_d12_th_5fa) * 100.0
    emp_norm_fa_1_d12 = np.mean(ae_d12_test_scores[norm_mask_test] >= ae_d12_th_1fa) * 100.0
    print(f"Empirical Test Normal False Alarm: @5% th = {emp_norm_fa_5_d12:.2f}%, @1% th = {emp_norm_fa_1_d12:.2f}%")

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
    emp_norm_fa_5_prim = np.mean(ae_prim_test_scores[norm_mask_test] >= ae_prim_th_5fa) * 100.0
    emp_norm_fa_1_prim = np.mean(ae_prim_test_scores[norm_mask_test] >= ae_prim_th_1fa) * 100.0
    print(f"Empirical Test Normal False Alarm: @5% th = {emp_norm_fa_5_prim:.2f}%, @1% th = {emp_norm_fa_1_prim:.2f}%")

    # -------------------------------------------------------------------------
    # PART 3: BASELINE METHODS & COMPARISON TABLES
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART 3: BASELINES & COMPREHENSIVE COMPARISON TABLES")
    print("=" * 80)

    # 1. Chi-Square (objective_J)
    chi_test_scores = test_meta["objective_J"].values
    chi_th_5fa = float(np.percentile(chi_test_scores[norm_mask_test], 95.0))
    chi_th_1fa = float(np.percentile(chi_test_scores[norm_mask_test], 99.0))

    # 2. Calibrated LNR (max_norm_residual)
    lnr_test_scores = test_meta["max_norm_residual"].values
    lnr_th_5fa = float(np.percentile(lnr_test_scores[norm_mask_test], 95.0))
    lnr_th_1fa = float(np.percentile(lnr_test_scores[norm_mask_test], 99.0))

    # 3. Range-Check (Operating Envelope)
    train_norm = df_attacks[(df_splits["split"] == "train") & (df_attacks["label"] == 0)]
    v_cols = [f"x_hat_{i}" for i in range(33, 66)]
    v_min_norm = float(train_norm[v_cols].min().min())
    v_max_norm = float(train_norm[v_cols].max().max())
    test_v = test_meta[v_cols].values
    rc_viol = np.maximum(0.0, np.maximum(v_min_norm - test_v, test_v - v_max_norm)).max(axis=1)
    rc_th_5fa = float(np.percentile(rc_viol[norm_mask_test], 95.0))
    rc_th_1fa = float(np.percentile(rc_viol[norm_mask_test], 99.0))

    # 4. Per-Hour Max-Z on x_hat
    x_cols = [f"x_hat_{i}" for i in range(1, 66)]
    hourly_stats = {}
    for h in range(24):
        sub = train_norm[train_norm["hour_of_day"] == h][x_cols]
        mu = sub.mean(axis=0).values
        std = sub.std(axis=0).values.copy()
        std[std < 1e-9] = 1.0
        hourly_stats[h] = (mu, std)

    max_abs_z = np.zeros(len(test_meta))
    for idx, row in test_meta.iterrows():
        h = int(row["hour_of_day"])
        mu, std = hourly_stats[h]
        x_val = row[x_cols].values.astype(float)
        max_abs_z[idx] = np.max(np.abs((x_val - mu) / std))
    z_th_5fa = float(np.percentile(max_abs_z[norm_mask_test], 95.0))
    z_th_1fa = float(np.percentile(max_abs_z[norm_mask_test], 99.0))

    # 5. Mahalanobis (Ledoit-Wolf)
    hourly_lw = {}
    for h in range(24):
        sub = train_norm[train_norm["hour_of_day"] == h][x_cols].values
        hourly_lw[h] = LedoitWolf().fit(sub)

    mahal_dist = np.zeros(len(test_meta))
    test_x = test_meta[x_cols].values
    test_hours = test_meta["hour_of_day"].values
    for idx in range(len(test_meta)):
        h = int(test_hours[idx])
        lw = hourly_lw[h]
        diff = test_x[idx:idx+1] - lw.location_
        dist_sq = diff @ lw.precision_ @ diff.T
        mahal_dist[idx] = np.sqrt(dist_sq[0, 0])
    mahal_th_5fa = float(np.percentile(mahal_dist[norm_mask_test], 95.0))
    mahal_th_1fa = float(np.percentile(mahal_dist[norm_mask_test], 99.0))

    # Collect methods for standard comparison
    methods = {
        "Chi-Square": (chi_test_scores, chi_th_5fa, chi_th_1fa),
        "Calibrated LNR": (lnr_test_scores, lnr_th_5fa, lnr_th_1fa),
        "Range-Check (Envelope)": (rc_viol, rc_th_5fa, rc_th_1fa),
        "Per-Hour Max-Z": (max_abs_z, z_th_5fa, z_th_1fa),
        "Mahalanobis (Ledoit-Wolf)": (mahal_dist, mahal_th_5fa, mahal_th_1fa),
        "Autoencoder (D1+D2)": (ae_d12_test_scores, ae_d12_th_5fa, ae_d12_th_1fa),
        "Autoencoder (All: A-D2)": (ae_prim_test_scores, ae_prim_th_5fa, ae_prim_th_1fa),
        "RF w/o D3 (Primary)": (rf_prim_test_scores, rf_prim_th_5fa, rf_prim_th_1fa),
        "RF with D3 (Clean Prev)": (rf_d3_test_scores, rf_d3_th_5fa, rf_d3_th_1fa),
    }

    # Evaluate table on standard test split
    attack_types = ["exact_ac_single_bus", "exact_ac_multi_bus", "limited_meter", "linearised_single_bus", "linearised_multi_bus", "random"]
    std_table_rows = []

    for m_name, (scores, th5, th1) in methods.items():
        flags5 = scores >= th5
        flags1 = scores >= th1

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
    print("\n--- TABLE 1: STANDARD TEST SPLIT COMPARISON (N=2,000) ---")
    hdr = f"{'Method':<25} | {'Overall AUC [95% CI]':<25} | {'Exact AC AUC [95% CI]':<25} | {'Det @ 5% FA':<14} | {'Det @ 1% FA':<14} | {'Exact @ 5% FA':<14} | {'Eff @ 5% FA':<14}"
    print(hdr)
    print("-" * len(hdr))
    for r in std_table_rows:
        auc_s = f"{r['AUC_All'][0]:.4f} [{r['AUC_All'][1][0]:.4f},{r['AUC_All'][1][1]:.4f}]"
        auc_e = f"{r['AUC_Exact'][0]:.4f} [{r['AUC_Exact'][1][0]:.4f},{r['AUC_Exact'][1][1]:.4f}]"
        d5_s = f"{r['Det5_All'][0]:5.2f}% [{r['Det5_All'][1][0]:.1f},{r['Det5_All'][1][1]:.1f}]"
        d1_s = f"{r['Det1_All'][0]:5.2f}% [{r['Det1_All'][1][0]:.1f},{r['Det1_All'][1][1]:.1f}]"
        ex5_s = f"{r['Det5_Exact'][0]:5.2f}% [{r['Det5_Exact'][1][0]:.1f},{r['Det5_Exact'][1][1]:.1f}]"
        eff5_s = f"{r['Det5_Eff'][0]:5.2f}% [{r['Det5_Eff'][1][0]:.1f},{r['Det5_Eff'][1][1]:.1f}]"
        print(f"{r['Method']:<25} | {auc_s:<25} | {auc_e:<25} | {d5_s:<14} | {d1_s:<14} | {ex5_s:<14} | {eff5_s:<14}")

    # Per Type Breakdown for Table 1
    print("\n--- Table 1 Per Attack Type Detection @ 5% FA ---")
    hdr_pt = f"{'Method':<25} | {'exact_single':<12} | {'exact_multi':<12} | {'limited':<10} | {'lin_single':<10} | {'lin_multi':<10} | {'random':<10}"
    print(hdr_pt)
    print("-" * len(hdr_pt))
    for r in std_table_rows:
        pt = r["PerType5"]
        print(f"{r['Method']:<25} | {pt['exact_ac_single_bus']:6.2f}%     | {pt['exact_ac_multi_bus']:6.2f}%     | {pt['limited_meter']:6.2f}%  | {pt['linearised_single_bus']:6.2f}%  | {pt['linearised_multi_bus']:6.2f}%  | {pt['random']:6.2f}%")

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
    X_unseen_test_prim = df_all_feat[cols_primary].values[unseen_test_mask]

    rf_unseen = RandomForestClassifier(
        n_estimators=best_rf_params["n_estimators"],
        max_depth=best_rf_params["max_depth"],
        min_samples_leaf=best_rf_params["min_samples_leaf"],
        random_state=seed,
        n_jobs=-1
    )
    rf_unseen.fit(X_unseen_train_prim, y_unseen_train)
    rf_unseen_scores = rf_unseen.predict_proba(X_unseen_test_prim)[:, 1]
    rf_unseen_th_5fa = float(np.percentile(rf_unseen_scores[norm_mask_unseen_test], 95.0))
    rf_unseen_th_1fa = float(np.percentile(rf_unseen_scores[norm_mask_unseen_test], 99.0))

    # Baselines on Unseen Test Split
    chi_unseen_scores = unseen_test_meta["objective_J"].values
    chi_unseen_th5 = float(np.percentile(chi_unseen_scores[norm_mask_unseen_test], 95.0))
    chi_unseen_th1 = float(np.percentile(chi_unseen_scores[norm_mask_unseen_test], 99.0))

    lnr_unseen_scores = unseen_test_meta["max_norm_residual"].values
    lnr_unseen_th5 = float(np.percentile(lnr_unseen_scores[norm_mask_unseen_test], 95.0))
    lnr_unseen_th1 = float(np.percentile(lnr_unseen_scores[norm_mask_unseen_test], 99.0))

    unseen_test_v = unseen_test_meta[v_cols].values
    rc_unseen_viol = np.maximum(0.0, np.maximum(v_min_norm - unseen_test_v, unseen_test_v - v_max_norm)).max(axis=1)
    rc_unseen_th5 = float(np.percentile(rc_unseen_viol[norm_mask_unseen_test], 95.0))
    rc_unseen_th1 = float(np.percentile(rc_unseen_viol[norm_mask_unseen_test], 99.0))

    # Mahalanobis on Unseen Test Split
    mahal_unseen = np.zeros(len(unseen_test_meta))
    unseen_test_x = unseen_test_meta[x_cols].values
    unseen_test_hours = unseen_test_meta["hour_of_day"].values
    for idx in range(len(unseen_test_meta)):
        h = int(unseen_test_hours[idx])
        lw = hourly_lw[h]
        diff = unseen_test_x[idx:idx+1] - lw.location_
        dist_sq = diff @ lw.precision_ @ diff.T
        mahal_unseen[idx] = np.sqrt(dist_sq[0, 0])
    mahal_unseen_th5 = float(np.percentile(mahal_unseen[norm_mask_unseen_test], 95.0))
    mahal_unseen_th1 = float(np.percentile(mahal_unseen[norm_mask_unseen_test], 99.0))

    # Autoencoders on Unseen Test Split
    ae_d12_unseen_scores = best_ae_d12.score_samples(df_all_feat[cols_d12].values[unseen_test_mask])
    ae_prim_unseen_scores = best_ae_prim.score_samples(X_unseen_test_prim)

    unseen_methods = {
        "Chi-Square": (chi_unseen_scores, chi_unseen_th5, chi_unseen_th1),
        "Calibrated LNR": (lnr_unseen_scores, lnr_unseen_th5, lnr_unseen_th1),
        "Range-Check (Envelope)": (rc_unseen_viol, rc_unseen_th5, rc_unseen_th1),
        "Mahalanobis (Ledoit-Wolf)": (mahal_unseen, mahal_unseen_th5, mahal_unseen_th1),
        "Autoencoder (D1+D2)": (ae_d12_unseen_scores, ae_d12_th_5fa, ae_d12_th_1fa),
        "Autoencoder (All: A-D2)": (ae_prim_unseen_scores, ae_prim_th_5fa, ae_prim_th_1fa),
        "RF (Seen Only -> Held-Out)": (rf_unseen_scores, rf_unseen_th_5fa, rf_unseen_th_1fa),
    }

    unseen_table_rows = []
    unseen_exact_sub = exact_mask_unseen_test | norm_mask_unseen_test

    for m_name, (scores, th5, th1) in unseen_methods.items():
        flags5 = scores >= th5
        flags1 = scores >= th1

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
            "AUC_All": (auc_all, ci_auc_all),
            "AUC_Exact": (auc_exact, ci_auc_exact),
            "Det5_All": (det5_all, ci_det5_all),
            "Det1_All": (det1_all, ci_det1_all),
            "Det5_Exact": (det5_exact, ci_det5_exact),
            "Det5_Eff": (det5_eff, ci_det5_eff),
            "PerType5": per_type_det5,
        })

    print("\n--- TABLE 2: UNSEEN-ATTACK SPLIT COMPARISON (N=4,026) ---")
    print(hdr)
    print("-" * len(hdr))
    for r in unseen_table_rows:
        auc_s = f"{r['AUC_All'][0]:.4f} [{r['AUC_All'][1][0]:.4f},{r['AUC_All'][1][1]:.4f}]"
        auc_e = f"{r['AUC_Exact'][0]:.4f} [{r['AUC_Exact'][1][0]:.4f},{r['AUC_Exact'][1][1]:.4f}]"
        d5_s = f"{r['Det5_All'][0]:5.2f}% [{r['Det5_All'][1][0]:.1f},{r['Det5_All'][1][1]:.1f}]"
        d1_s = f"{r['Det1_All'][0]:5.2f}% [{r['Det1_All'][1][0]:.1f},{r['Det1_All'][1][1]:.1f}]"
        ex5_s = f"{r['Det5_Exact'][0]:5.2f}% [{r['Det5_Exact'][1][0]:.1f},{r['Det5_Exact'][1][1]:.1f}]"
        eff5_s = f"{r['Det5_Eff'][0]:5.2f}% [{r['Det5_Eff'][1][0]:.1f},{r['Det5_Eff'][1][1]:.1f}]"
        print(f"{r['Method']:<25} | {auc_s:<25} | {auc_e:<25} | {d5_s:<14} | {d1_s:<14} | {ex5_s:<14} | {eff5_s:<14}")

    print("\n--- Table 2 Held-Out and Seen Attack Detection @ 5% FA ---")
    hdr_pt2 = f"{'Method':<25} | {'HELD exact_sin':<14} | {'HELD exact_mul':<14} | {'HELD limited':<12} | {'lin_single':<10} | {'lin_multi':<10} | {'random':<10}"
    print(hdr_pt2)
    print("-" * len(hdr_pt2))
    for r in unseen_table_rows:
        pt = r["PerType5"]
        print(f"{r['Method']:<25} | {pt['exact_ac_single_bus']:6.2f}%       | {pt['exact_ac_multi_bus']:6.2f}%       | {pt['limited_meter']:6.2f}%    | {pt['linearised_single_bus']:6.2f}%  | {pt['linearised_multi_bus']:6.2f}%  | {pt['random']:6.2f}%")

    # -------------------------------------------------------------------------
    # PART 4: EXTRA EXPERIMENTS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART 4: EXTRA EXPERIMENTS")
    print("=" * 80)

    # 1. Detection vs Shift Sigmas Bins
    print("\n--- (1) Detection vs Shift Sigmas Bins (@ 5% FA) ---")
    sig_bins = cfg["evaluation"]["sigma_bins"]
    sig_labels = cfg["evaluation"]["sigma_labels"]
    test_meta["sig_bin"] = pd.cut(test_meta["shift_sigmas"], bins=sig_bins, labels=sig_labels)

    sig_det_results = {"RF_Primary": [], "AE_All": [], "Mahalanobis": [], "Max_Z": [], "Counts": []}
    print(f"{'Sigma Bin':<15} | {'Count':<6} | {'RF (Primary)':<14} | {'Autoencoder':<14} | {'Mahalanobis':<14} | {'Per-Hour Max-Z':<14}")
    print("-" * 85)
    for sl in sig_labels:
        sub_mask = (test_meta["sig_bin"] == sl).values & (test_meta["label"] == 1).values
        cnt = np.sum(sub_mask)
        det_rf = np.mean(rf_prim_test_scores[sub_mask] >= rf_prim_th_5fa) * 100.0 if cnt > 0 else 0.0
        det_ae = np.mean(ae_prim_test_scores[sub_mask] >= ae_prim_th_5fa) * 100.0 if cnt > 0 else 0.0
        det_m = np.mean(mahal_dist[sub_mask] >= mahal_th_5fa) * 100.0 if cnt > 0 else 0.0
        det_z = np.mean(max_abs_z[sub_mask] >= z_th_5fa) * 100.0 if cnt > 0 else 0.0

        sig_det_results["Counts"].append(cnt)
        sig_det_results["RF_Primary"].append(det_rf)
        sig_det_results["AE_All"].append(det_ae)
        sig_det_results["Mahalanobis"].append(det_m)
        sig_det_results["Max_Z"].append(det_z)
        print(f"{sl:<15} | {cnt:<6} | {det_rf:6.2f}%       | {det_ae:6.2f}%       | {det_m:6.2f}%       | {det_z:6.2f}%")

    # 2. Feature Group Ablations
    print("\n--- (2) Feature Group Ablation: Each Group Alone & Each Removed ---")
    all_groups = ["a", "b", "c", "d1", "d2"]
    ablation_results = []

    # Alone
    for g in all_groups + ["d3"]:
        sw = {k: (k == g) for k in (all_groups + ["d3"])}
        cols_g = select_feature_columns(groups_dict, sw)
        rf_g = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=seed, n_jobs=-1)
        rf_g.fit(df_all_feat[cols_g].values[train_mask], y_train)
        p_val_g = rf_g.predict_proba(df_all_feat[cols_g].values[val_mask])[:, 1]
        p_test_g = rf_g.predict_proba(df_all_feat[cols_g].values[test_mask])[:, 1]

        val_auc_g = roc_auc_score(y_val, p_val_g)
        test_auc_g = roc_auc_score(y_test, p_test_g)
        th5_g = float(np.percentile(p_test_g[norm_mask_test], 95.0))
        test_det5_g = np.mean(p_test_g[att_mask_test] >= th5_g) * 100.0
        test_exact5_g = np.mean(p_test_g[exact_mask_test] >= th5_g) * 100.0

        lbl = f"Group {g.upper()} alone" + (" (clean prev)" if g == "d3" else "")
        ablation_results.append({
            "Config": lbl,
            "N_Feat": len(cols_g),
            "Val_AUC": val_auc_g,
            "Test_AUC": test_auc_g,
            "Test_Det5": test_det5_g,
            "Test_Exact5": test_exact5_g,
        })

    # Removed (leave-one-group-out from primary A+B+C+D1+D2)
    for g in all_groups:
        sw = {k: (k != g and k in all_groups) for k in (all_groups + ["d3"])}
        cols_rem = select_feature_columns(groups_dict, sw)
        rf_rem = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=seed, n_jobs=-1)
        rf_rem.fit(df_all_feat[cols_rem].values[train_mask], y_train)
        p_val_rem = rf_rem.predict_proba(df_all_feat[cols_rem].values[val_mask])[:, 1]
        p_test_rem = rf_rem.predict_proba(df_all_feat[cols_rem].values[test_mask])[:, 1]

        val_auc_rem = roc_auc_score(y_val, p_val_rem)
        test_auc_rem = roc_auc_score(y_test, p_test_rem)
        th5_rem = float(np.percentile(p_test_rem[norm_mask_test], 95.0))
        test_det5_rem = np.mean(p_test_rem[att_mask_test] >= th5_rem) * 100.0
        test_exact5_rem = np.mean(p_test_rem[exact_mask_test] >= th5_rem) * 100.0

        ablation_results.append({
            "Config": f"Without {g.upper()} (All - {g.upper()})",
            "N_Feat": len(cols_rem),
            "Val_AUC": val_auc_rem,
            "Test_AUC": test_auc_rem,
            "Test_Det5": test_det5_rem,
            "Test_Exact5": test_exact5_rem,
        })

    print(f"{'Ablation Setting':<32} | {'Features':<8} | {'Val AUC':<8} | {'Test AUC':<8} | {'Overall Det@5%':<15} | {'Exact AC Det@5%':<15}")
    print("-" * 95)
    for ar in ablation_results:
        print(f"{ar['Config']:<32} | {ar['N_Feat']:<8} | {ar['Val_AUC']:.4f}   | {ar['Test_AUC']:.4f}   | {ar['Test_Det5']:6.2f}%         | {ar['Test_Exact5']:6.2f}%")

    # 3. Limited-Meter: Detection vs Shift Ratio
    print("\n--- (3) Limited-Meter: Detection vs Shift Ratio ---")
    lm_mask = (test_meta["attack_type"] == "limited_meter").values
    test_meta["ratio_bin"] = pd.cut(test_meta["shift_ratio"], bins=cfg["evaluation"]["shift_ratio_bins"], labels=cfg["evaluation"]["shift_ratio_labels"])

    lm_ratio_results = {"RF_Primary": [], "AE_All": [], "Mahalanobis": [], "Calibrated_LNR": [], "Chi_Square": [], "Counts": []}
    print(f"{'Shift Ratio Bin':<18} | {'Count':<6} | {'RF Primary':<12} | {'Autoencoder':<12} | {'Mahalanobis':<12} | {'Calibrated LNR':<15} | {'Chi-Square':<12}")
    print("-" * 100)
    for rl in cfg["evaluation"]["shift_ratio_labels"]:
        sub_mask = lm_mask & (test_meta["ratio_bin"] == rl).values
        cnt = np.sum(sub_mask)
        det_rf = np.mean(rf_prim_test_scores[sub_mask] >= rf_prim_th_5fa) * 100.0 if cnt > 0 else 0.0
        det_ae = np.mean(ae_prim_test_scores[sub_mask] >= ae_prim_th_5fa) * 100.0 if cnt > 0 else 0.0
        det_m = np.mean(mahal_dist[sub_mask] >= mahal_th_5fa) * 100.0 if cnt > 0 else 0.0
        det_lnr = np.mean(lnr_test_scores[sub_mask] >= lnr_th_5fa) * 100.0 if cnt > 0 else 0.0
        det_chi = np.mean(chi_test_scores[sub_mask] >= chi_th_5fa) * 100.0 if cnt > 0 else 0.0

        lm_ratio_results["Counts"].append(cnt)
        lm_ratio_results["RF_Primary"].append(det_rf)
        lm_ratio_results["AE_All"].append(det_ae)
        lm_ratio_results["Mahalanobis"].append(det_m)
        lm_ratio_results["Calibrated_LNR"].append(det_lnr)
        lm_ratio_results["Chi_Square"].append(det_chi)
        print(f"{rl:<18} | {cnt:<6} | {det_rf:6.2f}%     | {det_ae:6.2f}%     | {det_m:6.2f}%     | {det_lnr:6.2f}%         | {det_chi:6.2f}%")

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
    for m_name in ["RF w/o D3 (Primary)", "Autoencoder (All: A-D2)", "Mahalanobis (Ledoit-Wolf)", "Per-Hour Max-Z", "Calibrated LNR", "Chi-Square"]:
        sc, _, _ = methods[m_name]
        fpr, tpr, _ = roc_curve(y_test, sc)
        auc_val = roc_auc_score(y_test, sc)
        plt.plot(fpr, tpr, label=f"{m_name} (AUC={auc_val:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel("False Positive Rate (Normal Rows)")
    plt.ylabel("True Positive Rate (All Attacks)")
    plt.title("ROC Curves: All Attacks (N=2,000)")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    for m_name in ["RF w/o D3 (Primary)", "Autoencoder (All: A-D2)", "Mahalanobis (Ledoit-Wolf)", "Per-Hour Max-Z", "Calibrated LNR", "Chi-Square"]:
        sc, _, _ = methods[m_name]
        fpr, tpr, _ = roc_curve(y_test[exact_sub_mask], sc[exact_sub_mask])
        auc_val = roc_auc_score(y_test[exact_sub_mask], sc[exact_sub_mask])
        plt.plot(fpr, tpr, label=f"{m_name} (AUC={auc_val:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel("False Positive Rate (Normal Rows)")
    plt.ylabel("True Positive Rate (Exact AC Attacks)")
    plt.title("ROC Curves: Exact AC Attacks Only (N=1,332)")
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
    plt.axvline(rf_prim_th_5fa, color="black", linestyle="--", label="5% FA Thresh")
    plt.xlabel("RF Probability P(Attack)")
    plt.ylabel("Density")
    plt.title("RF Score Distribution")
    plt.legend(fontsize=7)

    plt.subplot(1, 3, 2)
    plt.hist(ae_prim_test_scores[norm_mask_test], bins=30, density=True, alpha=0.5, label="Normal", color="gray")
    plt.hist(ae_prim_test_scores[exact_mask_test], bins=30, density=True, alpha=0.6, label="Exact AC", color="crimson")
    plt.hist(ae_prim_test_scores[att_mask_test & ~exact_mask_test], bins=30, density=True, alpha=0.5, label="Other Attacks", color="royalblue")
    plt.axvline(ae_prim_th_5fa, color="black", linestyle="--", label="5% FA Thresh")
    plt.xlabel("Autoencoder Reconstruction MSE")
    plt.ylabel("Density")
    plt.title("Autoencoder Score Distribution")
    plt.legend(fontsize=7)

    plt.subplot(1, 3, 3)
    plt.hist(mahal_dist[norm_mask_test], bins=30, density=True, alpha=0.5, label="Normal", color="gray")
    plt.hist(mahal_dist[exact_mask_test], bins=30, density=True, alpha=0.6, label="Exact AC", color="crimson")
    plt.hist(mahal_dist[att_mask_test & ~exact_mask_test], bins=30, density=True, alpha=0.5, label="Other Attacks", color="royalblue")
    plt.axvline(mahal_th_5fa, color="black", linestyle="--", label="5% FA Thresh")
    plt.xlabel("Mahalanobis Distance D_M")
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
    plt.ylabel("Permutation Importance Drop (Delta Validation ROC-AUC)")
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

    print("\nTask 8 pipeline completed successfully.")


if __name__ == "__main__":
    main()
