"""
python/splits.py

Task 7, Part 1: Dataset Splitting Module.
Implements:
 1. Standard Group-Aware Split: 60% Train, 20% Validation, 20% Test by group_id.
    Both rows of each group (normal and attacked) remain strictly together.
 2. Unseen Attack Scheme: Normal rows follow the group-aware split.
    Attacked rows with held-out attack types (exact_ac_single_bus, exact_ac_multi_bus, limited_meter)
    are excluded from training and assigned to the test evaluation split.
 3. Exports split assignments to data/splits.csv (row_id, group_id, split, split_unseen).
"""

import sys
import os
import tomllib
import numpy as np
import pandas as pd


def load_config(config_path="configs/features.toml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def create_splits(df, cfg):
    seed = cfg["general"]["seed"]
    train_ratio = cfg["splits"]["train_ratio"]
    val_ratio = cfg["splits"]["val_ratio"]
    test_ratio = cfg["splits"]["test_ratio"]
    
    held_out_types = set(cfg["splits"]["held_out_attack_types"])
    train_types = set(cfg["splits"]["train_attack_types"])

    # Unique groups
    unique_groups = np.sort(df["group_id"].unique())
    n_groups = len(unique_groups)

    # Seeded shuffle of groups
    rng = np.random.default_rng(seed)
    shuffled_groups = rng.permutation(unique_groups)

    n_train = int(round(train_ratio * n_groups))
    n_val = int(round(val_ratio * n_groups))
    n_test = n_groups - n_train - n_val

    train_groups = set(shuffled_groups[:n_train])
    val_groups = set(shuffled_groups[n_train:n_train + n_val])
    test_groups = set(shuffled_groups[n_train + n_val:])

    print(f"Total groups: {n_groups} -> Train: {len(train_groups)}, Val: {len(val_groups)}, Test: {len(test_groups)}")

    # 1. Standard Group-Aware Split
    group_to_split = {}
    for g in train_groups:
        group_to_split[g] = "train"
    for g in val_groups:
        group_to_split[g] = "val"
    for g in test_groups:
        group_to_split[g] = "test"

    splits_standard = [group_to_split[g] for g in df["group_id"]]

    # 2. Unseen Attack Scheme
    # Normal rows follow the group-aware split.
    # Attacked rows:
    # - If attack type is in train_attack_types: follow group-aware split.
    # - If attack type is in held_out_attack_types: cannot be in train; assigned to test.
    splits_unseen = []
    for _, row in df.iterrows():
        g_split = group_to_split[row["group_id"]]
        if row["label"] == 0:
            # Normal row follows group-aware split
            splits_unseen.append(g_split)
        else:
            atype = row["attack_type"]
            if atype in held_out_types:
                # Held out of training entirely -> goes to test
                splits_unseen.append("test")
            else:
                # Allowed in training -> follows group split
                splits_unseen.append(g_split)

    df_splits = pd.DataFrame({
        "row_id": df["row_id"],
        "group_id": df["group_id"],
        "split": splits_standard,
        "split_unseen": splits_unseen
    })

    return df_splits


def report_split_statistics(df, df_splits):
    df_merged = df.copy()
    df_merged["split"] = df_splits["split"]
    df_merged["split_unseen"] = df_splits["split_unseen"]

    for col, title in [("split", "STANDARD GROUP-AWARE SPLIT"), ("split_unseen", "UNSEEN ATTACK GENERALIZATION SPLIT")]:
        print(f"\n{'='*80}\n{title}\n{'='*80}")
        
        # Summary by split and label
        pivot_label = pd.crosstab(df_merged[col], df_merged["label"], margins=True)
        pivot_label.columns = ["Normal (0)", "Attacked (1)", "Total"]
        print("\nRow counts by Split and Label:")
        print(pivot_label)

        # Summary by split and attack type
        pivot_attack = pd.crosstab(df_merged["attack_type"], df_merged[col], margins=True)
        print("\nRow counts by Attack Type and Split:")
        print(pivot_attack)


def main():
    cfg = load_config()
    dataset_path = cfg["general"]["dataset_path"]
    out_splits_path = cfg["general"]["splits_output_path"]

    print(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"Loaded {len(df)} rows across {df['group_id'].nunique()} groups.")

    df_splits = create_splits(df, cfg)
    report_split_statistics(df, df_splits)

    # Save to data/splits.csv
    os.makedirs(os.path.dirname(out_splits_path), exist_ok=True)
    df_splits.to_csv(out_splits_path, index=False)
    print(f"\nSaved split assignments to: {out_splits_path}")


if __name__ == "__main__":
    main()
