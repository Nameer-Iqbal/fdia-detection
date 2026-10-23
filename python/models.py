"""
Task 8 Models: Supervised Random Forest and Unsupervised PyTorch Autoencoder.
Includes hyperparameter tuning on validation split, grouped permutation importance,
and candidate bottleneck selection for Autoencoder on normal validation reconstruction loss.
"""

import copy
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# -----------------------------------------------------------------------------
# 1. Feature Selection Utilities
# -----------------------------------------------------------------------------

def get_feature_columns_by_group(meta_path="data/features_meta_v2.json"):
    """
    Returns a dictionary mapping group identifier to its list of feature column names.
    Groups:
      - 'a': group_a_residuals (328 cols)
      - 'b': group_b_consistency (163 cols)
      - 'c': group_c_sensitivity (163 cols)
      - 'd1': group_d1_raw (228 cols)
      - 'd2': group_d2_hourly_forecast (228 cols)
      - 'd3': group_d3_change_prev (228 cols)
    """
    with open(meta_path, "r") as f:
        meta = json.load(f)

    fg = meta["feature_groups"]
    return {
        "a": fg["group_a_residuals"]["columns"],
        "b": fg["group_b_consistency"]["columns"],
        "c": fg["group_c_sensitivity"]["columns"],
        "d1": fg["group_d1_raw"]["columns"],
        "d2": fg["group_d2_hourly_forecast"]["columns"],
        "d3": fg["group_d3_change_prev"]["columns"],
    }


def select_feature_columns(groups_dict, switches):
    """
    Concatenates feature column names according to boolean switches dict.
    e.g. switches = {'a': True, 'b': True, 'c': True, 'd1': True, 'd2': True, 'd3': False}
    """
    cols = []
    for g_key, active in switches.items():
        if active and g_key in groups_dict:
            cols.extend(groups_dict[g_key])
    return cols


# -----------------------------------------------------------------------------
# 2. Supervised Random Forest & Validation Tuning
# -----------------------------------------------------------------------------

def tune_random_forest(X_train, y_train, X_val, y_val, param_grid, seed=42, n_jobs=-1):
    """
    Tunes Random Forest on the VALIDATION split only using ROC-AUC.
    Never evaluates on test data.
    """
    best_score = -1.0
    best_params = None
    best_rf = None
    history = []

    # Grid search combinations
    for n_est in param_grid.get("grid_n_estimators", [100, 300]):
        for max_d in param_grid.get("grid_max_depth", [6, 8, 12]):
            for min_leaf in param_grid.get("grid_min_samples_leaf", [1, 3]):
                rf = RandomForestClassifier(
                    n_estimators=n_est,
                    max_depth=max_d,
                    min_samples_leaf=min_leaf,
                    random_state=seed,
                    n_jobs=n_jobs,
                )
                rf.fit(X_train, y_train)
                p_val = rf.predict_proba(X_val)[:, 1]
                val_auc = roc_auc_score(y_val, p_val)

                history.append({
                    "n_estimators": n_est,
                    "max_depth": max_d,
                    "min_samples_leaf": min_leaf,
                    "val_auc": float(val_auc),
                })

                if val_auc > best_score:
                    best_score = val_auc
                    best_params = {
                        "n_estimators": n_est,
                        "max_depth": max_d,
                        "min_samples_leaf": min_leaf,
                    }
                    best_rf = rf

    return best_rf, best_params, history


def grouped_permutation_importance(model, X_val, y_val, feature_names, groups_dict, active_groups, n_repeats=5, seed=42):
    """
    Computes permutation importance on validation set, grouped by feature group.
    Permutes all columns in a feature group simultaneously to capture joint information.
    """
    rng = np.random.default_rng(seed)
    p_val_base = model.predict_proba(X_val)[:, 1]
    base_auc = roc_auc_score(y_val, p_val_base)

    col_to_idx = {name: i for i, name in enumerate(feature_names)}
    group_indices = {}
    for g in active_groups:
        if g in groups_dict:
            idxs = [col_to_idx[c] for c in groups_dict[g] if c in col_to_idx]
            if len(idxs) > 0:
                group_indices[g] = np.array(idxs)

    importance_results = {}
    for g, idxs in group_indices.items():
        drops = []
        for rep in range(n_repeats):
            X_perm = X_val.copy()
            perm_order = rng.permutation(len(X_val))
            X_perm[:, idxs] = X_perm[perm_order, :][:, idxs]

            p_perm = model.predict_proba(X_perm)[:, 1]
            perm_auc = roc_auc_score(y_val, p_perm)
            drops.append(base_auc - perm_auc)

        importance_results[g] = {
            "mean_drop": float(np.mean(drops)),
            "std_drop": float(np.std(drops)),
            "n_features": len(idxs),
        }

    return base_auc, importance_results


# -----------------------------------------------------------------------------
# 3. Unsupervised PyTorch Autoencoder
# -----------------------------------------------------------------------------

class PyTorchAutoencoder(nn.Module):
    """
    Multi-layer Perceptron Autoencoder for tabular power grid measurements/states.
    Symmetric encoder-decoder with ReLU non-linearities.
    """
    def __init__(self, input_dim, hidden_dims, bottleneck_dim):
        super().__init__()
        # Encoder
        enc_layers = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers.append(nn.Linear(prev, h))
            enc_layers.append(nn.ReLU())
            prev = h
        enc_layers.append(nn.Linear(prev, bottleneck_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder
        dec_layers = []
        prev = bottleneck_dim
        for h in reversed(hidden_dims):
            dec_layers.append(nn.Linear(prev, h))
            dec_layers.append(nn.ReLU())
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        code = self.encoder(x)
        rec = self.decoder(code)
        return rec


class AutoencoderDetector:
    """
    Wrapper for training and scoring with PyTorch Autoencoder.
    Trains strictly on normal train rows, early stops on normal validation loss.
    """
    def __init__(self, input_dim, hidden_dims, bottleneck_dim, seed=42, lr=0.001, weight_decay=1e-5):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.bottleneck_dim = bottleneck_dim
        self.seed = seed
        self.lr = lr
        self.weight_decay = weight_decay
        self.scaler = StandardScaler()
        self.model = None

    def fit(self, X_train_norm, X_val_norm, max_epochs=60, batch_size=64, patience=8):
        """
        Fits StandardScaler on X_train_norm only.
        Trains PyTorchAutoencoder on X_train_norm only with early stopping on X_val_norm MSE loss.
        """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Fit scaler on train normal only
        X_tr_scaled = self.scaler.fit_transform(X_train_norm)
        X_va_scaled = self.scaler.transform(X_val_norm)

        train_tensor = torch.tensor(X_tr_scaled, dtype=torch.float32)
        val_tensor = torch.tensor(X_va_scaled, dtype=torch.float32)

        train_loader = DataLoader(
            TensorDataset(train_tensor),
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )

        model = PyTorchAutoencoder(self.input_dim, self.hidden_dims, self.bottleneck_dim)
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        best_state = None
        no_improve = 0

        for epoch in range(max_epochs):
            model.train()
            for (batch_x,) in train_loader:
                optimizer.zero_grad()
                out = model(batch_x)
                loss = criterion(out, batch_x)
                loss.backward()
                optimizer.step()

            # Validation loss on normal validation rows
            model.eval()
            with torch.no_grad():
                val_out = model(val_tensor)
                val_loss = float(criterion(val_out, val_tensor).item())

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        model.load_state_dict(best_state)
        self.model = model
        self.best_val_loss = best_val_loss
        return self

    def score_samples(self, X):
        """
        Computes sample-wise mean squared reconstruction error:
        score_i = mean((x_i - x_rec_i)^2)
        """
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        with torch.no_grad():
            x_tensor = torch.tensor(X_scaled, dtype=torch.float32)
            rec = self.model(x_tensor)
            diff = (x_tensor - rec).numpy()
            mse_scores = np.mean(diff ** 2, axis=1)
        return mse_scores


def select_best_autoencoder(X_train_norm, X_val_norm, input_dim, hidden_dims, bottleneck_candidates,
                            seed=42, lr=0.001, max_epochs=60, batch_size=64, patience=8):
    """
    Tries bottleneck candidates and chooses strictly based on lowest normal validation
    reconstruction loss (never by attack detection).
    """
    best_ae = None
    best_val_loss = float("inf")
    candidate_results = {}

    for b_dim in bottleneck_candidates:
        ae = AutoencoderDetector(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            bottleneck_dim=b_dim,
            seed=seed,
            lr=lr,
        )
        ae.fit(X_train_norm, X_val_norm, max_epochs=max_epochs, batch_size=batch_size, patience=patience)
        candidate_results[b_dim] = ae.best_val_loss

        if ae.best_val_loss < best_val_loss:
            best_val_loss = ae.best_val_loss
            best_ae = ae

    return best_ae, candidate_results
