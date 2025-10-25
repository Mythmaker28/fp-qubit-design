#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script for v2.2.2 - Blending to Family Median
Uses RandomForest + family median blending for improved MAE
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
import hashlib
import warnings
import argparse
from collections import Counter
import os
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_predict
import joblib

def balanced_group_kfold(groups, n_splits=5, seed=1337):
    """Balanced Group K-Fold without np.unique"""
    rng = np.random.RandomState(seed)
    fam_counts = Counter(groups)
    fams = list(fam_counts.keys())
    rng.shuffle(fams)
    fams.sort(key=lambda f: fam_counts[f], reverse=True)
    folds = [set() for _ in range(n_splits)]
    load = [0]*n_splits
    for f in fams:
        i = min(range(n_splits), key=lambda k: load[k])
        folds[i].add(f)
        load[i] += fam_counts[f]
    fam_to_fold = {}
    for k, fs in enumerate(folds):
        for f in fs: 
            fam_to_fold[f] = k
    return np.array([fam_to_fold[g] for g in groups], dtype=int)

def clean_data(df):
    """Clean data according to specifications"""
    print("=== CLEANING DATA ===")
    
    # Clean family column
    df["family"] = df["family"].fillna("Other").astype(str).str.strip()
    
    # Clean numerical columns
    for col in ["excitation_nm", "emission_nm", "stokes_shift_nm"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Impute missing values with median
    df[["excitation_nm", "emission_nm", "stokes_shift_nm"]] = (
        df[["excitation_nm", "emission_nm", "stokes_shift_nm"]]
        .fillna(df[["excitation_nm", "emission_nm", "stokes_shift_nm"]].median())
    )
    
    # Clean categorical columns
    for col in ["method", "context_type"]:
        df[col] = df[col].fillna("NA").astype(str).str.strip()
    
    print(f"Data shape after cleaning: {df.shape}")
    print(f"Family distribution: {df['family'].value_counts().head()}")
    
    return df

def prepare_features_and_target(df):
    """Prepare features and target with proper encoding"""
    print("\n=== PREPARING FEATURES ===")
    
    # Target: log1p(contrast_normalized)
    y_log = np.log1p(df['contrast_normalized'].values)
    y_original = df['contrast_normalized'].values
    
    # Sample weights
    if 'sample_weight' in df.columns:
        sample_weights = df['sample_weight'].fillna(1.0).values
    else:
        sample_weights = np.ones(len(df))
    
    # Groups for CV
    groups = df['family'].values
    
    # Feature columns
    numerical_features = ['excitation_nm', 'emission_nm', 'stokes_shift_nm']
    categorical_features = ['method', 'context_type', 'family']
    
    # Create feature matrix
    X = df[numerical_features + categorical_features].copy()
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target range (original): [{y_original.min():.3f}, {y_original.max():.3f}]")
    print(f"Target range (log1p): [{y_log.min():.3f}, {y_log.max():.3f}]")
    print(f"Groups: {len(set(groups))} families")
    print(f"Sample weights range: {sample_weights.min():.3f} - {sample_weights.max():.3f}")
    
    return X, y_original, y_log, groups, sample_weights

def create_preprocessor():
    """Create ColumnTransformer for feature preprocessing"""
    numerical_features = ['excitation_nm', 'emission_nm', 'stokes_shift_nm']
    categorical_features = ['method', 'context_type', 'family']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, min_frequency=2), categorical_features)
        ]
    )
    
    return preprocessor

def train_model_with_cv(X, y_log, groups, sample_weights):
    """Train RandomForest with balanced GroupKFold CV"""
    print("\n=== TRAINING MODEL WITH CV ===")
    
    # Create preprocessor
    preprocessor = create_preprocessor()
    
    # Create RandomForest
    rf = RandomForestRegressor(
        n_estimators=1200,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=1337
    )
    
    # Create balanced GroupKFold
    fold_indices = balanced_group_kfold(groups, n_splits=5, seed=1337)
    
    # Cross-validation predictions
    cv_predictions = []
    cv_results = []
    
    for fold in range(5):
        print(f"Fold {fold + 1}/5")
        
        # Get train/test indices for this fold
        train_mask = fold_indices != fold
        test_mask = fold_indices == fold
        
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y_log[train_mask], y_log[test_mask]
        weights_train = sample_weights[train_mask]
        weights_test = sample_weights[test_mask]
        groups_test = groups[test_mask]
        
        # Fit preprocessor and model
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        rf.fit(X_train_processed, y_train, sample_weight=weights_train)
        y_pred_log = rf.predict(X_test_processed)
        y_pred_orig = np.expm1(y_pred_log)
        y_test_orig = np.expm1(y_test)
        
        # Calculate metrics
        r2 = r2_score(y_test_orig, y_pred_orig)
        mae = mean_absolute_error(y_test_orig, y_pred_orig)
        
        cv_results.append({
            'fold': fold + 1,
            'r2': r2,
            'mae': mae,
            'y_true': y_test_orig,
            'y_pred': y_pred_orig,
            'family': groups_test,
            'weights': weights_test
        })
        
        print(f"  R²: {r2:.3f}, MAE: {mae:.3f}")
    
    return cv_results, fold_indices

def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description='FP-DESIGN v2.2.2 Blending to Family Median')
    parser.add_argument('--data', required=True, help='Path to balanced training data CSV')
    parser.add_argument('--out', required=True, help='Output directory')
    
    args = parser.parse_args()
    
    print("=== FP-DESIGN v2.2.2 BLENDING TO FAMILY MEDIAN ===")
    
    # Load data
    df = pd.read_csv(args.data)
    print(f"N_balanced: {len(df)}")
    print(f"Families: {df['family'].nunique()}")
    print(f"Calcium share: {(df['family'] == 'Calcium').mean()*100:.1f}%")
    
    # Clean data
    df = clean_data(df)
    
    # Prepare features
    X, y_original, y_log, groups, sample_weights = prepare_features_and_target(df)
    
    # Train model with CV
    cv_results, fold_indices = train_model_with_cv(X, y_log, groups, sample_weights)
    
    # Aggregate all predictions for blending
    all_y_true = np.concatenate([r['y_true'] for r in cv_results])
    all_y_pred = np.concatenate([r['y_pred'] for r in cv_results])
    all_families = np.concatenate([r['family'] for r in cv_results])
    
    # --- Blending to family-median learned on train of each fold ---
    print("\n=== APPLYING FAMILY MEDIAN BLENDING ===")
    
    families = df["family"].values
    alphas = [0.2, 0.4, 0.5, 0.6, 0.8]
    best_alpha = None
    best_mae = 1e9
    
    def fold_blend(alpha):
        y_blend = np.zeros_like(all_y_pred)
        for k in range(5):
            tr = (fold_indices != k)
            te = (fold_indices == k)
            # median per-family on TRAIN fold only
            fam_med = (
                pd.DataFrame({"fam": families[tr], "y": y_original[tr]})
                .groupby("fam")["y"].median()
            )
            # apply on TEST fold
            med_te = pd.Series(families[te]).map(fam_med).fillna(np.median(y_original[tr])).values
            y_blend[te] = alpha * all_y_pred[te] + (1 - alpha) * med_te
        return y_blend
    
    for a in alphas:
        y_b = fold_blend(a)
        mae_b = float(np.mean(np.abs(all_y_true - y_b)))
        if mae_b < best_mae:
            best_mae, best_alpha = mae_b, a
        print(f"  Alpha {a}: MAE = {mae_b:.3f}")
    
    y_blend = fold_blend(best_alpha)
    print(f"Best alpha: {best_alpha} (MAE: {best_mae:.3f})")
    
    # recompute metrics with blended central
    from sklearn.metrics import r2_score, mean_absolute_error
    r2_blend  = float(r2_score(all_y_true, y_blend))
    mae_blend = float(mean_absolute_error(all_y_true, y_blend))
    mae_mean  = float(mean_absolute_error(all_y_true, np.full_like(all_y_true, all_y_true.mean())))
    mae_med   = float(mean_absolute_error(all_y_true, np.full_like(all_y_true, np.median(all_y_true))))
    dmean     = float((mae_mean - mae_blend)/mae_mean*100.0)
    dmedian   = float((mae_med  - mae_blend)/mae_med *100.0)
    
    # simple conformal intervals around blended preds
    resid = np.abs(all_y_true - y_blend)
    q = float(np.quantile(resid, 0.90))
    pi_low  = y_blend - q
    pi_high = y_blend + q
    covered = ((all_y_true >= pi_low) & (all_y_true <= pi_high)).mean()
    coverage = float(covered*100.0)
    ece = float(abs(covered - 0.90))
    
    # overwrite outputs for this variant
    out_bl = {
      "model":"RF+family_median_blend",
      "alpha": best_alpha,
      "r2": r2_blend, "mae": mae_blend,
      "baseline_mae_mean": mae_mean, "baseline_mae_median": mae_med,
      "delta_mae_percent_vs_mean": dmean,
      "delta_mae_percent_vs_median": dmedian,
      "coverage_90_percent": coverage, "ece_abs_error": ece
    }
    
    # Create output directory
    Path(args.out).mkdir(parents=True, exist_ok=True)
    
    with open(os.path.join(args.out, "cv_metrics_v2_2_2_blend.json"), "w", encoding="utf-8") as f:
        json.dump(out_bl, f, indent=2)
    
    pd.DataFrame({
        "fold": fold_indices, "family": all_families,
        "y_true": all_y_true, "y_pred_blend": y_blend,
        "pi_low": pi_low, "pi_high": pi_high
    }).to_csv(os.path.join(args.out, "cv_predictions_uq_v2_2_2_blend.csv"), index=False, encoding="utf-8")
    
    print("\n=== FINAL RESULTS ===")
    print(json.dumps(out_bl, indent=2))

if __name__ == "__main__":
    main()
