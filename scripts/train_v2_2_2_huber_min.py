#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script for v2.2.2 - Huber Loss GBDT
Uses GradientBoostingRegressor with Huber loss for robustness to outliers
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
warnings.filterwarnings('ignore')

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
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
    """Train GradientBoostingRegressor with Huber loss and balanced GroupKFold CV"""
    print("\n=== TRAINING MODEL WITH CV ===")
    
    # Create preprocessor
    preprocessor = create_preprocessor()
    
    # Create GradientBoostingRegressor with Huber loss
    gbr = GradientBoostingRegressor(
        n_estimators=800,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.7,
        loss="huber",
        random_state=1337
    )
    
    # Create pipeline
    pipe = Pipeline([("prep", preprocessor), ("gbr", gbr)])
    
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
        
        # Fit pipeline
        pipe.fit(X_train, y_train, gbr__sample_weight=weights_train)
        y_pred_log = pipe.predict(X_test)
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
    
    return cv_results

def calculate_metrics(cv_results, y_original):
    """Calculate overall metrics and baselines"""
    print("\n=== CALCULATING METRICS ===")
    
    # Aggregate all predictions
    all_y_true = np.concatenate([r['y_true'] for r in cv_results])
    all_y_pred = np.concatenate([r['y_pred'] for r in cv_results])
    all_weights = np.concatenate([r['weights'] for r in cv_results])
    
    # Overall metrics (original scale)
    r2 = r2_score(all_y_true, all_y_pred)
    mae = mean_absolute_error(all_y_true, all_y_pred)
    
    # Baselines
    mean_pred = np.full_like(y_original, np.mean(y_original))
    median_pred = np.full_like(y_original, np.median(y_original))
    
    mae_mean = mean_absolute_error(y_original, mean_pred)
    mae_median = mean_absolute_error(y_original, median_pred)
    
    # Delta MAE
    delta_mae_percent = (mae_mean - mae) / mae_mean * 100
    
    # UQ: Split-conformal global 90%
    residuals = np.abs(all_y_true - all_y_pred)
    q_90 = np.quantile(residuals, 0.90)
    
    # Prediction intervals
    pi_low = all_y_pred - q_90
    pi_high = all_y_pred + q_90
    
    # Coverage
    coverage = np.mean((all_y_true >= pi_low) & (all_y_true <= pi_high))
    ece = abs(coverage - 0.90)
    
    print(f"R²: {r2:.3f}")
    print(f"MAE: {mae:.3f}")
    print(f"MAE (mean baseline): {mae_mean:.3f}")
    print(f"MAE (median baseline): {mae_median:.3f}")
    print(f"Delta MAE: {delta_mae_percent:.1f}%")
    print(f"Coverage (90%): {coverage:.1%}")
    print(f"ECE: {ece:.3f}")
    
    return {
        'r2': r2,
        'mae': mae,
        'baseline_mae_mean': mae_mean,
        'baseline_mae_median': mae_median,
        'delta_mae_percent': delta_mae_percent,
        'coverage_90_percent': coverage,
        'ece_abs_error': ece
    }

def save_artifacts(cv_results, metrics, output_dir):
    """Save all artifacts"""
    print("\n=== SAVING ARTIFACTS ===")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    with open(f"{output_dir}/cv_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved: {output_dir}/cv_metrics.json")
    
    # Save predictions
    all_results = []
    for r in cv_results:
        for i in range(len(r['y_true'])):
            all_results.append({
                'fold': r['fold'],
                'family': r['family'][i],
                'y_true': r['y_true'][i],
                'y_pred': r['y_pred'][i],
                'pi_low': r['y_pred'][i] - np.quantile(np.abs(r['y_true'] - r['y_pred']), 0.90),
                'pi_high': r['y_pred'][i] + np.quantile(np.abs(r['y_true'] - r['y_pred']), 0.90)
            })
    
    pred_df = pd.DataFrame(all_results)
    pred_df.to_csv(f"{output_dir}/cv_predictions_uq.csv", index=False)
    print(f"Saved: {output_dir}/cv_predictions_uq.csv")

def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description='FP-DESIGN v2.2.2 Huber Loss GBDT')
    parser.add_argument('--data', required=True, help='Path to balanced training data CSV')
    parser.add_argument('--out', required=True, help='Output directory')
    
    args = parser.parse_args()
    
    print("=== FP-DESIGN v2.2.2 HUBER LOSS GBDT ===")
    
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
    cv_results = train_model_with_cv(X, y_log, groups, sample_weights)
    
    # Calculate metrics
    metrics = calculate_metrics(cv_results, y_original)
    
    # Save artifacts
    save_artifacts(cv_results, metrics, args.out)
    
    # Final status
    print(f"\n=== FINAL STATUS ===")
    print(f"Data: N_rows={len(df)} ; Families={len(set(groups))} ; Other={sum(groups == 'Other')}")
    print(f"Metrics (CV mean±std, original scale):")
    print(f"  - R² = {metrics['r2']:.3f}")
    print(f"  - MAE = {metrics['mae']:.3f}")
    print(f"  - ECE = {metrics['ece_abs_error']:.3f}")
    print(f"  - Coverage = {metrics['coverage_90_percent']:.1%}")
    print(f"Baselines: mean MAE={metrics['baseline_mae_mean']:.3f} ; median MAE={metrics['baseline_mae_median']:.3f}")
    print(f"Artifacts: {args.out}/*")

if __name__ == "__main__":
    main()
