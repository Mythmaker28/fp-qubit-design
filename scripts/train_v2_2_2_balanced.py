#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script for v2.2.2 - Balanced Evaluation & Release Gate
Uses sample_weight for balanced training with RandomForest + robust CV
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
        
        print(f"  RÂ²: {r2:.3f}, MAE: {mae:.3f}")
    
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
    
    print(f"RÂ²: {r2:.3f}")
    print(f"MAE: {mae:.3f}")
    print(f"MAE (mean baseline): {mae_mean:.3f}")
    print(f"MAE (median baseline): {mae_median:.3f}")
    print(f"Delta MAE: {delta_mae_percent:.1f}%")
    print(f"Coverage (90%): {coverage:.1%}")
    print(f"ECE: {ece:.3f}")
    
    return {
        'r2': r2,
        'mae': mae,
        'mae_mean': mae_mean,
        'mae_median': mae_median,
        'delta_mae_percent': delta_mae_percent,
        'coverage_90_percent': coverage,
        'ece_abs_error': ece
    }

def save_artifacts(cv_results, metrics, output_dir):
    """Save all artifacts"""
    print("\n=== SAVING ARTIFACTS ===")
    
    # Save metrics
    with open(f"{output_dir}/cv_metrics_v2_2_2.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved: {output_dir}/cv_metrics_v2_2_2.json")
    
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
    pred_df.to_csv(f"{output_dir}/cv_predictions_uq_v2_2_2.csv", index=False)
    print(f"Saved: {output_dir}/cv_predictions_uq_v2_2_2.csv")
    
    # Generate SHA256SUMS
    files_to_hash = [
        f"{output_dir}/cv_metrics_v2_2_2.json",
        f"{output_dir}/cv_predictions_uq_v2_2_2.csv"
    ]
    
    sha256sums = []
    for file_path in files_to_hash:
        if Path(file_path).exists():
            with open(file_path, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            sha256sums.append(f"{file_hash}  {file_path}")
    
    with open(f"{output_dir}/SHA256SUMS_v2_2_2.txt", "w") as f:
        f.write("\n".join(sha256sums))
    print(f"Saved: {output_dir}/SHA256SUMS_v2_2_2.txt")

def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description='FP-DESIGN v2.2.2 Balanced Evaluation')
    parser.add_argument('--data', required=True, help='Path to balanced training data CSV')
    parser.add_argument('--out', required=True, help='Output directory')
    
    args = parser.parse_args()
    
    print("=== FP-DESIGN v2.2.2 BALANCED EVALUATION ===")
    
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
    print(f"Metrics (CV meanÂ±std, original scale):")
    print(f"  - RÂ² = {metrics['r2']:.3f}  (â‰¥0.20) â†’ {'PASS' if metrics['r2'] >= 0.20 else 'FAIL'}")
    print(f"  - MAE = {metrics['mae']:.3f} (<7.810) â†’ {'PASS' if metrics['mae'] < 7.810 else 'FAIL'}")
    print(f"  - ECE = {metrics['ece_abs_error']:.3f} (â‰¤0.15) â†’ {'PASS' if metrics['ece_abs_error'] <= 0.15 else 'FAIL'}")
    print(f"  - Coverage = {metrics['coverage_90_percent']:.1%} (90Â±5) â†’ {'PASS' if 0.85 <= metrics['coverage_90_percent'] <= 0.95 else 'FAIL'}")
    print(f"Baselines: mean MAE={metrics['mae_mean']:.3f} ; median MAE={metrics['mae_median']:.3f} ; Î”MAE={metrics['delta_mae_percent']:.1f}% â†’ {'PASS' if metrics['delta_mae_percent'] >= 10 else 'FAIL'}")
    print(f"Artifacts: {args.out}/*")
    
    # Decision
    criteria_passed = (
        metrics['r2'] >= 0.20 and
        metrics['mae'] < 7.810 and
        metrics['ece_abs_error'] <= 0.15 and
        0.85 <= metrics['coverage_90_percent'] <= 0.95 and
        metrics['delta_mae_percent'] >= 10
    )
    
    print(f"Decision: {'GO' if criteria_passed else 'NO-GO'}")
    
    if not criteria_passed:
        print("Notes: Some criteria failed - detailed analysis needed")
    else:
        print("Notes: All criteria passed - ready for release")

if __name__ == "__main__":
    main()
