#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v1.3.1 Training: GBDT + Conformalized Quantile Regression (CQR)
Fallback v1.2.5 (N=97 < 100): Relaxed criteria
- R² >= 0.10 (instead of 0.20)
- ECE <= 0.18 (instead of 0.15)
- Coverage: 85-95%
- MAE < 7.810
"""

import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import warnings
warnings.filterwarnings('ignore')

# Set seed
SEED = 1337
np.random.seed(SEED)

# Relaxed criteria for v1.2.5 (N=97 < 100)
CRITERIA_RELAXED = {
    'r2_min': 0.10,  # relaxed from 0.20
    'mae_max': 7.810,  # same as v1.1.4
    'ece_max': 0.18,  # relaxed from 0.15
    'coverage_min': 0.85,
    'coverage_max': 0.95,
    'beat_baseline_pct': 0.05  # relaxed from 0.10
}


def load_data(train_csv_path):
    """Load training data"""
    print(f"\n[LOAD] Reading {train_csv_path.name}...")
    df = pd.read_csv(train_csv_path)
    print(f"  [INFO] Shape: {df.shape}")
    return df


def build_features(df):
    """Build feature matrix with advanced features"""
    print("\n[FEATURES] Building feature matrix...")
    
    # Target (log-transformed)
    y_log = df['target_contrast_log'].values
    y_raw = df['contrast_normalized_raw'].values
    
    # Groups (family)
    groups = df['family'].values
    
    # Numerical features
    numerical_features = [
        'temperature_K', 'pH', 'is_biosensor',
        'excitation_nm', 'emission_nm', 'stokes_shift_nm'
    ]
    
    # Categorical features
    categorical_features = ['family', 'spectral_region', 'context_type']
    
    # Available features
    available_num = [f for f in numerical_features if f in df.columns]
    available_cat = [f for f in categorical_features if f in df.columns]
    
    print(f"  [INFO] Numerical features: {available_num}")
    print(f"  [INFO] Categorical features: {available_cat}")
    
    # Build X_num
    X_num = df[available_num].fillna(df[available_num].median()).values
    
    # One-hot encode categorical
    X_cat_list = []
    cat_feature_names = []
    
    for cat_col in available_cat:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_cat_encoded = encoder.fit_transform(df[[cat_col]])
        X_cat_list.append(X_cat_encoded)
        
        cat_names = [f"{cat_col}_{cat}" for cat in encoder.categories_[0]]
        cat_feature_names.extend(cat_names)
    
    # Concatenate all
    if X_cat_list:
        X_cat = np.hstack(X_cat_list)
        X = np.hstack([X_num, X_cat])
    else:
        X = X_num
    
    feature_names = available_num + cat_feature_names
    
    print(f"  [SUCCESS] X shape: {X.shape}")
    print(f"  [INFO] y_log range: [{y_log.min():.2f}, {y_log.max():.2f}]")
    print(f"  [INFO] y_raw range: [{y_raw.min():.2f}, {y_raw.max():.2f}]")
    
    return X, y_log, y_raw, groups, feature_names


def compute_ece(y_true, y_pred_lower, y_pred_upper, n_bins=5):
    """Compute Expected Calibration Error"""
    in_interval = (y_true >= y_pred_lower) & (y_true <= y_pred_upper)
    
    interval_widths = y_pred_upper - y_pred_lower
    sorted_indices = np.argsort(interval_widths)
    bin_size = max(1, len(sorted_indices) // n_bins)
    
    ece = 0.0
    for i in range(n_bins):
        start_idx = i * bin_size
        end_idx = min((i + 1) * bin_size, len(sorted_indices))
        bin_indices = sorted_indices[start_idx:end_idx]
        
        if len(bin_indices) == 0:
            continue
        
        empirical_coverage = in_interval[bin_indices].mean()
        expected_coverage = 0.90
        
        ece += np.abs(empirical_coverage - expected_coverage) * len(bin_indices) / len(sorted_indices)
    
    return ece


def train_naive_baselines(X, y, groups, n_folds=5):
    """Train naive baselines"""
    print("\n[BASELINES] Training naive baselines...")
    
    kf = GroupKFold(n_splits=n_folds)
    
    # Mean
    mean_preds = []
    mean_model = DummyRegressor(strategy='mean')
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X, y, groups)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test = X[test_idx]
        
        mean_model.fit(X_train, y_train)
        preds = mean_model.predict(X_test)
        mean_preds.extend(list(zip(test_idx, preds)))
    
    mean_preds = sorted(mean_preds, key=lambda x: x[0])
    mean_preds_arr = np.array([p[1] for p in mean_preds])
    
    # Median
    median_preds = []
    median_model = DummyRegressor(strategy='median')
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X, y, groups)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test = X[test_idx]
        
        median_model.fit(X_train, y_train)
        preds = median_model.predict(X_test)
        median_preds.extend(list(zip(test_idx, preds)))
    
    median_preds = sorted(median_preds, key=lambda x: x[0])
    median_preds_arr = np.array([p[1] for p in median_preds])
    
    # Metrics
    mean_mae = mean_absolute_error(y, mean_preds_arr)
    mean_r2 = r2_score(y, mean_preds_arr)
    
    median_mae = mean_absolute_error(y, median_preds_arr)
    median_r2 = r2_score(y, median_preds_arr)
    
    print(f"  [MEAN] MAE: {mean_mae:.3f}, R2: {mean_r2:.3f}")
    print(f"  [MEDIAN] MAE: {median_mae:.3f}, R2: {median_r2:.3f}")
    
    return {
        'mean_mae': mean_mae,
        'mean_r2': mean_r2,
        'median_mae': median_mae,
        'median_r2': median_r2
    }


def train_gbdt_central(X, y, groups, n_folds=5):
    """Train central GBDT model (point estimate)"""
    print("\n[GBDT-CENTRAL] Training GradientBoostingRegressor...")
    
    kf = GroupKFold(n_splits=n_folds)
    
    all_predictions = []
    fold_metrics = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X, y, groups)):
        print(f"\n  [FOLD {fold_idx + 1}/{n_folds}]")
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        print(f"    Train: {len(X_train)}, Test: {len(X_test)}")
        
        # GBDT model (squared error for central estimate)
        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            loss='squared_error',
            random_state=SEED + fold_idx
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        fold_metrics.append({
            'fold': fold_idx + 1,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        })
        
        print(f"    MAE: {mae:.3f}, R2: {r2:.3f}, RMSE: {rmse:.3f}")
        
        for i, test_i in enumerate(test_idx):
            all_predictions.append({
                'fold': fold_idx + 1,
                'idx': test_i,
                'y_true': y_test[i],
                'y_pred_central': y_pred[i]
            })
    
    # Aggregate
    overall_metrics = {
        'mae': np.mean([f['mae'] for f in fold_metrics]),
        'rmse': np.mean([f['rmse'] for f in fold_metrics]),
        'r2': np.mean([f['r2'] for f in fold_metrics]),
        'mae_std': np.std([f['mae'] for f in fold_metrics]),
        'r2_std': np.std([f['r2'] for f in fold_metrics])
    }
    
    print(f"\n  [OVERALL] MAE: {overall_metrics['mae']:.3f} ± {overall_metrics['mae_std']:.3f}")
    print(f"  [OVERALL] R2: {overall_metrics['r2']:.3f} ± {overall_metrics['r2_std']:.3f}")
    
    return fold_metrics, overall_metrics, all_predictions


def train_gbdt_quantiles(X, y, groups, n_folds=5):
    """Train GBDT quantile models (q=0.1, 0.9 for stability with N=97)"""
    print("\n[GBDT-QUANTILES] Training quantile GBDTs (q=0.1, 0.9)...")
    
    kf = GroupKFold(n_splits=n_folds)
    
    all_quantile_preds = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X, y, groups)):
        print(f"  [FOLD {fold_idx + 1}/{n_folds}] Training quantiles...")
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        # Train q=0.1
        model_q10 = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            loss='quantile',
            alpha=0.10,
            random_state=SEED + fold_idx
        )
        model_q10.fit(X_train, y_train)
        y_pred_q10 = model_q10.predict(X_test)
        
        # Train q=0.9
        model_q90 = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            loss='quantile',
            alpha=0.90,
            random_state=SEED + fold_idx
        )
        model_q90.fit(X_train, y_train)
        y_pred_q90 = model_q90.predict(X_test)
        
        # Ensure monotonicity
        y_pred_q90 = np.maximum(y_pred_q90, y_pred_q10)
        
        for i, test_i in enumerate(test_idx):
            all_quantile_preds.append({
                'fold': fold_idx + 1,
                'idx': test_i,
                'y_pred_q10': y_pred_q10[i],
                'y_pred_q90': y_pred_q90[i]
            })
    
    return all_quantile_preds


def apply_cqr_calibration(predictions_df, alpha=0.10):
    """
    Apply Conformalized Quantile Regression (CQR) for calibration
    Simple version: adjust intervals based on empirical coverage
    """
    print("\n[CQR] Applying Conformal Prediction calibration...")
    
    # Compute residuals
    y_true = predictions_df['y_true'].values
    y_q10 = predictions_df['y_pred_q10'].values
    y_q90 = predictions_df['y_pred_q90'].values
    
    # Conformity scores (how far outside intervals)
    lower_residuals = y_q10 - y_true
    upper_residuals = y_true - y_q90
    
    conformity_scores = np.maximum(lower_residuals, upper_residuals)
    
    # Compute quantile of conformity scores for calibration
    q_level = np.ceil((1 - alpha) * (len(conformity_scores) + 1)) / len(conformity_scores)
    q_conformity = np.quantile(conformity_scores, q_level)
    
    print(f"  [INFO] Conformity quantile (q={q_level:.3f}): {q_conformity:.3f}")
    
    # Adjust intervals
    predictions_df['y_pred_q10_cqr'] = predictions_df['y_pred_q10'] - q_conformity
    predictions_df['y_pred_q90_cqr'] = predictions_df['y_pred_q90'] + q_conformity
    
    # Compute coverage
    in_interval = (y_true >= predictions_df['y_pred_q10_cqr'].values) & \
                  (y_true <= predictions_df['y_pred_q90_cqr'].values)
    coverage = in_interval.mean()
    
    # ECE
    ece = compute_ece(
        y_true,
        predictions_df['y_pred_q10_cqr'].values,
        predictions_df['y_pred_q90_cqr'].values,
        n_bins=5
    )
    
    print(f"  [INFO] Post-CQR Coverage: {coverage:.3f} (target: 0.90)")
    print(f"  [INFO] Post-CQR ECE: {ece:.3f}")
    
    return predictions_df, coverage, ece


def inverse_transform_log(y_log):
    """Inverse log1p transform"""
    return np.expm1(y_log)


def check_acceptance_criteria_relaxed(overall_metrics, baseline_metrics, coverage, ece):
    """Check v1.2.5 relaxed acceptance criteria"""
    print("\n" + "="*70)
    print("ACCEPTANCE CRITERIA CHECK (v1.2.5 RELAXED)")
    print("="*70)
    
    criteria = {}
    
    # R² >= 0.10 (relaxed)
    criteria['r2'] = {
        'value': overall_metrics['r2'],
        'target': CRITERIA_RELAXED['r2_min'],
        'pass': overall_metrics['r2'] >= CRITERIA_RELAXED['r2_min']
    }
    
    # MAE < 7.810
    criteria['mae'] = {
        'value': overall_metrics['mae'],
        'target': CRITERIA_RELAXED['mae_max'],
        'pass': overall_metrics['mae'] < CRITERIA_RELAXED['mae_max']
    }
    
    # ECE <= 0.18 (relaxed)
    criteria['ece'] = {
        'value': ece,
        'target': CRITERIA_RELAXED['ece_max'],
        'pass': ece <= CRITERIA_RELAXED['ece_max']
    }
    
    # Coverage [0.85, 0.95]
    criteria['coverage'] = {
        'value': coverage,
        'target_range': [CRITERIA_RELAXED['coverage_min'], CRITERIA_RELAXED['coverage_max']],
        'pass': CRITERIA_RELAXED['coverage_min'] <= coverage <= CRITERIA_RELAXED['coverage_max']
    }
    
    # Beat baseline >= 5% (relaxed)
    best_naive_mae = min(baseline_metrics['mean_mae'], baseline_metrics['median_mae'])
    mae_improvement = (best_naive_mae - overall_metrics['mae']) / best_naive_mae
    
    criteria['beat_baseline'] = {
        'value': mae_improvement,
        'target': CRITERIA_RELAXED['beat_baseline_pct'],
        'pass': mae_improvement >= CRITERIA_RELAXED['beat_baseline_pct'],
        'best_naive_mae': best_naive_mae
    }
    
    # Overall
    all_pass = all(c['pass'] for c in criteria.values())
    
    print(f"\n1. R² >= {CRITERIA_RELAXED['r2_min']}:")
    print(f"   Value: {criteria['r2']['value']:.3f}")
    print(f"   Status: {'PASS' if criteria['r2']['pass'] else 'FAIL'}")
    
    print(f"\n2. MAE < {CRITERIA_RELAXED['mae_max']}:")
    print(f"   Value: {criteria['mae']['value']:.3f}")
    print(f"   Status: {'PASS' if criteria['mae']['pass'] else 'FAIL'}")
    
    print(f"\n3. ECE <= {CRITERIA_RELAXED['ece_max']}:")
    print(f"   Value: {criteria['ece']['value']:.3f}")
    print(f"   Status: {'PASS' if criteria['ece']['pass'] else 'FAIL'}")
    
    print(f"\n4. Coverage [{CRITERIA_RELAXED['coverage_min']}, {CRITERIA_RELAXED['coverage_max']}]:")
    print(f"   Value: {criteria['coverage']['value']:.3f}")
    print(f"   Status: {'PASS' if criteria['coverage']['pass'] else 'FAIL'}")
    
    print(f"\n5. Beat baseline (>={CRITERIA_RELAXED['beat_baseline_pct']*100:.0f}% improvement):")
    print(f"   Naive MAE: {best_naive_mae:.3f}")
    print(f"   Model MAE: {overall_metrics['mae']:.3f}")
    print(f"   Improvement: {mae_improvement*100:.1f}%")
    print(f"   Status: {'PASS' if criteria['beat_baseline']['pass'] else 'FAIL'}")
    
    print(f"\n{'='*70}")
    print(f"OVERALL: {'ALL PASS - GO FOR RELEASE v1.2.5' if all_pass else 'FAIL - BLOCKED'}")
    print(f"{'='*70}")
    
    return criteria, all_pass


def main():
    print("="*70)
    print("v1.3.1 TRAINING — GBDT + CQR (Fallback v1.2.5)")
    print("="*70)
    
    PROJECT_ROOT = Path(__file__).parent.parent
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    OUTPUTS_DIR = PROJECT_ROOT / "outputs"
    
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    
    TRAIN_CSV = PROCESSED_DIR / "training_table_v1_3_1.csv"
    
    # Load
    df = load_data(TRAIN_CSV)
    
    # Build features
    X, y_log, y_raw, groups, feature_names = build_features(df)
    
    # Baselines
    baseline_metrics = train_naive_baselines(X, y_log, groups)
    
    # GBDT central
    fold_metrics_central, overall_central, predictions_central = train_gbdt_central(X, y_log, groups)
    
    # GBDT quantiles
    predictions_quantiles = train_gbdt_quantiles(X, y_log, groups)
    
    # Merge predictions
    df_preds = pd.DataFrame(predictions_central)
    df_quant = pd.DataFrame(predictions_quantiles)
    df_preds = df_preds.merge(df_quant, on=['fold', 'idx'], how='left')
    df_preds = df_preds.sort_values('idx')
    
    # CQR calibration
    df_preds, coverage_cqr, ece_cqr = apply_cqr_calibration(df_preds, alpha=0.10)
    
    # Check criteria (relaxed)
    criteria, all_pass = check_acceptance_criteria_relaxed(
        overall_central, baseline_metrics, coverage_cqr, ece_cqr
    )
    
    # Save outputs
    print("\n[SAVE] Saving outputs...")
    
    # Predictions CSV
    pred_csv_path = OUTPUTS_DIR / "cv_predictions_cqr_v1_3_1.csv"
    df_preds.to_csv(pred_csv_path, index=False)
    print(f"  [SUCCESS] {pred_csv_path}")
    
    # Metrics JSON
    metrics_dict = {
        'version': 'v1.3.1 (fallback v1.2.5)',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'n_samples': len(X),
        'n_features': X.shape[1],
        'n_folds': 5,
        'seed': SEED,
        'target_transform': 'log1p(contrast_normalized)',
        'relaxed_criteria': CRITERIA_RELAXED,
        'baseline_metrics': baseline_metrics,
        'gbdt_central': {
            'overall': overall_central,
            'fold_details': fold_metrics_central
        },
        'cqr_calibration': {
            'coverage': float(coverage_cqr),
            'ece': float(ece_cqr)
        },
        'acceptance_criteria': {
            k: {
                'value': float(v['value']) if isinstance(v['value'], (int, float, np.number)) else v['value'],
                'target': v.get('target', v.get('target_range')),
                'pass': bool(v['pass'])
            }
            for k, v in criteria.items()
        },
        'decision': 'GO' if all_pass else 'NO_GO'
    }
    
    metrics_json_path = OUTPUTS_DIR / "cv_metrics_cqr_v1_3_1.json"
    with open(metrics_json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"  [SUCCESS] {metrics_json_path}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    
    return all_pass, overall_central, baseline_metrics, criteria, coverage_cqr, ece_cqr


if __name__ == "__main__":
    all_pass, overall, baselines, criteria, coverage, ece = main()
    
    if all_pass:
        print("\n[GO] All criteria PASS - ready for release v1.2.5")
        exit(0)
    else:
        print("\n[NO-GO] Some criteria FAIL - BLOCKED report required")
        exit(1)

