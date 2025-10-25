#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v1.2.5 RETRY: RandomForest + CQR with Balanced Splits
NO R² RELAX - Strict criteria
Metrics on ORIGINAL SCALE (inverse log transform)
"""

import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import warnings
warnings.filterwarnings('ignore')

# Seed
SEED = 1337
np.random.seed(SEED)

# Strict criteria (NO RELAX)
CRITERIA_STRICT = {
    'r2_min': 0.10,
    'mae_max': 7.810,
    'ece_max': 0.18,
    'coverage_min': 0.85,
    'coverage_max': 0.95,
    'beat_baseline_pct': 0.10
}


def load_data():
    """Load training data v1.3.1"""
    PROJECT_ROOT = Path(__file__).parent.parent
    csv_path = PROJECT_ROOT / "data" / "processed" / "training_table_v1_3_1.csv"
    
    print(f"\n[LOAD] Reading {csv_path.name}...")
    df = pd.read_csv(csv_path)
    print(f"  [INFO] Shape: {df.shape}")
    return df


def aggregate_rare_families(df, min_samples=3):
    """Aggregate families with N<min_samples into 'Other'"""
    print(f"\n[AGGREGATE] Aggregating families with N<{min_samples}...")
    
    family_counts = df['family'].value_counts()
    rare_families = family_counts[family_counts < min_samples].index.tolist()
    
    print(f"  [INFO] Total families: {len(family_counts)}")
    print(f"  [INFO] Families with N>={min_samples}: {sum(family_counts >= min_samples)}")
    print(f"  [INFO] Rare families (N<{min_samples}): {len(rare_families)}")
    
    df['family_original'] = df['family'].copy()
    df.loc[df['family'].isin(rare_families), 'family'] = 'Other'
    
    family_counts_new = df['family'].value_counts()
    print(f"  [SUCCESS] Aggregated families: {len(family_counts_new)}")
    print(f"  [INFO] 'Other' count: {family_counts_new.get('Other', 0)}")
    
    return df


def create_balanced_folds(df, n_splits=5):
    """
    Create balanced GroupKFold splits
    Ensure each fold has diverse families, no single family domination
    """
    print(f"\n[SPLITS] Creating balanced {n_splits}-fold splits...")
    
    families = df['family'].unique()
    family_counts = df['family'].value_counts().to_dict()
    
    # Sort families by count (descending)
    sorted_families = sorted(families, key=lambda x: family_counts[x], reverse=True)
    
    # Initialize folds
    folds = [[] for _ in range(n_splits)]
    fold_sizes = [0] * n_splits
    
    # Greedy assignment: assign each family to the fold with smallest current size
    for family in sorted_families:
        family_size = family_counts[family]
        
        # Find fold with smallest size
        min_fold_idx = np.argmin(fold_sizes)
        
        # Assign family to that fold
        folds[min_fold_idx].append(family)
        fold_sizes[min_fold_idx] += family_size
    
    print(f"  [INFO] Fold sizes: {fold_sizes}")
    
    # Create fold assignments
    fold_assignments = np.zeros(len(df), dtype=int)
    for fold_idx, families_in_fold in enumerate(folds):
        mask = df['family'].isin(families_in_fold)
        fold_assignments[mask] = fold_idx
    
    print(f"  [SUCCESS] Balanced folds created")
    
    # Validate
    for fold_idx in range(n_splits):
        mask = fold_assignments == fold_idx
        families_in_fold = df[mask]['family'].unique()
        print(f"    Fold {fold_idx}: n={mask.sum()}, families={len(families_in_fold)}")
    
    return fold_assignments


def build_features(df):
    """Build feature matrix"""
    print("\n[FEATURES] Building feature matrix...")
    
    # Target (log)
    y_log = df['target_contrast_log'].values
    y_raw = df['contrast_normalized_raw'].values
    
    # Numerical
    numerical_features = [
        'temperature_K', 'pH', 'is_biosensor',
        'excitation_nm', 'emission_nm', 'stokes_shift_nm'
    ]
    
    # Categorical
    categorical_features = ['family', 'spectral_region', 'context_type']
    
    available_num = [f for f in numerical_features if f in df.columns]
    available_cat = [f for f in categorical_features if f in df.columns]
    
    # Build X_num (with imputation)
    X_num = df[available_num].fillna(df[available_num].median()).values
    
    # One-hot encode
    X_cat_list = []
    cat_feature_names = []
    
    for cat_col in available_cat:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_cat_encoded = encoder.fit_transform(df[[cat_col]])
        X_cat_list.append(X_cat_encoded)
        
        cat_names = [f"{cat_col}_{cat}" for cat in encoder.categories_[0]]
        cat_feature_names.extend(cat_names)
    
    if X_cat_list:
        X_cat = np.hstack(X_cat_list)
        X = np.hstack([X_num, X_cat])
    else:
        X = X_num
    
    feature_names = available_num + cat_feature_names
    
    print(f"  [SUCCESS] X shape: {X.shape}")
    print(f"  [INFO] Features: {len(feature_names)}")
    
    return X, y_log, y_raw, feature_names


def train_naive_baselines(X, y_raw, fold_assignments):
    """Train naive baselines on ORIGINAL scale"""
    print("\n[BASELINES] Training naive baselines (original scale)...")
    
    n_folds = len(np.unique(fold_assignments))
    
    # Mean
    mean_preds = []
    for fold_idx in range(n_folds):
        train_mask = fold_assignments != fold_idx
        test_mask = fold_assignments == fold_idx
        
        mean_val = y_raw[train_mask].mean()
        mean_preds.extend([(i, mean_val) for i in np.where(test_mask)[0]])
    
    mean_preds = sorted(mean_preds, key=lambda x: x[0])
    mean_preds_arr = np.array([p[1] for p in mean_preds])
    
    # Median
    median_preds = []
    for fold_idx in range(n_folds):
        train_mask = fold_assignments != fold_idx
        test_mask = fold_assignments == fold_idx
        
        median_val = np.median(y_raw[train_mask])
        median_preds.extend([(i, median_val) for i in np.where(test_mask)[0]])
    
    median_preds = sorted(median_preds, key=lambda x: x[0])
    median_preds_arr = np.array([p[1] for p in median_preds])
    
    # Metrics
    mean_mae = mean_absolute_error(y_raw, mean_preds_arr)
    mean_r2 = r2_score(y_raw, mean_preds_arr)
    
    median_mae = mean_absolute_error(y_raw, median_preds_arr)
    median_r2 = r2_score(y_raw, median_preds_arr)
    
    print(f"  [MEAN] MAE: {mean_mae:.3f}, R2: {mean_r2:.3f}")
    print(f"  [MEDIAN] MAE: {median_mae:.3f}, R2: {median_r2:.3f}")
    
    return {
        'mean_mae': mean_mae,
        'mean_r2': mean_r2,
        'median_mae': median_mae,
        'median_r2': median_r2
    }


def train_randomforest(X, y_log, y_raw, fold_assignments, feature_names):
    """Train RandomForest on log, evaluate on original scale"""
    print("\n[RANDOMFOREST] Training RandomForestRegressor...")
    
    n_folds = len(np.unique(fold_assignments))
    
    all_predictions = []
    fold_metrics_log = []
    fold_metrics_orig = []
    
    for fold_idx in range(n_folds):
        print(f"\n  [FOLD {fold_idx + 1}/{n_folds}]")
        
        train_mask = fold_assignments != fold_idx
        test_mask = fold_assignments == fold_idx
        
        X_train, y_train_log = X[train_mask], y_log[train_mask]
        X_test, y_test_log, y_test_raw = X[test_mask], y_log[test_mask], y_raw[test_mask]
        
        print(f"    Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Train RF on log-scale
        rf = RandomForestRegressor(
            n_estimators=1000,
            max_depth=None,
            min_samples_leaf=2,
            oob_score=True,
            random_state=SEED + fold_idx,
            n_jobs=-1
        )
        
        rf.fit(X_train, y_train_log)
        
        # Predict log
        y_pred_log = rf.predict(X_test)
        
        # Inverse transform to original scale
        y_pred_raw = np.expm1(y_pred_log)
        
        # Metrics log-space
        mae_log = mean_absolute_error(y_test_log, y_pred_log)
        r2_log = r2_score(y_test_log, y_pred_log)
        
        # Metrics original scale
        mae_orig = mean_absolute_error(y_test_raw, y_pred_raw)
        rmse_orig = np.sqrt(mean_squared_error(y_test_raw, y_pred_raw))
        r2_orig = r2_score(y_test_raw, y_pred_raw)
        
        fold_metrics_log.append({
            'fold': fold_idx + 1,
            'mae': mae_log,
            'r2': r2_log
        })
        
        fold_metrics_orig.append({
            'fold': fold_idx + 1,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'mae': mae_orig,
            'rmse': rmse_orig,
            'r2': r2_orig,
            'oob_score': rf.oob_score_ if hasattr(rf, 'oob_score_') else None
        })
        
        print(f"    [LOG] MAE: {mae_log:.3f}, R2: {r2_log:.3f}")
        print(f"    [ORIG] MAE: {mae_orig:.3f}, R2: {r2_orig:.3f}, RMSE: {rmse_orig:.3f}")
        if rf.oob_score_:
            print(f"    [OOB] Score: {rf.oob_score_:.3f}")
        
        for i, test_i in enumerate(np.where(test_mask)[0]):
            all_predictions.append({
                'fold': fold_idx + 1,
                'idx': test_i,
                'y_true_log': y_test_log[i],
                'y_true_raw': y_test_raw[i],
                'y_pred_log': y_pred_log[i],
                'y_pred_raw': y_pred_raw[i]
            })
    
    # Aggregate original scale
    overall_orig = {
        'mae': np.mean([f['mae'] for f in fold_metrics_orig]),
        'rmse': np.mean([f['rmse'] for f in fold_metrics_orig]),
        'r2': np.mean([f['r2'] for f in fold_metrics_orig]),
        'mae_std': np.std([f['mae'] for f in fold_metrics_orig]),
        'r2_std': np.std([f['r2'] for f in fold_metrics_orig])
    }
    
    # Aggregate log scale
    overall_log = {
        'mae': np.mean([f['mae'] for f in fold_metrics_log]),
        'r2': np.mean([f['r2'] for f in fold_metrics_log])
    }
    
    print(f"\n  [OVERALL-ORIG] MAE: {overall_orig['mae']:.3f} ± {overall_orig['mae_std']:.3f}")
    print(f"  [OVERALL-ORIG] R2: {overall_orig['r2']:.3f} ± {overall_orig['r2_std']:.3f}")
    print(f"  [OVERALL-LOG] MAE: {overall_log['mae']:.3f}, R2: {overall_log['r2']:.3f}")
    
    return fold_metrics_orig, overall_orig, overall_log, all_predictions


def train_quantiles_and_cqr(X, y_log, y_raw, fold_assignments, predictions_df):
    """Train GBDT quantiles + CQR, evaluate on original scale"""
    print("\n[QUANTILES+CQR] Training GBDT quantiles + CQR...")
    
    n_folds = len(np.unique(fold_assignments))
    
    all_quantile_preds = []
    
    for fold_idx in range(n_folds):
        print(f"  [FOLD {fold_idx + 1}/{n_folds}] Training quantiles...")
        
        train_mask = fold_assignments != fold_idx
        test_mask = fold_assignments == fold_idx
        
        X_train, y_train_log = X[train_mask], y_log[train_mask]
        X_test, y_test_log, y_test_raw = X[test_mask], y_log[test_mask], y_raw[test_mask]
        
        # q=0.1
        model_q10 = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            loss='quantile',
            alpha=0.10,
            random_state=SEED + fold_idx
        )
        model_q10.fit(X_train, y_train_log)
        y_pred_q10_log = model_q10.predict(X_test)
        
        # q=0.9
        model_q90 = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            loss='quantile',
            alpha=0.90,
            random_state=SEED + fold_idx
        )
        model_q90.fit(X_train, y_train_log)
        y_pred_q90_log = model_q90.predict(X_test)
        
        # Ensure monotonicity (log-space)
        y_pred_q90_log = np.maximum(y_pred_q90_log, y_pred_q10_log)
        
        # Inverse transform to original scale
        y_pred_q10_raw = np.expm1(y_pred_q10_log)
        y_pred_q90_raw = np.expm1(y_pred_q90_log)
        
        for i, test_i in enumerate(np.where(test_mask)[0]):
            all_quantile_preds.append({
                'fold': fold_idx + 1,
                'idx': test_i,
                'y_pred_q10_log': y_pred_q10_log[i],
                'y_pred_q90_log': y_pred_q90_log[i],
                'y_pred_q10_raw': y_pred_q10_raw[i],
                'y_pred_q90_raw': y_pred_q90_raw[i]
            })
    
    # Merge with RF predictions
    df_quant = pd.DataFrame(all_quantile_preds)
    predictions_df = predictions_df.merge(df_quant, on=['fold', 'idx'], how='left')
    
    # Apply CQR on ORIGINAL scale
    print("\n[CQR] Applying Conformal Prediction (original scale)...")
    
    y_true_raw = predictions_df['y_true_raw'].values
    y_q10_raw = predictions_df['y_pred_q10_raw'].values
    y_q90_raw = predictions_df['y_pred_q90_raw'].values
    
    # Conformity scores (original scale)
    lower_residuals = y_q10_raw - y_true_raw
    upper_residuals = y_true_raw - y_q90_raw
    conformity_scores = np.maximum(lower_residuals, upper_residuals)
    
    # Calibration quantile
    alpha = 0.10  # for 90% coverage
    q_level = np.ceil((1 - alpha) * (len(conformity_scores) + 1)) / len(conformity_scores)
    q_conformity = np.quantile(conformity_scores, q_level)
    
    print(f"  [INFO] Conformity quantile: {q_conformity:.3f}")
    
    # Adjust intervals (original scale)
    predictions_df['y_pred_q10_cqr'] = predictions_df['y_pred_q10_raw'] - q_conformity
    predictions_df['y_pred_q90_cqr'] = predictions_df['y_pred_q90_raw'] + q_conformity
    
    # Clip to positive (contrast cannot be negative)
    predictions_df['y_pred_q10_cqr'] = predictions_df['y_pred_q10_cqr'].clip(lower=0)
    
    # Coverage & ECE (original scale)
    in_interval = (y_true_raw >= predictions_df['y_pred_q10_cqr'].values) & \
                  (y_true_raw <= predictions_df['y_pred_q90_cqr'].values)
    coverage = in_interval.mean()
    
    ece = compute_ece(
        y_true_raw,
        predictions_df['y_pred_q10_cqr'].values,
        predictions_df['y_pred_q90_cqr'].values,
        n_bins=5
    )
    
    print(f"  [INFO] Coverage (original scale): {coverage:.3f}")
    print(f"  [INFO] ECE (original scale): {ece:.3f}")
    
    return predictions_df, coverage, ece


def compute_ece(y_true, y_pred_lower, y_pred_upper, n_bins=5):
    """Compute ECE"""
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


def check_criteria_strict(overall_orig, baseline_metrics, coverage, ece):
    """Check STRICT criteria (NO RELAX)"""
    print("\n" + "="*70)
    print("ACCEPTANCE CRITERIA CHECK (v1.2.5 STRICT - NO RELAX)")
    print("="*70)
    
    criteria = {}
    
    # R² >= 0.10
    criteria['r2'] = {
        'value': overall_orig['r2'],
        'target': CRITERIA_STRICT['r2_min'],
        'pass': overall_orig['r2'] >= CRITERIA_STRICT['r2_min']
    }
    
    # MAE < 7.810
    criteria['mae'] = {
        'value': overall_orig['mae'],
        'target': CRITERIA_STRICT['mae_max'],
        'pass': overall_orig['mae'] < CRITERIA_STRICT['mae_max']
    }
    
    # ECE <= 0.18
    criteria['ece'] = {
        'value': ece,
        'target': CRITERIA_STRICT['ece_max'],
        'pass': ece <= CRITERIA_STRICT['ece_max']
    }
    
    # Coverage [0.85, 0.95]
    criteria['coverage'] = {
        'value': coverage,
        'target_range': [CRITERIA_STRICT['coverage_min'], CRITERIA_STRICT['coverage_max']],
        'pass': CRITERIA_STRICT['coverage_min'] <= coverage <= CRITERIA_STRICT['coverage_max']
    }
    
    # Beat baseline >= 10%
    best_naive_mae = min(baseline_metrics['mean_mae'], baseline_metrics['median_mae'])
    mae_improvement = (best_naive_mae - overall_orig['mae']) / best_naive_mae
    
    criteria['beat_baseline'] = {
        'value': mae_improvement,
        'target': CRITERIA_STRICT['beat_baseline_pct'],
        'pass': mae_improvement >= CRITERIA_STRICT['beat_baseline_pct'],
        'best_naive_mae': best_naive_mae
    }
    
    all_pass = all(c['pass'] for c in criteria.values())
    
    print(f"\n1. R² >= {CRITERIA_STRICT['r2_min']} (original scale):")
    print(f"   Value: {criteria['r2']['value']:.3f}")
    print(f"   Status: {'PASS' if criteria['r2']['pass'] else 'FAIL'}")
    
    print(f"\n2. MAE < {CRITERIA_STRICT['mae_max']} (original scale):")
    print(f"   Value: {criteria['mae']['value']:.3f}")
    print(f"   Status: {'PASS' if criteria['mae']['pass'] else 'FAIL'}")
    
    print(f"\n3. ECE <= {CRITERIA_STRICT['ece_max']}:")
    print(f"   Value: {criteria['ece']['value']:.3f}")
    print(f"   Status: {'PASS' if criteria['ece']['pass'] else 'FAIL'}")
    
    print(f"\n4. Coverage [{CRITERIA_STRICT['coverage_min']}, {CRITERIA_STRICT['coverage_max']}]:")
    print(f"   Value: {criteria['coverage']['value']:.3f}")
    print(f"   Status: {'PASS' if criteria['coverage']['pass'] else 'FAIL'}")
    
    print(f"\n5. Beat baseline (>={CRITERIA_STRICT['beat_baseline_pct']*100:.0f}%):")
    print(f"   Naive MAE: {best_naive_mae:.3f}")
    print(f"   RF MAE: {overall_orig['mae']:.3f}")
    print(f"   Improvement: {mae_improvement*100:.1f}%")
    print(f"   Status: {'PASS' if criteria['beat_baseline']['pass'] else 'FAIL'}")
    
    print(f"\n{'='*70}")
    if all_pass:
        print(f"OVERALL: ALL PASS (6/6) - GO FOR RELEASE v1.2.5")
    else:
        print(f"OVERALL: FAIL ({sum(c['pass'] for c in criteria.values())}/6 PASS) - BLOCKED")
    print(f"{'='*70}")
    
    return criteria, all_pass


def generate_figures(predictions_df, feature_names, fold_metrics_orig, overall_orig):
    """Generate diagnostic figures"""
    print("\n[FIGURES] Generating diagnostic plots...")
    
    PROJECT_ROOT = Path(__file__).parent.parent
    FIGURES_DIR = PROJECT_ROOT / "figures_v1_2_5_retry"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Pred vs True (original scale)
    plt.figure(figsize=(8, 6))
    plt.scatter(predictions_df['y_true_raw'], predictions_df['y_pred_raw'], alpha=0.6, s=50)
    plt.plot([0, predictions_df['y_true_raw'].max()], [0, predictions_df['y_true_raw'].max()], 
             'r--', linewidth=2, label='Perfect prediction')
    plt.xlabel('True Contrast (original scale)', fontsize=12)
    plt.ylabel('Predicted Contrast (original scale)', fontsize=12)
    plt.title(f'Predicted vs True (RF)\nR²={overall_orig["r2"]:.3f}, MAE={overall_orig["mae"]:.2f}', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pred_vs_true.png", dpi=150)
    plt.close()
    print(f"  [SUCCESS] {FIGURES_DIR}/pred_vs_true.png")
    
    # 2. Interval coverage
    plt.figure(figsize=(10, 6))
    x_plot = np.arange(len(predictions_df))
    plt.fill_between(x_plot, 
                     predictions_df['y_pred_q10_cqr'].values, 
                     predictions_df['y_pred_q90_cqr'].values,
                     alpha=0.3, label='90% PI (CQR)', color='blue')
    plt.scatter(x_plot, predictions_df['y_true_raw'].values, 
                s=20, color='red', alpha=0.6, label='True values')
    plt.xlabel('Sample index', fontsize=12)
    plt.ylabel('Contrast (original scale)', fontsize=12)
    plt.title('90% Prediction Intervals (CQR calibrated)', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "interval_coverage.png", dpi=150)
    plt.close()
    print(f"  [SUCCESS] {FIGURES_DIR}/interval_coverage.png")
    
    # 3. R² distribution by fold
    plt.figure(figsize=(8, 6))
    fold_r2s = [f['r2'] for f in fold_metrics_orig]
    plt.bar(range(1, len(fold_r2s) + 1), fold_r2s, color='steelblue', alpha=0.7)
    plt.axhline(0.10, color='red', linestyle='--', linewidth=2, label='Target R²=0.10')
    plt.axhline(overall_orig['r2'], color='green', linestyle='--', linewidth=2, label=f'Mean R²={overall_orig["r2"]:.3f}')
    plt.xlabel('Fold', fontsize=12)
    plt.ylabel('R² (original scale)', fontsize=12)
    plt.title('R² Distribution by Fold', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fold_r2_distribution.png", dpi=150)
    plt.close()
    print(f"  [SUCCESS] {FIGURES_DIR}/fold_r2_distribution.png")
    
    print(f"\n  [INFO] All figures saved to {FIGURES_DIR}/")


def main():
    print("="*70)
    print("v1.2.5 RETRY — RandomForest + CQR (STRICT CRITERIA)")
    print("="*70)
    
    PROJECT_ROOT = Path(__file__).parent.parent
    OUTPUTS_DIR = PROJECT_ROOT / "outputs"
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load
    df = load_data()
    
    # Aggregate rare families
    df = aggregate_rare_families(df, min_samples=3)
    
    # Create balanced folds
    fold_assignments = create_balanced_folds(df, n_splits=5)
    df['fold'] = fold_assignments
    
    # Build features
    X, y_log, y_raw, feature_names = build_features(df)
    
    # Baselines (original scale)
    baseline_metrics = train_naive_baselines(X, y_raw, fold_assignments)
    
    # RandomForest
    fold_metrics_orig, overall_orig, overall_log, predictions = train_randomforest(
        X, y_log, y_raw, fold_assignments, feature_names
    )
    
    # Quantiles + CQR
    df_preds = pd.DataFrame(predictions).sort_values('idx')
    df_preds, coverage, ece = train_quantiles_and_cqr(
        X, y_log, y_raw, fold_assignments, df_preds
    )
    
    # Check criteria (STRICT)
    criteria, all_pass = check_criteria_strict(overall_orig, baseline_metrics, coverage, ece)
    
    # Generate figures
    generate_figures(df_preds, feature_names, fold_metrics_orig, overall_orig)
    
    # Save outputs
    print("\n[SAVE] Saving outputs...")
    
    # Predictions
    pred_csv_path = OUTPUTS_DIR / "cv_predictions_cqr_v1_2_5_retry.csv"
    df_preds.to_csv(pred_csv_path, index=False)
    print(f"  [SUCCESS] {pred_csv_path}")
    
    # Metrics
    metrics_dict = {
        'version': 'v1.2.5 RETRY (RandomForest + CQR, strict)',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'n_samples': len(X),
        'n_features': X.shape[1],
        'n_folds': 5,
        'seed': SEED,
        'target_transform': 'log1p (training only)',
        'metrics_scale': 'ORIGINAL (inverse log for reporting)',
        'splits': 'Custom balanced GroupKFold (families N<3 aggregated)',
        'strict_criteria': CRITERIA_STRICT,
        'baseline_metrics_original': baseline_metrics,
        'randomforest_original': {
            'overall': overall_orig,
            'fold_details': fold_metrics_orig
        },
        'randomforest_log': overall_log,
        'cqr_calibration_original': {
            'coverage': float(coverage),
            'ece': float(ece)
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
    
    metrics_json_path = OUTPUTS_DIR / "cv_metrics_v1_2_5_retry.json"
    with open(metrics_json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"  [SUCCESS] {metrics_json_path}")
    
    print("\n" + "="*70)
    print("v1.2.5 RETRY COMPLETE")
    print("="*70)
    
    # Final status report
    print("\n" + "="*70)
    print("STATUS REPORT - v1.2.5 RETRY (RF + CQR, splits corriges)")
    print("="*70)
    
    families_aggregated = (df['family'] == 'Other').sum()
    
    print(f"\nData: N_total={len(df)} ; N_utiles={len(df)}")
    print(f"      Families={len(df['family'].unique())} (N<3 agregees={families_aggregated})")
    print(f"Splits: Custom GroupKFold balanced, seed={SEED}")
    print(f"\nMetrics (original scale, CV mean±std):")
    print(f"  - R² = {overall_orig['r2']:.3f} ± {overall_orig['r2_std']:.3f}  (target >=0.10) -> {'PASS' if criteria['r2']['pass'] else 'FAIL'}")
    print(f"  - MAE = {overall_orig['mae']:.3f} ± {overall_orig['mae_std']:.3f}  (target <7.810) -> {'PASS' if criteria['mae']['pass'] else 'FAIL'}")
    print(f"  - ECE = {ece:.3f}  (target <=0.18) -> {'PASS' if criteria['ece']['pass'] else 'FAIL'}")
    print(f"  - Coverage = {coverage*100:.1f}%  (target 90±5) -> {'PASS' if criteria['coverage']['pass'] else 'FAIL'}")
    print(f"\nBaselines (original scale):")
    print(f"  mean MAE={baseline_metrics['mean_mae']:.3f} ; median MAE={baseline_metrics['median_mae']:.3f}")
    print(f"  RF MAE={overall_orig['mae']:.3f} ; DeltaMAE={criteria['beat_baseline']['value']*100:.1f}% -> {'PASS' if criteria['beat_baseline']['pass'] else 'FAIL'}")
    print(f"\nAnnexe (log-space, informative only):")
    print(f"  R²={overall_log['r2']:.3f} ; MAE={overall_log['mae']:.3f}")
    print(f"\nDecision: {'GO' if all_pass else 'NO-GO'}")
    
    if not all_pass:
        print(f"\nRoot cause: {sum(c['pass'] for c in criteria.values())}/6 criteria PASS")
        if not criteria['r2']['pass']:
            print(f"  - R² = {criteria['r2']['value']:.3f} < 0.10 (FAIL)")
        print(f"\nNext step: Option C - FPbase API + literature mining -> v1.3.2 (N>=120)")
    
    print("="*70)
    
    return all_pass, overall_orig, baseline_metrics, criteria, coverage, ece


if __name__ == "__main__":
    all_pass, overall, baselines, criteria, coverage, ece = main()
    
    if all_pass:
        print("\n[GO] 6/6 PASS - Ready for release v1.2.5")
        exit(0)
    else:
        print("\n[NO-GO] Criteria FAIL - BLOCKED, proceed to Option C")
        exit(1)

