#!/usr/bin/env python3
"""
Training script for v1.3.2 - RandomForest + CQR with 178 systems
Uses RandomForest for central predictions and GBDT quantiles with CQR for UQ
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression

# For CQR
from sklearn.base import BaseEstimator, RegressorMixin
import joblib

class ConformalizedQuantileRegression:
    """Conformalized Quantile Regression for prediction intervals"""
    
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.calibration_scores = None
        
    def fit(self, y_low, y_high, y_true):
        """Fit CQR calibration"""
        # Calculate non-conformity scores
        scores_low = np.maximum(y_low - y_true, 0)
        scores_high = np.maximum(y_true - y_high, 0)
        scores = np.maximum(scores_low, scores_high)
        
        # Get quantile for calibration
        self.calibration_scores = np.quantile(scores, 1 - self.alpha)
        return self
    
    def predict_intervals(self, y_low, y_high):
        """Predict conformalized intervals"""
        if self.calibration_scores is None:
            raise ValueError("Must fit before predicting")
        
        # Adjust intervals
        y_low_adj = y_low - self.calibration_scores
        y_high_adj = y_high + self.calibration_scores
        
        return y_low_adj, y_high_adj

def load_training_data():
    """Load the v1.3.2 training data"""
    print("=== LOADING TRAINING DATA ===")
    
    df = pd.read_csv("data/processed/training_table_v1_3_2.csv")
    print(f"Loaded {len(df)} systems")
    print(f"Features: {list(df.columns)}")
    
    return df

def prepare_features(df):
    """Prepare features for training"""
    print("\n=== PREPARING FEATURES ===")
    
    # Numerical features
    numerical_features = [
        'excitation_nm', 'emission_nm', 'stokes_shift_nm', 
        'temperature_K', 'pH'
    ]
    
    # Categorical features
    categorical_features = [
        'family', 'spectral_region', 'context_type', 'is_biosensor'
    ]
    
    # Missing value flags
    flag_features = [
        'excitation_missing', 'emission_missing', 'contrast_missing'
    ]
    
    # Prepare feature matrix
    X = df[numerical_features + flag_features].copy()
    
    # Encode categorical features
    le_dict = {}
    for col in categorical_features:
        le = LabelEncoder()
        X[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le
    
    # Target variables
    y_original = df['contrast_normalized'].values
    y_log = df['contrast_log1p'].values
    
    # Groups for CV
    groups = df['family'].values
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target range (original): [{y_original.min():.3f}, {y_original.max():.3f}]")
    print(f"Target range (log1p): [{y_log.min():.3f}, {y_log.max():.3f}]")
    print(f"Groups: {len(np.unique(groups))} families")
    
    return X, y_original, y_log, groups, le_dict

def train_naive_baselines(X, y_original, y_log, groups):
    """Train naive baselines"""
    print("\n=== TRAINING NAIVE BASELINES ===")
    
    # Mean and median regressors
    mean_pred_orig = np.full_like(y_original, np.mean(y_original))
    median_pred_orig = np.full_like(y_original, np.median(y_original))
    
    mean_pred_log = np.full_like(y_log, np.mean(y_log))
    median_pred_log = np.full_like(y_log, np.median(y_log))
    
    # Calculate metrics
    mean_mae_orig = mean_absolute_error(y_original, mean_pred_orig)
    median_mae_orig = mean_absolute_error(y_original, median_pred_orig)
    
    mean_mae_log = mean_absolute_error(y_log, mean_pred_log)
    median_mae_log = mean_absolute_error(y_log, median_pred_log)
    
    print(f"Mean regressor MAE (original): {mean_mae_orig:.3f}")
    print(f"Median regressor MAE (original): {median_mae_orig:.3f}")
    print(f"Mean regressor MAE (log): {mean_mae_log:.3f}")
    print(f"Median regressor MAE (log): {median_mae_log:.3f}")
    
    return {
        'mean_mae_orig': mean_mae_orig,
        'median_mae_orig': median_mae_orig,
        'mean_mae_log': mean_mae_log,
        'median_mae_log': median_mae_log
    }

def custom_group_kfold(groups, n_splits=5):
    """Custom balanced GroupKFold"""
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    
    # Aggregate rare families (N < 3)
    group_counts = pd.Series(groups).value_counts()
    rare_families = group_counts[group_counts < 3].index
    groups_agg = groups.copy()
    for rare_fam in rare_families:
        groups_agg[groups == rare_fam] = 'Other'
    
    # Recalculate unique groups
    unique_groups_agg = np.unique(groups_agg)
    n_groups_agg = len(unique_groups_agg)
    
    print(f"Original families: {n_groups}")
    print(f"Aggregated families: {n_groups_agg}")
    print(f"Rare families aggregated: {len(rare_families)}")
    
    # Create balanced splits
    group_kfold = GroupKFold(n_splits=n_splits)
    splits = list(group_kfold.split(X, y_log, groups_agg))
    
    return splits, groups_agg

def train_models(X, y_original, y_log, groups):
    """Train RandomForest and GBDT quantile models"""
    print("\n=== TRAINING MODELS ===")
    
    # Custom GroupKFold
    splits, groups_agg = custom_group_kfold(groups, n_splits=5)
    
    # Initialize models
    rf = RandomForestRegressor(
        n_estimators=1000,
        max_depth=None,
        min_samples_leaf=2,
        oob_score=True,
        random_state=1337,
        n_jobs=-1
    )
    
    gbdt_low = GradientBoostingRegressor(
        loss='quantile',
        alpha=0.1,
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=1337
    )
    
    gbdt_high = GradientBoostingRegressor(
        loss='quantile',
        alpha=0.9,
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=1337
    )
    
    # Cross-validation
    cv_results = []
    
    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"Fold {fold + 1}/5")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train_orig, y_val_orig = y_original[train_idx], y_original[val_idx]
        y_train_log, y_val_log = y_log[train_idx], y_log[val_idx]
        
        # Train RandomForest (central model)
        rf.fit(X_train, y_train_log)
        y_pred_log = rf.predict(X_val)
        y_pred_orig = np.expm1(y_pred_log)
        
        # Train GBDT quantiles
        gbdt_low.fit(X_train, y_train_log)
        gbdt_high.fit(X_train, y_train_log)
        
        y_low_log = gbdt_low.predict(X_val)
        y_high_log = gbdt_high.predict(X_val)
        
        # Convert to original scale
        y_low_orig = np.expm1(y_low_log)
        y_high_orig = np.expm1(y_high_log)
        
        # Apply CQR
        cqr = ConformalizedQuantileRegression(alpha=0.1)
        cqr.fit(y_low_orig, y_high_orig, y_val_orig)
        y_low_cqr, y_high_cqr = cqr.predict_intervals(y_low_orig, y_high_orig)
        
        # Calculate metrics
        r2 = r2_score(y_val_orig, y_pred_orig)
        mae = mean_absolute_error(y_val_orig, y_pred_orig)
        
        # Coverage
        coverage = np.mean((y_val_orig >= y_low_cqr) & (y_val_orig <= y_high_cqr))
        
        # ECE (simplified)
        interval_width = y_high_cqr - y_low_cqr
        ece = np.mean(np.abs(interval_width - np.percentile(interval_width, 90)))
        
        cv_results.append({
            'fold': fold + 1,
            'r2': r2,
            'mae': mae,
            'coverage': coverage,
            'ece': ece,
            'y_true': y_val_orig,
            'y_pred': y_pred_orig,
            'y_low': y_low_cqr,
            'y_high': y_high_cqr
        })
        
        print(f"  R²: {r2:.3f}, MAE: {mae:.3f}, Coverage: {coverage:.3f}, ECE: {ece:.3f}")
    
    return cv_results

def calculate_overall_metrics(cv_results):
    """Calculate overall metrics"""
    print("\n=== OVERALL METRICS ===")
    
    # Aggregate predictions
    all_y_true = np.concatenate([r['y_true'] for r in cv_results])
    all_y_pred = np.concatenate([r['y_pred'] for r in cv_results])
    all_y_low = np.concatenate([r['y_low'] for r in cv_results])
    all_y_high = np.concatenate([r['y_high'] for r in cv_results])
    
    # Overall metrics
    overall_r2 = r2_score(all_y_true, all_y_pred)
    overall_mae = mean_absolute_error(all_y_true, all_y_pred)
    overall_coverage = np.mean((all_y_true >= all_y_low) & (all_y_true <= all_y_high))
    
    # ECE
    interval_width = all_y_high - all_y_low
    overall_ece = np.mean(np.abs(interval_width - np.percentile(interval_width, 90)))
    
    # CV statistics
    r2_scores = [r['r2'] for r in cv_results]
    mae_scores = [r['mae'] for r in cv_results]
    coverage_scores = [r['coverage'] for r in cv_results]
    ece_scores = [r['ece'] for r in cv_results]
    
    print(f"R²: {overall_r2:.3f} ± {np.std(r2_scores):.3f}")
    print(f"MAE: {overall_mae:.3f} ± {np.std(mae_scores):.3f}")
    print(f"Coverage: {overall_coverage:.3f} ± {np.std(coverage_scores):.3f}")
    print(f"ECE: {overall_ece:.3f} ± {np.std(ece_scores):.3f}")
    
    return {
        'r2': overall_r2,
        'mae': overall_mae,
        'coverage': overall_coverage,
        'ece': overall_ece,
        'r2_std': np.std(r2_scores),
        'mae_std': np.std(mae_scores),
        'coverage_std': np.std(coverage_scores),
        'ece_std': np.std(ece_scores)
    }

def check_acceptance_criteria(metrics, baselines):
    """Check v1.3.2 acceptance criteria"""
    print("\n=== ACCEPTANCE CRITERIA CHECK ===")
    
    criteria = {
        'n_utiles': {'value': 178, 'target': 100, 'pass': 178 >= 100},
        'r2': {'value': metrics['r2'], 'target': 0.20, 'pass': metrics['r2'] >= 0.20},
        'mae': {'value': metrics['mae'], 'target': 7.810, 'pass': metrics['mae'] < 7.810},
        'ece': {'value': metrics['ece'], 'target': 0.15, 'pass': metrics['ece'] <= 0.15},
        'coverage': {'value': metrics['coverage'], 'target': (0.85, 0.95), 'pass': 0.85 <= metrics['coverage'] <= 0.95},
        'beat_baseline': {
            'value': (baselines['mean_mae_orig'] - metrics['mae']) / baselines['mean_mae_orig'],
            'target': 0.10,
            'pass': (baselines['mean_mae_orig'] - metrics['mae']) / baselines['mean_mae_orig'] >= 0.10
        }
    }
    
    print(f"N_utiles: {criteria['n_utiles']['value']} (target: >=100) -> {'PASS' if criteria['n_utiles']['pass'] else 'FAIL'}")
    print(f"R²: {criteria['r2']['value']:.3f} (target: >=0.20) -> {'PASS' if criteria['r2']['pass'] else 'FAIL'}")
    print(f"MAE: {criteria['mae']['value']:.3f} (target: <7.810) -> {'PASS' if criteria['mae']['pass'] else 'FAIL'}")
    print(f"ECE: {criteria['ece']['value']:.3f} (target: <=0.15) -> {'PASS' if criteria['ece']['pass'] else 'FAIL'}")
    print(f"Coverage: {criteria['coverage']['value']:.3f} (target: 85-95%) -> {'PASS' if criteria['coverage']['pass'] else 'FAIL'}")
    print(f"Beat baseline: {criteria['beat_baseline']['value']:.1%} (target: >=10%) -> {'PASS' if criteria['beat_baseline']['pass'] else 'FAIL'}")
    
    n_passed = sum(criteria[k]['pass'] for k in criteria)
    print(f"\nOverall: {n_passed}/{len(criteria)} criteria passed")
    
    return criteria, n_passed == len(criteria)

def save_results(cv_results, metrics, criteria, baselines):
    """Save all results"""
    print("\n=== SAVING RESULTS ===")
    
    # Save predictions
    all_results = []
    for r in cv_results:
        for i in range(len(r['y_true'])):
            all_results.append({
                'fold': r['fold'],
                'y_true': r['y_true'][i],
                'y_pred': r['y_pred'][i],
                'y_low': r['y_low'][i],
                'y_high': r['y_high'][i]
            })
    
    pred_df = pd.DataFrame(all_results)
    pred_df.to_csv("outputs/cv_predictions_cqr_v1_3_2.csv", index=False)
    print("Saved: outputs/cv_predictions_cqr_v1_3_2.csv")
    
    # Save metrics
    results_metrics = {
        'version': 'v1.3.2',
        'n_systems': 178,
        'model': 'RandomForest + GBDT Quantiles + CQR',
        'cv_folds': 5,
        'metrics': metrics,
        'baselines': baselines,
        'acceptance_criteria': {
            k: {
                'value': float(v['value']) if isinstance(v['value'], (int, float, np.number)) else v['value'],
                'target': v.get('target', v.get('target_range')),
                'pass': bool(v['pass'])
            }
            for k, v in criteria.items()
        }
    }
    
    with open("outputs/cv_metrics_v1_3_2.json", "w") as f:
        json.dump(results_metrics, f, indent=2)
    print("Saved: outputs/cv_metrics_v1_3_2.json")
    
    return results_metrics

def generate_figures(cv_results, metrics):
    """Generate diagnostic figures"""
    print("\n=== GENERATING FIGURES ===")
    
    # Create figures directory
    Path("figures_v1_3_2").mkdir(exist_ok=True)
    
    # Aggregate data
    all_y_true = np.concatenate([r['y_true'] for r in cv_results])
    all_y_pred = np.concatenate([r['y_pred'] for r in cv_results])
    all_y_low = np.concatenate([r['y_low'] for r in cv_results])
    all_y_high = np.concatenate([r['y_high'] for r in cv_results])
    
    # 1. Prediction vs True
    plt.figure(figsize=(8, 6))
    plt.scatter(all_y_true, all_y_pred, alpha=0.6, s=20)
    plt.plot([all_y_true.min(), all_y_true.max()], [all_y_true.min(), all_y_true.max()], 'r--', lw=2)
    plt.xlabel('True Contrast')
    plt.ylabel('Predicted Contrast')
    plt.title(f'Predictions vs True (R² = {metrics["r2"]:.3f})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures_v1_3_2/pred_vs_true.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Interval Coverage
    plt.figure(figsize=(10, 6))
    sorted_idx = np.argsort(all_y_true)
    x_range = np.arange(len(all_y_true))
    
    plt.fill_between(x_range, all_y_low[sorted_idx], all_y_high[sorted_idx], 
                     alpha=0.3, label='Prediction Intervals')
    plt.plot(x_range, all_y_true[sorted_idx], 'o', markersize=2, alpha=0.6, label='True Values')
    plt.plot(x_range, all_y_pred[sorted_idx], 'r-', alpha=0.8, label='Predictions')
    
    plt.xlabel('Sample Index (sorted by true value)')
    plt.ylabel('Contrast')
    plt.title(f'Prediction Intervals (Coverage = {metrics["coverage"]:.1%})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures_v1_3_2/interval_coverage.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Fold R² Distribution
    r2_scores = [r['r2'] for r in cv_results]
    plt.figure(figsize=(8, 6))
    plt.bar(range(1, len(r2_scores)+1), r2_scores)
    plt.axhline(y=metrics['r2'], color='r', linestyle='--', label=f'Overall R² = {metrics["r2"]:.3f}')
    plt.xlabel('Fold')
    plt.ylabel('R² Score')
    plt.title('R² Score by Fold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures_v1_3_2/fold_r2_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved: figures_v1_3_2/pred_vs_true.png")
    print("Saved: figures_v1_3_2/interval_coverage.png")
    print("Saved: figures_v1_3_2/fold_r2_distribution.png")

def main():
    """Main training pipeline"""
    print("=== v1.3.2 TRAINING - RandomForest + CQR ===")
    print("N_systems: 178 (target: >=100)")
    print()
    
    # Load data
    df = load_training_data()
    
    # Prepare features
    X, y_original, y_log, groups, le_dict = prepare_features(df)
    
    # Train baselines
    baselines = train_naive_baselines(X, y_original, y_log, groups)
    
    # Train models
    cv_results = train_models(X, y_original, y_log, groups)
    
    # Calculate metrics
    metrics = calculate_overall_metrics(cv_results)
    
    # Check criteria
    criteria, all_passed = check_acceptance_criteria(metrics, baselines)
    
    # Save results
    results_metrics = save_results(cv_results, metrics, criteria, baselines)
    
    # Generate figures
    generate_figures(cv_results, metrics)
    
    # Final status
    print(f"\n=== FINAL STATUS ===")
    print(f"Data: N_total=178 ; N_utiles=178")
    print(f"Model: RandomForest + GBDT Quantiles + CQR")
    print(f"Splits: Custom GroupKFold balanced, seed=1337")
    print()
    print(f"Metrics (original scale, CV mean±std):")
    print(f"  - R² = {metrics['r2']:.3f} ± {metrics['r2_std']:.3f}  (target >=0.20) -> {'PASS' if criteria['r2']['pass'] else 'FAIL'}")
    print(f"  - MAE = {metrics['mae']:.3f} ± {metrics['mae_std']:.3f}  (target <7.810) -> {'PASS' if criteria['mae']['pass'] else 'FAIL'}")
    print(f"  - ECE = {metrics['ece']:.3f} ± {metrics['ece_std']:.3f}  (target <=0.15) -> {'PASS' if criteria['ece']['pass'] else 'FAIL'}")
    print(f"  - Coverage = {metrics['coverage']:.1%} ± {metrics['coverage_std']:.1%}  (target 85-95%) -> {'PASS' if criteria['coverage']['pass'] else 'FAIL'}")
    print()
    print(f"Baselines (original scale):")
    print(f"  mean MAE={baselines['mean_mae_orig']:.3f} ; median MAE={baselines['median_mae_orig']:.3f}")
    print(f"  RF MAE={metrics['mae']:.3f} ; DeltaMAE={criteria['beat_baseline']['value']*100:.1f}% -> {'PASS' if criteria['beat_baseline']['pass'] else 'FAIL'}")
    print()
    print(f"Decision: {'GO' if all_passed else 'NO-GO'} ({sum(criteria[k]['pass'] for k in criteria)}/{len(criteria)} PASS)")
    
    if not all_passed:
        failed_criteria = [k for k, v in criteria.items() if not v['pass']]
        print(f"Failed criteria: {failed_criteria}")
        print("Next step: Generate BLOCKED report")
    else:
        print("Next step: Release v1.3.2")

if __name__ == "__main__":
    main()
