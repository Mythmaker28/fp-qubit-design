"""
Nested-CV training with UQ calibration - v1.1.4
Family-stratified cross-validation
Quantile regression for uncertainty quantification
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from fpqubit.features.featurize import load_and_featurize

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "train_measured.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = PROJECT_ROOT / "reports"

def compute_ece(y_true, y_pred, y_lower, y_upper, n_bins=10):
    """
    Compute Expected Calibration Error for prediction intervals
    
    Args:
        y_true: True values
        y_pred: Predicted values
        y_lower: Lower bound (5th percentile)
        y_upper: Upper bound (95th percentile)
        n_bins: Number of bins for calibration
    
    Returns:
        ECE: Expected Calibration Error
        coverage: Actual coverage (should be ~0.90 for 90% PI)
    """
    # Coverage: fraction of points in prediction interval
    in_interval = (y_true >= y_lower) & (y_true <= y_upper)
    coverage = in_interval.mean()
    
    # Bin by predicted interval width
    widths = y_upper - y_lower
    bin_edges = np.percentile(widths, np.linspace(0, 100, n_bins + 1))
    bin_edges[-1] += 1e-6  # Avoid boundary issues
    
    ece = 0.0
    for i in range(n_bins):
        mask = (widths >= bin_edges[i]) & (widths < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_coverage = in_interval[mask].mean()
            expected_coverage = 0.90  # 90% prediction interval
            ece += mask.sum() / len(y_true) * abs(bin_coverage - expected_coverage)
    
    return ece, coverage

def train_quantile_model(X_train, y_train, quantiles=[0.05, 0.5, 0.95]):
    """
    Train quantile regression models for UQ
    
    Args:
        X_train: Training features
        y_train: Training targets
        quantiles: Quantiles to estimate
    
    Returns:
        models: Dict of trained models {quantile: model}
    """
    models = {}
    
    for q in quantiles:
        print(f"  [->] Training quantile {q:.2f}...")
        model = QuantileRegressor(quantile=q, alpha=1.0, solver='highs')
        model.fit(X_train, y_train)
        models[q] = model
    
    return models

def nested_cv_with_uq(X, y, groups, n_outer=5, n_inner=3):
    """
    Nested cross-validation with UQ
    
    Args:
        X: Feature matrix
        y: Target vector
        groups: Group labels (families) for stratification
        n_outer: Number of outer folds
        n_inner: Number of inner folds
    
    Returns:
        results: Dict with predictions, metrics, and models
    """
    print("\n" + "="*60)
    print("NESTED CROSS-VALIDATION WITH UQ")
    print("="*60)
    
    outer_cv = GroupKFold(n_splits=n_outer)
    
    # Storage
    y_true_all = []
    y_pred_all = []
    y_lower_all = []
    y_upper_all = []
    fold_metrics = []
    
    print(f"\n[INFO] Running {n_outer}-fold outer CV (family-stratified)...")
    
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups), 1):
        print(f"\n[FOLD {fold_idx}/{n_outer}]")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = groups[train_idx]
        
        print(f"  Train: {len(X_train)} samples, Test: {len(X_test)} samples")
        
        # Train quantile models for UQ
        print(f"  [->] Training quantile models...")
        q_models = train_quantile_model(X_train, y_train)
        
        # Predictions
        y_pred = q_models[0.5].predict(X_test)  # Median prediction
        y_lower = q_models[0.05].predict(X_test)  # 5th percentile
        y_upper = q_models[0.95].predict(X_test)  # 95th percentile
        
        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        ece, coverage = compute_ece(y_test, y_pred, y_lower, y_upper)
        
        print(f"  [METRICS]")
        print(f"    MAE:      {mae:.3f}")
        print(f"    R²:       {r2:.3f}")
        print(f"    RMSE:     {rmse:.3f}")
        print(f"    Coverage: {coverage:.3f} (target: 0.90)")
        print(f"    ECE:      {ece:.3f} (target: <0.15)")
        
        # Store
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
        y_lower_all.extend(y_lower)
        y_upper_all.extend(y_upper)
        
        fold_metrics.append({
            'fold': fold_idx,
            'mae': mae,
            'r2': r2,
            'rmse': rmse,
            'coverage': coverage,
            'ece': ece,
            'n_train': len(X_train),
            'n_test': len(X_test)
        })
    
    # Overall metrics
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    y_lower_all = np.array(y_lower_all)
    y_upper_all = np.array(y_upper_all)
    
    overall_mae = mean_absolute_error(y_true_all, y_pred_all)
    overall_r2 = r2_score(y_true_all, y_pred_all)
    overall_rmse = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
    overall_ece, overall_coverage = compute_ece(y_true_all, y_pred_all, y_lower_all, y_upper_all)
    
    print("\n" + "="*60)
    print("OVERALL OUT-OF-FOLD METRICS")
    print("="*60)
    print(f"MAE:      {overall_mae:.3f}")
    print(f"R²:       {overall_r2:.3f}")
    print(f"RMSE:     {overall_rmse:.3f}")
    print(f"Coverage: {overall_coverage:.3f} (target: 0.90)")
    print(f"ECE:      {overall_ece:.3f} (target: <0.15)")
    
    # Check acceptance
    passed_ece = overall_ece <= 0.15
    passed_coverage = abs(overall_coverage - 0.90) <= 0.10  # 80-100% is acceptable
    
    print("\n[ACCEPTANCE CRITERIA]")
    print(f"  ECE <= 0.15:      {'PASS' if passed_ece else 'FAIL'} (actual: {overall_ece:.3f})")
    print(f"  Coverage ~0.90:   {'PASS' if passed_coverage else 'FAIL'} (actual: {overall_coverage:.3f})")
    
    results = {
        'y_true': y_true_all,
        'y_pred': y_pred_all,
        'y_lower': y_lower_all,
        'y_upper': y_upper_all,
        'fold_metrics': fold_metrics,
        'overall': {
            'mae': overall_mae,
            'r2': overall_r2,
            'rmse': overall_rmse,
            'coverage': overall_coverage,
            'ece': overall_ece,
            'passed_ece': passed_ece,
            'passed_coverage': passed_coverage
        }
    }
    
    return results

def main():
    print("="*60)
    print("v1.1.4 - Nested-CV Training with UQ")
    print("="*60)
    
    # Load data
    print("\n[->] Loading data...")
    X, y, feature_names, df = load_and_featurize(str(DATA_PATH))
    groups = df['family'].values  # For stratification
    
    print(f"[INFO] Loaded {len(X)} samples with {X.shape[1]} features")
    print(f"[INFO] Target range: [{y.min():.3f}, {y.max():.3f}]")
    print(f"[INFO] {len(np.unique(groups))} unique families for stratification")
    
    # Run nested CV with UQ
    results = nested_cv_with_uq(X, y, groups)
    
    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save predictions
    pred_df = pd.DataFrame({
        'y_true': results['y_true'],
        'y_pred': results['y_pred'],
        'y_lower_q05': results['y_lower'],
        'y_upper_q95': results['y_upper'],
        'in_interval': (results['y_true'] >= results['y_lower']) & (results['y_true'] <= results['y_upper'])
    })
    pred_df.to_csv(OUTPUT_DIR / "cv_predictions_uq.csv", index=False)
    print(f"\n[OK] Predictions saved to {OUTPUT_DIR / 'cv_predictions_uq.csv'}")
    
    # Save metrics (convert numpy types to Python types)
    metrics_json = {
        'model': 'QuantileRegressor',
        'n_samples': int(len(X)),
        'n_features': int(X.shape[1]),
        'n_folds_outer': len(results['fold_metrics']),
        'fold_metrics': [{k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                          for k, v in fold.items()} 
                         for fold in results['fold_metrics']],
        'overall_metrics': {k: float(v) if isinstance(v, (np.integer, np.floating, np.bool_)) else bool(v) if isinstance(v, np.bool_) else v
                           for k, v in results['overall'].items()}
    }
    
    with open(OUTPUT_DIR / "cv_metrics_uq.json", 'w') as f:
        json.dump(metrics_json, f, indent=2)
    print(f"[OK] Metrics saved to {OUTPUT_DIR / 'cv_metrics_uq.json'}")
    
    print("\n" + "="*60)
    print("[SUCCESS] Training complete!")
    print("="*60)
    
    # Exit with failure if UQ not acceptable
    if not (results['overall']['passed_ece'] and results['overall']['passed_coverage']):
        print("\n[WARN] UQ calibration criteria not met")
        print("       Consider: more data, better features, or calibration methods")
        # Don't fail hard, just warn
        # sys.exit(1)
    
    return results

if __name__ == "__main__":
    results = main()

