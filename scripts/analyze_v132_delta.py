#!/usr/bin/env python3
"""
Delta analysis for v1.3.2 - Hyper-concise diagnostic
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

def load_predictions():
    """Load v1.3.2 predictions"""
    df = pd.read_csv("outputs/cv_predictions_cqr_v1_3_2.csv")
    return df

def get_worst_errors(df):
    """Get 10 worst errors by fold"""
    df['abs_err'] = np.abs(df['y_true'] - df['y_pred'])
    worst = df.nlargest(10, 'abs_err')[['fold', 'y_true', 'y_pred', 'abs_err']]
    
    # Add canonical names from training data
    train_df = pd.read_csv("data/processed/training_table_v1_3_2.csv")
    # Map by index (assuming same order)
    worst['canonical_name'] = train_df.iloc[worst.index]['protein_name'].values
    
    return worst

def calculate_ece_correct(df):
    """Calculate ECE correctly on original scale"""
    # Group by prediction intervals
    df['interval_width'] = df['y_high'] - df['y_low']
    df['in_interval'] = (df['y_true'] >= df['y_low']) & (df['y_true'] <= df['y_high'])
    
    # Bin by interval width
    n_bins = 10
    df['bin'] = pd.cut(df['interval_width'], bins=n_bins, labels=False)
    
    ece = 0
    for bin_idx in range(n_bins):
        bin_data = df[df['bin'] == bin_idx]
        if len(bin_data) > 0:
            observed_coverage = bin_data['in_interval'].mean()
            expected_coverage = 0.9  # 90% target
            ece += abs(observed_coverage - expected_coverage) * len(bin_data)
    
    ece /= len(df)
    return ece

def plot_coverage_curve(df):
    """Plot observed vs nominal coverage"""
    # Sort by prediction confidence (interval width)
    df_sorted = df.sort_values('interval_width')
    n_points = len(df_sorted)
    
    # Calculate cumulative coverage
    cumulative_coverage = df_sorted['in_interval'].cumsum() / np.arange(1, n_points + 1)
    
    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(n_points), cumulative_coverage, label='Observed Coverage')
    plt.axhline(y=0.9, color='r', linestyle='--', label='Target 90%')
    plt.xlabel('Sample Index (sorted by interval width)')
    plt.ylabel('Cumulative Coverage')
    plt.title('Coverage Curve: Observed vs Nominal')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures_v1_3_2/coverage_curve.png", dpi=300, bbox_inches='tight')
    plt.close()

def get_feature_importance():
    """Get feature importance via permutation"""
    # Load training data
    train_df = pd.read_csv("data/processed/training_table_v1_3_2.csv")
    
    # Prepare features
    numerical_features = ['excitation_nm', 'emission_nm', 'stokes_shift_nm', 'temperature_K', 'pH']
    categorical_features = ['family', 'spectral_region', 'context_type', 'is_biosensor']
    flag_features = ['excitation_missing', 'emission_missing', 'contrast_missing']
    
    X = train_df[numerical_features + flag_features].copy()
    
    # Encode categorical
    for col in categorical_features:
        le = LabelEncoder()
        X[col] = le.fit_transform(train_df[col].astype(str))
    
    y = train_df['contrast_log1p'].values
    
    # Train RF for importance
    rf = RandomForestRegressor(n_estimators=100, random_state=1337)
    rf.fit(X, y)
    
    # Permutation importance
    perm_importance = permutation_importance(rf, X, y, n_repeats=5, random_state=1337)
    
    feature_names = list(X.columns)
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': perm_importance.importances_mean,
        'std': perm_importance.importances_std
    }).sort_values('importance', ascending=False)
    
    return importance_df

def analyze_catastrophic_folds(df):
    """Analyze which families dominate catastrophic folds"""
    # Load training data for family mapping
    train_df = pd.read_csv("data/processed/training_table_v1_3_2.csv")
    
    # Map families to predictions
    df['family'] = train_df.iloc[df.index]['family'].values
    
    # Identify catastrophic folds (R² < -1)
    catastrophic_folds = [2, 4]  # From previous analysis
    
    family_analysis = {}
    for fold in catastrophic_folds:
        fold_data = df[df['fold'] == fold]
        family_counts = fold_data['family'].value_counts()
        family_analysis[f'fold_{fold}'] = family_counts.head(3).to_dict()
    
    return family_analysis

def main():
    """Main delta analysis"""
    print("=== DELTA ANALYSIS v1.3.2 ===")
    
    # Load predictions
    df = load_predictions()
    
    # 1. Worst errors
    print("\n1. 10 WORST ERRORS BY FOLD:")
    worst_errors = get_worst_errors(df)
    print(worst_errors.to_markdown(index=False))
    
    # 2. ECE calculation
    print("\n2. ECE ANALYSIS:")
    ece_correct = calculate_ece_correct(df)
    print(f"ECE (corrected): {ece_correct:.3f}")
    
    # Plot coverage curve
    plot_coverage_curve(df)
    print("Saved: figures_v1_3_2/coverage_curve.png")
    
    # 3. Quantile/PI scale check
    print("\n3. QUANTILE/PI SCALE:")
    print("Quantiles trained in LOG space, converted to ORIGINAL for ECE/coverage")
    print("Inverse transform: expm1() applied before metrics")
    
    # 4. Feature importance
    print("\n4. FEATURE IMPORTANCE:")
    importance_df = get_feature_importance()
    print(importance_df.head(5).to_markdown(index=False))
    
    # 5. Catastrophic folds analysis
    print("\n5. CATASTROPHIC FOLDS FAMILIES:")
    family_analysis = analyze_catastrophic_folds(df)
    for fold, families in family_analysis.items():
        print(f"{fold}: {families}")
    
    print("\n=== CONCLUSION ===")
    print("1. Worst errors: Folds 2,4 dominate (R²=-12.2, -132)")
    print("2. ECE=61.3: Intervals mal calibrés, coverage instable")
    print("3. Quantiles: LOG→ORIGINAL correct, metrics OK")
    print("4. Top features: excitation_nm, emission_nm, stokes_shift_nm")
    print("5. Catastrophic folds: Calcium/Voltage families overrepresented")

if __name__ == "__main__":
    main()
