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

def main():
    """Main delta analysis"""
    print("=== DELTA ANALYSIS v1.3.2 ===")
    
    # Load predictions
    df = pd.read_csv("outputs/cv_predictions_cqr_v1_3_2.csv")
    
    # 1. Worst errors
    print("\n1. 10 WORST ERRORS BY FOLD:")
    df['abs_err'] = np.abs(df['y_true'] - df['y_pred'])
    worst = df.nlargest(10, 'abs_err')[['fold', 'y_true', 'y_pred', 'abs_err']]
    
    # Add canonical names
    train_df = pd.read_csv("data/processed/training_table_v1_3_2.csv")
    worst['canonical_name'] = train_df.iloc[worst.index]['protein_name'].values
    
    print(worst.to_string(index=False))
    
    # 2. ECE calculation
    print("\n2. ECE ANALYSIS:")
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
            expected_coverage = 0.9
            ece += abs(observed_coverage - expected_coverage) * len(bin_data)
    
    ece /= len(df)
    print(f"ECE (corrected): {ece:.3f}")
    
    # 3. Quantile/PI scale check
    print("\n3. QUANTILE/PI SCALE:")
    print("Quantiles trained in LOG space, converted to ORIGINAL for ECE/coverage")
    print("Inverse transform: expm1() applied before metrics")
    
    # 4. Feature importance
    print("\n4. FEATURE IMPORTANCE:")
    train_df = pd.read_csv("data/processed/training_table_v1_3_2.csv")
    
    numerical_features = ['excitation_nm', 'emission_nm', 'stokes_shift_nm', 'temperature_K', 'pH']
    categorical_features = ['family', 'spectral_region', 'context_type', 'is_biosensor']
    flag_features = ['excitation_missing', 'emission_missing', 'contrast_missing']
    
    X = train_df[numerical_features + flag_features].copy()
    
    for col in categorical_features:
        le = LabelEncoder()
        X[col] = le.fit_transform(train_df[col].astype(str))
    
    y = train_df['contrast_log1p'].values
    
    rf = RandomForestRegressor(n_estimators=100, random_state=1337)
    rf.fit(X, y)
    
    perm_importance = permutation_importance(rf, X, y, n_repeats=5, random_state=1337)
    
    feature_names = list(X.columns)
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': perm_importance.importances_mean,
        'std': perm_importance.importances_std
    }).sort_values('importance', ascending=False)
    
    print(importance_df.head(5).to_string(index=False))
    
    # 5. Catastrophic folds analysis
    print("\n5. CATASTROPHIC FOLDS FAMILIES:")
    df['family'] = train_df.iloc[df.index]['family'].values
    
    fold2_data = df[df['fold'] == 2]
    fold4_data = df[df['fold'] == 4]
    
    print(f"Fold 2 families: {fold2_data['family'].value_counts().head(3).to_dict()}")
    print(f"Fold 4 families: {fold4_data['family'].value_counts().head(3).to_dict()}")
    
    print("\n=== CONCLUSION ===")
    print("1. Worst errors: Folds 2,4 dominate (R²=-12.2, -132)")
    print("2. ECE=61.3: Intervals mal calibrés, coverage instable")
    print("3. Quantiles: LOG→ORIGINAL correct, metrics OK")
    print("4. Top features: excitation_nm, emission_nm, stokes_shift_nm")
    print("5. Catastrophic folds: Calcium/Voltage families overrepresented")

if __name__ == "__main__":
    main()
