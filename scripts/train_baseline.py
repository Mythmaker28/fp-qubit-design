#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train baseline ML models (Random Forest) for FP mutant property prediction.

This script:
1. Loads Atlas snapshot and maps to proxies
2. Creates a simple synthetic dataset for training
3. Trains a Random Forest model
4. Performs cross-validation
5. Saves metrics to outputs/metrics.json
"""

import argparse
import yaml
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fpqubit.utils.seed import set_seed


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train baseline ML models for FP-Qubit Design"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/example.yaml",
        help="Path to config file (YAML)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run (no actual training)"
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_atlas_snapshot(filepath: str) -> pd.DataFrame:
    """Load Atlas snapshot CSV."""
    print(f"[INFO] Loading Atlas snapshot from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"[INFO] Loaded {len(df)} systems")
    return df


def create_synthetic_dataset(atlas_df: pd.DataFrame, n_samples: int = 200, seed: int = 42) -> tuple:
    """
    Create a synthetic dataset for training.
    
    In a real implementation, this would extract features from FP sequences.
    Here we create a simple synthetic dataset based on Atlas proxies.
    """
    np.random.seed(seed)
    
    # Extract simple features from Atlas
    atlas_features = []
    atlas_targets = []
    
    for _, row in atlas_df.iterrows():
        # Skip rows with missing contrast
        if pd.isna(row.get('Contraste_%', np.nan)):
            continue
            
        # Simple features (one-hot encoding of method, temperature, flags)
        features = {
            'temperature': row.get('Temperature_K', 295),
            'method_odmr': 1 if 'ODMR' in str(row.get('Methode_lecture', '')) else 0,
            'method_esr': 1 if 'ESR' in str(row.get('Methode_lecture', '')) else 0,
            'method_nmr': 1 if 'NMR' in str(row.get('Methode_lecture', '')) else 0,
            'in_vivo': int(row.get('In_vivo_flag', 0)),
            'quality': int(row.get('Qualite', 1)),
        }
        
        target = float(row.get('Contraste_%', 10))
        
        atlas_features.append(list(features.values()))
        atlas_targets.append(target)
    
    # Create synthetic augmented dataset
    X_base = np.array(atlas_features)
    y_base = np.array(atlas_targets)
    
    # Augment with noise
    X_synthetic = []
    y_synthetic = []
    
    for _ in range(n_samples):
        idx = np.random.randint(0, len(X_base))
        x = X_base[idx].copy()
        y = y_base[idx]
        
        # Add small noise
        x[0] += np.random.normal(0, 5)  # temperature noise
        y += np.random.normal(0, 2)  # target noise
        
        X_synthetic.append(x)
        y_synthetic.append(max(0, y))  # ensure non-negative
    
    X = np.array(X_synthetic)
    y = np.array(y_synthetic)
    
    feature_names = ['temperature', 'method_odmr', 'method_esr', 'method_nmr', 'in_vivo', 'quality']
    
    return X, y, feature_names


def train_model(X_train, y_train, config: dict):
    """Train Random Forest model."""
    print(f"[INFO] Training Random Forest model...")
    
    model = RandomForestRegressor(
        n_estimators=config['baseline']['n_estimators'],
        max_depth=config['baseline']['max_depth'],
        random_state=config['seed'],
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    print(f"[INFO] Model trained successfully")
    return model


def evaluate_model(model, X_train, y_train, X_test, y_test, feature_names, config: dict) -> dict:
    """Evaluate model and return metrics."""
    print(f"[INFO] Evaluating model...")
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Metrics
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    # Cross-validation
    cv_scores = cross_val_score(
        model, X_train, y_train, 
        cv=config['baseline']['cv_folds'], 
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )
    cv_mae = -cv_scores.mean()
    cv_mae_std = cv_scores.std()
    
    # Feature importance
    feature_importance = dict(zip(feature_names, model.feature_importances_))
    
    metrics = {
        'model_type': 'RandomForest',
        'n_estimators': config['baseline']['n_estimators'],
        'max_depth': config['baseline']['max_depth'],
        'seed': config['seed'],
        'train_size': len(X_train),
        'test_size': len(X_test),
        'train_mae': float(train_mae),
        'test_mae': float(test_mae),
        'train_rmse': float(train_rmse),
        'test_rmse': float(test_rmse),
        'train_r2': float(train_r2),
        'test_r2': float(test_r2),
        'cv_mae_mean': float(cv_mae),
        'cv_mae_std': float(cv_mae_std),
        'feature_importance': {k: float(v) for k, v in feature_importance.items()},
        'date_trained': datetime.now().isoformat(),
    }
    
    print(f"[INFO] Test MAE: {test_mae:.3f}")
    print(f"[INFO] Test R²: {test_r2:.3f}")
    print(f"[INFO] CV MAE: {cv_mae:.3f} ± {cv_mae_std:.3f}")
    
    return metrics


def main():
    """Main training pipeline."""
    args = parse_args()
    
    if args.dry_run:
        print("[DRY-RUN] train_baseline.py - OK")
        return
    
    config = load_config(args.config)
    
    # Set seed for reproducibility
    set_seed(config['seed'])
    
    print("=" * 60)
    print("FP-Qubit Design - Train Baseline (REAL)")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Seed: {config['seed']}")
    print()
    
    # Load Atlas snapshot
    atlas_df = load_atlas_snapshot(config['data']['atlas_snapshot'])
    
    # Create synthetic dataset
    print(f"[INFO] Creating synthetic dataset...")
    X, y, feature_names = create_synthetic_dataset(atlas_df, n_samples=200, seed=config['seed'])
    print(f"[INFO] Dataset shape: X={X.shape}, y={y.shape}")
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config['seed']
    )
    
    # Train model
    model = train_model(X_train, y_train, config)
    
    # Evaluate model
    metrics = evaluate_model(model, X_train, y_train, X_test, y_test, feature_names, config)
    
    # Save metrics
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    
    metrics_path = outputs_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Metrics saved to: {metrics_path}")
    
    # Save model
    model_path = outputs_dir / "model_rf.pkl"
    joblib.dump(model, model_path)
    print(f"[INFO] Model saved to: {model_path}")
    
    print()
    print("=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
