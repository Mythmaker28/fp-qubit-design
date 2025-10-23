#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate figures for FP-Qubit Design.

This script generates:
1. Feature importance plot
2. Predicted gains histogram
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_metrics(metrics_path: str = "outputs/metrics.json") -> dict:
    """Load metrics from JSON."""
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    return metrics


def load_shortlist(shortlist_path: str = "outputs/shortlist.csv") -> pd.DataFrame:
    """Load shortlist from CSV."""
    df = pd.DataFrame(pd.read_csv(shortlist_path))
    return df


def plot_feature_importance(metrics: dict, output_path: str = "figures/feature_importance.png"):
    """Plot feature importance from trained model."""
    feature_importance = metrics['feature_importance']
    
    # Sort by importance
    features = list(feature_importance.keys())
    importances = list(feature_importance.values())
    
    sorted_idx = np.argsort(importances)[::-1]
    features = [features[i] for i in sorted_idx]
    importances = [importances[i] for i in sorted_idx]
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(features, importances, color='steelblue')
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title('Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Feature importance plot saved to: {output_path}")


def plot_predicted_gains_histogram(shortlist: pd.DataFrame, output_path: str = "figures/predicted_gains_histogram.png"):
    """Plot histogram of predicted gains."""
    # Extract predicted_gain (convert from string "+X.XX" to float)
    gains = shortlist['predicted_gain'].apply(lambda x: float(x))
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.hist(gains, bins=15, color='coral', edgecolor='black', alpha=0.7)
    plt.xlabel('Predicted Gain (%)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Distribution of Predicted Gains (Contrast Proxy)', fontsize=14, fontweight='bold')
    plt.axvline(gains.mean(), color='darkred', linestyle='--', linewidth=2, label=f'Mean: {gains.mean():+.2f}%')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Predicted gains histogram saved to: {output_path}")


def main():
    """Main figure generation pipeline."""
    print("=" * 60)
    print("FP-Qubit Design - Generate Figures")
    print("=" * 60)
    
    # Create figures directory
    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)
    
    # Load data
    print("[INFO] Loading metrics and shortlist...")
    metrics = load_metrics()
    shortlist = load_shortlist()
    
    # Generate figures
    print("[INFO] Generating feature importance plot...")
    plot_feature_importance(metrics)
    
    print("[INFO] Generating predicted gains histogram...")
    plot_predicted_gains_histogram(shortlist)
    
    print()
    print("=" * 60)
    print("Figure generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()


