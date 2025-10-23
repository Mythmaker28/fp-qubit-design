#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Statistical analysis on REAL data ONLY (N=19 Atlas systems).

This script performs rigorous analysis to test for "candidate discovery":
1. Correlation analysis (Pearson + Spearman) with bootstrap CIs
2. Partial correlations controlling for confounders
3. Baseline models vs ML models (RF + XGBoost)
4. SHAP interpretability + permutation importance
5. UQ/Calibration (quantile regression)
6. Stability analysis (feature ranking across folds)

CRITICAL: Only uses REAL Atlas data. Synthetic data NOT included.

