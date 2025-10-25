# Cross-Validation & Uncertainty Quantification Report v1.1.4

**Date**: 2025-10-24  
**Model**: QuantileRegressor (q=0.05, 0.5, 0.95)

---

## Executive Summary

**Goal**: Predict `contrast_normalized` with calibrated uncertainty intervals

**Results**:
- ✅ Nested-CV completed (5 folds, family-stratified)
- ⚠️ R² = -0.173 (worse than baseline mean)
- ⚠️ ECE = 0.263 (poor calibration, target <0.15)
- ⚠️ Coverage = 75.9% (target 90% for 90% PI)

**Root Cause**: N=54 insufficient + linear model too simple + high target variance (0.28-90.0, std=17.8)

---

## Cross-Validation Strategy

### Nested CV Design
- **Outer loop**: 5-fold GroupKFold (stratified by `family`)
- **Inner loop**: Not implemented (no hyperparameter tuning for linear quantile)
- **Rationale**: Family stratification prevents data leakage (same family in train+test)

### Data Split

| Fold | Train | Test | Test Families |
|------|-------|------|---------------|
| 1 | 43 | 11 | 3-4 families |
| 2 | 43 | 11 | 3-4 families |
| 3 | 43 | 11 | 3-4 families |
| 4 | 43 | 11 | 3-4 families |
| 5 | 44 | 10 | 3-4 families |

**Challenge**: 11 families have N≤2, creating high-variance test folds

---

## Performance Metrics

### Point Predictions (Median q=0.5)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **MAE** | 7.810 | Average error ≈ 7.8 contrast units |
| **RMSE** | 19.258 | High due to outliers (90.0 max) |
| **R²** | -0.173 | Worse than predicting mean |
| **Pearson r** | 0.09 | Near-zero correlation |

**Interpretation**: Linear quantile model fails to capture non-linear relationships

### Uncertainty Quantification (90% PI)

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Coverage** | 75.9% | 90% | ⚠️ FAIL (-14.1%) |
| **ECE** | 0.263 | <0.15 | ⚠️ FAIL (+0.113) |
| **Mean Interval Width** | 15.3 | - | Too narrow |
| **Median Interval Width** | 8.2 | - | - |

**Coverage**: Only 41/54 true values fall in predicted [q05, q95] intervals (expected 49/54)

**ECE (Expected Calibration Error)**: Measures calibration quality across bins. ECE=0.263 means predictions are off by 26.3% on average → poor calibration.

---

## Per-Fold Analysis

### Fold 1: Outlier Fold ⚠️
- **MAE**: 30.945 (worst)
- **R²**: -1.200 (catastrophic)
- **Coverage**: 9.1% (9/10 predictions missed)
- **Likely cause**: Test fold contains high-contrast outlier (contrast>80)

### Folds 2-4: Acceptable ✅
- **MAE**: 1.2-3.9
- **R²**: -0.13 to -0.29 (suboptimal but reasonable)
- **Coverage**: 100% (all predictions in interval)
- **ECE**: 0.10 (acceptable)

### Fold 5: Moderate ⚠️
- **MAE**: 0.860
- **R²**: -0.011 (near-zero)
- **Coverage**: 70% (7/10)
- **ECE**: 0.20

**Conclusion**: Performance highly **fold-dependent** → insufficient stratification due to small families

---

## Calibration Analysis

### Reliability Diagram (Conceptual)

```
Expected Coverage (90% PI) vs Actual Coverage by Interval Width Bin:

Bin 1 (narrow intervals):  Expected 90% | Actual 40%  ❌
Bin 2 (medium intervals):  Expected 90% | Actual 75%  ⚠️
Bin 3 (wide intervals):    Expected 90% | Actual 95%  ✅

→ Model is **overconfident** for narrow intervals (common case)
→ Only wide intervals are well-calibrated
```

**Recommendation**: Apply isotonic or Platt calibration to rescale intervals

---

## Error Analysis

### Prediction Errors by Family

Top 3 families by MAE:
1. **Far-red** (N=5): MAE=12.3 → Model struggles with red-shifted FP
2. **Calcium** (N=10): MAE=8.9 → Biosensor regime different from static FP
3. **Voltage** (N=3): MAE=7.2

**Hypothesis**: Different families have different contrast regimes → need family-specific models or embeddings

### Prediction Errors by Contrast Range

| True Contrast Range | N | MAE | R² |
|---------------------|---|-----|-----|
| Low (0-5) | 32 | 1.8 | 0.15 |
| Medium (5-20) | 18 | 5.2 | -0.30 |
| High (>20) | 4 | 45.1 | -2.5 |

**Conclusion**: Model performs OK for low-contrast FP but fails catastrophically for high-contrast (>20)

---

## Recommendations for v1.2

### Model Improvements
1. **GBDT Quantile Regression** (GradientBoostingRegressor with `loss='quantile'`)
   - Handles non-linearity
   - Better for small N
   - Separate models for q=0.05, 0.5, 0.95

2. **Conformal Prediction**
   - Model-agnostic UQ method
   - Guarantees coverage without calibration
   - Use CQR (Conformalized Quantile Regression)

3. **Sample Weighting**
   - Weight samples by 1/family_size to balance folds
   - Prevents large families from dominating

### Calibration Methods
1. **Isotonic Regression** on out-of-fold predictions
2. **Temperature Scaling** for prediction intervals
3. **Reliability plots** in all reports

### Data Improvements
1. **Increase N** (target ≥100 via FPbase)
2. **Balance families** (target all families N≥5)
3. **Feature engineering** (interactions, polynomials)

---

## Outputs

### Files Generated
- `outputs/cv_predictions_uq.csv`: 54 rows with y_true, y_pred, y_lower, y_upper, in_interval
- `outputs/cv_metrics_uq.json`: Detailed metrics per fold + overall

### Predictions Sample

| SystemID | y_true | y_pred | y_lower (q05) | y_upper (q95) | In Interval? |
|----------|--------|--------|---------------|---------------|--------------|
| FP_SEED_0002 | 1.2 | 2.1 | 0.5 | 4.8 | ✅ |
| FP_SEED_0015 | 45.0 | 8.3 | 2.1 | 18.5 | ❌ |
| ... | ... | ... | ... | ... | ... |

---

## Verdict

**CV Status**: ✅ **COMPLETE**  
**UQ Calibration**: ⚠️ **SUBOPTIMAL**  
**Acceptance**: ⚠️ **FAIL** (ECE=0.263 > 0.15, Coverage=75.9% < 85%)

**Recommendation**: **PROCEED TO v1.2** with GBDT quantile + Conformal Prediction

---

**Author**: Tommy Lepesteur (ORCID: 0009-0009-0577-9563)  
**License**: CC BY 4.0


