# Audit Report v1.1.4

**Date**: 2025-10-24  
**Pipeline**: v1.1.4 "Measured-Only, Clean & Ship"

---

## Data Audit

### Source Validation ✅
- **File**: `atlas_fp_optical.csv`
- **Source**: `https://raw.githubusercontent.com/Mythmaker28/biological-qubits-atlas/main/data/processed/atlas_fp_optical.csv`
- **SHA256**: `4b847f48eef6d65efc819e5bb54451bd0ab124faa4d3538e83c396794df3ac90`
- **Size**: 7930 bytes
- **Format**: CSV, 20 columns

### Count Validation ✅

| Metric | Expected | Actual | Status |
|--------|----------|--------|--------|
| **Total FP systems** | 66 | 66 | ✅ PASS |
| **Measured A/B tier** | 54 | 54 | ✅ PASS |
| **Families (≥3 samples)** | 7 | 7 | ✅ PASS |

### Schema Validation ✅

**Required columns (20)**:
- ✅ `SystemID`, `protein_name`, `variant`, `family`
- ✅ `excitation_nm`, `emission_nm`, `temperature_K`, `pH`
- ✅ `contrast_ratio`, `contrast_ci_low`, `contrast_ci_high`
- ✅ `contrast_source`, `contrast_normalized`, `contrast_quality_tier`
- ✅ `is_biosensor`, `uniprot_id`, `pdb_id`
- ✅ `condition_text`, `source_refs`, `license_source`

### Training Data Audit

**train_measured.csv** (filtered from atlas_fp_optical.csv):
- **Filter**: `contrast_quality_tier in ['A', 'B']`
- **N_input**: 66
- **N_output**: 54
- **Families**: 18 total, 7 with ≥3 samples

#### Family Distribution (N≥3)

| Family | Count | Status |
|--------|-------|--------|
| Calcium | 10 | ✅ |
| GFP-like | 8 | ✅ |
| Far-red | 5 | ✅ |
| RFP | 5 | ✅ |
| CFP-like | 3 | ✅ |
| Dopamine | 3 | ✅ |
| Voltage | 3 | ✅ |

**Total**: 37/54 samples in families with ≥3 (68.5%)

---

## Model Audit

### Training Configuration
- **Model**: QuantileRegressor (q=0.05, 0.5, 0.95)
- **CV**: 5-fold GroupKFold (family-stratified)
- **N_samples**: 54
- **N_features**: 39
- **Target**: `contrast_normalized` (range: 0.28 to 90.0)

### Performance Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **MAE** | 7.810 | - | - |
| **R²** | -0.173 | ≥0.10 | ⚠️ FAIL |
| **RMSE** | 19.258 | - | - |
| **Coverage** | 0.759 | 0.90 | ⚠️ FAIL |
| **ECE** | 0.263 | <0.15 | ⚠️ FAIL |

### Per-Fold Metrics

| Fold | N_train | N_test | MAE | R² | Coverage | ECE |
|------|---------|--------|-----|-----|----------|-----|
| 1 | 43 | 11 | 30.945 | -1.200 | 0.091 | 0.809 |
| 2 | 43 | 11 | 1.209 | -0.128 | 1.000 | 0.100 |
| 3 | 43 | 11 | 1.527 | -0.153 | 1.000 | 0.100 |
| 4 | 43 | 11 | 3.877 | -0.291 | 1.000 | 0.100 |
| 5 | 44 | 10 | 0.860 | -0.011 | 0.700 | 0.200 |

**Observation**: High variance across folds → small N issue

---

## Acceptance Criteria

| Criterion | Status | Notes |
|-----------|--------|-------|
| **Data available** | ✅ PASS | 66 FP found, SHA256 verified |
| **N≥40 measured** | ✅ PASS | 54 tier A/B |
| **Families≥5 (N≥3)** | ✅ PASS | 7 families |
| **Featurization** | ✅ PASS | 39 features implemented |
| **Nested-CV** | ✅ PASS | 5-fold family-stratified |
| **R²≥0.10** | ⚠️ FAIL | R²=-0.17 (need better model) |
| **ECE≤0.15** | ⚠️ FAIL | ECE=0.263 (UQ not calibrated) |
| **Coverage~0.90** | ⚠️ FAIL | 75.9% (intervals too narrow) |

---

## Recommendations for v1.2

### Priority 1: Model Upgrade
- Replace linear QuantileRegressor with GBDT quantile or Conformal Prediction
- Add isotonic calibration for prediction intervals
- Implement sample weighting by family size

### Priority 2: Data Augmentation
- Integrate FPbase (target +50 FP)
- Target total N≥100 for robust UQ

### Priority 3: Feature Engineering
- Add interaction terms (e.g., T × pH, ex × em)
- Polynomial features for non-linear relationships
- Family-specific baseline offsets

---

## Verdict

**Pipeline Status**: ✅ **FUNCTIONAL**  
**Data Quality**: ✅ **HIGH**  
**Model Performance**: ⚠️ **SUBOPTIMAL BUT DOCUMENTED**

**Action Required**: Proceed to v1.2 with improved modeling

---

**Author**: Tommy Lepesteur (ORCID: 0009-0009-0577-9563)  
**License**: CC BY 4.0


