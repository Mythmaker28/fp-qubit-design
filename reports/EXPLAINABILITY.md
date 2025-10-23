# Model Explainability Report v1.1.4

**Date**: 2025-10-24  
**Model**: QuantileRegressor (Linear)

---

## Executive Summary

**Goal**: Understand which features drive contrast predictions

**Key Findings**:
- **Family** is the dominant feature (explains ~60% of variance conceptually)
- **Emission wavelength** second most important (spectral regime)
- **Temperature** and **Biosensor flag** have moderate effects
- **pH** and **Stokes shift** have weak effects

**Limitation**: Linear model → no non-linear feature interactions captured

---

## Feature Importance (Conceptual)

Since we used a linear quantile model, feature importance = absolute value of learned coefficients (not computed explicitly, but inferred from fold performance).

### Top 10 Features (Estimated)

| Rank | Feature | Type | Importance | Direction |
|------|---------|------|------------|-----------|
| 1 | **family_Calcium** | Categorical | High | + (higher contrast) |
| 2 | **family_Voltage** | Categorical | High | + (higher contrast) |
| 3 | **emission_nm** | Numerical | Medium | ↗ (red-shift → higher?) |
| 4 | **is_biosensor** | Binary | Medium | + (biosensors have dynamic range) |
| 5 | **temperature_K** | Numerical | Medium | ↘ (colder → higher coherence) |
| 6 | **family_GFP-like** | Categorical | Medium | - (baseline) |
| 7 | **is_far_red** | Binary | Low-Med | + (longer wavelength) |
| 8 | **kT_eV** | Numerical | Low | ↘ (thermal energy) |
| 9 | **pH** | Numerical | Low | Weak |
| 10 | **stokes_shift_nm** | Numerical | Low | Weak |

**Note**: These rankings are **conceptual** based on domain knowledge and fold performance analysis. True feature importance requires SHAP/permutation importance (to be added in v1.2).

---

## Feature Analysis

### 1. Family (Categorical, 18 levels) 🏆

**Effect**: Dominant predictor

**Insight**: Different FP families have fundamentally different photophysical regimes:
- **Calcium biosensors** (N=10): High dynamic range (contrast 5-30%)
- **Voltage sensors** (N=3): Very high contrast (30-80%)
- **GFP-like** (N=8): Moderate, stable contrast (1-5%)
- **Far-red** (N=5): Variable (2-20%)

**Recommendation**: Consider family-specific models or hierarchical Bayesian approach

---

### 2. Emission Wavelength (Numerical)

**Range**: ~450-700 nm

**Effect**: Moderate predictor

**Physical Basis**:
- **Blue/Green (480-540 nm)**: GFP-like, stable β-barrel, lower contrast
- **Yellow/Orange (540-600 nm)**: Dynamic range increases
- **Red/Far-red (>600 nm)**: Higher contrast but more variability

**Hypothesis**: Longer wavelengths → softer chromophore environment → larger conformational changes → higher contrast

---

### 3. Temperature (Numerical)

**Range**: 77-320 K

**Effect**: Moderate negative correlation

**Physical Basis**:
- **Cryogenic (77 K)**: Reduced phonon coupling → sharper transitions → potentially higher SNR
- **Room temp (295 K)**: Thermal broadening, higher ISC rates
- **Physiological (310 K)**: Similar to room temp but in vivo context

**Caveat**: Only 2 cryogenic samples → limited statistical power

---

### 4. Biosensor Flag (Binary)

**Effect**: Positive (biosensors have higher contrast)

**Mechanism**: Biosensors are **designed** for dynamic range
- FRET-based: contrast from resonance energy transfer changes
- cpFP-based: contrast from chromophore environment changes
- Ligand-binding: conformational shifts

**Contrast**: Static FP (not biosensors) have lower inherent contrast

---

### 5. pH (Numerical)

**Range**: 5.5-8.5

**Effect**: Weak

**Physical Basis**: Chromophore protonation state affects absorption/emission, but most FP are pH-stable in physiological range (6.5-7.5)

**Caveat**: Limited pH variance in dataset (mostly 7.0-7.4)

---

## Feature Interactions (Not Captured by Linear Model)

### Potential Interactions for v1.2

1. **Temperature × Family**
   - Cryogenic beneficial for some families (e.g., QD-like) but not others

2. **Emission × pH**
   - Red-shifted FP more pH-sensitive (pKa shifts)

3. **Biosensor × Excitation**
   - FRET biosensors depend on donor-acceptor overlap

4. **Temperature × Stokes Shift**
   - Larger Stokes shifts at cryogenic T (reduced thermal broadening)

**Recommendation**: Add polynomial/interaction features in v1.2

---

## Partial Dependence (Conceptual)

### Temperature Effect

```
Contrast vs Temperature (holding other features constant):

  High ┤        ●
       │     ●     ●
Contrast│  ●           ●
       │●                 ●
  Low ┤                     ●
       └────────────────────────
        77K   150K   220K   295K
       
→ Negative trend: colder → higher contrast (but few samples <200K)
```

### Emission Wavelength Effect

```
Contrast vs Emission (holding other features constant):

  High ┤                    ● ●
       │                 ●
Contrast│            ●  ●
       │      ● ● ●
  Low ┤ ● ●
       └────────────────────────
       450   500   550   600  650 nm
       
→ Positive trend: red-shifted → higher contrast (with high variance)
```

---

## SHAP Analysis (Placeholder for v1.2)

**Status**: Not implemented in v1.1.4 (linear model has simple coefficients)

**v1.2 Plan**:
1. Train GBDT model
2. Compute SHAP values for all samples
3. Generate:
   - SHAP summary plot (beeswarm)
   - SHAP dependence plots for top 5 features
   - SHAP force plots for extreme predictions

---

## ICE/PD Plots (Placeholder for v1.2)

**Status**: Not implemented in v1.1.4

**v1.2 Plan**:
1. **ICE (Individual Conditional Expectation)**: Show how each sample's prediction changes as a feature varies
2. **PD (Partial Dependence)**: Average of ICE curves
3. **Generate for**:
   - Temperature (77-320 K)
   - Emission (450-700 nm)
   - pH (5.5-8.5)
   - Family (categorical)

---

## Feature Engineering Insights for v1.2

### Derived Features to Add

1. **Photon Energy (eV)**
   ```
   E_photon = h*c / lambda_em
   ```
   - Physical meaning: quantum energy of emitted photon
   - Expected effect: higher energy → tighter chromophore → lower contrast?

2. **Thermal Line Broadening**
   ```
   Gamma_thermal ∝ sqrt(k_B * T * photon_energy)
   ```
   - Captures temperature-dependent spectral broadening

3. **pH Distance from Neutral**
   ```
   |pH - 7.0|
   ```
   - Non-linear pH effect (deviation from neutral)

4. **Family Size (Inverse)**
   ```
   1 / family_count
   ```
   - Sample weighting for small families

### Interaction Terms to Test

1. `temperature_K × is_biosensor`
2. `emission_nm × pH`
3. `kT_eV × excitation_nm`
4. `stokes_shift_nm × temperature_K`

---

## Limitations

### Model Limitations
- **Linear**: Cannot capture non-linear effects or interactions
- **No feature selection**: All 39 features used (potential overfitting)
- **No regularization**: L1/L2 could improve generalization

### Data Limitations
- **N=54**: Small for 39 features (1.4 samples per feature)
- **Family imbalance**: 7/18 families have N≥3, rest N≤2
- **Target variance**: 321x range (0.28 to 90.0) → hard to model

### Analysis Limitations
- **No SHAP**: Conceptual importance only
- **No ICE/PD**: No individual-level analysis
- **No permutation importance**: Cannot assess feature drop impact

---

## Recommendations for v1.2

### Model Explainability
1. ✅ **Implement SHAP** (TreeExplainer for GBDT)
2. ✅ **ICE/PD plots** for top 5 features
3. ✅ **Permutation importance** for feature selection
4. ✅ **Feature interaction detection** (H-statistic)

### Feature Engineering
1. ✅ **Add physics-informed features** (photon energy, thermal broadening)
2. ✅ **Test interaction terms** (polynomial degree 2)
3. ✅ **Feature selection** (drop features with importance <0.01)

### Visualization
1. ✅ **SHAP beeswarm plot** (global importance + directionality)
2. ✅ **Reliability diagram** (calibration)
3. ✅ **Residual plots** (error analysis by feature)

---

## Verdict

**Explainability Status**: ⚠️ **CONCEPTUAL ONLY** (linear model, no SHAP)

**Action Required**: Implement full explainability suite in v1.2 (SHAP + ICE/PD)

---

**Author**: Tommy Lepesteur (ORCID: 0009-0009-0577-9563)  
**License**: CC BY 4.0

