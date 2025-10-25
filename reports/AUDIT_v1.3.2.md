# AUDIT REPORT v1.3.2 - Atlas v2.2 Integration

## Summary
- **Version**: v1.3.2
- **Data Source**: Atlas v2.2 (atlas_fp_optical_v2_2.csv)
- **Total Systems**: 178
- **Families**: 30
- **Target Variable**: contrast_normalized (log1p transformed for training)

## Data Quality
- **Complete Systems**: 178 (100%)
- **Missing Contrast**: 0
- **Missing Family**: 0
- **Missing Temperature**: 0
- **Missing pH**: 0

## Family Distribution
- **Calcium**: 37 systems
- **Voltage**: 20 systems
- **Dopamine**: 13 systems
- **GFP-like**: 11 systems
- **RFP**: 11 systems
- **pH**: 10 systems
- **Glutamate**: 9 systems
- **Far-red**: 7 systems
- **CFP-like**: 7 systems
- **NIR**: 6 systems
- **H2O2**: 5 systems
- **cAMP**: 5 systems
- **GABA**: 4 systems
- **YFP**: 4 systems
- **NADH/NAD+**: 3 systems
- **BFP-like**: 3 systems
- **Acetylcholine**: 3 systems
- **ATP**: 3 systems
- **ATP/ADP**: 2 systems
- **Redox**: 2 systems
- **cGMP**: 2 systems
- **Norepinephrine**: 2 systems
- **Zinc**: 2 systems
- **Serotonin**: 1 systems
- **Histamine**: 1 systems
- **Opioid**: 1 systems
- **NADPH/NADP+**: 1 systems
- **Oxygen**: 1 systems
- **Teal**: 1 systems
- **Orange**: 1 systems

## Context Distribution
- **in_cellulo**: 99 systems
- **in_vivo**: 79 systems

## Spectral Distribution
- **cyan**: 79 systems
- **yellow**: 28 systems
- **blue**: 28 systems
- **unknown**: 19 systems
- **green**: 17 systems
- **orange**: 5 systems
- **red**: 2 systems

## Target Statistics
- **Mean**: 9.093
- **Std**: 14.814
- **Min**: 0.750
- **Max**: 90.000
- **Median**: 3.500

## Features
- **Numerical**: excitation_nm, emission_nm, stokes_shift_nm, temperature_K, pH
- **Categorical**: family, spectral_region, context_type, is_biosensor
- **Flags**: excitation_missing, emission_missing, contrast_missing

## Sources
- **metabolic_preseed**: 6 systems
- **geci_db_preseed**: 6 systems
- **neurotransmitter_preseed**: 6 systems
- **Literature_v2.2**: 5 systems
- **voltage_preseed**: 3 systems
- **pmc_fulltext**: 2 systems

## Gate Check: N_utiles >= 100
- **Current N_utiles**: 178
- **Target**: >= 100
- **Status**: PASS

## Decision
GO - Proceed to v1.3.2 training
