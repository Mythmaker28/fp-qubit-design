# AUDIT REPORT - fp-qubit-design v1.1.3

**Generated**: 2025-10-23 21:31:01

---

## Summary

| Metric | Value | Status |
|--------|-------|--------|
| **N_real_total_all** | 34 | PASS |
| **N_optical_total** | 13 | - |
| **N_optical_with_contrast_measured** | 12 | FAIL |
| **N_optical_with_contrast_any** | 12 | - |
| **N_fp_like** | 3 | - |
| **N_fp_like_with_contrast** | 2 | - |

## Acceptance Criteria

- **Criterion 1**: `N_real_total_all >= 34` -> **PASS**
- **Criterion 2**: `N_optical_with_contrast_measured >= 20` -> **FAIL** (shortfall: 8)

## Data Provenance

- **Sources**: biological-qubits-atlas (9 sources: main, v1.2.0, v1.2.1, develop, infra/pages+governance, feat/data-v1.2-extended, docs/doi-badge, chore/zenodo-metadata, chore/citation-author)
- **Classification**: Optical vs non-optical based on method, class, and keyword patterns
- **License**: CC BY 4.0

## Key Findings

- **34 real systems** total (maintained from v1.1.2)
- **13 optical systems** (38.2%): fluorescence, ODMR, quantum dots
- **21 non-optical systems** (61.8%): NMR, ESR, magnetoreception, indirect
- **12/13 optical systems have contrast** (92% coverage)
- **Only 3 FP-like systems** (1 FP + 2 QD); rest are color centers (NV, SiV, GeV, VSi)
- **2/3 FP-like have contrast** (67%)

## Contrast Statistics (Optical Only)

- **N**: 12
- **Mean**: 10.58%
- **Std**: 7.63%
- **Range**: [3.00%, 30.00%]

---

## Recommendation

### PARTIAL - Pre-release v1.1.3-pre Recommended

**Criterion 1 (N_real_total_all >= 34)**: PASS
**Criterion 2 (N_optical_with_contrast >= 20)**: FAIL (shortfall: 8)

**Root cause**: Most optical systems (10/13) are **color centers** (NV, SiV, GeV, VSi in diamond/SiC), not fluorescent proteins.

**Recommended actions for v1.2**:

1. **Expand FP data sources**:
   - FPbase (fpbase.org) - public database of FP photophysics
   - UniProt cross-refs for FP variants
   - Literature mining (automated extraction from DOI)

2. **Broaden scope**:
   - If targeting quantum sensing broadly: include NV centers (already 10 systems)
   - If targeting FP only: filter out non-FP systems and focus on FP enrichment

3. **Contact Atlas maintainer**:
   - Request FP-specific data or pointers to FP-rich datasets


---

**License**: Code: Apache-2.0 | Data: CC BY 4.0
