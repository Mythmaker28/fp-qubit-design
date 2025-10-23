## TARGET GAP REPORT - fp-qubit-design v1.1.3

**Generated**: 2025-10-23  
**Status**: ⚠️ **PARTIAL FAIL** - Criterion 2 not met

---

## Summary

| Criterion | Target | Achieved | Gap | Status |
|-----------|--------|----------|-----|--------|
| **N_real_total_all** | ≥ 34 | **34** | 0 | ✅ PASS |
| **N_optical_total** | (no target) | **13** | - | ℹ️ INFO |
| **N_optical_with_contrast_measured** | ≥ 20 | **12** | **-8** | ❌ **FAIL** |
| **N_fp_like** | (no target) | **3** | - | ℹ️ INFO |
| **N_fp_like_with_contrast** | (no target) | **2** | - | ℹ️ INFO |

---

## Root Cause Analysis

### Why N_optical_with_contrast = 12 < 20?

The optical systems (13 total) consist of:

1. **10 Color centers** (NV, SiV, GeV, VSi in diamond/SiC) - **NOT fluorescent proteins**
   - These are point defects in semiconductors
   - Used for ODMR-based quantum sensing
   - **All 10 have contrast data**

2. **1 Fluorescent protein** with ODMR readout - **HAS contrast** (12%)

3. **2 Quantum dots** (CdSe, InP/ZnS) - **1 has contrast** (CdSe: 3%), 1 missing (InP/ZnS)

**Conclusion**: Only **3/13 optical systems are FP-like** (fluorescent proteins or quantum dots). The rest are color centers in semiconductors, which are **out of scope** for "FP-qubit design" (fluorescent protein design).

---

## Data Composition (Optical Systems)

| Type | Count | With Contrast | % of Optical |
|------|-------|---------------|--------------|
| **Color centers (NV, SiV, etc.)** | 10 | 10 | 76.9% |
| **Fluorescent proteins** | 1 | 1 | 7.7% |
| **Quantum dots** | 2 | 1 | 15.4% |
| **TOTAL Optical** | **13** | **12** | **100%** |

---

## Scope Mismatch

The **fp-qubit-design** project aims to design **fluorescent protein mutants** optimized for quantum sensing applications. However, the Atlas data is dominated by:

1. **NMR/hyperpolarized systems** (10 systems) - Class C
2. **Color centers in diamond/SiC** (10 systems) - Class B, ODMR
3. **ESR/EPR systems** (6 systems) - Class A/B/C
4. **Magnetoreception (indirect)** (4 systems) - Class D

Only **3 systems** are relevant to FP design:
- 1 fluorescent protein
- 2 quantum dots (similar optical properties)

---

## Recommended Actions for v1.2

### Priority 1: Expand FP Data Sources ⭐⭐⭐

1. **FPbase** (https://www.fpbase.org/)
   - Public database of fluorescent proteins
   - ~1000+ FP variants with photophysical properties
   - Includes: brightness, QY, lifetime, photostability, **ΔF/F0** for sensors
   - API available for programmatic access

2. **UniProt cross-references**
   - Map FP names → UniProt accessions
   - Retrieve linked publications and experimental data
   - Filter for "fluorescent protein" keyword

3. **Literature mining**
   - Automated extraction from DOI (via Atlas provenance)
   - Focus on FP characterization papers
   - Extract: contrast/ΔF/F0, QY, lifetime, temperature, pH

### Priority 2: Clarify Project Scope ⭐⭐

**Option A**: **FP-only** (recommended for "FP-qubit design")
- Filter out color centers (NV, SiV, etc.)
- Focus on biological fluorescent proteins + quantum dots
- Target: N_fp_like ≥ 30 with contrast
- Sources: FPbase, UniProt, FP literature

**Option B**: **Quantum sensing broadly**
- Include color centers (already 10 systems with contrast)
- Rename project to "quantum-bio-design" or similar
- Target: N_optical ≥ 20 already achieved (12 with contrast)
- Expand to: diamond NV engineering, SiC defect design

### Priority 3: Contact Atlas Maintainer ⭐

- Request FP-specific subset or pointers to FP-rich datasets
- Propose collaboration for FP-focused Atlas extension
- Share findings from this gap analysis

---

## Short-term Workaround

For immediate progress with limited FP data:

1. **Data augmentation**:
   - Generate synthetic FP variants based on 1 real FP + literature rules
   - Use FPbase data (if available) to constrain synthetic distributions

2. **Transfer learning**:
   - Train on color centers (10 systems) to learn structure-property relationships
   - Fine-tune on FP (1 system) with domain adaptation

3. **Proof-of-concept**:
   - Demonstrate pipeline on color centers (well-represented)
   - Document limitations for FP generalization
   - Set stage for FP-rich v1.2

---

## Proposed Roadmap

### v1.2 (FP Enrichment)
- **Goal**: N_fp_like ≥ 30 with contrast
- **Actions**:
  1. Integrate FPbase (API/scraping)
  2. UniProt cross-refs
  3. Literature mining (semi-auto)
- **Timeline**: 2-4 weeks

### v1.3 (ML Training)
- **Goal**: Train RF/XGBoost on enriched FP data
- **Actions**:
  1. Featurization (AAindex, structure)
  2. Nested CV + UQ
  3. Generate FP mutant shortlist (≥30)
- **Timeline**: 2-3 weeks

### v2.0 (Advanced)
- **Goal**: GNN + active learning
- **Actions**:
  1. Structure-aware GNN
  2. Active learning loop (predict → validate → retrain)
  3. Experimental validation roadmap
- **Timeline**: 2-3 months

---

## Conclusion

**v1.1.3** successfully achieved:
- ✅ N_real_total = 34 (maintained from v1.1.2)
- ✅ Optical/non-optical classification
- ✅ Separate tables (all vs optical)

**v1.1.3** did NOT achieve:
- ❌ N_optical_with_contrast ≥ 20 (only 12, shortfall: 8)
- ❌ Sufficient FP data (only 3 FP-like systems)

**Root cause**: Scope mismatch between Atlas (broad quantum bio-systems) and fp-qubit-design (FP-specific).

**Recommendation**: **Pre-release v1.1.3-pre** + roadmap for v1.2 (FP enrichment).

---

**License**: Code: Apache-2.0 | Data: CC BY 4.0


