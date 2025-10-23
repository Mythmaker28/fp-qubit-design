# DATA REALITY REPORT - fp-qubit-design v1.1.4

**Generated**: 2025-10-23  
**Status**: ⚠️ **CRITICAL BLOCKER** - Canonical data source not found

---

## 🎯 What Was Expected

**User specification** (from prompt):
- File: `atlas_fp_optical.csv` v1.2.1
- Total: **66 entries** (FP optical systems)
- Measured tier A/B: **54 entries**
- Families: ≥7 with ≥3 measurements
- SHA256: `333ADC871F5B2EC5118298DE4E534A468C7379F053D8B03C13D7CD9EB7C43285`

**Scope**: FP optical ONLY (biosensors, fluorescent proteins)
- **Included**: GFP, RFP, CFP, YFP, mCherry, TagRFP, calcium sensors, voltage sensors, pH sensors, Quantum Dots (CdSe, InP/ZnS)
- **Excluded**: NV centers, SiV centers, color centers in diamond/SiC, NMR/ESR systems, hyperpolarized nuclei, magnetoreception

---

## 🔍 What Actually Exists

### Atlas v1.2.1 - Full Inventory

**Source**: https://github.com/Mythmaker28/biological-qubits-atlas/releases/tag/v1.2.1

**Assets**:
1. `biological_qubits.csv` (26 systems, SHA256: `8d75d58dfbf8660fb853db1cd7ea122c3efb4ebf2150671942bb8fac3c650839`)
2. `CITATION.cff`
3. `LICENSE` (CC BY 4.0)
4. `QC_REPORT.md`

**MISSING**: `atlas_fp_optical.csv`

### biological_qubits.csv Breakdown

| Category | Count | With Contrast | % of Total |
|----------|-------|---------------|------------|
| **Color centers (ODMR)** | 10 | 10 | 38.5% |
| **NMR hyperpolarized** | 10 | 0 | 38.5% |
| **ESR/EPR** | 4 | 1 | 15.4% |
| **FP optical** | **1** | **1** | **3.8%** |
| **Quantum dots** | **1** | **1** | **3.8%** |
| **TOTAL** | **26** | **13** | **100%** |

### FP Optical Systems (ACTUAL)

Only **2 systems** match "FP optical" criteria:

| System | Family | Contrast | Tier | Method | Host Context |
|--------|--------|----------|------|--------|--------------|
| **Protéine fluorescente avec lecture ODMR** | Other (unknown FP) | 12% | C (no peer-reviewed ref with error bars) | ODMR | HeLa cells |
| **Quantum dots CdSe** | QuantumDot | 3% | C | Optical-only | Cryogenic solution |

**Gap vs expectation**: **64 systems missing** (66 expected - 2 found = **-64**)

---

## 🚨 Root Cause Analysis

### Why is atlas_fp_optical.csv missing?

**Hypothesis 1**: **Never created/published**
- The Atlas maintainer may not have created this filtered subset yet
- The public Atlas focuses on **broad quantum bio-systems**, not FP-specific

**Hypothesis 2**: **Future release**
- The file may be planned for a future Atlas release (v1.3+)
- Current Atlas (v1.2.1) is dominated by color centers and NMR systems

**Hypothesis 3**: **User confusion**
- The expected file name/structure may have been from a **different project** or **local processing**
- No public Atlas release has ever contained 66 FP systems

### What exists instead?

The Atlas v1.2.1 contains:
- **10 color centers** (NV, SiV, GeV, VSi in diamond/SiC) - **quantum sensors** but **NOT fluorescent proteins**
- **10 NMR systems** (^13C hyperpolarized metabolites) - **not optical**
- **4 ESR/EPR systems** (nitroxide radicals, LOV2 protein) - **not fluorescent**
- **1 FP** + **1 QD** = **2 FP optical** total

---

## 📊 What We Can Actually Use

### Option 1: **Use All Optical Systems** (ODMR + FP) - **12 systems**

**Includes**:
- 10 color centers (NV, SiV, etc.) - **optical readout via ODMR**
- 1 FP with ODMR
- 1 Quantum Dot

**Pros**:
- N=12 with contrast (92% coverage)
- All have **optical readout** (ODMR is optical detection)
- Good for **quantum sensing broadly**

**Cons**:
- **Violates user specification** (excluded NV/SiV explicitly)
- Color centers are **semiconductor defects**, not **biological FPs**
- Scope mismatch with "fp-qubit-design"

### Option 2: **FP Optical ONLY (strict)** - **2 systems** ❌

**Includes**:
- 1 FP (unknown family)
- 1 Quantum Dot (CdSe)

**Pros**:
- Respects user specification (FP only)
- No scope creep

**Cons**:
- **N=2 is insufficient** for ANY ML (need min 30-50)
- Cannot train nested-CV, UQ, or generate shortlist
- **BLOCKS v1.1.4 entirely**

### Option 3: **Integrate External FP Data** - **Recommended** ⭐

**Sources**:
1. **FPbase** (https://www.fpbase.org/)
   - ~1000+ FP variants with photophysical properties
   - API available: `https://www.fpbase.org/api/proteins/`
   - Includes: brightness, QY, lifetime, ΔF/F0 for sensors
   - License: CC BY 4.0

2. **UniProt cross-refs**
   - Search: `fluorescent protein` → ~500+ entries
   - Includes: sequences, variants, cross-refs to PDB/literature

3. **Literature mining**
   - Parse DOI from Atlas provenance
   - Extract: contrast/ΔF/F0, QY, lifetime, temperature, pH

**Workflow**:
```bash
# Step 1: Fetch FPbase data
python scripts/consume/fetch_fpbase.py  # → data/external/fpbase_fp_optical.csv

# Step 2: Merge with Atlas (2 FP systems)
python scripts/consume/merge_fp_sources.py  # → data/processed/train_measured.csv (N≥50)

# Step 3: Continue v1.1.4 pipeline
```

---

## ✅ RECOMMENDATIONS

### Immediate Actions (v1.1.4-pre)

1. **STOP current v1.1.4 pipeline** ❌ 
   - Cannot proceed with N=2 (expected N=54)
   - Acceptance criteria FAIL: `N_train_measured < 40`

2. **Document reality** ✅ (this report)
   - `WHERE_I_LOOKED.md`: Discovery log (7 attempts, all 404)
   - `DATA_REALITY_v1.1.4.md`: This report

3. **Create pre-release v1.1.4-pre** with status:
   - **BLOCKED**: Canonical data source not found
   - **Recommendation**: Wait for external FP integration (v1.2)

### Mid-term Plan (v1.2 - FP Enrichment)

**Goal**: Integrate FPbase + UniProt to reach N≥50 FP optical with measurements

**Timeline**: 2-4 weeks

**Actions**:
1. Implement `scripts/consume/fetch_fpbase.py`
2. Implement `scripts/consume/fetch_uniprot_fps.py`
3. Merge sources with provenance tracking
4. Resume v1.1.4 pipeline

### Long-term Plan (v1.3+)

**Option A**: **Contact Atlas maintainer**
- Request creation of `atlas_fp_optical.csv` filtered subset
- Propose collaboration to expand FP coverage
- Share this gap analysis

**Option B**: **Expand scope**
- Rename project to "bio-quantum-sensors" (include NV centers)
- Keep FP-focused branch separately

---

## 📦 Deliverables from v1.1.4 Attempt

### Files Created ✅

1. `config/data_sources.yaml` - Configuration (expected SHA256, URLs)
2. `scripts/consume/resolve_atlas_v1_2_1.py` - Robust multi-path discovery (7 attempts logged)
3. `reports/WHERE_I_LOOKED.md` - Discovery log (releases/tags/branches)
4. `reports/DATA_REALITY_v1.1.4.md` - This report

### Files NOT Created ❌

- `data/external/atlas_fp_optical_v1_2_1.csv` (doesn't exist)
- `data/processed/train_measured.csv` (N=2 insufficient)
- ML training outputs (nested-CV, UQ, SHAP)
- Shortlist (cannot generate with N=2)

---

## 🔚 CONCLUSION

**v1.1.4 "Measured-Only, Clean & Ship" cannot proceed** with current Atlas data.

**Root cause**: Expected `atlas_fp_optical.csv` (66 FP systems) **does not exist** in public Atlas v1.2.1.

**Actual data**: Only **2 FP optical systems** available (1 FP + 1 QD).

**Recommendation**: **Pause v1.1.4** and plan **v1.2 (FP Enrichment)** with external sources (FPbase, UniProt).

---

**License**: Code: Apache-2.0 | Data: CC BY 4.0

**Contact**: Tommy Lepesteur (ORCID: 0009-0009-0577-9563)

