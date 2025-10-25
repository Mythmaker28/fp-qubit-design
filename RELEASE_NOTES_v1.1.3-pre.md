# Release Notes - fp-qubit-design v1.1.3-pre (PRE-RELEASE)

**Release Date**: 2025-10-23  
**Release Type**: ⚠️ **PRE-RELEASE** (Partial criteria met)  
**Branch**: `release/v1.1.3-data-extend`

---

## ⚠️ Pre-Release Status

This is a **pre-release** because:
- ✅ **Criterion 1** (N_real_total ≥ 34): **PASS** (34 systems)
- ❌ **Criterion 2** (N_optical_with_contrast ≥ 20): **FAIL** (12 systems, shortfall: 8)

**Root cause**: Only 3/13 optical systems are **fluorescent proteins or quantum dots**. The remaining 10 are **color centers** (NV, SiV, GeV, VSi in diamond/SiC), which are out of scope for "FP-qubit design".

---

## 🎯 Objectives v1.1.3

1. ✅ **Classify optical vs non-optical** systems
2. ✅ **Separate tables**: `atlas_all_real.csv` (all) vs `training_table_optical.csv` (optical only)
3. ❌ **Achieve N_optical_with_contrast ≥ 20** (only 12, shortfall: 8)

---

## 📊 Final Metrics

| Metric | Value | Status |
|--------|-------|--------|
| **N_real_total_all** | **34** | ✅ PASS (≥34) |
| **N_optical_total** | **13** (38.2%) | ℹ️ INFO |
| **N_non_optical** | **21** (61.8%) | ℹ️ INFO |
| **N_optical_with_contrast** | **12** (92% of optical) | ❌ FAIL (<20) |
| **N_fp_like** | **3** (1 FP + 2 QD) | ⚠️ LOW |
| **N_fp_like_with_contrast** | **2** (67% of FP-like) | ⚠️ LOW |

---

## 🚀 What's New in v1.1.3-pre

### 1. **Modality Classification** 🔍

- **Script**: `scripts/etl/classify_modality.py`
- **Logic**:
  - **Optical**: Fluorescence, FRET, ODMR, quantum dots, GFP family, excitation/emission
  - **Non-optical**: NMR, ESR, hyperpolarized, magnetoreception, indirect readout
- **Results**:
  - 13 optical (38.2%)
  - 21 non-optical (61.8%)
- **Report**: `reports/MODALITY_SPLIT.md`

### 2. **Separate Training Tables** 📊

| Table | Systems | Description |
|-------|---------|-------------|
| **`atlas_all_real.csv`** | **34** | ALL real Atlas systems (optical + non-optical) |
| **`training_table_optical.csv`** | **13** | ONLY optical systems (filtered for FP-qubit design) |

**Why separate?**
- `atlas_all_real.csv`: Complete provenance, all Atlas data preserved
- `training_table_optical.csv`: Focus on optical FP/QD for training (excludes NMR, ESR, etc.)

### 3. **Optical Systems Breakdown** 🔬

| Type | Count | With Contrast | % of Optical |
|------|-------|---------------|--------------|
| **Color centers** (NV, SiV, GeV, VSi in diamond/SiC) | 10 | 10 | 76.9% |
| **Fluorescent proteins** | 1 | 1 | 7.7% |
| **Quantum dots** | 2 | 1 | 15.4% |
| **TOTAL** | **13** | **12** | **100%** |

**Key insight**: Most optical systems are **color centers**, not FP!

### 4. **Audit with Optical Metrics** ✅❌

- **Script**: `scripts/qa/audit_counts_v1.1.3.py`
- **Exit codes**:
  - 0: All criteria met
  - 1: N_real_total < 34
  - 2: N_optical_with_contrast < 20 (triggered)
- **Reports**: `reports/AUDIT_v1.1.3.md`, `reports/TARGET_GAP_v1.1.3.md`

---

## 📦 Assets

1. **`data/processed/atlas_all_real.csv`** (34 systems, 24 columns)
2. **`data/processed/training_table_optical.csv`** (13 optical systems, 24 columns)
3. **`data/processed/TRAINING.METADATA.json`** (schema v1.1.3)
4. **`reports/MODALITY_SPLIT.md`** (classification details + lists)
5. **`reports/AUDIT_v1.1.3.md`** (audit metrics + recommendation)
6. **`reports/TARGET_GAP_v1.1.3.md`** (gap analysis + roadmap)

---

## 🔍 Root Cause Analysis

### Why N_optical_with_contrast = 12 < 20?

The **Biological Qubits Atlas** covers **broad quantum bio-systems**:
- NMR hyperpolarized (10 systems)
- Color centers in diamond/SiC (10 systems)
- ESR/EPR (6 systems)
- Magnetoreception (4 systems)
- **Fluorescent proteins (1 system)**
- Quantum dots (2 systems)

The **fp-qubit-design** project targets **fluorescent protein design**, but Atlas has only **3 FP-like systems**.

**Scope mismatch** → Insufficient FP data.

---

## 🛠️ Recommended Actions for v1.2

### Priority 1: **Expand FP Data Sources** ⭐⭐⭐

1. **FPbase** (https://www.fpbase.org/)
   - ~1000+ FP variants with photophysical properties
   - Includes: brightness, QY, lifetime, ΔF/F0 for sensors
   - API available

2. **UniProt cross-references**
   - Map FP names → UniProt accessions
   - Retrieve linked publications

3. **Literature mining**
   - Automated extraction from DOI
   - Focus on FP characterization papers

### Priority 2: **Clarify Project Scope** ⭐⭐

**Option A**: **FP-only** (recommended)
- Filter out color centers
- Focus on biological FP + QD
- Target: N_fp_like ≥ 30

**Option B**: **Quantum sensing broadly**
- Include color centers (already 10 with contrast)
- Rename project to "quantum-bio-design"

### Priority 3: **Contact Atlas Maintainer** ⭐

- Request FP-specific subset
- Propose collaboration for FP-focused extension

---

## 📈 Comparison v1.1.2 → v1.1.3-pre

| Metric | v1.1.2 | v1.1.3-pre | Change |
|--------|--------|------------|--------|
| **Total systems** | 34 | 34 | - |
| **With contrast** | 17 | 17 | - |
| **Optical classified** | - | **13** | ✅ NEW |
| **Non-optical classified** | - | **21** | ✅ NEW |
| **Optical with contrast** | - | **12** | ✅ NEW |
| **FP-like** | - | **3** | ✅ NEW |
| **Tables** | 1 | **2** (all + optical) | ✅ NEW |

---

## 🎓 Citation

```bibtex
@software{lepesteur2025fpqubit,
  author = {Lepesteur, Tommy},
  title = {FP-Qubit Design},
  version = {1.1.3-pre},
  year = {2025},
  url = {https://github.com/Mythmaker28/fp-qubit-design},
  note = {Pre-release: Optical classification + separate tables (N_optical=13, N_fp_like=3)}
}
```

---

## 🔄 Workflow

```bash
# 1. Classify modality
python scripts/etl/classify_modality.py

# 2. Build separate tables
python scripts/etl/build_training_tables_v1.1.3.py

# 3. Audit (fails if N_optical_with_contrast < 20)
python scripts/qa/audit_counts_v1.1.3.py
```

---

## 🚀 Next Steps

1. **Push to GitHub**:
   ```bash
   git push origin master --tags
   ```

2. **Create GitHub Pre-Release v1.1.3-pre** (manually or via `gh`)

3. **Plan v1.2**: FP enrichment (FPbase, UniProt, literature mining)

---

## 📄 License

- **Code**: Apache-2.0
- **Data**: CC BY 4.0 (Biological Qubits Atlas)

---

## 🙏 Acknowledgments

- **Biological Qubits Atlas** (Tommy Lepesteur): Source data (CC BY 4.0)
- **FPbase** (planned for v1.2): FP photophysics database

---

**⚠️ This is a PRE-RELEASE. Use with caution for production training.**

**Recommendation**: Wait for v1.2 (FP enrichment) for robust FP mutant design.



