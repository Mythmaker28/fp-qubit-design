## Context

I'm working on **fp-qubit-design** (https://github.com/Mythmaker28/fp-qubit-design), a project that designs fluorescent protein mutants optimized for quantum sensing applications.

This project uses **biological-qubits-atlas** as its canonical data source for FP optical systems.

## Problem

The project expects **`atlas_fp_optical.csv`** v1.2.1 with the following characteristics:
- **Total FP optical systems**: 66
- **Measured (tier A/B)**: 54
- **Families with ≥3 measurements**: ≥7

However, after exhaustive search across:
- ✅ Releases API (v1.2.1 found, but asset absent)
- ❌ Direct download URL (404)
- ❌ Branches (`release/v1.2.1-fp-optical-push`, `main`) (404)

**Result**: `atlas_fp_optical.csv` **does not exist** in the public repository.

## Current Atlas v1.2.1 Assets

The v1.2.1 release currently includes:
- `biological_qubits.csv` (26 systems total, only 2 FP optical)
- `CITATION.cff`
- `LICENSE`
- `QC_REPORT.md`

## Request

Could you please **publish `atlas_fp_optical.csv`** as an asset in the v1.2.1 release (or a new release)?

**Expected structure**:
- Filtered subset: FP optical systems only (biosensors, fluorescent proteins, quantum dots)
- Excludes: NV centers, SiV centers, color centers, NMR, ESR, magnetoreception
- Columns: `protein_name`, `variant`, `family`, `is_biosensor`, `excitation_nm`, `emission_nm`, `temperature_K`, `pH`, `contrast_ratio`, `contrast_normalized`, `contrast_source`, `contrast_quality_tier`, `source_refs`, `license_source`, `evidence_type`

**Expected counts**:
- Total: 66 FP optical systems
- Measured tier A/B: 54 (contrast_source=="measured" AND contrast_quality_tier ∈ {A, B})
- Families: ≥7 with ≥3 measurements each

**SHA256 checksum** (if available): `333ADC871F5B2EC5118298DE4E534A468C7379F053D8B03C13D7CD9EB7C43285`

## Supporting Documents

I've attached:
- `WHERE_I_LOOKED.md`: Discovery log (25 attempts across releases/tags/branches)
- `DATA_REALITY_v1.1.4.md`: Gap analysis showing only 2 FP systems currently in Atlas
- `SUGGESTIONS.md`: Recommendations including FPbase integration as fallback

## Impact

**Current status**: fp-qubit-design v1.1.4 is **BLOCKED** (cannot proceed with ML pipeline with N=2).

**Workarounds considered**:
1. ❌ Recreate locally from `biological_qubits.csv` → violates "canonical source" principle
2. ❌ Expand scope to include NV/SiV centers → violates "FP optical only" specification
3. ⏳ Integrate external sources (FPbase) → planned for v1.2, but increases maintenance burden

**Preferred solution**: Publish canonical `atlas_fp_optical.csv` from Atlas repository.

## Alternative Solutions

If creating a 66-system FP dataset is not feasible:

1. **Option A**: Publish current FP subset (N=2) with clear documentation
   - Label: `atlas_fp_optical_v1.2.1_limited.csv`
   - Update README with realistic expectations

2. **Option B**: Collaborate on FP enrichment
   - I can help integrate FPbase data into Atlas
   - Expand FP coverage to 50+ systems
   - Maintain provenance & licenses (CC BY 4.0)

3. **Option C**: Point to external FP sources
   - Document recommended FP databases (FPbase, UniProt)
   - Provide integration guidance

## Questions

1. Does `atlas_fp_optical.csv` (66 systems) exist internally?
2. If yes, can it be published as a release asset?
3. If no, would you be interested in collaboration to create it?

Thank you for maintaining this valuable resource! 🙏

---

**Project**: fp-qubit-design v1.1.4  
**Author**: Tommy Lepesteur (ORCID: 0009-0009-0577-9563)  
**License**: Code: Apache-2.0 | Data: CC BY 4.0
