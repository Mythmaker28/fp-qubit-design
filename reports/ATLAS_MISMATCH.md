# Atlas Mismatch Report v1.1.4

**Date**: 2025-10-24  
**File**: `atlas_fp_optical.csv`  
**Source**: Fallback Local (Chemin B)

## Expected vs Actual Counts

| Metric | Expected | Actual | Delta | Status |
|--------|----------|--------|-------|--------|
| **Total entries** | 66 | 2 | -64 | FAIL |
| **Measured A/B** | 54 | 0 | -54 | FAIL |
| **Families (>=3)** | 7 | 0 | -7 | FAIL |

## Detailed Breakdown

### Actual Data

- **Total rows**: 2
- **Measured (tier A/B)**: 0
- **Unique families**: 2
- **Families with ≥3 entries**: 0

### Family Distribution

```
family
Other         1
QuantumDot    1
```

## Root Cause

The file `atlas_fp_optical.csv` found in the local fallback **does NOT match** the v1.2.1 specification.

**Gap**: 64 missing FP systems (97.0% of expected)

## Verdict

**STATUS**: ❌ **VALIDATION FAILED**

The counts do not match the v1.2.1 specification (66 total, 54 measured A/B, ≥7 families).

## Recommendations

1. **Wait for Atlas publication**: The canonical `atlas_fp_optical.csv` is not yet available in the public repository.
2. **Integrate FPbase**: Use FPbase API to fetch ≥50 FP optical systems with measured photophysical properties.
3. **Literature mining**: Extract data from primary sources.

See `reports/SUGGESTIONS.md` for detailed alternatives.

---

**License**: Data CC BY 4.0  
**Author**: Tommy Lepesteur (ORCID: 0009-0009-0577-9563)
