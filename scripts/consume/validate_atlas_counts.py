"""
Validate atlas_fp_optical.csv counts against expected v1.2.1 schema
FAIL if counts don't match
"""
import pandas as pd
from pathlib import Path
import sys

# Expected counts (from v1.2.1 specification)
EXPECTED = {
    "N_total": 66,
    "N_measured_AB": 54,
    "N_families_min": 7,
    "families_with_3plus": 7
}

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
CSV_PATH = PROJECT_ROOT / "data" / "processed" / "atlas_fp_optical.csv"
MISMATCH_REPORT = PROJECT_ROOT / "reports" / "ATLAS_MISMATCH.md"

def validate_counts():
    """Validate CSV against expected counts"""
    print("="*60)
    print("VALIDATION: atlas_fp_optical.csv v1.2.1")
    print("="*60)
    
    # Read CSV
    if not CSV_PATH.exists():
        print(f"\n[FAIL] File not found: {CSV_PATH}")
        sys.exit(1)
    
    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        print(f"\n[FAIL] Cannot read CSV: {e}")
        sys.exit(1)
    
    print(f"\n[INFO] Loaded {len(df)} rows")
    
    # Calculate actual counts
    N_total = len(df)
    
    # Try to find measured A/B column
    measured_col = None
    for col in ['measured_AB', 'contrast_quality_tier', 'evidence_type']:
        if col in df.columns:
            measured_col = col
            break
    
    if measured_col == 'contrast_quality_tier':
        N_measured_AB = len(df[df[measured_col].isin(['A', 'B'])])
    elif measured_col == 'measured_AB':
        N_measured_AB = len(df[df[measured_col] == True])
    elif measured_col == 'evidence_type':
        N_measured_AB = len(df[df[measured_col] == 'verifie'])
    else:
        N_measured_AB = 0
    
    # Family counts
    if 'family' in df.columns:
        family_counts = df['family'].value_counts()
        N_families = len(family_counts)
        families_with_3plus = len(family_counts[family_counts >= 3])
    else:
        N_families = 0
        families_with_3plus = 0
    
    # Display results
    print("\n" + "-"*60)
    print("EXPECTED vs ACTUAL")
    print("-"*60)
    print(f"Total entries:        {EXPECTED['N_total']:3d} expected | {N_total:3d} actual")
    print(f"Measured A/B:         {EXPECTED['N_measured_AB']:3d} expected | {N_measured_AB:3d} actual")
    print(f"Families (>=3 each):  {EXPECTED['families_with_3plus']:3d} expected | {families_with_3plus:3d} actual")
    print("-"*60)
    
    # Check for mismatches
    mismatches = []
    
    if N_total != EXPECTED['N_total']:
        delta = N_total - EXPECTED['N_total']
        mismatches.append(f"N_total: {N_total} != {EXPECTED['N_total']} (delta: {delta:+d})")
    
    if N_measured_AB != EXPECTED['N_measured_AB']:
        delta = N_measured_AB - EXPECTED['N_measured_AB']
        mismatches.append(f"N_measured_AB: {N_measured_AB} != {EXPECTED['N_measured_AB']} (delta: {delta:+d})")
    
    if families_with_3plus < EXPECTED['families_with_3plus']:
        delta = families_with_3plus - EXPECTED['families_with_3plus']
        mismatches.append(f"families_with_3plus: {families_with_3plus} < {EXPECTED['families_with_3plus']} (delta: {delta:+d})")
    
    # Generate mismatch report if needed
    if mismatches:
        print("\n[FAIL] MISMATCHES DETECTED!\n")
        
        MISMATCH_REPORT.parent.mkdir(parents=True, exist_ok=True)
        
        report = f"""# Atlas Mismatch Report v1.1.4

**Date**: 2025-10-24  
**File**: `{CSV_PATH.name}`  
**Source**: Fallback Local (Chemin B)

## Expected vs Actual Counts

| Metric | Expected | Actual | Delta | Status |
|--------|----------|--------|-------|--------|
| **Total entries** | {EXPECTED['N_total']} | {N_total} | {N_total - EXPECTED['N_total']:+d} | {'PASS' if N_total == EXPECTED['N_total'] else 'FAIL'} |
| **Measured A/B** | {EXPECTED['N_measured_AB']} | {N_measured_AB} | {N_measured_AB - EXPECTED['N_measured_AB']:+d} | {'PASS' if N_measured_AB == EXPECTED['N_measured_AB'] else 'FAIL'} |
| **Families (>=3)** | {EXPECTED['families_with_3plus']} | {families_with_3plus} | {families_with_3plus - EXPECTED['families_with_3plus']:+d} | {'PASS' if families_with_3plus >= EXPECTED['families_with_3plus'] else 'FAIL'} |

## Detailed Breakdown

### Actual Data

- **Total rows**: {N_total}
- **Measured (tier A/B)**: {N_measured_AB}
- **Unique families**: {N_families}
- **Families with ≥3 entries**: {families_with_3plus}

### Family Distribution

```
{family_counts.to_string() if N_families > 0 else 'No family data'}
```

## Root Cause

The file `atlas_fp_optical.csv` found in the local fallback **does NOT match** the v1.2.1 specification.

**Gap**: {EXPECTED['N_total'] - N_total} missing FP systems ({(EXPECTED['N_total'] - N_total) / EXPECTED['N_total'] * 100:.1f}% of expected)

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
"""
        
        MISMATCH_REPORT.write_text(report, encoding='utf-8')
        print(f"[->] Mismatch report saved: {MISMATCH_REPORT}")
        
        print("\nMISMATCHES:")
        for mismatch in mismatches:
            print(f"  - {mismatch}")
        
        print("\n" + "="*60)
        print("VALIDATION FAILED - See reports/ATLAS_MISMATCH.md")
        print("="*60)
        
        return False
    
    else:
        print("\n[SUCCESS] All counts match! ✓")
        print("="*60)
        return True

def main():
    success = validate_counts()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

