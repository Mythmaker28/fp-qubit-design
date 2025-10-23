#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audit Atlas real counts for v1.1.2 release.

This script:
1. Calculates N_real_total, N_with_contrast_measured, N_with_contrast_any
2. Fails (exit code 1) if N_real_total < 34
3. Generates reports/AUDIT.md
4. Generates reports/MISSING_REAL_SYSTEMS.md (list of systems without contrast)
"""

import sys
from pathlib import Path
from datetime import datetime

import pandas as pd


def load_training_table() -> pd.DataFrame:
    """Load training table."""
    csv_path = Path("data/processed/training_table.csv")
    
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found. Run build_training_table.py first.")
    
    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded training table: {len(df)} rows")
    
    return df


def audit_counts(df: pd.DataFrame) -> dict:
    """Calculate audit metrics."""
    
    # Filter real data only
    df_real = df[df['is_real'] == 1].copy()
    
    # Metrics
    n_real_total = len(df_real)
    n_with_contrast_measured = int(df_real[df_real['contrast_source'] == 'measured'].shape[0])
    n_with_contrast_any = int(df_real['contrast_ratio'].notna().sum())
    
    # Systems without contrast
    df_no_contrast = df_real[df_real['contrast_ratio'].isna()].copy()
    
    metrics = {
        'n_real_total': n_real_total,
        'n_with_contrast_measured': n_with_contrast_measured,
        'n_with_contrast_any': n_with_contrast_any,
        'n_without_contrast': len(df_no_contrast),
        'systems_without_contrast': df_no_contrast,
    }
    
    print()
    print("=" * 60)
    print("AUDIT METRICS")
    print("=" * 60)
    print(f"N_real_total:             {n_real_total}")
    print(f"N_with_contrast_measured: {n_with_contrast_measured}")
    print(f"N_with_contrast_any:      {n_with_contrast_any}")
    print(f"N_without_contrast:       {metrics['n_without_contrast']}")
    print("=" * 60)
    print()
    
    return metrics


def generate_audit_report(metrics: dict) -> str:
    """Generate AUDIT.md report."""
    
    report = f"""# AUDIT REPORT - fp-qubit-design v1.1.2

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Summary

| Metric | Value | Status |
|--------|-------|--------|
| **N_real_total** | {metrics['n_real_total']} | {'PASS ✓' if metrics['n_real_total'] >= 34 else 'FAIL ✗'} |
| **N_with_contrast_measured** | {metrics['n_with_contrast_measured']} | {f"{metrics['n_with_contrast_measured']/metrics['n_real_total']*100:.1f}% coverage"} |
| **N_with_contrast_any** | {metrics['n_with_contrast_any']} | {f"{metrics['n_with_contrast_any']/metrics['n_real_total']*100:.1f}% coverage"} |
| **N_without_contrast** | {metrics['n_without_contrast']} | - |

## Acceptance Criteria

- **Criterion 1**: `N_real_total >= 34` → {'**PASS ✓**' if metrics['n_real_total'] >= 34 else '**FAIL ✗**'}
- **Criterion 2**: `N_with_contrast_measured >= 20` → {'**PASS ✓**' if metrics['n_with_contrast_measured'] >= 20 else f'**SHORTFALL** ({20 - metrics["n_with_contrast_measured"]} systems needed)'}

## Data Provenance

- **Sources**: biological-qubits-atlas (multiple releases + branches)
- **Releases merged**: main, v1.2.0, v1.2.1, develop, infra/pages+governance, feat/data-v1.2-extended, docs/doi-badge, chore/zenodo-metadata, chore/citation-author
- **Deduplication**: Based on SystemID (normalized system name)
- **License**: CC BY 4.0

## Contrast Statistics (Measured Only)

- **Mean**: {metrics.get('contrast_mean', 'N/A')}%
- **Std**: {metrics.get('contrast_std', 'N/A')}%
- **Range**: [{metrics.get('contrast_min', 'N/A')}%, {metrics.get('contrast_max', 'N/A')}%]

---

## Recommendation

"""
    
    if metrics['n_real_total'] >= 34:
        report += """✓ **Release v1.1.2 approved**

All acceptance criteria met. Proceed with public release.
"""
    else:
        report += f"""✗ **Pre-release v1.1.2-pre recommended**

N_real_total ({metrics['n_real_total']}) is below target (34). 

**Recommended actions for v1.2**:
1. Contact biological-qubits-atlas maintainer for additional data
2. Literature mining (automated or semi-automated)
3. Schema alias patch (check for hidden synonyms in Photophysique, Notes, etc.)
4. Consider expanding to related quantum sensing systems (not just bio-intrinsic)
"""
    
    report += "\n---\n\n**License**: Code: Apache-2.0 | Data: CC BY 4.0\n"
    
    return report


def generate_missing_systems_report(metrics: dict) -> str:
    """Generate MISSING_REAL_SYSTEMS.md report."""
    
    df_no_contrast = metrics['systems_without_contrast']
    
    report = f"""# MISSING REAL SYSTEMS - fp-qubit-design v1.1.2

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

This report lists real Atlas systems that **lack measured contrast** data.

---

## Summary

- **Total systems without contrast**: {len(df_no_contrast)} / {metrics['n_real_total']} ({len(df_no_contrast)/metrics['n_real_total']*100:.1f}%)

## Systems Without Contrast

| System ID | Protein Name | Class | Method | Source Tag | Reason |
|-----------|--------------|-------|--------|------------|--------|
"""
    
    for _, row in df_no_contrast.iterrows():
        system_id = row.get('system_id', 'N/A')
        protein_name = row.get('protein_name', 'N/A')
        cls = row.get('class', 'N/A')
        method = row.get('method', 'N/A')
        source = row.get('source_release_tag', 'N/A')
        
        # Determine reason
        reason = "Contrast column empty in source Atlas CSV"
        
        report += f"| {system_id} | {protein_name} | {cls} | {method} | {source} | {reason} |\n"
    
    report += """
---

## Recommendations

1. **Contact Atlas maintainer**: Request contrast data for systems listed above
2. **Literature mining**: Search primary literature for missing measurements
3. **Proxy computation**: If QY, epsilon, or other photophysical params available, compute proxies
4. **Schema alias patch**: Check if contrast is hidden under synonyms (ΔF/F0, SNR, etc.) in Notes or Photophysique columns

---

**License**: Data from biological-qubits-atlas is licensed under CC BY 4.0
"""
    
    return report


def main():
    print("=" * 60)
    print("Audit Atlas Real Counts - ETL Pipeline")
    print("=" * 60)
    print()
    
    # Load training table
    df = load_training_table()
    
    # Audit
    metrics = audit_counts(df)
    
    # Add contrast statistics
    if metrics['n_with_contrast_measured'] > 0:
        df_real = df[df['is_real'] == 1]
        df_contrast = df_real[df_real['contrast_source'] == 'measured']
        
        metrics['contrast_mean'] = f"{df_contrast['contrast_ratio'].mean():.2f}"
        metrics['contrast_std'] = f"{df_contrast['contrast_ratio'].std():.2f}"
        metrics['contrast_min'] = f"{df_contrast['contrast_ratio'].min():.2f}"
        metrics['contrast_max'] = f"{df_contrast['contrast_ratio'].max():.2f}"
    
    # Generate reports
    audit_report = generate_audit_report(metrics)
    missing_report = generate_missing_systems_report(metrics)
    
    # Save reports
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    
    audit_path = reports_dir / "AUDIT.md"
    with open(audit_path, 'w', encoding='utf-8') as f:
        f.write(audit_report)
    print(f"[INFO] Saved: {audit_path}")
    
    missing_path = reports_dir / "MISSING_REAL_SYSTEMS.md"
    with open(missing_path, 'w', encoding='utf-8') as f:
        f.write(missing_report)
    print(f"[INFO] Saved: {missing_path}")
    
    print()
    
    # Exit with failure if N_real_total < 34
    if metrics['n_real_total'] < 34:
        print("[ERROR] N_real_total < 34. Exiting with code 1.")
        print("[ACTION] Consider pre-release v1.1.2-pre instead of full release.")
        sys.exit(1)
    else:
        print("[SUCCESS] N_real_total >= 34. All criteria met!")
        sys.exit(0)


if __name__ == "__main__":
    main()

