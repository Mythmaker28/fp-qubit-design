#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fetch and filter Atlas v1.2.1 for FP optical systems only.

This script:
1. Downloads biological_qubits.csv from Atlas v1.2.1
2. Validates SHA256
3. Filters for FP optical systems (is_optical=True AND is_fp_like=True)
4. Adds contrast_normalized (ΔF/F₀) and quality tiers
5. Saves to data/external/atlas_fp_optical_v1_2_1.csv

Exit codes:
- 0: Success
- 1: Download/SHA256 failure
- 2: Filtering/validation failure
"""

import sys
import hashlib
import urllib.request
from pathlib import Path

import pandas as pd
import yaml


def calculate_sha256(file_path: Path) -> str:
    """Calculate SHA256 checksum of a file."""
    sha256 = hashlib.sha256()
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            sha256.update(chunk)
    
    return sha256.hexdigest()


def download_atlas(url: str, output_path: Path) -> None:
    """Download Atlas CSV from GitHub."""
    print(f"[INFO] Downloading from: {url}")
    
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"[INFO] Downloaded to: {output_path}")
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        sys.exit(1)


def filter_fp_optical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter for FP optical systems only.
    
    Criteria:
    - Optical readout (ODMR, Optical-only, or FP-related methods)
    - FP-like (fluorescent proteins or quantum dots)
    - Exclude: NMR, ESR, magnetoreception, indirect
    """
    print("\n[INFO] Filtering for FP optical systems...")
    print(f"  Total input rows: {len(df)}")
    
    # Step 1: Filter by method (optical readout)
    optical_methods = ['ODMR', 'Optical-only', 'Fluorescence', 'FRET']
    
    # Check column names (case-insensitive)
    method_col = None
    for col in df.columns:
        if col.lower() in ['methode_lecture', 'method', 'methode']:
            method_col = col
            break
    
    if method_col:
        df_optical = df[df[method_col].isin(optical_methods) | 
                       df[method_col].str.contains('fluor|optical|fret', case=False, na=False)].copy()
    else:
        # If no method column, use system name patterns
        df_optical = df[df['Systeme'].str.contains('fluoresc|GFP|quantum dot|QD', case=False, na=False)].copy()
    
    print(f"  After optical filter: {len(df_optical)} rows")
    
    # Step 2: Exclude non-FP systems
    exclude_patterns = [
        r'NV', r'SiV', r'GeV', r'VSi',  # Color centers
        r'diamant|diamond', r'SiC',  # Semiconductors
        r'NMR', r'ESR', r'EPR',  # Non-optical
        r'hyperpolariz', r'magneto', r'\^13C', r'\^15N',  # NMR/magnetism
    ]
    
    exclude_regex = '|'.join(exclude_patterns)
    
    df_fp = df_optical[~df_optical['Systeme'].str.contains(exclude_regex, case=False, na=False, regex=True)].copy()
    
    print(f"  After FP-like filter: {len(df_fp)} rows")
    
    # Step 3: Keep systems with photophysical data OR contrast data
    # At minimum: (excitation/emission OR quantum yield OR lifetime) OR contrast
    df_fp['has_photo_data'] = (
        df_fp.get('Excitation_nm', pd.Series()).notna() |
        df_fp.get('Emission_nm', pd.Series()).notna() |
        df_fp.get('Photophysique', pd.Series()).notna() |
        df_fp.get('Contraste_%', pd.Series()).notna()
    )
    
    df_fp = df_fp[df_fp['has_photo_data']].copy()
    
    print(f"  After photophysics/contrast filter: {len(df_fp)} rows")
    
    # If still 0, keep ANY FP-like system (fallback)
    if len(df_fp) == 0:
        print("  [WARN] No systems with photophysics/contrast, keeping all FP-like")
        df_fp = df_optical[~df_optical['Systeme'].str.contains(exclude_regex, case=False, na=False, regex=True)].copy()
    
    return df_fp


def add_normalized_contrast(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add contrast_normalized (ΔF/F₀ format) and quality tiers.
    
    contrast_ratio (%) → contrast_normalized (ΔF/F₀)
    
    Quality tiers:
    - A: Measured + peer-reviewed + error bars
    - B: Measured + peer-reviewed + no error bars
    - C: Estimated/computed
    """
    print("\n[INFO] Adding normalized contrast and quality tiers...")
    
    # Contrast normalization: % → ΔF/F₀
    if 'Contraste_%' in df.columns:
        df['contrast_ratio'] = df['Contraste_%']
        # ΔF/F₀ = (I_on - I_off) / I_off = Contrast% / 100
        df['contrast_normalized'] = df['contrast_ratio'] / 100.0
    else:
        df['contrast_ratio'] = None
        df['contrast_normalized'] = None
    
    # Quality tier
    df['contrast_quality_tier'] = 'C'  # Default: computed/estimated
    
    # Tier B: Measured + peer-reviewed (no error bars)
    if 'Source_Contraste' in df.columns:
        has_source = df['Source_Contraste'].notna() & (df['Source_Contraste'] != '')
        is_measured = df.get('contrast_source', pd.Series()) == 'measured'
        
        df.loc[has_source & is_measured & df['contrast_ratio'].notna(), 'contrast_quality_tier'] = 'B'
    
    # Tier A: Measured + peer-reviewed + error bars
    if 'Contraste_err' in df.columns:
        has_error = df['Contraste_err'].notna()
        is_tier_b = df['contrast_quality_tier'] == 'B'
        
        df.loc[has_error & is_tier_b, 'contrast_quality_tier'] = 'A'
    
    # Count by tier
    tier_counts = df['contrast_quality_tier'].value_counts()
    print(f"  Quality tiers: {tier_counts.to_dict()}")
    
    # Contrast source
    if 'contrast_source' not in df.columns:
        df['contrast_source'] = df['contrast_ratio'].apply(
            lambda x: 'measured' if pd.notna(x) else 'unknown'
        )
    
    return df


def build_output_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Build output CSV with required schema."""
    print("\n[INFO] Building output schema...")
    
    # System ID
    if 'SystemID' in df.columns:
        df['system_id'] = df['SystemID']
    elif 'Systeme' in df.columns:
        df['system_id'] = df['Systeme'].str.lower().str.replace(r'[^a-z0-9]+', '_', regex=True)
    
    # Protein name
    df['protein_name'] = df.get('Systeme', 'Unknown')
    
    # Family (try to infer from name)
    def infer_family(name):
        name_lower = str(name).lower()
        if 'gfp' in name_lower or 'egfp' in name_lower:
            return 'GFP'
        elif 'rfp' in name_lower or 'mcherry' in name_lower or 'dsred' in name_lower:
            return 'RFP'
        elif 'yfp' in name_lower:
            return 'YFP'
        elif 'cfp' in name_lower:
            return 'CFP'
        elif 'quantum dot' in name_lower or 'qd' in name_lower:
            return 'QuantumDot'
        else:
            return 'Other'
    
    df['family'] = df['protein_name'].apply(infer_family)
    
    # Excitation/Emission
    df['excitation_nm'] = df.get('Excitation_nm', None)
    df['emission_nm'] = df.get('Emission_nm', None)
    
    # Temperature/pH
    df['temperature_K'] = df.get('Temperature_K', None)
    df['pH'] = None  # Not available in current Atlas schema
    
    # Biosensor flag
    df['is_biosensor'] = df['protein_name'].str.contains('sensor|indicator', case=False, na=False)
    
    # Source refs
    df['source_refs'] = df.get('DOI', '')
    df['license_source'] = 'CC BY 4.0 (Biological Qubits Atlas)'
    df['evidence_type'] = df.get('Verification_statut', 'a_confirmer')
    
    # Select columns
    output_cols = [
        'system_id', 'protein_name', 'family',
        'contrast_ratio', 'contrast_normalized', 'contrast_quality_tier', 'contrast_source',
        'excitation_nm', 'emission_nm', 'temperature_K', 'pH', 'is_biosensor',
        'source_refs', 'license_source', 'evidence_type',
    ]
    
    # Add optional columns if present
    optional_cols = ['quantum_yield', 'lifetime_ns', 'photostability', 'host_context', 'method']
    for col in optional_cols:
        atlas_col_map = {
            'host_context': 'Hote_contexte',
            'method': 'Methode_lecture',
        }
        source_col = atlas_col_map.get(col, col)
        if source_col in df.columns:
            df[col] = df[source_col]
            output_cols.append(col)
    
    df_output = df[output_cols].copy()
    
    print(f"  Output shape: {df_output.shape}")
    print(f"  Columns: {list(df_output.columns)}")
    
    return df_output


def main():
    print("=" * 60)
    print("Fetch & Filter Atlas v1.2.1 - FP Optical Only")
    print("=" * 60)
    print()
    
    # Load config
    config_path = Path("config/data_sources.yaml")
    
    if not config_path.exists():
        print(f"[ERROR] Config not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    atlas_config = config['atlas']
    
    # Download full Atlas CSV
    download_url = atlas_config['full_csv_url']
    expected_sha256 = atlas_config['full_csv_sha256']
    
    temp_path = Path("data/external/atlas_v1_2_1_full.csv")
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    
    download_atlas(download_url, temp_path)
    
    # Validate SHA256
    print("\n[INFO] Validating SHA256...")
    actual_sha256 = calculate_sha256(temp_path)
    
    print(f"  Expected: {expected_sha256}")
    print(f"  Actual:   {actual_sha256}")
    
    if actual_sha256 != expected_sha256:
        print("[ERROR] SHA256 mismatch!")
        sys.exit(1)
    
    print("  [OK] SHA256 valid")
    
    # Load CSV
    print("\n[INFO] Loading Atlas CSV...")
    df = pd.read_csv(temp_path)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Filter for FP optical
    df_fp = filter_fp_optical(df)
    
    if len(df_fp) == 0:
        print("[ERROR] No FP optical systems found after filtering!")
        sys.exit(2)
    
    # Add normalized contrast
    df_fp = add_normalized_contrast(df_fp)
    
    # Build output schema
    df_output = build_output_schema(df_fp)
    
    # Save
    output_path = Path(atlas_config['fp_optical_csv_local'])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df_output.to_csv(output_path, index=False)
    
    print(f"\n[INFO] Saved: {output_path}")
    print(f"  Total FP optical systems: {len(df_output)}")
    print(f"  With contrast (any tier): {int(df_output['contrast_ratio'].notna().sum())}")
    print(f"  Tier A: {int((df_output['contrast_quality_tier'] == 'A').sum())}")
    print(f"  Tier B: {int((df_output['contrast_quality_tier'] == 'B').sum())}")
    print(f"  Tier C: {int((df_output['contrast_quality_tier'] == 'C').sum())}")
    
    print()
    print("=" * 60)
    print("Fetch complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

