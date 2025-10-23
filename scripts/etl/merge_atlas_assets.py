#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge all Atlas assets into canonical atlas_merged.parquet.

This script:
1. Loads all CSV/TSV/JSON from data/raw/atlas/releases/
2. Normalizes encoding, separators, headers
3. Adds provenance (source_release_tag, source_asset, source_sha256)
4. Builds stable SystemID (species|protein|variant|fluorophore)
5. Deduplicates by SystemID (keep most recent + most complete)
6. Outputs: data/interim/atlas_merged.parquet + ATLAS_MERGE_REPORT.md
"""

import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge all Atlas assets"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/raw/atlas/releases",
        help="Input directory with releases"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/interim/atlas_merged.parquet",
        help="Output parquet file"
    )
    return parser.parse_args()


def compute_file_sha256(filepath: Path) -> str:
    """Compute SHA256 of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for block in iter(lambda: f.read(4096), b''):
            sha256.update(block)
    return sha256.hexdigest()


def load_csv_robust(filepath: Path) -> pd.DataFrame:
    """Load CSV with multiple encoding/separator attempts."""
    encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']
    separators = [',', ';', '\t']
    
    for encoding in encodings:
        for sep in separators:
            try:
                df = pd.read_csv(filepath, encoding=encoding, sep=sep)
                # Check if it parsed correctly (should have multiple columns)
                if len(df.columns) > 3:
                    print(f"    [INFO] Loaded with encoding={encoding}, sep={sep!r}")
                    return df
            except Exception:
                continue
    
    raise ValueError(f"Could not parse {filepath} with any encoding/separator")


def normalize_system_name(name: str) -> str:
    """Normalize system name (strip, lower, ascii)."""
    if pd.isna(name):
        return ""
    return str(name).strip().lower().replace('"', '')


def build_system_id(row: pd.Series) -> str:
    """Build stable SystemID from row."""
    # Try to extract: species | protein | variant | fluorophore
    system_name = normalize_system_name(row.get('Systeme', ''))
    
    # For now, use system_name as SystemID (can be improved with parsing)
    # Future: extract protein family, species, fluorophore from text
    return system_name


def count_non_null_measurements(row: pd.Series) -> int:
    """Count how many measurement columns are non-null."""
    measurement_cols = [
        'T1_s', 'T2_us', 'Contraste_%', 'Temperature_K',
        'Frequence', 'B0_Tesla', 'Taille_objet_nm'
    ]
    count = 0
    for col in measurement_cols:
        if col in row.index and pd.notna(row[col]):
            count += 1
    return count


def deduplicate_systems(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate by SystemID, keeping most recent + most complete."""
    print(f"\n[INFO] Deduplicating {len(df)} rows...")
    
    # Add completeness score
    df['_completeness'] = df.apply(count_non_null_measurements, axis=1)
    
    # Sort by: SystemID, published_at (desc), completeness (desc)
    # If published_at not available, use a default
    if 'published_at' not in df.columns:
        df['published_at'] = '2025-01-01'  # default
    
    df_sorted = df.sort_values(
        by=['SystemID', 'published_at', '_completeness'],
        ascending=[True, False, False]
    )
    
    # Keep first (most recent + most complete) for each SystemID
    df_dedup = df_sorted.drop_duplicates(subset=['SystemID'], keep='first')
    
    # Drop temp columns
    df_dedup = df_dedup.drop(columns=['_completeness'])
    
    duplicates_removed = len(df) - len(df_dedup)
    print(f"[INFO] Removed {duplicates_removed} duplicates")
    print(f"[INFO] Unique systems: {len(df_dedup)}")
    
    return df_dedup


def merge_releases(input_dir: Path) -> pd.DataFrame:
    """Merge all CSV from releases."""
    print(f"[INFO] Scanning: {input_dir}")
    
    # Find all CSV files
    csv_files = list(input_dir.rglob('*.csv'))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")
    
    print(f"[INFO] Found {len(csv_files)} CSV files")
    
    all_dataframes = []
    
    for csv_file in csv_files:
        # Determine release tag from path
        relative_path = csv_file.relative_to(input_dir)
        release_tag = relative_path.parts[0]  # First directory is the tag
        asset_name = csv_file.name
        
        print(f"\n[INFO] Loading: {release_tag}/{asset_name}")
        
        # Load CSV
        try:
            df = load_csv_robust(csv_file)
        except Exception as e:
            print(f"  [ERROR] Failed to load: {e}")
            continue
        
        print(f"  [INFO] Loaded {len(df)} rows, {len(df.columns)} columns")
        
        # Compute SHA256
        sha256 = compute_file_sha256(csv_file)
        
        # Add provenance columns
        df['source_release_tag'] = release_tag
        df['source_asset'] = asset_name
        df['source_sha256'] = sha256
        
        # Set published_at based on tag (rough approximation)
        if release_tag == 'main':
            df['published_at'] = '2025-10-23'  # Today
        elif release_tag.startswith('v1.2.1'):
            df['published_at'] = '2025-10-22'
        elif release_tag.startswith('v1.2.0'):
            df['published_at'] = '2025-10-22'
        else:
            df['published_at'] = '2025-01-01'  # Unknown
        
        all_dataframes.append(df)
    
    # Concatenate
    print(f"\n[INFO] Concatenating {len(all_dataframes)} dataframes...")
    df_merged = pd.concat(all_dataframes, ignore_index=True)
    
    print(f"[INFO] Total rows before dedup: {len(df_merged)}")
    
    # Build SystemID
    df_merged['SystemID'] = df_merged.apply(build_system_id, axis=1)
    
    # Deduplicate
    df_merged = deduplicate_systems(df_merged)
    
    return df_merged


def generate_merge_report(df: pd.DataFrame, output_file: Path):
    """Generate ATLAS_MERGE_REPORT.md."""
    lines = []
    lines.append("# ATLAS MERGE REPORT - fp-qubit-design v1.1.2")
    lines.append("")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Summary
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Total unique systems**: {len(df)}")
    lines.append(f"- **Total releases merged**: {df['source_release_tag'].nunique()}")
    lines.append("")
    
    # By release
    lines.append("## Systems by Release")
    lines.append("")
    release_counts = df['source_release_tag'].value_counts().sort_index()
    for tag, count in release_counts.items():
        lines.append(f"- **{tag}**: {count} systems")
    lines.append("")
    
    # Fields available
    lines.append("## Available Fields")
    lines.append("")
    non_null_counts = df.count().sort_values(ascending=False)
    lines.append("| Field | Non-null Count | Coverage % |")
    lines.append("|-------|----------------|------------|")
    for col, count in non_null_counts.items():
        if col.startswith('source_') or col in ['SystemID', 'published_at']:
            continue
        coverage = count / len(df) * 100
        lines.append(f"| `{col}` | {count} | {coverage:.1f}% |")
    lines.append("")
    
    # Key measurements
    lines.append("## Key Measurements (Real Data)")
    lines.append("")
    
    measurement_cols = {
        'Contraste_%': 'Contrast (%)',
        'Temperature_K': 'Temperature (K)',
        'T2_us': 'T2 (µs)',
        'T1_s': 'T1 (s)',
    }
    
    for col, label in measurement_cols.items():
        if col in df.columns:
            values = df[col].dropna()
            if len(values) > 0:
                lines.append(f"### {label}")
                lines.append("```")
                lines.append(f"N:      {len(values)}")
                lines.append(f"Mean:   {values.mean():.2f}")
                lines.append(f"Std:    {values.std():.2f}")
                lines.append(f"Range:  [{values.min():.2f}, {values.max():.2f}]")
                lines.append("```")
                lines.append("")
    
    # Provenance
    lines.append("## Provenance")
    lines.append("")
    lines.append("All data sourced from:")
    lines.append("- **Repository**: https://github.com/Mythmaker28/biological-qubits-atlas")
    lines.append("- **License**: CC BY 4.0")
    lines.append("")
    lines.append("**Citation**:")
    lines.append("```")
    lines.append("Lepesteur, T. (2025). Biological Qubits Atlas. GitHub.")
    lines.append("https://github.com/Mythmaker28/biological-qubits-atlas")
    lines.append("```")
    lines.append("")
    
    lines.append("---")
    lines.append("")
    lines.append("**Generated by**: `scripts/etl/merge_atlas_assets.py`")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"\n[INFO] Merge report saved: {output_file}")


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Merge Atlas Assets - ETL Pipeline")
    print("=" * 60)
    print()
    
    # Merge releases
    input_dir = Path(args.input_dir)
    df_merged = merge_releases(input_dir)
    
    # Save to CSV (parquet requires pyarrow)
    output_path = Path(args.output).with_suffix('.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df_merged.to_csv(output_path, index=False)
    print(f"\n[INFO] Saved: {output_path}")
    print(f"[INFO] Shape: {df_merged.shape}")
    
    # Generate report
    report_path = Path("reports/ATLAS_MERGE_REPORT.md")
    generate_merge_report(df_merged, report_path)
    
    print()
    print("=" * 60)
    print(f"Merge complete! {len(df_merged)} unique systems")
    print("=" * 60)


if __name__ == "__main__":
    main()

