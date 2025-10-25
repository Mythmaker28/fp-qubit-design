#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build final training_table.csv from atlas_merged.csv.

This script:
1. Loads atlas_merged.csv
2. Selects and renames columns
3. Adds is_real flag (all=1 for Atlas data)
4. Writes training_table.csv + TRAINING.METADATA.json
"""

import json
from pathlib import Path
from datetime import datetime

import pandas as pd


def load_merged_atlas() -> pd.DataFrame:
    """Load merged Atlas data."""
    csv_path = Path("data/interim/atlas_merged.csv")
    
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found. Run merge_atlas_assets.py first.")
    
    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded {len(df)} systems from atlas_merged.csv")
    
    return df


def build_training_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build training table with minimal columns."""
    
    # Map Atlas columns to training table columns
    column_mapping = {
        'SystemID': 'system_id',
        'Systeme': 'protein_name',
        'Classe': 'class',
        'Hote_contexte': 'host_context',
        'Methode_lecture': 'method',
        'Contraste_%': 'contrast_ratio',
        'Contraste_err': 'contrast_ci',
        'Temperature_K': 'temperature_K',
        'T1_s': 't1_s',
        'T2_us': 't2_us',
        'Frequence': 'frequency',
        'B0_Tesla': 'b0_tesla',
        'Qualite': 'quality',
        'Verification_statut': 'verification_status',
        'In_vivo_flag': 'in_vivo_flag',
        'source_release_tag': 'source_release_tag',
        'source_asset': 'source_asset',
        'source_sha256': 'source_sha256',
        'published_at': 'published_at',
    }
    
    # Select and rename columns
    available_cols = [col for col in column_mapping.keys() if col in df.columns]
    df_training = df[available_cols].rename(columns=column_mapping)
    
    # Add is_real flag (all Atlas data is real)
    df_training['is_real'] = 1
    
    # Add contrast_source (measured if non-null, else unknown)
    if 'contrast_ratio' in df_training.columns:
        df_training['contrast_source'] = df_training['contrast_ratio'].apply(
            lambda x: 'measured' if pd.notna(x) else 'unknown'
        )
    
    print(f"[INFO] Training table shape: {df_training.shape}")
    print(f"[INFO] Columns: {list(df_training.columns)}")
    
    return df_training


def generate_metadata(df: pd.DataFrame) -> dict:
    """Generate TRAINING.METADATA.json."""
    
    metadata = {
        'schema_version': 'v1.1.2',
        'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'source': 'biological-qubits-atlas (multiple releases + branches)',
        'license': 'CC BY 4.0',
        'citation': 'Lepesteur, T. (2025). Biological Qubits Atlas. GitHub. https://github.com/Mythmaker28/biological-qubits-atlas',
        'total_systems': len(df),
        'real_systems': int((df['is_real'] == 1).sum()),
        'synthetic_systems': 0,
        'with_contrast_measured': int(df[df['contrast_source'] == 'measured'].shape[0]) if 'contrast_source' in df.columns else 0,
        'columns': {
            'system_id': 'Unique identifier (normalized system name)',
            'protein_name': 'Original system name from Atlas',
            'class': 'System class (A/B/C/D)',
            'host_context': 'Biological context (in_vitro, in_cellulo, in_vivo, ex_vivo)',
            'method': 'Readout method (ODMR, ESR, NMR, Optical-only, Indirect)',
            'contrast_ratio': 'Contrast (%), directly from Atlas Contraste_% column',
            'contrast_ci': 'Contrast error/CI from Atlas',
            'contrast_source': 'measured (if Atlas has Contraste_%), unknown otherwise',
            'temperature_K': 'Temperature in Kelvin',
            't1_s': 'T1 relaxation time (seconds)',
            't2_us': 'T2 coherence time (microseconds)',
            'frequency': 'Operating frequency',
            'b0_tesla': 'Magnetic field (Tesla)',
            'quality': 'Quality rating (1-3)',
            'verification_status': 'verifie or a_confirmer',
            'in_vivo_flag': 'In vivo demonstration (0/1)',
            'is_real': 'Real data (1) vs synthetic (0)',
            'source_release_tag': 'Git tag/branch of origin',
            'source_asset': 'Asset filename',
            'source_sha256': 'SHA256 checksum',
            'published_at': 'Publication date (YYYY-MM-DD)',
        },
        'statistics': {
            'contrast_ratio': {
                'n': int(df['contrast_ratio'].notna().sum()) if 'contrast_ratio' in df.columns else 0,
                'mean': float(df['contrast_ratio'].mean()) if 'contrast_ratio' in df.columns and df['contrast_ratio'].notna().any() else None,
                'std': float(df['contrast_ratio'].std()) if 'contrast_ratio' in df.columns and df['contrast_ratio'].notna().any() else None,
                'min': float(df['contrast_ratio'].min()) if 'contrast_ratio' in df.columns and df['contrast_ratio'].notna().any() else None,
                'max': float(df['contrast_ratio'].max()) if 'contrast_ratio' in df.columns and df['contrast_ratio'].notna().any() else None,
            },
            'temperature_K': {
                'n': int(df['temperature_K'].notna().sum()) if 'temperature_K' in df.columns else 0,
                'mean': float(df['temperature_K'].mean()) if 'temperature_K' in df.columns and df['temperature_K'].notna().any() else None,
                'std': float(df['temperature_K'].std()) if 'temperature_K' in df.columns and df['temperature_K'].notna().any() else None,
                'min': float(df['temperature_K'].min()) if 'temperature_K' in df.columns and df['temperature_K'].notna().any() else None,
                'max': float(df['temperature_K'].max()) if 'temperature_K' in df.columns and df['temperature_K'].notna().any() else None,
            },
        },
        'notes': [
            'All data sourced from biological-qubits-atlas (multiple releases and branches)',
            'Deduplication performed based on SystemID (normalized system name)',
            'contrast_ratio comes directly from Atlas Contraste_% column (no computation)',
            'No synthetic data included in v1.1.2',
        ],
    }
    
    return metadata


def main():
    print("=" * 60)
    print("Build Training Table - ETL Pipeline")
    print("=" * 60)
    print()
    
    # Load merged Atlas
    df = load_merged_atlas()
    
    # Build training table
    df_training = build_training_table(df)
    
    # Save CSV
    output_csv = Path("data/processed/training_table.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    df_training.to_csv(output_csv, index=False)
    print(f"\n[INFO] Saved: {output_csv}")
    
    # Generate metadata
    metadata = generate_metadata(df_training)
    
    output_json = Path("data/processed/TRAINING.METADATA.json")
    with open(output_json, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"[INFO] Saved: {output_json}")
    
    print()
    print("=" * 60)
    print(f"Training table complete! {len(df_training)} systems")
    print(f"  - With contrast: {metadata['with_contrast_measured']}")
    print(f"  - Real systems: {metadata['real_systems']}")
    print("=" * 60)


if __name__ == "__main__":
    main()



