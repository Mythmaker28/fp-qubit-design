#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build 2 separate training tables for v1.1.3:
1. atlas_all_real.csv - ALL real Atlas systems (incl. non-optical)
2. training_table_optical.csv - ONLY optical systems with contrast_ratio

This separation allows:
- Keeping all Atlas data traceable (atlas_all_real)
- Focus on optical FP/QD for training (training_table_optical)
"""

import json
from pathlib import Path
from datetime import datetime

import pandas as pd


def main():
    print("=" * 60)
    print("Build Training Tables v1.1.3 (Separate All/Optical)")
    print("=" * 60)
    print()
    
    # Load classified merged data
    merged_path = Path("data/interim/atlas_merged_classified.csv")
    
    if not merged_path.exists():
        print(f"[ERROR] {merged_path} not found. Run classify_modality.py first.")
        return
    
    df = pd.read_csv(merged_path)
    print(f"[INFO] Loaded {len(df)} classified systems")
    
    # ============================
    # TABLE 1: atlas_all_real.csv
    # ============================
    print("\n[INFO] Building atlas_all_real.csv...")
    
    # Select ALL systems (no filter)
    df_all = df.copy()
    
    # Minimal column set (keep provenance)
    cols_all = [
        'SystemID', 'Systeme', 'Classe', 'Hote_contexte', 'Methode_lecture',
        'Contraste_%', 'Contraste_err', 'Source_Contraste',
        'Temperature_K', 'T1_s', 'T1_s_err', 'T2_us', 'T2_us_err',
        'Frequence', 'B0_Tesla', 'Qualite', 'Verification_statut', 'In_vivo_flag',
        'source_release_tag', 'source_asset', 'source_sha256', 'published_at',
        'is_optical', 'is_fp_like', 'in_scope_training',
    ]
    
    # Keep only available columns
    cols_all_available = [col for col in cols_all if col in df_all.columns]
    df_all_export = df_all[cols_all_available].copy()
    
    # Save
    output_all = Path("data/processed/atlas_all_real.csv")
    output_all.parent.mkdir(parents=True, exist_ok=True)
    df_all_export.to_csv(output_all, index=False)
    
    print(f"  [INFO] Saved: {output_all}")
    print(f"  [INFO] Total systems: {len(df_all_export)}")
    print(f"    - Optical: {int(df_all_export['is_optical'].sum())}")
    print(f"    - Non-optical: {len(df_all_export) - int(df_all_export['is_optical'].sum())}")
    
    # ====================================
    # TABLE 2: training_table_optical.csv
    # ====================================
    print("\n[INFO] Building training_table_optical.csv...")
    
    # Filter: optical systems ONLY
    df_optical = df[df['is_optical'] == True].copy()
    
    print(f"  [INFO] Filtered to {len(df_optical)} optical systems")
    
    # Rename columns for consistency
    rename_map = {
        'Systeme': 'protein_name',
        'Classe': 'class',
        'Hote_contexte': 'host_context',
        'Methode_lecture': 'method',
        'Contraste_%': 'contrast_ratio',
        'Contraste_err': 'contrast_ci',
        'Source_Contraste': 'contrast_source_col',
        'Temperature_K': 'temperature_K',
        'T1_s': 't1_s',
        'T2_us': 't2_us',
        'Frequence': 'frequency',
        'B0_Tesla': 'b0_tesla',
        'Qualite': 'quality',
        'Verification_statut': 'verification_status',
        'In_vivo_flag': 'in_vivo_flag',
    }
    
    # Rename available columns
    for old, new in rename_map.items():
        if old in df_optical.columns:
            df_optical.rename(columns={old: new}, inplace=True)
    
    # Add contrast_source if not present
    if 'contrast_source' not in df_optical.columns:
        if 'contrast_ratio' in df_optical.columns:
            df_optical['contrast_source'] = df_optical['contrast_ratio'].apply(
                lambda x: 'measured' if pd.notna(x) else 'unknown'
            )
        else:
            df_optical['contrast_source'] = 'unknown'
    
    # Add is_real flag (all=1 for Atlas)
    df_optical['is_real'] = 1
    
    # Select minimal columns for training
    cols_training = [
        'SystemID', 'protein_name', 'class', 'host_context', 'method',
        'contrast_ratio', 'contrast_ci', 'contrast_source',
        'temperature_K', 't1_s', 't2_us', 'frequency', 'b0_tesla',
        'quality', 'verification_status', 'in_vivo_flag',
        'source_release_tag', 'source_asset', 'source_sha256', 'published_at',
        'is_optical', 'is_fp_like', 'in_scope_training', 'is_real',
    ]
    
    # Keep only available columns
    cols_training_available = [col for col in cols_training if col in df_optical.columns]
    df_optical_export = df_optical[cols_training_available].copy()
    
    # Save
    output_optical = Path("data/processed/training_table_optical.csv")
    df_optical_export.to_csv(output_optical, index=False)
    
    print(f"  [INFO] Saved: {output_optical}")
    print(f"  [INFO] Optical systems: {len(df_optical_export)}")
    
    # Count contrast
    if 'contrast_ratio' in df_optical_export.columns:
        n_with_contrast = int(df_optical_export['contrast_ratio'].notna().sum())
    else:
        n_with_contrast = 0
    
    print(f"    - With contrast: {n_with_contrast} / {len(df_optical_export)}")
    
    # Count FP-like
    if 'is_fp_like' in df_optical_export.columns:
        n_fp_like = int(df_optical_export['is_fp_like'].sum())
        print(f"    - FP-like: {n_fp_like}")
        
        # FP-like with contrast
        df_fp = df_optical_export[df_optical_export['is_fp_like'] == True]
        if 'contrast_ratio' in df_fp.columns:
            n_fp_with_contrast = int(df_fp['contrast_ratio'].notna().sum())
        else:
            n_fp_with_contrast = 0
        
        print(f"    - FP-like with contrast: {n_fp_with_contrast} / {n_fp_like}")
    
    # ============================
    # METADATA
    # ============================
    print("\n[INFO] Generating TRAINING.METADATA.json...")
    
    metadata = {
        'schema_version': 'v1.1.3',
        'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'source': 'biological-qubits-atlas (9 sources reconciled)',
        'license': 'CC BY 4.0',
        'citation': 'Lepesteur, T. (2025). Biological Qubits Atlas. GitHub. https://github.com/Mythmaker28/biological-qubits-atlas',
        
        'tables': {
            'atlas_all_real.csv': {
                'description': 'ALL real Atlas systems (optical + non-optical)',
                'n_systems': len(df_all_export),
                'n_optical': int(df_all_export['is_optical'].sum()) if 'is_optical' in df_all_export.columns else None,
                'n_non_optical': len(df_all_export) - int(df_all_export['is_optical'].sum()) if 'is_optical' in df_all_export.columns else None,
            },
            'training_table_optical.csv': {
                'description': 'ONLY optical systems (filtered from atlas_all_real)',
                'n_systems': len(df_optical_export),
                'n_with_contrast': n_with_contrast,
                'n_fp_like': n_fp_like if 'is_fp_like' in df_optical_export.columns else None,
                'n_fp_like_with_contrast': n_fp_with_contrast if 'is_fp_like' in df_optical_export.columns else None,
            },
        },
        
        'columns': {
            'SystemID': 'Unique identifier (normalized system name)',
            'protein_name': 'Original system name from Atlas',
            'class': 'System class (A/B/C/D)',
            'host_context': 'Biological context (in_vitro, in_cellulo, in_vivo, ex_vivo)',
            'method': 'Readout method (ODMR, ESR, NMR, Optical-only, Indirect)',
            'contrast_ratio': 'Contrast (%) from Atlas Contraste_% column',
            'contrast_ci': 'Contrast error/CI',
            'contrast_source': 'measured (if Atlas has Contraste_%), unknown otherwise',
            'temperature_K': 'Temperature (K)',
            't1_s': 'T1 relaxation (s)',
            't2_us': 'T2 coherence (µs)',
            'frequency': 'Operating frequency',
            'b0_tesla': 'Magnetic field (T)',
            'quality': 'Quality rating (1-3)',
            'verification_status': 'verifie or a_confirmer',
            'in_vivo_flag': 'In vivo demonstration (0/1)',
            'is_optical': 'Optical readout system (1/0)',
            'is_fp_like': 'Fluorescent protein or quantum dot (1/0)',
            'in_scope_training': 'Suitable for FP-qubit design (optical + FP-like)',
            'is_real': 'Real data (1) vs synthetic (0)',
            'source_release_tag': 'Git tag/branch of origin',
            'source_asset': 'Asset filename',
            'source_sha256': 'SHA256 checksum',
            'published_at': 'Publication date',
        },
        
        'notes': [
            'v1.1.3 separates ALL real systems (atlas_all_real) from OPTICAL training slice (training_table_optical)',
            'Modality classification based on method, class, and keyword patterns',
            'Optical: fluorescence, ODMR, quantum dots, FP families',
            'Non-optical: NMR, ESR, hyperpolarized, magnetoreception, indirect',
            'Only 3 FP-like systems found (1 FP + 2 QD); rest are color centers (NV, SiV, etc.)',
        ],
    }
    
    # Add statistics
    if 'contrast_ratio' in df_optical_export.columns and df_optical_export['contrast_ratio'].notna().any():
        df_contrast = df_optical_export[df_optical_export['contrast_ratio'].notna()]
        metadata['statistics'] = {
            'optical_contrast_ratio': {
                'n': len(df_contrast),
                'mean': float(df_contrast['contrast_ratio'].mean()),
                'std': float(df_contrast['contrast_ratio'].std()),
                'min': float(df_contrast['contrast_ratio'].min()),
                'max': float(df_contrast['contrast_ratio'].max()),
            },
        }
    
    # Save metadata
    metadata_path = Path("data/processed/TRAINING.METADATA.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  [INFO] Saved: {metadata_path}")
    
    print()
    print("=" * 60)
    print("Training tables complete!")
    print("=" * 60)
    print()
    print("SUMMARY:")
    print(f"  - atlas_all_real.csv:          {len(df_all_export)} systems (all)")
    print(f"  - training_table_optical.csv:  {len(df_optical_export)} systems (optical only)")
    print(f"    -> With contrast: {n_with_contrast}")
    print(f"    -> FP-like: {n_fp_like if 'is_fp_like' in df_optical_export.columns else 'N/A'}")
    print("=" * 60)


if __name__ == "__main__":
    main()

