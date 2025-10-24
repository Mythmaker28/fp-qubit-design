#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v1.3.1 ETL: Build training table with advanced features
- Filter useful systems (contrast > 0, complete features)
- Feature engineering: excitation_nm, emission_nm, Stokes shift
- Log-transform target: log1p(contrast_normalized)
- Gate check: N_utiles ≥ 100
"""

import pandas as pd
import numpy as np
import json
import hashlib
from pathlib import Path
from datetime import datetime
from collections import Counter

# Set seed
np.random.seed(1337)


def compute_sha256(filepath):
    """Calculate SHA256"""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_augmented_data():
    """Load augmented dataset"""
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    csv_path = PROJECT_ROOT / "data" / "raw" / "atlas" / "atlas_fp_optical_v2_1_augmented.csv"
    
    print(f"\n[LOAD] Reading {csv_path.name}...")
    df = pd.read_csv(csv_path, encoding='utf-8')
    
    # Clean
    df = df.dropna(subset=['SystemID'])
    df = df[df['SystemID'].str.strip() != '']
    
    print(f"  [INFO] Total records: {len(df)}")
    
    return df


def filter_useful_systems(df):
    """Filter useful systems for ML"""
    print("\n[FILTER] Filtering useful systems...")
    
    # Criterion 1: contrast > 0
    mask_contrast = (df['contrast_normalized'].notna()) & (df['contrast_normalized'] > 0)
    print(f"  [INFO] With contrast > 0: {mask_contrast.sum()}")
    
    # Criterion 2: family exists
    if 'family' not in df.columns:
        df['family'] = 'Unknown'
    
    # Fill missing family
    for idx, row in df.iterrows():
        if pd.isna(row.get('family')) or row.get('family', '').strip() == '':
            if row.get('is_biosensor') == 1.0:
                pname = str(row.get('protein_name', '')).split('-')[0].split('_')[0]
                df.at[idx, 'family'] = pname if pname else 'Biosensor'
            else:
                df.at[idx, 'family'] = 'Unknown'
    
    mask_family = df['family'].notna() & (df['family'] != '') & (df['family'] != 'Unknown')
    print(f"  [INFO] With known family: {mask_family.sum()}")
    
    # Criterion 3: temperature & pH
    mask_temp = df['temperature_K'].notna()
    mask_ph = df['pH'].notna()
    print(f"  [INFO] With temperature_K: {mask_temp.sum()}")
    print(f"  [INFO] With pH: {mask_ph.sum()}")
    
    # Combined
    mask_useful = mask_contrast & mask_family & mask_temp & mask_ph
    
    df_useful = df[mask_useful].copy()
    df_excluded = df[~mask_useful].copy()
    
    print(f"  [SUCCESS] Useful systems: {len(df_useful)}")
    print(f"  [INFO] Excluded systems: {len(df_excluded)}")
    
    return df_useful, df_excluded


def engineer_features(df):
    """
    Engineer advanced features:
    - excitation_nm, emission_nm (already exist)
    - Stokes shift = emission - excitation
    - log1p transform target
    """
    print("\n[FEATURES] Engineering advanced features...")
    
    # Stokes shift
    if 'excitation_nm' in df.columns and 'emission_nm' in df.columns:
        df['stokes_shift_nm'] = df['emission_nm'] - df['excitation_nm']
        print(f"  [INFO] Stokes shift: {df['stokes_shift_nm'].notna().sum()} values")
    else:
        df['stokes_shift_nm'] = np.nan
        print(f"  [WARN] excitation_nm or emission_nm missing - Stokes shift set to NaN")
    
    # Log-transform target
    df['contrast_normalized_raw'] = df['contrast_normalized'].copy()
    df['target_contrast_log'] = np.log1p(df['contrast_normalized'])
    
    print(f"  [INFO] Target transformed: log1p(contrast)")
    print(f"  [INFO] Raw range: [{df['contrast_normalized_raw'].min():.2f}, {df['contrast_normalized_raw'].max():.2f}]")
    print(f"  [INFO] Log range: [{df['target_contrast_log'].min():.2f}, {df['target_contrast_log'].max():.2f}]")
    
    # Spectral region (based on emission)
    def classify_spectral_region(emission):
        if pd.isna(emission):
            return 'unknown'
        elif emission < 490:
            return 'blue'
        elif emission < 520:
            return 'green'
        elif emission < 580:
            return 'yellow'
        elif emission < 620:
            return 'orange'
        elif emission < 700:
            return 'red'
        else:
            return 'far_red'
    
    df['spectral_region'] = df['emission_nm'].apply(classify_spectral_region)
    print(f"  [INFO] Spectral regions classified")
    
    # Parse context (in_vivo vs in_cellulo)
    def parse_context(ctx):
        if pd.isna(ctx):
            return 'unknown'
        ctx_str = str(ctx).lower()
        if 'in_vivo' in ctx_str:
            return 'in_vivo'
        elif 'in_cellulo' in ctx_str:
            return 'in_cellulo'
        elif 'in_vitro' in ctx_str:
            return 'in_vitro'
        else:
            return 'unknown'
    
    df['context_type'] = df['context'].apply(parse_context)
    print(f"  [INFO] Context types parsed")
    
    # Feature summary
    feature_cols = [
        'temperature_K', 'pH', 'is_biosensor', 'excitation_nm', 'emission_nm',
        'stokes_shift_nm', 'spectral_region', 'context_type', 'family'
    ]
    
    available_features = [f for f in feature_cols if f in df.columns]
    print(f"  [SUCCESS] Total features: {len(available_features)}")
    
    return df, available_features


def build_training_table(df, feature_cols):
    """Build final training table"""
    print("\n[BUILD] Constructing training table...")
    
    # Core columns
    core_cols = [
        'SystemID', 'protein_name', 'family', 'is_biosensor',
        'temperature_K', 'pH', 'context', 'context_type',
        'excitation_nm', 'emission_nm', 'stokes_shift_nm',
        'spectral_region',
        'target_contrast_log',  # transformed target
        'contrast_normalized_raw',  # original for reference
        'quality_tier', 'source'
    ]
    
    # Keep only existing columns
    available_cols = [c for c in core_cols if c in df.columns]
    df_train = df[available_cols].copy()
    
    # Add provenance
    df_train['data_version'] = 'v1.3.1'
    df_train['ingestion_date'] = datetime.now().strftime('%Y-%m-%d')
    
    print(f"  [INFO] Training table shape: {df_train.shape}")
    print(f"  [INFO] Columns: {list(df_train.columns)}")
    
    return df_train


def generate_metadata(df_train, raw_sha256, exclusion_details):
    """Generate metadata"""
    
    # Family distribution
    family_counts = df_train['family'].value_counts().to_dict()
    families_3plus = sum(1 for count in family_counts.values() if count >= 3)
    
    # Target statistics (log-transformed)
    target_stats_log = {
        'mean': float(df_train['target_contrast_log'].mean()),
        'std': float(df_train['target_contrast_log'].std()),
        'min': float(df_train['target_contrast_log'].min()),
        'max': float(df_train['target_contrast_log'].max()),
        'median': float(df_train['target_contrast_log'].median()),
    }
    
    # Target statistics (raw)
    target_stats_raw = {
        'mean': float(df_train['contrast_normalized_raw'].mean()),
        'std': float(df_train['contrast_normalized_raw'].std()),
        'min': float(df_train['contrast_normalized_raw'].min()),
        'max': float(df_train['contrast_normalized_raw'].max()),
        'median': float(df_train['contrast_normalized_raw'].median()),
    }
    
    # Feature completeness
    feature_completeness = {}
    for col in ['excitation_nm', 'emission_nm', 'stokes_shift_nm']:
        if col in df_train.columns:
            feature_completeness[col] = {
                'count': int(df_train[col].notna().sum()),
                'missing': int(df_train[col].isna().sum()),
                'pct_complete': float(df_train[col].notna().mean() * 100)
            }
    
    metadata = {
        'version': 'v1.3.1',
        'source': 'atlas_fp_optical_v2_1_augmented.csv',
        'source_sha256': raw_sha256,
        'ingestion_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'n_total_raw': len(df_train) + len(exclusion_details),
        'n_useful': len(df_train),
        'n_excluded': len(exclusion_details),
        'families_total': len(family_counts),
        'families_with_3plus_samples': families_3plus,
        'family_distribution': family_counts,
        'target_statistics_log': target_stats_log,
        'target_statistics_raw': target_stats_raw,
        'feature_completeness': feature_completeness,
        'features': list(df_train.columns),
        'target_transform': 'log1p(contrast_normalized)',
        'filtering_criteria': {
            'contrast_normalized': '> 0 and NOT NULL',
            'family': 'NOT NULL and != Unknown',
            'temperature_K': 'NOT NULL',
            'pH': 'NOT NULL'
        },
        'license': 'CC BY 4.0',
        'curator': 'v1.3.1_autonomous_agent'
    }
    
    return metadata


def gate_check(n_useful):
    """Check if N_utiles ≥ 100 (GO/NO-GO gate)"""
    print("\n" + "="*70)
    print("GATE CHECK: N_utiles >= 100")
    print("="*70)
    
    print(f"\n  N_utiles = {n_useful}")
    
    if n_useful >= 100:
        decision = "GO - v1.3.1 FULL PIPELINE"
        status = "PASS"
        next_step = "Proceed to training with GBDT + CQR"
    else:
        decision = "FALLBACK - v1.2.5 (RELAXED CRITERIA)"
        status = "WARN"
        next_step = f"N={n_useful} < 100 - Use relaxed acceptance criteria"
    
    print(f"\n  DECISION: {decision}")
    print(f"  Status: {status}")
    print(f"  Next: {next_step}")
    
    return status, decision


def main():
    print("="*70)
    print("v1.3.1 ETL PIPELINE — Advanced Feature Engineering")
    print("="*70)
    
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    REPORTS_DIR = PROJECT_ROOT / "reports"
    
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load augmented data
    df = load_augmented_data()
    
    # Compute SHA256 of augmented CSV
    raw_csv_path = PROJECT_ROOT / "data" / "raw" / "atlas" / "atlas_fp_optical_v2_1_augmented.csv"
    raw_sha256 = compute_sha256(raw_csv_path)
    print(f"\n[SHA256] {raw_sha256}")
    
    # Filter useful
    df_useful, df_excluded = filter_useful_systems(df)
    
    # Engineer features
    df_featured, feature_cols = engineer_features(df_useful)
    
    # Build training table
    df_train = build_training_table(df_featured, feature_cols)
    
    # Generate metadata
    exclusion_details = []
    for idx, row in df_excluded.iterrows():
        reasons = []
        if pd.isna(row.get('contrast_normalized')) or row.get('contrast_normalized', 0) <= 0:
            reasons.append('no_contrast')
        if pd.isna(row.get('family')) or row.get('family', '') in ['', 'Unknown']:
            reasons.append('no_family')
        if pd.isna(row.get('temperature_K')):
            reasons.append('no_temperature')
        if pd.isna(row.get('pH')):
            reasons.append('no_pH')
        
        exclusion_details.append({
            'SystemID': row.get('SystemID', 'UNKNOWN'),
            'protein_name': row.get('protein_name', 'UNKNOWN'),
            'reasons': ', '.join(reasons) if reasons else 'unknown'
        })
    
    metadata = generate_metadata(df_train, raw_sha256, exclusion_details)
    
    # Save outputs
    print("\n[SAVE] Writing outputs...")
    
    # Training table
    train_csv_path = PROCESSED_DIR / "training_table_v1_3_1.csv"
    df_train.to_csv(train_csv_path, index=False, encoding='utf-8')
    print(f"  [SUCCESS] {train_csv_path}")
    
    # Metadata
    metadata_path = PROCESSED_DIR / "TRAINING.METADATA_v1_3_1.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  [SUCCESS] {metadata_path}")
    
    # Target metadata
    target_metadata = {
        'n_samples': len(df_train),
        'target_column': 'target_contrast_log',
        'target_transform': 'log1p(contrast_normalized)',
        'statistics_log': metadata['target_statistics_log'],
        'statistics_raw': metadata['target_statistics_raw'],
        'version': 'v1.3.1'
    }
    target_meta_path = PROCESSED_DIR / "TRAIN_MEASURED.METADATA_v1_3_1.json"
    with open(target_meta_path, 'w', encoding='utf-8') as f:
        json.dump(target_metadata, f, indent=2, ensure_ascii=False)
    print(f"  [SUCCESS] {target_meta_path}")
    
    # Gate check
    status, decision = gate_check(len(df_train))
    
    print("\n" + "="*70)
    print("ETL PIPELINE COMPLETE")
    print("="*70)
    
    return status, len(df_train), metadata


if __name__ == "__main__":
    status, n_useful, metadata = main()
    
    if status == "PASS":
        print("\n[GO] N_utiles >= 100 - Full v1.3.1 pipeline authorized")
        exit(0)
    else:
        print(f"\n[FALLBACK] N_utiles = {n_useful} < 100 - Use v1.2.5 relaxed criteria")
        exit(0)

