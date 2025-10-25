#!/usr/bin/env python3
"""
Integration script for Atlas v2.2 data (189 systems) - v1.3.2 mission
Integrates atlas_fp_optical_v2_2.csv and builds training table with advanced features
"""

import pandas as pd
import numpy as np
import json
import hashlib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_atlas_v2_2():
    """Load and validate Atlas v2.2 data"""
    print("=== LOADING ATLAS v2.2 DATA ===")
    
    # Load the new v2.2 data
    df = pd.read_csv("data/raw/atlas_fp_optical_v2_2.csv")
    print(f"Loaded {len(df)} total systems from Atlas v2.2")
    
    # Basic validation
    print(f"Columns: {list(df.columns)}")
    print(f"Shape: {df.shape}")
    
    return df

def clean_and_harmonize(df):
    """Clean and harmonize the data"""
    print("\n=== CLEANING & HARMONIZING ===")
    
    # Handle SystemID issues - create unique IDs for rows without SystemID
    initial_count = len(df)
    
    # Create SystemID for rows that don't have one
    mask_no_systemid = df['SystemID'].isna() | (df['SystemID'] == '')
    n_no_systemid = mask_no_systemid.sum()
    print(f"Found {n_no_systemid} rows without SystemID")
    
    if n_no_systemid > 0:
        df.loc[mask_no_systemid, 'SystemID'] = [f"FP_{i+10000:04d}" for i in range(n_no_systemid)]
    
    # Remove duplicates based on SystemID
    df = df.drop_duplicates(subset=['SystemID'], keep='first')
    print(f"Processed {initial_count} rows, kept {len(df)} unique systems")
    
    # Clean family names
    df['family'] = df['family'].fillna('Unknown')
    df['family'] = df['family'].str.strip()
    
    # Clean protein names
    df['protein_name'] = df['protein_name'].fillna('Unknown')
    df['protein_name'] = df['protein_name'].str.strip()
    
    # Ensure contrast_normalized is numeric
    df['contrast_normalized'] = pd.to_numeric(df['contrast_normalized'], errors='coerce')
    
    # Clean context
    df['context'] = df['context'].fillna('unknown')
    df['context'] = df['context'].str.strip()
    
    print(f"After cleaning: {len(df)} systems")
    return df

def filter_useful_systems(df):
    """Filter for useful systems (measured target + required features)"""
    print("\n=== FILTERING USEFUL SYSTEMS ===")
    
    initial_count = len(df)
    
    # Filter criteria for useful systems
    useful_mask = (
        (df['contrast_normalized'].notna()) &  # Has measured contrast
        (df['contrast_normalized'] > 0) &      # Positive contrast
        (df['family'].notna()) &               # Has family
        (df['family'] != 'Unknown') &          # Family is known
        (df['temperature_K'].notna()) &        # Has temperature
        (df['pH'].notna())                     # Has pH
    )
    
    df_useful = df[useful_mask].copy()
    
    print(f"Initial systems: {initial_count}")
    print(f"Useful systems: {len(df_useful)}")
    print(f"Filtered out: {initial_count - len(df_useful)} systems")
    
    # Show family distribution
    family_counts = df_useful['family'].value_counts()
    print(f"\nFamily distribution:")
    for family, count in family_counts.items():
        print(f"  {family}: {count}")
    
    return df_useful

def engineer_features(df):
    """Engineer advanced features"""
    print("\n=== FEATURE ENGINEERING ===")
    
    # Optical features
    df['excitation_nm'] = pd.to_numeric(df['excitation_nm'], errors='coerce')
    df['emission_nm'] = pd.to_numeric(df['emission_nm'], errors='coerce')
    
    # Calculate Stokes shift
    df['stokes_shift_nm'] = df['emission_nm'] - df['excitation_nm']
    
    # Spectral regions
    def get_spectral_region(excitation):
        if pd.isna(excitation):
            return 'unknown'
        elif excitation < 450:
            return 'blue'
        elif excitation < 500:
            return 'cyan'
        elif excitation < 550:
            return 'green'
        elif excitation < 600:
            return 'yellow'
        elif excitation < 650:
            return 'orange'
        else:
            return 'red'
    
    df['spectral_region'] = df['excitation_nm'].apply(get_spectral_region)
    
    # Context type
    def get_context_type(context):
        context_lower = str(context).lower()
        if 'in_vivo' in context_lower:
            return 'in_vivo'
        elif 'in_cellulo' in context_lower:
            return 'in_cellulo'
        else:
            return 'other'
    
    df['context_type'] = df['context'].apply(get_context_type)
    
    # Missing value flags
    df['excitation_missing'] = df['excitation_nm'].isna()
    df['emission_missing'] = df['emission_nm'].isna()
    df['contrast_missing'] = df['contrast_normalized'].isna()
    
    # Fill missing optical values with median
    df['excitation_nm'] = df['excitation_nm'].fillna(df['excitation_nm'].median())
    df['emission_nm'] = df['emission_nm'].fillna(df['emission_nm'].median())
    df['stokes_shift_nm'] = df['stokes_shift_nm'].fillna(df['stokes_shift_nm'].median())
    
    print(f"Features engineered:")
    print(f"  - excitation_nm: {df['excitation_nm'].notna().sum()}/{len(df)} available")
    print(f"  - emission_nm: {df['emission_nm'].notna().sum()}/{len(df)} available")
    print(f"  - stokes_shift_nm: {df['stokes_shift_nm'].notna().sum()}/{len(df)} available")
    print(f"  - spectral_region: {df['spectral_region'].value_counts().to_dict()}")
    print(f"  - context_type: {df['context_type'].value_counts().to_dict()}")
    
    return df

def build_training_table(df):
    """Build the final training table"""
    print("\n=== BUILDING TRAINING TABLE ===")
    
    # Select and order columns for training
    training_cols = [
        'SystemID', 'protein_name', 'family', 'is_biosensor',
        'contrast_normalized', 'context', 'temperature_K', 'pH',
        'excitation_nm', 'emission_nm', 'stokes_shift_nm',
        'spectral_region', 'context_type',
        'excitation_missing', 'emission_missing', 'contrast_missing',
        'doi', 'source', 'year'
    ]
    
    # Ensure all columns exist
    missing_cols = [col for col in training_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns: {missing_cols}")
        for col in missing_cols:
            df[col] = None
    
    training_table = df[training_cols].copy()
    
    # Apply log1p transformation to target
    training_table['contrast_log1p'] = np.log1p(training_table['contrast_normalized'])
    
    print(f"Training table shape: {training_table.shape}")
    print(f"Target range (original): [{training_table['contrast_normalized'].min():.3f}, {training_table['contrast_normalized'].max():.3f}]")
    print(f"Target range (log1p): [{training_table['contrast_log1p'].min():.3f}, {training_table['contrast_log1p'].max():.3f}]")
    
    return training_table

def save_artifacts(training_table, df_useful):
    """Save all artifacts"""
    print("\n=== SAVING ARTIFACTS ===")
    
    # Save training table
    training_table.to_csv("data/processed/training_table_v1_3_2.csv", index=False)
    print("Saved: data/processed/training_table_v1_3_2.csv")
    
    # Create metadata
    metadata = {
        "version": "v1.3.2",
        "description": "Training table for v1.3.2 with Atlas v2.2 data (189 systems)",
        "n_total": len(df_useful),
        "n_families": df_useful['family'].nunique(),
        "target_variable": "contrast_normalized",
        "target_transformation": "log1p",
        "features": {
            "numerical": ["excitation_nm", "emission_nm", "stokes_shift_nm", "temperature_K", "pH"],
            "categorical": ["family", "spectral_region", "context_type", "is_biosensor"],
            "flags": ["excitation_missing", "emission_missing", "contrast_missing"]
        },
        "family_distribution": df_useful['family'].value_counts().to_dict(),
        "context_distribution": df_useful['context_type'].value_counts().to_dict(),
        "spectral_distribution": df_useful['spectral_region'].value_counts().to_dict(),
        "target_stats": {
            "mean": float(df_useful['contrast_normalized'].mean()),
            "std": float(df_useful['contrast_normalized'].std()),
            "min": float(df_useful['contrast_normalized'].min()),
            "max": float(df_useful['contrast_normalized'].max()),
            "median": float(df_useful['contrast_normalized'].median())
        }
    }
    
    # Save metadata
    with open("data/processed/TRAINING.METADATA_v1_3_2.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("Saved: data/processed/TRAINING.METADATA_v1_3_2.json")
    
    # Create measured metadata
    measured_metadata = {
        "version": "v1.3.2",
        "description": "Measured systems metadata for v1.3.2",
        "n_measured": len(df_useful),
        "measurement_stats": {
            "contrast_mean": float(df_useful['contrast_normalized'].mean()),
            "contrast_std": float(df_useful['contrast_normalized'].std()),
            "temperature_mean": float(df_useful['temperature_K'].mean()),
            "ph_mean": float(df_useful['pH'].mean())
        },
        "sources": df_useful['source'].value_counts().to_dict(),
        "years": df_useful['year'].value_counts().to_dict()
    }
    
    with open("data/processed/TRAIN_MEASURED.METADATA_v1_3_2.json", "w") as f:
        json.dump(measured_metadata, f, indent=2)
    print("Saved: data/processed/TRAIN_MEASURED.METADATA_v1_3_2.json")
    
    return metadata, measured_metadata

def generate_audit_report(df_useful, metadata):
    """Generate audit report"""
    print("\n=== GENERATING AUDIT REPORT ===")
    
    report = f"""# AUDIT REPORT v1.3.2 - Atlas v2.2 Integration

## Summary
- **Version**: v1.3.2
- **Data Source**: Atlas v2.2 (atlas_fp_optical_v2_2.csv)
- **Total Systems**: {len(df_useful)}
- **Families**: {df_useful['family'].nunique()}
- **Target Variable**: contrast_normalized (log1p transformed for training)

## Data Quality
- **Complete Systems**: {len(df_useful)} (100%)
- **Missing Contrast**: {df_useful['contrast_normalized'].isna().sum()}
- **Missing Family**: {df_useful['family'].isna().sum()}
- **Missing Temperature**: {df_useful['temperature_K'].isna().sum()}
- **Missing pH**: {df_useful['pH'].isna().sum()}

## Family Distribution
"""
    
    for family, count in df_useful['family'].value_counts().items():
        report += f"- **{family}**: {count} systems\n"
    
    report += f"""
## Context Distribution
"""
    
    for context, count in df_useful['context_type'].value_counts().items():
        report += f"- **{context}**: {count} systems\n"
    
    report += f"""
## Spectral Distribution
"""
    
    for region, count in df_useful['spectral_region'].value_counts().items():
        report += f"- **{region}**: {count} systems\n"
    
    report += f"""
## Target Statistics
- **Mean**: {df_useful['contrast_normalized'].mean():.3f}
- **Std**: {df_useful['contrast_normalized'].std():.3f}
- **Min**: {df_useful['contrast_normalized'].min():.3f}
- **Max**: {df_useful['contrast_normalized'].max():.3f}
- **Median**: {df_useful['contrast_normalized'].median():.3f}

## Features
- **Numerical**: excitation_nm, emission_nm, stokes_shift_nm, temperature_K, pH
- **Categorical**: family, spectral_region, context_type, is_biosensor
- **Flags**: excitation_missing, emission_missing, contrast_missing

## Sources
"""
    
    for source, count in df_useful['source'].value_counts().items():
        report += f"- **{source}**: {count} systems\n"
    
    report += f"""
## Gate Check: N_utiles >= 100
- **Current N_utiles**: {len(df_useful)}
- **Target**: >= 100
- **Status**: {'PASS' if len(df_useful) >= 100 else 'FAIL'}

## Decision
{'GO' if len(df_useful) >= 100 else 'NO-GO'} - {'Proceed to v1.3.2 training' if len(df_useful) >= 100 else 'Insufficient data for v1.3.2'}
"""
    
    with open("reports/AUDIT_v1.3.2.md", "w") as f:
        f.write(report)
    print("Saved: reports/AUDIT_v1.3.2.md")
    
    return report

def main():
    """Main integration pipeline"""
    print("=== ATLAS v2.2 INTEGRATION - v1.3.2 MISSION ===")
    print("Target: N_utiles >= 100 for v1.3.2 release")
    print()
    
    # Load data
    df = load_atlas_v2_2()
    
    # Clean and harmonize
    df = clean_and_harmonize(df)
    
    # Filter useful systems
    df_useful = filter_useful_systems(df)
    
    # Check gate
    n_useful = len(df_useful)
    print(f"\n=== GATE CHECK ===")
    print(f"N_utiles = {n_useful}")
    print(f"Target: >= 100")
    print(f"Status: {'PASS' if n_useful >= 100 else 'FAIL'}")
    
    if n_useful < 100:
        print("\nGATE FAILED - Insufficient data for v1.3.2")
        print("Falling back to v1.2.5 with relaxed criteria")
        return
    
    print("\nGATE PASSED - Proceeding to v1.3.2")
    
    # Engineer features
    df_useful = engineer_features(df_useful)
    
    # Build training table
    training_table = build_training_table(df_useful)
    
    # Save artifacts
    metadata, measured_metadata = save_artifacts(training_table, df_useful)
    
    # Generate audit report
    generate_audit_report(df_useful, metadata)
    
    print(f"\n=== INTEGRATION COMPLETE ===")
    print(f"N_utiles: {n_useful} (target: >=100)")
    print(f"Training table: data/processed/training_table_v1_3_2.csv")
    print(f"Metadata: data/processed/TRAINING.METADATA_v1_3_2.json")
    print(f"Audit: reports/AUDIT_v1.3.2.md")
    print(f"\nNext: Proceed to v1.3.2 training with RandomForest + CQR")

if __name__ == "__main__":
    main()
