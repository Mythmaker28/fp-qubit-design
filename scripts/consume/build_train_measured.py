"""
Build train_measured.csv from atlas_fp_optical.csv
Filter for tiers A/B only (measured, high quality)
"""
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "atlas_fp_optical.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "train_measured.csv"
METADATA_JSON = PROJECT_ROOT / "data" / "processed" / "TRAIN_MEASURED.METADATA.json"

def build_train_measured():
    """Filter for measured (A/B tier) only"""
    print("="*60)
    print("Building train_measured.csv")
    print("="*60)
    
    # Load full dataset
    df = pd.read_csv(INPUT_CSV)
    print(f"\n[INFO] Loaded {len(df)} total FP systems")
    
    # Filter for tier A or B (measured, high quality)
    df_measured = df[df['contrast_quality_tier'].isin(['A', 'B'])].copy()
    print(f"[INFO] Filtered to {len(df_measured)} tier A/B systems")
    
    # Check family distribution
    family_counts = df_measured['family'].value_counts()
    families_with_3plus = len(family_counts[family_counts >= 3])
    
    print(f"\n[INFO] Family distribution (tier A/B):")
    for family, count in family_counts.items():
        marker = " [OK]" if count >= 3 else " [WARN: <3]"
        print(f"  {family}: {count}{marker}")
    
    print(f"\n[INFO] Families with >=3 samples: {families_with_3plus}")
    
    if families_with_3plus < 3:
        print(f"\n[WARN] Only {families_with_3plus} families with >=3 samples")
        print("       Cross-validation may be challenging")
    
    # Sort by family for readability
    df_measured = df_measured.sort_values('family').reset_index(drop=True)
    
    # Save
    df_measured.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[OK] Saved to {OUTPUT_CSV}")
    
    # Metadata
    metadata = {
        "source_file": "atlas_fp_optical.csv",
        "filter_criteria": "contrast_quality_tier in ['A', 'B']",
        "n_total_input": len(df),
        "n_measured_output": len(df_measured),
        "families": family_counts.to_dict(),
        "families_with_3plus": families_with_3plus,
        "columns": list(df_measured.columns),
        "created_date": datetime.now().isoformat(),
        "purpose": "Training dataset for ML pipeline (measured contrast only)"
    }
    
    with open(METADATA_JSON, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"[OK] Metadata saved to {METADATA_JSON}")
    
    print("\n" + "="*60)
    print("[SUCCESS] train_measured.csv ready!")
    print(f"N = {len(df_measured)} measured FP systems")
    print(f"Families: {len(family_counts)} ({families_with_3plus} with >=3 samples)")
    print("="*60)
    
    return df_measured

if __name__ == "__main__":
    df_measured = build_train_measured()


