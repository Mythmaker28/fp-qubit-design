#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Select top-12 candidates for wet-lab testing
Apply diversity constraints and uncertainty criteria
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def select_top12_candidates(lab_sheet_file, output_dir, max_calcium=3, max_per_family=6, min_non_calcium=6):
    """Select top-12 candidates with diversity constraints"""
    
    print("=== SELECTING TOP-12 CANDIDATES ===")
    
    # Load lab sheet
    df = pd.read_csv(lab_sheet_file)
    print(f"Loaded {len(df)} candidates from lab sheet")
    
    # Sort by criteria: high y_pred, low PI90_width
    df_sorted = df.sort_values(['y_pred', 'PI90_width'], ascending=[False, True])
    
    print(f"Sorting criteria: y_pred (DESC), PI90_width (ASC)")
    print(f"y_pred range: {df['y_pred'].min():.3f} - {df['y_pred'].max():.3f}")
    print(f"PI90_width range: {df['PI90_width'].min():.3f} - {df['PI90_width'].max():.3f}")
    
    # Apply selection constraints
    selected = []
    family_counts = {}
    calcium_count = 0
    non_calcium_count = 0
    
    print(f"\nSelection constraints:")
    print(f"- Max Calcium: {max_calcium}")
    print(f"- Max per family: {max_per_family}")
    print(f"- Min non-Calcium: {min_non_calcium}")
    
    for idx, row in df_sorted.iterrows():
        family = row['family']
        is_calcium = (family == 'Calcium')
        
        # Check constraints
        can_add = True
        reasons = []
        
        # Check Calcium limit
        if is_calcium and calcium_count >= max_calcium:
            can_add = False
            reasons.append(f"Calcium limit ({max_calcium}) reached")
        
        # Check family limit
        if family in family_counts and family_counts[family] >= max_per_family:
            can_add = False
            reasons.append(f"Family {family} limit ({max_per_family}) reached")
        
        # Check if we need more non-Calcium
        remaining_slots = 12 - len(selected)
        needed_non_calcium = max(0, min_non_calcium - non_calcium_count)
        if is_calcium and remaining_slots <= needed_non_calcium and non_calcium_count < min_non_calcium:
            can_add = False
            reasons.append(f"Need {needed_non_calcium} more non-Calcium candidates")
        
        if can_add:
            selected.append(row)
            family_counts[family] = family_counts.get(family, 0) + 1
            if is_calcium:
                calcium_count += 1
            else:
                non_calcium_count += 1
            
            print(f"  Selected #{len(selected)}: {row['canonical_name']} ({family}) - y_pred={row['y_pred']:.3f}, PI90_width={row['PI90_width']:.1f}")
            
            if len(selected) >= 12:
                break
        else:
            print(f"  Skipped: {row['canonical_name']} ({family}) - {'; '.join(reasons)}")
    
    # Create final selection
    top12_df = pd.DataFrame(selected)
    
    # Save top-12
    top12_path = Path(output_dir) / "shortlist_top12_final.csv"
    top12_df.to_csv(top12_path, index=False)
    
    # Generate rationale
    create_selection_rationale(top12_df, df_sorted, output_dir)
    
    print(f"\n=== SELECTION COMPLETE ===")
    print(f"Selected: {len(top12_df)} candidates")
    print(f"Calcium: {calcium_count}/{max_calcium}")
    print(f"Non-Calcium: {non_calcium_count}/{min_non_calcium}")
    print(f"Family distribution:")
    for family, count in family_counts.items():
        print(f"  {family}: {count}")
    
    return top12_df

def create_selection_rationale(top12_df, original_sorted, output_dir):
    """Create selection rationale document"""
    
    print("\n=== CREATING SELECTION RATIONALE ===")
    
    # Calculate statistics
    y_pred_mean = top12_df['y_pred'].mean()
    y_pred_min = top12_df['y_pred'].min()
    y_pred_max = top12_df['y_pred'].max()
    pi90_mean = top12_df['PI90_width'].mean()
    pi90_min = top12_df['PI90_width'].min()
    pi90_max = top12_df['PI90_width'].max()
    
    # Create rationale content
    rationale = "# Selection Rationale for Top-12 Candidates\n\n"
    rationale += "## Selection Rules\n"
    rationale += "1. **Primary sorting**: High y_pred (DESC), Low PI90_width (ASC)\n"
    rationale += "2. **Calcium limit**: Maximum 3 Calcium candidates\n"
    rationale += "3. **Family diversity**: Maximum 6 candidates per family\n"
    rationale += "4. **Non-Calcium minimum**: At least 6 non-Calcium candidates\n"
    rationale += "5. **Uncertainty priority**: Lower PI90_width preferred for same y_pred\n\n"
    
    rationale += "## Selected Candidates\n\n"
    rationale += "| Rank | Name | Family | y_pred | PI90_width | Excitation | Emission |\n"
    rationale += "|------|------|--------|--------|------------|------------|----------|\n"
    
    for idx, row in top12_df.iterrows():
        rank = idx + 1
        name = row['canonical_name']
        family = row['family']
        y_pred = f"{row['y_pred']:.3f}"
        pi90 = f"{row['PI90_width']:.1f}"
        exc = f"{row['excitation_nm']:.0f}" if pd.notna(row['excitation_nm']) else "N/A"
        em = f"{row['emission_nm']:.0f}" if pd.notna(row['emission_nm']) else "N/A"
        
        rationale += f"| {rank} | {name} | {family} | {y_pred} | {pi90} | {exc} | {em} |\n"
    
    rationale += f"\n## Selection Statistics\n"
    rationale += f"- **Total selected**: {len(top12_df)}/20 candidates\n"
    rationale += f"- **Prediction range**: {y_pred_min:.3f} - {y_pred_max:.3f} (mean: {y_pred_mean:.3f})\n"
    rationale += f"- **Uncertainty range**: {pi90_min:.1f} - {pi90_max:.1f} (mean: {pi90_mean:.1f})\n"
    rationale += f"- **Families represented**: {top12_df['family'].nunique()}\n"
    rationale += f"- **Calcium candidates**: {sum(top12_df['family'] == 'Calcium')}\n"
    rationale += f"- **Non-Calcium candidates**: {sum(top12_df['family'] != 'Calcium')}\n"
    
    # Save rationale
    rationale_path = Path(output_dir) / "selection_rationale.md"
    with open(rationale_path, 'w', encoding='utf-8') as f:
        f.write(rationale)
    
    print(f"Saved rationale: {rationale_path}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Select top-12 candidates')
    parser.add_argument('--lab_sheet', required=True, help='Path to lab sheet CSV file')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--max_calcium', type=int, default=3, help='Maximum Calcium candidates')
    parser.add_argument('--max_per_family', type=int, default=6, help='Maximum per family')
    parser.add_argument('--min_non_calcium', type=int, default=6, help='Minimum non-Calcium candidates')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    # Select top-12
    top12_df = select_top12_candidates(
        args.lab_sheet, 
        args.output,
        args.max_calcium,
        args.max_per_family,
        args.min_non_calcium
    )
    
    print(f"\nTOP12 READY: {len(top12_df)}/20 retenus")

if __name__ == "__main__":
    main()
