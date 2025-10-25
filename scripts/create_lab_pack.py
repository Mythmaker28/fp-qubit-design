#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create lab pack from shortlist top-20
Enrich with Atlas data and filter recommendations
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def create_lab_pack(shortlist_file, atlas_file, output_dir):
    """Create enriched lab pack from shortlist"""
    
    print("=== CREATING LAB PACK ===")
    
    # Read shortlist
    shortlist_df = pd.read_csv(shortlist_file)
    print(f"Loaded shortlist: {len(shortlist_df)} candidates")
    
    # Read Atlas data
    atlas_df = pd.read_csv(atlas_file)
    print(f"Loaded Atlas: {len(atlas_df)} entries")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Try to join by canonical_name first, then by exact name match
    enriched_df = shortlist_df.copy()
    
    # Add placeholder columns
    enriched_df['rec_excitation_filter'] = 'NA'
    enriched_df['rec_emission_filter'] = 'NA'
    enriched_df['excitation_nm'] = np.nan
    enriched_df['emission_nm'] = np.nan
    enriched_df['stokes_shift_nm'] = np.nan
    enriched_df['method'] = 'NA'
    enriched_df['context_type'] = 'NA'
    enriched_df['doi'] = 'NA'
    enriched_df['provenance'] = 'Atlas'
    
    # Try to match with Atlas data
    matched_count = 0
    for idx, row in enriched_df.iterrows():
        # Try to find matching entry in Atlas
        # Look for entries with similar family or name
        atlas_match = None
        
        # First try: exact family match
        family_matches = atlas_df[atlas_df['family'] == row['family']]
        if len(family_matches) > 0:
            # Take the first match with highest contrast_normalized
            atlas_match = family_matches.loc[family_matches['contrast_normalized'].idxmax()]
        
        if atlas_match is not None:
            # Fill in the data
            enriched_df.at[idx, 'excitation_nm'] = atlas_match['excitation_nm']
            enriched_df.at[idx, 'emission_nm'] = atlas_match['emission_nm']
            enriched_df.at[idx, 'stokes_shift_nm'] = atlas_match['stokes_shift_nm']
            enriched_df.at[idx, 'method'] = atlas_match['method']
            enriched_df.at[idx, 'context_type'] = atlas_match['context_type']
            enriched_df.at[idx, 'doi'] = atlas_match.get('doi', 'NA')
            
            # Calculate filter recommendations
            exc_nm = atlas_match['excitation_nm']
            em_nm = atlas_match['emission_nm']
            
            if pd.notna(exc_nm):
                exc_low = max(0, exc_nm - 20)
                exc_high = exc_nm + 20
                enriched_df.at[idx, 'rec_excitation_filter'] = f"[{exc_low:.0f}, {exc_high:.0f}]"
            
            if pd.notna(em_nm):
                em_low = max(0, em_nm - 20)
                em_high = em_nm + 20
                enriched_df.at[idx, 'rec_emission_filter'] = f"[{em_low:.0f}, {em_high:.0f}]"
            
            matched_count += 1
    
    print(f"Matched {matched_count} entries with Atlas data")
    
    # Reorder columns for lab sheet
    lab_columns = [
        'canonical_name', 'family', 'y_pred', 'PI90_width', 'fold',
        'excitation_nm', 'emission_nm', 'stokes_shift_nm',
        'rec_excitation_filter', 'rec_emission_filter',
        'method', 'context_type', 'doi', 'provenance'
    ]
    
    lab_sheet = enriched_df[lab_columns].copy()
    
    # Save lab sheet
    lab_sheet_path = Path(output_dir) / "shortlist_lab_sheet.csv"
    lab_sheet.to_csv(lab_sheet_path, index=False)
    print(f"Saved lab sheet: {lab_sheet_path}")
    
    # Create filter recommendations markdown
    create_filter_recommendations(lab_sheet, output_dir)
    
    print(f"\n=== LAB PACK READY ===")
    print(f"Total candidates: {len(lab_sheet)}")
    print(f"Atlas matches: {matched_count}")
    print(f"Output directory: {output_dir}")
    
    return lab_sheet

def create_filter_recommendations(lab_sheet, output_dir):
    """Create filter recommendations markdown table"""
    
    print("\n=== CREATING FILTER RECOMMENDATIONS ===")
    
    # Create markdown table
    md_content = "# Filter Recommendations for Top-20 Shortlist\n\n"
    md_content += "| # | Name | Family | Excitation (nm) | Emission (nm) | Exc Filter | Em Filter |\n"
    md_content += "|---|------|--------|-----------------|---------------|-------------|----------|\n"
    
    for idx, row in lab_sheet.iterrows():
        num = idx + 1
        name = row['canonical_name']
        family = row['family']
        exc_nm = f"{row['excitation_nm']:.0f}" if pd.notna(row['excitation_nm']) else "N/A"
        em_nm = f"{row['emission_nm']:.0f}" if pd.notna(row['emission_nm']) else "N/A"
        exc_filter = row['rec_excitation_filter']
        em_filter = row['rec_emission_filter']
        
        md_content += f"| {num} | {name} | {family} | {exc_nm} | {em_nm} | {exc_filter} | {em_filter} |\n"
    
    # Add summary
    md_content += f"\n## Summary\n"
    md_content += f"- **Total candidates**: {len(lab_sheet)}\n"
    md_content += f"- **Families represented**: {lab_sheet['family'].nunique()}\n"
    md_content += f"- **Prediction range**: {lab_sheet['y_pred'].min():.3f} - {lab_sheet['y_pred'].max():.3f}\n"
    md_content += f"- **Average uncertainty**: {lab_sheet['PI90_width'].mean():.1f}\n"
    
    # Save markdown file
    md_path = Path(output_dir) / "filters_recommendations.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"Saved filter recommendations: {md_path}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Create lab pack from shortlist')
    parser.add_argument('--shortlist', required=True, help='Path to shortlist CSV file')
    parser.add_argument('--atlas', required=True, help='Path to Atlas CSV file')
    parser.add_argument('--output', required=True, help='Output directory')
    
    args = parser.parse_args()
    
    # Create lab pack
    lab_sheet = create_lab_pack(args.shortlist, args.atlas, args.output)
    
    print(f"\nLAB PACK READY: {len(lab_sheet)} lignes")

if __name__ == "__main__":
    main()
