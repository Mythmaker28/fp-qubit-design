#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate top-20 shortlist for experimental validation
Based on high predictions with minimal uncertainty intervals
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def generate_shortlist_top20(predictions_file, output_file, max_per_family=6, top_n=20):
    """Generate top-20 shortlist with family diversity constraints"""
    
    print("=== GENERATING TOP-20 SHORTLIST ===")
    
    # Read predictions
    df = pd.read_csv(predictions_file)
    print(f"Loaded {len(df)} predictions")
    
    # Calculate PI90 width
    df['PI90_width'] = df['pi_high_90'] - df['pi_low_90']
    
    # Sort by high predictions and low uncertainty (ascending PI90_width)
    df_sorted = df.sort_values(['y_pred', 'PI90_width'], ascending=[False, True])
    
    print(f"Prediction range: {df['y_pred'].min():.3f} - {df['y_pred'].max():.3f}")
    print(f"PI90 width range: {df['PI90_width'].min():.3f} - {df['PI90_width'].max():.3f}")
    
    # Apply family diversity constraint
    shortlist = []
    family_counts = {}
    
    for _, row in df_sorted.iterrows():
        family = row['family']
        
        # Check if we can add this family
        if family not in family_counts:
            family_counts[family] = 0
        
        if family_counts[family] < max_per_family:
            shortlist.append(row)
            family_counts[family] += 1
            
            if len(shortlist) >= top_n:
                break
    
    # Create shortlist DataFrame
    shortlist_df = pd.DataFrame(shortlist)
    
    # Add canonical_name (using family + index for now)
    shortlist_df['canonical_name'] = shortlist_df['family'] + '_' + (shortlist_df.index + 1).astype(str)
    
    # Select and reorder columns
    output_df = shortlist_df[['canonical_name', 'family', 'y_pred', 'PI90_width', 'fold']].copy()
    
    # Save shortlist
    output_df.to_csv(output_file, index=False)
    
    print(f"\n=== SHORTLIST GENERATED ===")
    print(f"Total candidates: {len(shortlist_df)}")
    print(f"Family distribution:")
    family_dist = shortlist_df['family'].value_counts()
    for family, count in family_dist.items():
        print(f"  {family}: {count}")
    
    print(f"\nPrediction statistics:")
    print(f"  Mean y_pred: {shortlist_df['y_pred'].mean():.3f}")
    print(f"  Mean PI90_width: {shortlist_df['PI90_width'].mean():.3f}")
    print(f"  Min PI90_width: {shortlist_df['PI90_width'].min():.3f}")
    print(f"  Max PI90_width: {shortlist_df['PI90_width'].max():.3f}")
    
    print(f"\nSaved to: {output_file}")
    
    return output_df

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Generate top-20 shortlist')
    parser.add_argument('--predictions', required=True, help='Path to predictions CSV file')
    parser.add_argument('--output', required=True, help='Output shortlist CSV file')
    parser.add_argument('--max_per_family', type=int, default=6, help='Maximum per family')
    parser.add_argument('--top_n', type=int, default=20, help='Number of top candidates')
    
    args = parser.parse_args()
    
    # Generate shortlist
    shortlist_df = generate_shortlist_top20(
        args.predictions, 
        args.output, 
        args.max_per_family, 
        args.top_n
    )
    
    print(f"\n=== SHORTLIST TOP-20 READY ===")
    print("Candidates selected based on:")
    print("- High predicted values (y_pred)")
    print("- Minimal uncertainty intervals (PI90_width)")
    print("- Family diversity (max 6 per family)")
    print("- Ready for experimental validation")

if __name__ == "__main__":
    main()
