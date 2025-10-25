#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create 96-well and 24-well plate layouts
Arrange candidates with replicates and controls
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def create_plate_layouts(top12_file, output_dir):
    """Create 96-well and 24-well plate layouts"""
    
    print("=== CREATING PLATE LAYOUTS ===")
    
    # Load top-12 candidates
    df = pd.read_csv(top12_file)
    print(f"Loaded {len(df)} candidates")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create 96-well layout
    layout_96 = create_96_well_layout(df)
    
    # Create 24-well layout
    layout_24 = create_24_well_layout(df)
    
    # Save layouts
    layout_96_path = Path(output_dir) / "plate_layout_96.csv"
    layout_24_path = Path(output_dir) / "plate_layout_24.csv"
    
    layout_96.to_csv(layout_96_path, index=False)
    layout_24.to_csv(layout_24_path, index=False)
    
    print(f"Saved 96-well layout: {layout_96_path}")
    print(f"Saved 24-well layout: {layout_24_path}")
    
    # Print summary
    print(f"\n=== PLATE LAYOUTS SUMMARY ===")
    print(f"96-well plate: {len(layout_96)} wells")
    print(f"  - Candidates: {len(layout_96[layout_96['type'] == 'candidate'])}")
    print(f"  - Replicates: {len(layout_96[layout_96['replicate'] > 0])}")
    print(f"  - Controls: {len(layout_96[layout_96['type'] == 'control'])}")
    print(f"  - Blanks: {len(layout_96[layout_96['type'] == 'blank'])}")
    
    print(f"24-well plate: {len(layout_24)} wells")
    print(f"  - Candidates: {len(layout_24[layout_24['type'] == 'candidate'])}")
    print(f"  - Replicates: {len(layout_24[layout_24['replicate'] > 0])}")
    print(f"  - Controls: {len(layout_24[layout_24['type'] == 'control'])}")
    
    return layout_96, layout_24

def create_96_well_layout(df):
    """Create 96-well plate layout (8x12)"""
    
    print("\n=== CREATING 96-WELL LAYOUT ===")
    
    # Sort candidates by family and spectral region to minimize spillover
    df_sorted = df.sort_values(['family', 'excitation_nm', 'emission_nm'])
    
    layout_data = []
    well_count = 0
    
    # 96-well plate: 8 rows (A-H) x 12 columns (1-12)
    rows = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    cols = list(range(1, 13))
    
    # Place candidates with 6 replicates each (72 wells total)
    for idx, candidate in df_sorted.iterrows():
        for replicate in range(1, 7):  # 6 replicates
            row = rows[well_count // 12]
            col = cols[well_count % 12]
            well = f"{row}{col}"
            
            layout_data.append({
                'well': well,
                'row': row,
                'col': col,
                'canonical_name': candidate['canonical_name'],
                'family': candidate['family'],
                'replicate': replicate,
                'type': 'candidate'
            })
            well_count += 1
    
    # Add 8 positive controls (CTRL+)
    for i in range(8):
        row = rows[well_count // 12]
        col = cols[well_count % 12]
        well = f"{row}{col}"
        
        layout_data.append({
            'well': well,
            'row': row,
            'col': col,
            'canonical_name': 'CTRL+',
            'family': 'Control',
            'replicate': 0,
            'type': 'control'
        })
        well_count += 1
    
    # Add 16 blanks
    for i in range(16):
        row = rows[well_count // 12]
        col = cols[well_count % 12]
        well = f"{row}{col}"
        
        layout_data.append({
            'well': well,
            'row': row,
            'col': col,
            'canonical_name': 'BLANK',
            'family': 'Blank',
            'replicate': 0,
            'type': 'blank'
        })
        well_count += 1
    
    # Fill remaining wells with blanks if needed
    while well_count < 96:
        row = rows[well_count // 12]
        col = cols[well_count % 12]
        well = f"{row}{col}"
        
        layout_data.append({
            'well': well,
            'row': row,
            'col': col,
            'canonical_name': 'BLANK',
            'family': 'Blank',
            'replicate': 0,
            'type': 'blank'
        })
        well_count += 1
    
    return pd.DataFrame(layout_data)

def create_24_well_layout(df):
    """Create 24-well plate layout (4x6)"""
    
    print("\n=== CREATING 24-WELL LAYOUT ===")
    
    # Sort candidates by family and spectral region
    df_sorted = df.sort_values(['family', 'excitation_nm', 'emission_nm'])
    
    layout_data = []
    well_count = 0
    
    # 24-well plate: 4 rows (A-D) x 6 columns (1-6)
    rows = ['A', 'B', 'C', 'D']
    cols = list(range(1, 7))
    
    # Option 1: 12 candidates x 2 replicates = 24 wells
    # Option 2: 8 candidates x 3 replicates = 24 wells
    # We'll use Option 1 (12 candidates x 2 replicates)
    
    # Place candidates with 2 replicates each (24 wells total)
    for idx, candidate in df_sorted.iterrows():
        for replicate in range(1, 3):  # 2 replicates
            row = rows[well_count // 6]
            col = cols[well_count % 6]
            well = f"{row}{col}"
            
            layout_data.append({
                'well': well,
                'row': row,
                'col': col,
                'canonical_name': candidate['canonical_name'],
                'family': candidate['family'],
                'replicate': replicate,
                'type': 'candidate'
            })
            well_count += 1
    
    # Fill remaining wells with blanks if needed
    while well_count < 24:
        row = rows[well_count // 6]
        col = cols[well_count % 6]
        well = f"{row}{col}"
        
        layout_data.append({
            'well': well,
            'row': row,
            'col': col,
            'canonical_name': 'BLANK',
            'family': 'Blank',
            'replicate': 0,
            'type': 'blank'
        })
        well_count += 1
    
    return pd.DataFrame(layout_data)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Create plate layouts')
    parser.add_argument('--top12', required=True, help='Path to top-12 CSV file')
    parser.add_argument('--output', required=True, help='Output directory')
    
    args = parser.parse_args()
    
    # Create plate layouts
    layout_96, layout_24 = create_plate_layouts(args.top12, args.output)
    
    print(f"\nPLATES READY")

if __name__ == "__main__":
    main()
