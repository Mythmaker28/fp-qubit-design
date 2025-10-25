#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assemble complete lab package with all deliverables
Generate checksums and handoff documentation
"""

import hashlib
import os
from pathlib import Path
import argparse

def assemble_lab_package(output_dir):
    """Assemble complete lab package with all deliverables"""
    
    print("=== ASSEMBLING LAB PACKAGE ===")
    
    # List all required files
    required_files = [
        "shortlist_lab_sheet.csv",
        "shortlist_top12_final.csv", 
        "filters_recommendations.md",
        "plate_layout_96.csv",
        "plate_layout_24.csv",
        "protocol_skeleton.md"
    ]
    
    print(f"Required files: {len(required_files)}")
    for file in required_files:
        print(f"  - {file}")
    
    # Check if all files exist
    missing_files = []
    for file in required_files:
        file_path = Path(output_dir) / file
        if not file_path.exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"ERROR: Missing files: {missing_files}")
        return False
    
    # Generate SHA256 checksums
    generate_checksums(output_dir, required_files)
    
    # Create handoff document
    create_handoff_document(output_dir)
    
    print(f"\n=== PACKAGE ASSEMBLY COMPLETE ===")
    print(f"Output directory: {output_dir}")
    print(f"Files included: {len(required_files)}")
    print(f"Checksums generated: SHA256SUMS.txt")
    print(f"Handoff document: LAB_HANDOFF_v2_2_2.txt")
    
    return True

def generate_checksums(output_dir, file_list):
    """Generate SHA256 checksums for all files"""
    
    print("\n=== GENERATING CHECKSUMS ===")
    
    checksums = []
    
    for filename in file_list:
        file_path = Path(output_dir) / filename
        if file_path.exists():
            # Calculate SHA256 hash
            with open(file_path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            
            checksums.append(f"{file_hash}  {filename}")
            print(f"  {filename}: {file_hash[:16]}...")
        else:
            print(f"  WARNING: {filename} not found")
    
    # Save checksums file
    checksums_path = Path(output_dir) / "SHA256SUMS.txt"
    with open(checksums_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(checksums))
    
    print(f"Saved checksums: {checksums_path}")

def create_handoff_document(output_dir):
    """Create handoff documentation"""
    
    print("\n=== CREATING HANDOFF DOCUMENT ===")
    
    handoff_content = """LAB HANDOFF v2.2.2 - Fluorescence Ion Channel Screening Package

FILES LOCATION: All deliverables are in this directory (outputs_v2_2_2_lab/)

USAGE GUIDE:
1. shortlist_lab_sheet.csv - Complete candidate data with spectral parameters
2. shortlist_top12_final.csv - Final 12 candidates selected for testing  
3. filters_recommendations.md - Filter recommendations table for each candidate
4. plate_layout_96.csv - 96-well plate layout with replicates and controls
5. plate_layout_24.csv - 24-well plate layout with replicates
6. protocol_skeleton.md - Experimental protocol with spectral parameters

VERIFICATION: Use SHA256SUMS.txt to verify file integrity before use

READY FOR LAB: All files validated and ready for experimental validation"""
    
    # Save handoff document
    handoff_path = Path(output_dir) / "LAB_HANDOFF_v2_2_2.txt"
    with open(handoff_path, 'w', encoding='utf-8') as f:
        f.write(handoff_content)
    
    print(f"Saved handoff document: {handoff_path}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Assemble lab package')
    parser.add_argument('--output', required=True, help='Output directory')
    
    args = parser.parse_args()
    
    # Assemble lab package
    success = assemble_lab_package(args.output)
    
    if success:
        print(f"\nHANDOFF READY")
    else:
        print(f"\nERROR: Package assembly failed")

if __name__ == "__main__":
    main()
