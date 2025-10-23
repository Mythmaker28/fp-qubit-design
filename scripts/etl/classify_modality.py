#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classify Atlas systems by modality (optical vs non-optical).

This script adds boolean flags to identify:
- is_optical: System uses optical readout (fluorescence, quantum dots, etc.)
- is_fp_like: Specifically fluorescent proteins or quantum dots
- in_scope_training: Suitable for FP-qubit design training (optical + FP-like)

Classification rules based on:
- Method/Methode_lecture column
- Classe column
- System name patterns
- Photophysique hints
"""

import re
from pathlib import Path
import pandas as pd


# Regex patterns (case-insensitive)
OPTICAL_PATTERNS = [
    r'\bfluoresc',  # fluorescence, fluorescent, fluorescein
    r'\bFRET\b',
    r'\bphotophys',
    r'\bbrightness',
    r'\bquantum\s+dot',
    r'\bGFP\b',
    r'\bmNG\b',
    r'\bmNeon',
    r'\bmCherry\b',
    r'\bTagRFP\b',
    r'\bEGFP\b',
    r'\bYFP\b',
    r'\bCFP\b',
    r'\bRFP\b',
    r'\bODMR\b',  # ODMR is optical (though often for NV centers)
    r'\boptical[_\s-]?read',
    r'excit.*emiss',  # excitation/emission
    r'\bchromophore\b',
    r'\bquantum\s+yield',
    r'\blifetime',  # photophysical lifetime
    r'\bphotostab',
]

NON_OPTICAL_PATTERNS = [
    r'\bNMR\b',
    r'\bESR\b',
    r'\bEPR\b',
    r'\bhyperpolariz',
    r'\bmagnetoreception\b',
    r'\bmagnetosome',
    r'\bcryptochrome.*magneto',  # cryptochrome for magnetoreception (not fluorescence)
    r'\bNV\s+center.*diamond',  # NV centers (not FP)
    r'\b\^13C\b',  # carbon-13 labeling
    r'\b\^15N\b',  # nitrogen-15 labeling
    r'\bindirect\b',  # indirect readout
]

FP_LIKE_PATTERNS = [
    r'\bGFP\b',
    r'\bfluorescent\s+protein',
    r'\bmNG\b',
    r'\bmNeon',
    r'\bmCherry\b',
    r'\bTagRFP\b',
    r'\bEGFP\b',
    r'\bYFP\b',
    r'\bCFP\b',
    r'\bRFP\b',
    r'\bquantum\s+dot',
    r'\bQD\b',
    r'\bInP/ZnS',
    r'\bCdSe',
]


def classify_system(row: pd.Series) -> dict:
    """Classify a single system."""
    
    # Combine relevant text fields
    text_fields = []
    
    for col in ['Systeme', 'Classe', 'Methode_lecture', 'Hote_contexte', 
                'Photophysique', 'Notes', 'protein_name', 'method', 'host_context']:
        if col in row.index and pd.notna(row[col]):
            text_fields.append(str(row[col]))
    
    combined_text = ' '.join(text_fields).lower()
    
    # Check patterns
    is_optical_match = any(re.search(pattern, combined_text, re.IGNORECASE) 
                          for pattern in OPTICAL_PATTERNS)
    is_non_optical_match = any(re.search(pattern, combined_text, re.IGNORECASE) 
                               for pattern in NON_OPTICAL_PATTERNS)
    is_fp_like_match = any(re.search(pattern, combined_text, re.IGNORECASE) 
                          for pattern in FP_LIKE_PATTERNS)
    
    # Decision rules
    is_optical = False
    is_fp_like = False
    
    # Rule 1: Explicit non-optical → not optical
    if is_non_optical_match:
        is_optical = False
        is_fp_like = False
    
    # Rule 2: Optical match → optical
    elif is_optical_match:
        is_optical = True
        is_fp_like = is_fp_like_match
    
    # Rule 3: Class-based heuristics
    else:
        classe = row.get('Classe', row.get('class', ''))
        if pd.notna(classe):
            classe_str = str(classe).upper()
            if classe_str in ['A', 'B']:  # A=bio-intrinsic, B=bio-compatible often optical
                is_optical = True
            elif classe_str in ['C', 'D']:  # C=hyperpol, D=indirect often non-optical
                is_optical = False
    
    # in_scope_training = optical AND fp_like
    in_scope_training = is_optical and is_fp_like
    
    return {
        'is_optical': is_optical,
        'is_fp_like': is_fp_like,
        'in_scope_training': in_scope_training,
    }


def main():
    print("=" * 60)
    print("Classify Modality (Optical vs Non-Optical)")
    print("=" * 60)
    print()
    
    # Load merged data
    merged_path = Path("data/interim/atlas_merged.csv")
    
    if not merged_path.exists():
        print(f"[ERROR] {merged_path} not found. Run merge_atlas_assets.py first.")
        return
    
    df = pd.read_csv(merged_path)
    print(f"[INFO] Loaded {len(df)} systems from atlas_merged.csv")
    
    # Classify each system
    print("\n[INFO] Classifying systems...")
    
    classifications = df.apply(classify_system, axis=1, result_type='expand')
    
    # Add flags to dataframe
    df['is_optical'] = classifications['is_optical']
    df['is_fp_like'] = classifications['is_fp_like']
    df['in_scope_training'] = classifications['in_scope_training']
    
    # Save updated merged file
    output_path = Path("data/interim/atlas_merged_classified.csv")
    df.to_csv(output_path, index=False)
    print(f"\n[INFO] Saved classified data: {output_path}")
    
    # Generate statistics
    n_optical = int(df['is_optical'].sum())
    n_non_optical = len(df) - n_optical
    n_fp_like = int(df['is_fp_like'].sum())
    n_in_scope = int(df['in_scope_training'].sum())
    
    # Count contrast by modality
    contrast_col = None
    for col in ['contrast_ratio', 'Contraste_%', 'Contraste_pourcent']:
        if col in df.columns:
            contrast_col = col
            break
    
    if contrast_col:
        n_optical_with_contrast = int(df[df['is_optical']][contrast_col].notna().sum())
        n_non_optical_with_contrast = int(df[~df['is_optical']][contrast_col].notna().sum())
    else:
        n_optical_with_contrast = 0
        n_non_optical_with_contrast = 0
    
    print()
    print("=" * 60)
    print("CLASSIFICATION SUMMARY")
    print("=" * 60)
    print(f"Total systems:              {len(df)}")
    print(f"  - Optical:                {n_optical} ({n_optical/len(df)*100:.1f}%)")
    print(f"  - Non-optical:            {n_non_optical} ({n_non_optical/len(df)*100:.1f}%)")
    print(f"  - FP-like:                {n_fp_like}")
    print(f"  - In scope (optical+FP):  {n_in_scope}")
    print()
    print("Contrast availability:")
    print(f"  - Optical with contrast:  {n_optical_with_contrast} / {n_optical}")
    print(f"  - Non-optical with contrast: {n_non_optical_with_contrast} / {n_non_optical}")
    print("=" * 60)
    
    # Generate detailed report
    report_path = Path("reports/MODALITY_SPLIT.md")
    report_path.parent.mkdir(exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# MODALITY SPLIT REPORT - fp-qubit-design v1.1.3\n\n")
        f.write("**Generated**: 2025-10-23\n\n")
        f.write("---\n\n")
        f.write("## Summary\n\n")
        f.write(f"- **Total systems**: {len(df)}\n")
        f.write(f"- **Optical systems**: {n_optical} ({n_optical/len(df)*100:.1f}%)\n")
        f.write(f"- **Non-optical systems**: {n_non_optical} ({n_non_optical/len(df)*100:.1f}%)\n")
        f.write(f"- **FP-like systems**: {n_fp_like}\n")
        f.write(f"- **In scope for training**: {n_in_scope}\n\n")
        
        f.write("## Optical Systems\n\n")
        f.write(f"- **With contrast measured**: {n_optical_with_contrast} / {n_optical}\n")
        f.write(f"- **Without contrast**: {n_optical - n_optical_with_contrast}\n\n")
        
        # List optical systems
        if n_optical > 0:
            df_optical = df[df['is_optical']].copy()
            f.write("### Optical Systems List\n\n")
            f.write("| System | Class | Method | Contrast | FP-like |\n")
            f.write("|--------|-------|--------|----------|----------|\n")
            
            for _, row in df_optical.iterrows():
                system = row.get('Systeme', row.get('protein_name', 'N/A'))
                classe = row.get('Classe', row.get('class', 'N/A'))
                method = row.get('Methode_lecture', row.get('method', 'N/A'))
                
                # Get contrast value (try multiple column names)
                contrast = None
                for col in ['contrast_ratio', 'Contraste_%', 'Contraste_pourcent']:
                    if col in row.index:
                        contrast = row[col]
                        break
                
                contrast_str = f"{contrast:.2f}%" if pd.notna(contrast) else "N/A"
                fp_like = "Yes" if row['is_fp_like'] else "No"
                
                f.write(f"| {system} | {classe} | {method} | {contrast_str} | {fp_like} |\n")
        
        f.write("\n## Non-Optical Systems\n\n")
        f.write(f"- **Total**: {n_non_optical}\n")
        f.write(f"- **With contrast** (unexpected): {n_non_optical_with_contrast}\n\n")
        
        # List non-optical systems
        if n_non_optical > 0:
            df_non_optical = df[~df['is_optical']].copy()
            f.write("### Non-Optical Systems List\n\n")
            f.write("| System | Class | Method | Reason |\n")
            f.write("|--------|-------|--------|--------|\n")
            
            for _, row in df_non_optical.iterrows():
                system = row.get('Systeme', row.get('protein_name', 'N/A'))
                classe = row.get('Classe', row.get('class', 'N/A'))
                method = row.get('Methode_lecture', row.get('method', 'N/A'))
                
                # Determine reason
                text = ' '.join([str(row.get(col, '')) for col in ['Systeme', 'Methode_lecture', 'Classe']]).lower()
                if 'nmr' in text or 'hyperpolariz' in text or '^13c' in text or '^15n' in text:
                    reason = "NMR/hyperpolarized"
                elif 'esr' in text or 'epr' in text:
                    reason = "ESR/EPR"
                elif 'magneto' in text:
                    reason = "Magnetoreception (indirect)"
                elif 'indirect' in text:
                    reason = "Indirect readout"
                else:
                    reason = "Non-optical (class-based)"
                
                f.write(f"| {system} | {classe} | {method} | {reason} |\n")
        
        f.write("\n---\n\n")
        f.write("## Classification Rules\n\n")
        f.write("### Optical Indicators\n\n")
        f.write("- Fluorescence/fluorescent\n")
        f.write("- FRET\n")
        f.write("- Photophysics keywords\n")
        f.write("- GFP family proteins\n")
        f.write("- Quantum dots\n")
        f.write("- Excitation/emission wavelengths\n")
        f.write("- Class A or B (bio-intrinsic/compatible)\n\n")
        
        f.write("### Non-Optical Indicators\n\n")
        f.write("- NMR, ESR, EPR\n")
        f.write("- Hyperpolarized nuclei (^13C, ^15N)\n")
        f.write("- Magnetoreception (cryptochrome, magnetosomes)\n")
        f.write("- Indirect readout\n")
        f.write("- Class C or D (hyperpolarized/indirect)\n\n")
        
        f.write("---\n\n")
        f.write("**License**: Data from biological-qubits-atlas is licensed under CC BY 4.0\n")
    
    print(f"\n[INFO] Report saved: {report_path}")
    print()
    print("=" * 60)
    print("Classification complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

