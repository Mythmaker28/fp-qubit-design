#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create protocol skeleton with spectral parameters
Generate experimental protocol for top-12 candidates
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def create_protocol_skeleton(top12_file, output_dir):
    """Create protocol skeleton with spectral parameters"""
    
    print("=== CREATING PROTOCOL SKELETON ===")
    
    # Load top-12 candidates
    df = pd.read_csv(top12_file)
    print(f"Loaded {len(df)} candidates")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Extract spectral parameters
    spectral_params = extract_spectral_parameters(df)
    
    # Create protocol content
    protocol_content = generate_protocol_content(df, spectral_params)
    
    # Save protocol
    protocol_path = Path(output_dir) / "protocol_skeleton.md"
    with open(protocol_path, 'w', encoding='utf-8') as f:
        f.write(protocol_content)
    
    print(f"Saved protocol: {protocol_path}")
    
    return protocol_content

def extract_spectral_parameters(df):
    """Extract spectral parameters for each candidate"""
    
    spectral_params = {}
    
    for idx, row in df.iterrows():
        name = row['canonical_name']
        family = row['family']
        
        # Extract excitation and emission wavelengths
        exc_nm = row['excitation_nm']
        em_nm = row['emission_nm']
        
        if pd.notna(exc_nm) and pd.notna(em_nm):
            # Calculate filter ranges (±20 nm)
            exc_low = max(0, exc_nm - 20)
            exc_high = exc_nm + 20
            em_low = max(0, em_nm - 20)
            em_high = em_nm + 20
            
            spectral_params[name] = {
                'family': family,
                'excitation_center': exc_nm,
                'emission_center': em_nm,
                'excitation_range': f"{exc_low:.0f}-{exc_high:.0f}",
                'emission_range': f"{em_low:.0f}-{em_high:.0f}",
                'excitation_filter': f"[{exc_low:.0f}, {exc_high:.0f}]",
                'emission_filter': f"[{em_low:.0f}, {em_high:.0f}]"
            }
        else:
            # Default values if spectral data not available
            spectral_params[name] = {
                'family': family,
                'excitation_center': 488,
                'emission_center': 510,
                'excitation_range': "468-508",
                'emission_range': "490-530",
                'excitation_filter': "[468, 508]",
                'emission_filter': "[490, 530]"
            }
    
    return spectral_params

def generate_protocol_content(df, spectral_params):
    """Generate protocol content"""
    
    # Group candidates by family for organization
    families = df['family'].unique()
    
    protocol = "# Experimental Protocol Skeleton\n"
    protocol += "## Fluorescence-based Ion Channel Screening\n\n"
    
    # Overview
    protocol += "### Overview\n"
    protocol += f"- **Total candidates**: {len(df)}\n"
    protocol += f"- **Families represented**: {len(families)}\n"
    protocol += f"- **Replicates per candidate**: 6 (96-well) / 2 (24-well)\n"
    protocol += f"- **Expected duration**: 2-3 days\n\n"
    
    # Instrument parameters
    protocol += "### Instrument Parameters\n\n"
    protocol += "#### Microplate Reader Settings\n"
    protocol += "- **Temperature**: 37°C (maintained)\n"
    protocol += "- **Read mode**: Fluorescence intensity\n"
    protocol += "- **Integration time**: 100-200 ms per well\n"
    protocol += "- **Gain**: Auto or optimized per filter set\n"
    protocol += "- **Number of flashes**: 10-20 per measurement\n\n"
    
    # Spectral parameters by family
    protocol += "### Spectral Parameters by Family\n\n"
    
    for family in sorted(families):
        family_candidates = df[df['family'] == family]
        protocol += f"#### {family} Family ({len(family_candidates)} candidates)\n\n"
        
        for idx, candidate in family_candidates.iterrows():
            name = candidate['canonical_name']
            params = spectral_params[name]
            
            protocol += f"**{name}**\n"
            protocol += f"- Excitation: {params['excitation_center']:.0f} nm ({params['excitation_range']} nm)\n"
            protocol += f"- Emission: {params['emission_center']:.0f} nm ({params['emission_range']} nm)\n"
            protocol += f"- Filter set: Exc {params['excitation_filter']}, Em {params['emission_filter']}\n\n"
    
    # Experimental procedure
    protocol += "### Experimental Procedure\n\n"
    
    protocol += "#### Day 1: Plate Preparation\n"
    protocol += "1. **Buffer preparation** (pH 7.4, 37°C)\n"
    protocol += "   - HEPES buffer: 10 mM HEPES, 140 mM NaCl, 5 mM KCl, 1 mM MgCl₂, 1 mM CaCl₂\n"
    protocol += "   - Adjust pH to 7.4 ± 0.1\n"
    protocol += "   - Filter sterilize (0.22 μm)\n\n"
    
    protocol += "2. **Cell seeding**\n"
    protocol += "   - Seed cells at 2×10⁴ cells/well (96-well) or 5×10⁴ cells/well (24-well)\n"
    protocol += "   - Incubate at 37°C, 5% CO₂ for 24-48 hours\n\n"
    
    protocol += "3. **Dye loading**\n"
    protocol += "   - Load fluorescent indicators according to manufacturer protocol\n"
    protocol += "   - Incubate for 30-60 minutes at 37°C\n"
    protocol += "   - Wash 2× with buffer\n\n"
    
    protocol += "#### Day 2: Experimental Measurements\n"
    protocol += "1. **Baseline measurement** (5-10 cycles)\n"
    protocol += "   - Read fluorescence for 2-5 minutes to establish baseline\n"
    protocol += "   - Record F₀ (baseline fluorescence)\n\n"
    
    protocol += "2. **Stimulus application**\n"
    protocol += "   - Add test compounds or controls\n"
    protocol += "   - Monitor fluorescence for 10-20 cycles\n"
    protocol += "   - Record F₁ (stimulated fluorescence)\n\n"
    
    protocol += "3. **Recovery measurement** (5-10 cycles)\n"
    protocol += "   - Wash with buffer\n"
    protocol += "   - Monitor fluorescence recovery\n"
    protocol += "   - Record F₂ (recovery fluorescence)\n\n"
    
    # Quality control
    protocol += "### Quality Control\n\n"
    protocol += "#### Data Validation\n"
    protocol += "- **Outlier detection**: Exclude wells with residuals > P90 threshold\n"
    protocol += "- **Replicate consistency**: CV < 20% between replicates\n"
    protocol += "- **Signal-to-noise ratio**: SNR > 3:1\n"
    protocol += "- **Minimum replicates**: n ≥ 3 per condition\n\n"
    
    protocol += "#### Controls\n"
    protocol += "- **Positive controls**: Known activators (n=8 per plate)\n"
    protocol += "- **Negative controls**: Vehicle only (n=16 per plate)\n"
    protocol += "- **Blank wells**: Buffer only (n=16 per plate)\n\n"
    
    # Data analysis
    protocol += "### Data Analysis\n\n"
    protocol += "#### Calculations\n"
    protocol += "- **ΔF/F₀**: (F₁ - F₀) / F₀ × 100\n"
    protocol += "- **Recovery**: (F₂ - F₀) / (F₁ - F₀) × 100\n"
    protocol += "- **EC₅₀**: Concentration for 50% maximal response\n"
    protocol += "- **Hill coefficient**: Steepness of dose-response curve\n\n"
    
    protocol += "#### Statistical Analysis\n"
    protocol += "- **ANOVA**: Compare between groups\n"
    protocol += "- **Dunnett's test**: Multiple comparisons vs control\n"
    protocol += "- **Dose-response fitting**: 4-parameter logistic model\n\n"
    
    # Documentation
    protocol += "### Documentation Requirements\n\n"
    protocol += "#### Experimental Log\n"
    protocol += "- **Date and time**: Record all measurements\n"
    protocol += "- **Operator**: Initials of person performing experiment\n"
    protocol += "- **Instrument settings**: Gain, integration time, filters\n"
    protocol += "- **Environmental conditions**: Temperature, humidity\n\n"
    
    protocol += "#### Data Storage\n"
    protocol += "- **Raw data**: Fluorescence values per well\n"
    protocol += "- **Metadata**: Plate layout, candidate information\n"
    protocol += "- **Analysis files**: Processed data and statistics\n"
    protocol += "- **DOI/Provenance**: Reference to Atlas database\n\n"
    
    # Safety and notes
    protocol += "### Safety Considerations\n\n"
    protocol += "- **Personal protective equipment**: Lab coat, gloves, safety glasses\n"
    protocol += "- **Chemical handling**: Follow SDS for all compounds\n"
    protocol += "- **Waste disposal**: Segregate chemical waste appropriately\n"
    protocol += "- **Emergency procedures**: Know location of safety equipment\n\n"
    
    protocol += "### Notes\n\n"
    protocol += "- **Buffer optimization**: May require pH/temperature adjustment\n"
    protocol += "- **Timing optimization**: Adjust cycle number based on kinetics\n"
    protocol += "- **Filter optimization**: Verify spectral overlap with indicators\n"
    protocol += "- **Automation**: Consider robotic liquid handling for high-throughput\n\n"
    
    return protocol

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Create protocol skeleton')
    parser.add_argument('--top12', required=True, help='Path to top-12 CSV file')
    parser.add_argument('--output', required=True, help='Output directory')
    
    args = parser.parse_args()
    
    # Create protocol skeleton
    protocol_content = create_protocol_skeleton(args.top12, args.output)
    
    print(f"\nPROTOCOL READY")

if __name__ == "__main__":
    main()
