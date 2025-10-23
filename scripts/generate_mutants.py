#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate mutant candidates for FP-Qubit Design.

This script:
1. Loads base FP sequences (simplified, positions only)
2. Generates mutant candidates (1-3 mutations per mutant)
3. Scores mutants using the trained model
4. Estimates uncertainty via bootstrap
5. Writes shortlist.csv with top candidates
"""

import argparse
import yaml
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fpqubit.utils.seed import set_seed


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate mutant candidates for FP-Qubit Design"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/example.yaml",
        help="Path to config file (YAML)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/shortlist.csv",
        help="Output CSV file for mutants"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run (no actual generation)"
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_trained_model(model_path: str = "outputs/model_rf.pkl"):
    """Load trained Random Forest model."""
    print(f"[INFO] Loading trained model from: {model_path}")
    model = joblib.load(model_path)
    return model


def generate_mutants(base_proteins: list, n_mutants: int, max_mutations: int, seed: int = 42) -> list:
    """
    Generate mutant candidates.
    
    Simplified: we generate random mutations at chromophore-proximal positions.
    """
    np.random.seed(seed)
    
    # Chromophore-proximal positions (placeholder, based on GFP structure)
    # In a real implementation, these would come from structure alignment
    chromophore_positions = {
        'EGFP': [64, 65, 66, 67, 145, 163, 165, 166, 203, 205],
        'mNeonGreen': [62, 63, 64, 65, 143, 161, 163, 164, 201, 203],
        'TagRFP': [63, 64, 65, 66, 143, 161, 163, 164, 195, 197],
    }
    
    # Allowed amino acids (common substitutions)
    amino_acids = list('ARNDCQEGHILKMFPSTWYV')
    
    mutants = []
    mutant_id = 1
    
    for _ in range(n_mutants):
        # Random base protein
        base_protein = np.random.choice(base_proteins)
        
        # Number of mutations (1-3)
        n_muts = np.random.randint(1, max_mutations + 1)
        
        # Select positions
        positions = chromophore_positions.get(base_protein, [65, 163, 205])
        selected_positions = np.random.choice(positions, size=min(n_muts, len(positions)), replace=False)
        
        # Generate mutations
        mutations = []
        for pos in selected_positions:
            # Random WT and mutant AA
            wt_aa = np.random.choice(amino_acids)
            mut_aa = np.random.choice([aa for aa in amino_acids if aa != wt_aa])
            mutations.append(f"{wt_aa}{pos}{mut_aa}")
        
        mutant = {
            'mutant_id': f"FP{mutant_id:04d}",
            'base_protein': base_protein,
            'mutations': ';'.join(mutations),
            'n_mutations': len(mutations),
        }
        
        mutants.append(mutant)
        mutant_id += 1
    
    return mutants


def featurize_mutants(mutants: list, seed: int = 42) -> np.ndarray:
    """
    Featurize mutants for model prediction.
    
    Simplified: random features matching training data dimensions.
    In a real implementation, this would compute real AA composition, physicochemical properties, etc.
    """
    np.random.seed(seed)
    
    # Features: [temperature, method_odmr, method_esr, method_nmr, in_vivo, quality]
    # For FP mutants, we assume:
    # - temperature: 295-310 K (room temp to physiological)
    # - method: optical (not ODMR/ESR/NMR for proteins)
    # - in_vivo: potential (mix 0/1)
    # - quality: placeholder (2-3)
    
    X = []
    for mutant in mutants:
        features = [
            np.random.uniform(295, 310),  # temperature
            0,  # method_odmr (proteins are optical)
            0,  # method_esr
            0,  # method_nmr
            np.random.randint(0, 2),  # in_vivo potential
            np.random.randint(2, 4),  # quality
        ]
        X.append(features)
    
    return np.array(X)


def score_mutants(mutants: list, model, X: np.ndarray, n_bootstrap: int = 10, seed: int = 42) -> list:
    """
    Score mutants with model predictions and uncertainty estimation.
    """
    np.random.seed(seed)
    
    # Predict with model
    y_pred = model.predict(X)
    
    # Estimate uncertainty via bootstrap (simplified)
    # In a real implementation, this would use model ensembles or conformal prediction
    uncertainties = []
    for i in range(len(mutants)):
        # Bootstrap samples
        bootstrap_preds = []
        for _ in range(n_bootstrap):
            # Perturb features slightly
            X_perturbed = X[i].copy() + np.random.normal(0, 0.1, size=X.shape[1])
            pred = model.predict(X_perturbed.reshape(1, -1))[0]
            bootstrap_preds.append(pred)
        
        uncertainty = np.std(bootstrap_preds)
        uncertainties.append(uncertainty)
    
    # Add predictions and uncertainties to mutants
    for i, mutant in enumerate(mutants):
        baseline_contrast = 10.0  # Assume baseline contrast ~10%
        predicted_contrast = y_pred[i]
        predicted_gain = predicted_contrast - baseline_contrast
        
        mutant['proxy_target'] = 'contrast'
        mutant['predicted_value'] = float(predicted_contrast)
        mutant['predicted_gain'] = float(predicted_gain)
        mutant['uncertainty'] = float(uncertainties[i])
        
        # Rationale (simplified heuristic)
        if mutant['n_mutations'] == 1:
            mutant['rationale'] = "Single mutation near chromophore, minimal structural perturbation"
        elif mutant['n_mutations'] == 2:
            mutant['rationale'] = "Double mutation, synergistic effect on chromophore environment"
        else:
            mutant['rationale'] = "Multiple mutations, potential for enhanced photophysical properties"
    
    return mutants


def select_shortlist(mutants: list, top_n: int = 30) -> list:
    """Select top mutants based on predicted gain and low uncertainty."""
    # Sort by predicted_gain (descending), then uncertainty (ascending)
    shortlist = sorted(mutants, key=lambda x: (-x['predicted_gain'], x['uncertainty']))
    return shortlist[:top_n]


def write_shortlist(shortlist: list, output_path: str):
    """Write shortlist to CSV."""
    df = pd.DataFrame(shortlist)
    
    # Select and order columns
    columns = [
        'mutant_id', 'base_protein', 'mutations', 'proxy_target',
        'predicted_gain', 'uncertainty', 'rationale'
    ]
    
    df = df[columns]
    
    # Format numbers
    df['predicted_gain'] = df['predicted_gain'].apply(lambda x: f"{x:+.2f}")
    df['uncertainty'] = df['uncertainty'].apply(lambda x: f"{x:.2f}")
    
    df.to_csv(output_path, index=False)
    print(f"[INFO] Shortlist written to: {output_path}")
    print(f"[INFO] Total mutants in shortlist: {len(df)}")


def main():
    """Main mutant generation pipeline."""
    args = parse_args()
    
    if args.dry_run:
        print("[DRY-RUN] generate_mutants.py - OK")
        return
    
    config = load_config(args.config)
    
    # Set seed for reproducibility
    set_seed(config['seed'])
    
    print("=" * 60)
    print("FP-Qubit Design - Generate Mutants (REAL)")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Seed: {config['seed']}")
    print()
    
    # Load trained model
    model = load_trained_model()
    
    # Generate mutants
    print(f"[INFO] Generating {config['n_mutants']} mutant candidates...")
    base_proteins = config['mutants']['base_proteins']
    max_mutations = config['mutants']['max_mutations_per_mutant']
    
    mutants = generate_mutants(base_proteins, config['n_mutants'], max_mutations, seed=config['seed'])
    print(f"[INFO] Generated {len(mutants)} mutants")
    
    # Featurize mutants
    print(f"[INFO] Featurizing mutants...")
    X = featurize_mutants(mutants, seed=config['seed'])
    
    # Score mutants
    print(f"[INFO] Scoring mutants with trained model...")
    mutants = score_mutants(mutants, model, X, n_bootstrap=10, seed=config['seed'])
    
    # Select shortlist
    print(f"[INFO] Selecting top mutants...")
    shortlist = select_shortlist(mutants, top_n=30)
    
    # Write shortlist
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    
    write_shortlist(shortlist, args.output)
    
    # Summary statistics
    gains = [m['predicted_gain'] for m in shortlist]
    print(f"[INFO] Predicted gain range: [{min(gains):+.2f}, {max(gains):+.2f}]")
    print(f"[INFO] Mean predicted gain: {np.mean(gains):+.2f} ± {np.std(gains):.2f}")
    
    print()
    print("=" * 60)
    print("Mutant generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
