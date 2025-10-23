"""
Featurization module for FP mutants.

TODO:
- Implement amino acid composition features
- Implement physicochemical property encodings (charge, hydrophobicity, etc.)
- Implement structure-based features (if PDB available)
- Implement chromophore proximity features
"""


def featurize_sequence(sequence: str) -> dict:
    """
    Featurize a protein sequence into numerical descriptors.
    
    Args:
        sequence: Amino acid sequence (single-letter code)
    
    Returns:
        Dictionary of features
    
    TODO:
    - AA composition (20 features)
    - Dipeptide composition (400 features)
    - Physicochemical properties (charge, hydrophobicity, aromaticity)
    - Chromophore-proximal residues (positions 64-67 for GFP-like)
    """
    # Placeholder
    features = {
        "length": len(sequence),
        "composition": {},  # TODO
        "properties": {},   # TODO
    }
    return features


def featurize_mutations(base_sequence: str, mutations: list) -> dict:
    """
    Featurize a set of mutations relative to base sequence.
    
    Args:
        base_sequence: Wild-type sequence
        mutations: List of mutations (e.g., ["K166R", "S205T"])
    
    Returns:
        Dictionary of mutation-specific features
    
    TODO:
    - Parse mutation strings (regex)
    - Compute ΔΔG predictions (placeholder: FoldX, Rosetta, or ML model)
    - Compute chromophore distance (if structure available)
    - Compute conservation scores (if MSA available)
    """
    # Placeholder
    features = {
        "n_mutations": len(mutations),
        "ddG": None,  # TODO: placeholder for stability prediction
        "chromophore_distance": None,  # TODO
    }
    return features


