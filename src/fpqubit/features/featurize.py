"""
Featurization for FP quantum design
Converts FP properties to ML-ready features
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import Tuple, List

class FPFeaturizer:
    """
    Featurizer for fluorescent protein properties
    
    Features include:
    - Family (one-hot encoded)
    - Photophysical properties (excitation, emission, Stokes shift)
    - Environmental conditions (temperature, pH)
    - Biosensor flag
    - Derived features (ex/em ratios, normalized values)
    """
    
    def __init__(self):
        self.family_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.scaler = StandardScaler()
        self.feature_names = []
        self.fitted = False
    
    def _extract_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract numerical and categorical features from DataFrame"""
        features = pd.DataFrame()
        
        # Photophysical properties (with fallback if all NaN)
        ex_median = df['excitation_nm'].median() if not df['excitation_nm'].isna().all() else 488.0
        em_median = df['emission_nm'].median() if not df['emission_nm'].isna().all() else 510.0
        features['excitation_nm'] = df['excitation_nm'].fillna(ex_median)
        features['emission_nm'] = df['emission_nm'].fillna(em_median)
        
        # Derived: Stokes shift (emission - excitation)
        features['stokes_shift_nm'] = features['emission_nm'] - features['excitation_nm']
        
        # Derived: Ex/Em ratio
        features['ex_em_ratio'] = features['excitation_nm'] / (features['emission_nm'] + 1e-6)
        
        # Environmental conditions
        features['temperature_K'] = df['temperature_K'].fillna(298.0)  # Room temp default
        features['pH'] = df['pH'].fillna(7.0)  # Neutral pH default
        
        # Derived: Thermal energy k*T (eV)
        k_B = 8.617e-5  # Boltzmann constant in eV/K
        features['kT_eV'] = k_B * features['temperature_K']
        
        # Derived: Temperature regime (categorical → numerical)
        features['is_cryogenic'] = (features['temperature_K'] < 150).astype(int)
        features['is_room_temp'] = ((features['temperature_K'] >= 280) & (features['temperature_K'] <= 310)).astype(int)
        features['is_physiological'] = ((features['temperature_K'] >= 310) & (features['temperature_K'] <= 320)).astype(int)
        
        # Derived: pH regime
        features['is_acidic'] = (features['pH'] < 6.5).astype(int)
        features['is_neutral'] = ((features['pH'] >= 6.5) & (features['pH'] <= 7.5)).astype(int)
        features['is_basic'] = (features['pH'] > 7.5).astype(int)
        
        # Biosensor flag
        features['is_biosensor'] = df['is_biosensor'].fillna(False).astype(int)
        
        # Spectral region (categorical → numerical)
        # Blue: <480nm, Cyan: 480-510, Green: 510-540, Yellow: 540-570, 
        # Orange: 570-600, Red: 600-650, Far-red: >650
        em = features['emission_nm']
        features['is_blue'] = (em < 480).astype(int)
        features['is_cyan'] = ((em >= 480) & (em < 510)).astype(int)
        features['is_green'] = ((em >= 510) & (em < 540)).astype(int)
        features['is_yellow'] = ((em >= 540) & (em < 570)).astype(int)
        features['is_orange'] = ((em >= 570) & (em < 600)).astype(int)
        features['is_red'] = ((em >= 600) & (em < 650)).astype(int)
        features['is_far_red'] = (em >= 650).astype(int)
        
        return features
    
    def _encode_family(self, df: pd.DataFrame) -> np.ndarray:
        """One-hot encode family"""
        family_array = df[['family']].values
        if not self.fitted:
            encoded = self.family_encoder.fit_transform(family_array)
        else:
            encoded = self.family_encoder.transform(family_array)
        return encoded
    
    def fit(self, df: pd.DataFrame) -> 'FPFeaturizer':
        """Fit featurizer on training data"""
        # Extract base features
        base_features = self._extract_base_features(df)
        
        # Encode family
        family_encoded = self._encode_family(df)
        
        # Combine
        X = np.hstack([base_features.values, family_encoded])
        
        # Fit scaler
        self.scaler.fit(X)
        
        # Store feature names
        family_names = [f"family_{cat}" for cat in self.family_encoder.categories_[0]]
        self.feature_names = list(base_features.columns) + family_names
        
        self.fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Transform DataFrame to feature matrix"""
        if not self.fitted:
            raise ValueError("Featurizer must be fitted before transform")
        
        # Extract base features
        base_features = self._extract_base_features(df)
        
        # Encode family
        family_encoded = self._encode_family(df)
        
        # Combine
        X = np.hstack([base_features.values, family_encoded])
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        return X_scaled, self.feature_names
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Fit and transform in one step"""
        self.fit(df)
        return self.transform(df)
    
    def get_feature_names(self) -> List[str]:
        """Get feature names"""
        return self.feature_names


def load_and_featurize(csv_path: str, fit: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    """
    Load CSV and featurize
    
    Args:
        csv_path: Path to train_measured.csv
        fit: Whether to fit the featurizer (True for training, False for prediction)
    
    Returns:
        X: Feature matrix (N x D)
        y: Target vector (N,) - contrast_normalized
        feature_names: List of feature names
        df: Original DataFrame
    """
    df = pd.read_csv(csv_path)
    
    # Target: contrast_normalized
    y = df['contrast_normalized'].values
    
    # Features
    featurizer = FPFeaturizer()
    if fit:
        X, feature_names = featurizer.fit_transform(df)
    else:
        X, feature_names = featurizer.transform(df)
    
    return X, y, feature_names, df


# Example usage
if __name__ == "__main__":
    from pathlib import Path
    
    # Path to training data
    project_root = Path(__file__).parent.parent.parent.parent
    train_csv = project_root / "data" / "processed" / "train_measured.csv"
    
    print("="*60)
    print("Featurization Demo")
    print("="*60)
    
    # Load and featurize
    X, y, feature_names, df = load_and_featurize(str(train_csv))
    
    print(f"\n[INFO] Loaded {len(df)} samples")
    print(f"[INFO] Feature matrix shape: {X.shape}")
    print(f"[INFO] Target vector shape: {y.shape}")
    
    print(f"\n[INFO] Features ({len(feature_names)}):")
    for i, name in enumerate(feature_names[:10]):
        print(f"  [{i}] {name}")
    if len(feature_names) > 10:
        print(f"  ... and {len(feature_names) - 10} more")
    
    print(f"\n[INFO] Target (contrast_normalized):")
    print(f"  Min: {y.min():.3f}")
    print(f"  Max: {y.max():.3f}")
    print(f"  Mean: {y.mean():.3f}")
    print(f"  Std: {y.std():.3f}")
    
    print("\n" + "="*60)
    print("[SUCCESS] Featurization complete!")
    print("="*60)
