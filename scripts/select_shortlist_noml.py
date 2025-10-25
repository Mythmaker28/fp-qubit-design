import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
import json
import matplotlib.pyplot as plt

EXTENDED = Path('data/processed/TRAINING_TABLE_v2_2_2_extended.csv')
SHORTLIST = Path('deliverables/lab_v2_2_3/shortlist_top12_final.csv')
FIG = Path('figures/spectral_coverage_v2_2_3.png')
FAMILY_WEIGHTS = Path('deliverables/family_weights_v2_2_2.json')

N_CLUSTERS = 12
MAX_CALCIUM = 3

def load_family_weights():
    if FAMILY_WEIGHTS.exists():
        with open(FAMILY_WEIGHTS, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def ensure_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def fallback_noml_selection(df):
    # Score using measured contrast + family rarity weights
    fam_w = load_family_weights()
    df['fam_w'] = df['family'].map(fam_w).fillna(1.0)
    # normalize contrast
    x = df['contrast_normalized']
    x = (x - x.mean()) / (x.std(ddof=0) + 1e-9)
    df['score_noml'] = 0.5 * x + 0.5 * df['fam_w']
    # cluster on spectrum for diversity
    k = min(N_CLUSTERS, len(df))
    if k == 0:
        return df.head(0)
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(df[['excitation_nm','emission_nm']])
    df['cluster'] = labels
    # pick best by score in each cluster
    picks = (df.sort_values('score_noml', ascending=False)
               .groupby('cluster', as_index=False)
               .head(1))
    return picks

def cap_calcium(dfp):
    fam = dfp['family'].fillna('Unknown').tolist()
    if fam.count('Calcium') <= MAX_CALCIUM:
        return dfp
    kept = []
    calcium_seen = 0
    for _, row in dfp.sort_values('brightness', ascending=False, na_position='last').iterrows():
        if row.get('family','') == 'Calcium':
            if calcium_seen < MAX_CALCIUM:
                kept.append(row)
                calcium_seen += 1
        else:
            kept.append(row)
    return pd.DataFrame(kept)

def main():
    df = pd.read_csv(EXTENDED)
    ensure_numeric(df, ['excitation_nm','emission_nm','QY','EC','brightness','contrast_normalized'])
    # Path A: use brightness if we have enough values
    bright = df.dropna(subset=['brightness']).copy()
    if len(bright) >= N_CLUSTERS:
        bright = bright.nlargest(max(30, N_CLUSTERS), 'brightness')
        k = min(N_CLUSTERS, len(bright))
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(bright[['excitation_nm','emission_nm']])
        bright['cluster'] = labels
        picks = (bright.sort_values('brightness', ascending=False)
                       .groupby('cluster', as_index=False)
                       .head(1))
    else:
        # Path B: fallback using measured contrast + family rarity weights
        picks = fallback_noml_selection(df)

    # Enforce Calcium cap
    if not picks.empty:
        picks = cap_calcium(picks)

    # Final top N_CLUSTERS
    picks = picks.sort_values(['brightness','score_noml'], ascending=False, na_position='last').head(N_CLUSTERS)

    # Save CSV
    SHORTLIST.parent.mkdir(parents=True, exist_ok=True)
    cols_keep = ['canonical_name','family','excitation_nm','emission_nm','contrast_normalized','QY','EC','brightness']
    for c in cols_keep:
        if c not in picks.columns:
            picks[c] = np.nan
    picks[cols_keep].to_csv(SHORTLIST, index=False)

    # Figure
    if len(picks) > 0:
        plt.figure()
        plt.scatter(picks['excitation_nm'], picks['emission_nm'])
        plt.xlabel('Excitation (nm)')
        plt.ylabel('Emission (nm)')
        plt.title('Spectral coverage — shortlist v2.2.3 (no-ML)')
        FIG.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(FIG, dpi=200, bbox_inches='tight')
    print(f'[OK] Wrote: {SHORTLIST} and {FIG}')

if __name__ == '__main__':
    main()
