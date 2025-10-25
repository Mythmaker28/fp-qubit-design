import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

EXTENDED = Path('data/processed/TRAINING_TABLE_v2_2_2_extended.csv')
SHORTLIST = Path('deliverables/lab_v2_2_3/shortlist_top12_final.csv')
FIG = Path('figures/spectral_coverage_v2_2_3.png')

N_CLUSTERS = 12
MAX_CALCIUM = 3

def main():
    df = pd.read_csv(EXTENDED)
    # step A: top brightness (fallback to dropna)
    sub = df.dropna(subset=['brightness']).copy()
    sub = sub.nlargest(30, 'brightness') if len(sub) > 30 else sub

    # step B: spectral clustering for diversity
    km = KMeans(n_clusters=min(N_CLUSTERS, len(sub)), n_init=10, random_state=42)
    labels = km.fit_predict(sub[['excitation_nm','emission_nm']])
    sub['cluster'] = labels

    # step C: 1 winner per cluster (max brightness)
    picks = (sub.sort_values('brightness', ascending=False)
               .groupby('cluster', as_index=False)
               .head(1))

    # step D: enforce calcium cap
    def cap_calcium(dfp):
        fam = dfp['family'].fillna('Unknown').tolist()
        if fam.count('Calcium') <= MAX_CALCIUM:
            return dfp
        kept, dropped = [], []
        calcium_seen = 0
        for _, row in dfp.sort_values('brightness', ascending=False).iterrows():
            if row.get('family','') == 'Calcium':
                if calcium_seen < MAX_CALCIUM:
                    kept.append(row)
                    calcium_seen += 1
                else:
                    dropped.append(row)
            else:
                kept.append(row)
        return pd.DataFrame(kept)

    picks = cap_calcium(picks)

    picks = picks.sort_values('brightness', ascending=False).head(N_CLUSTERS)
    SHORTLIST.parent.mkdir(parents=True, exist_ok=True)
    cols = ['name','family','excitation_nm','emission_nm','QY','EC','brightness']
    picks[cols].to_csv(SHORTLIST, index=False)

    # Figure
    plt.figure()
    plt.scatter(picks['excitation_nm'], picks['emission_nm'])
    plt.xlabel('Excitation (nm)')
    plt.ylabel('Emission (nm)')
    plt.title('Spectral coverage — shortlist v2.2.3 (no-ML)')
    FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG, dpi=200, bbox_inches='tight')
    print(f'Wrote: {SHORTLIST} and {FIG}')

if __name__ == '__main__':
    main()
