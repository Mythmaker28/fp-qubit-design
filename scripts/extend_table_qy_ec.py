import pandas as pd
import numpy as np
from pathlib import Path

INPUT = Path('data/processed/TRAINING_TABLE_v2_2_2_balanced.csv')
OUTPUT = Path('data/processed/TRAINING_TABLE_v2_2_2_extended.csv')

# Columns expected to exist in the balanced table
REQUIRED_COLS = ['name','family','excitation_nm','emission_nm']

NEW_COLS = ['QY','EC','brightness','photostability_min']

# Placeholder enrichment: this script only adds the columns with NA.
# Fill QY/EC later from FPbase/papers, brightness is computed when available.

def main():
    if not INPUT.exists():
        raise SystemExit(f'Input not found: {INPUT}. Provide the Atlas v2.2.2 balanced CSV.')
    df = pd.read_csv(INPUT)
    for c in REQUIRED_COLS:
        if c not in df.columns:
            raise SystemExit(f'Missing required column: {c}')
    for c in NEW_COLS:
        if c not in df.columns:
            df[c] = np.nan
    # Compute brightness where possible
    mask = df['QY'].notna() & df['EC'].notna()
    df.loc[mask, 'brightness'] = (df.loc[mask, 'QY'] * df.loc[mask, 'EC']) / 1000.0
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)
    print(f'Wrote: {OUTPUT} (rows={len(df)})')

if __name__ == '__main__':
    main()
