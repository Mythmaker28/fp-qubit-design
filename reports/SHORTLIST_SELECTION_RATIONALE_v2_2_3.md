# Shortlist Selection Rationale v2.2.3 (No-ML)

## Pourquoi sans modèle ML ?
Les variantes RF/ExtraTrees/Huber (v2.2.2) n’atteignent pas les critères : R²=-0.18, MAE=8.49%.
Les prédictions y_pred ne sont donc pas un signal fiable pour ordonner des candidats.

## Stratégie adoptée (sans ML)

1. **Brightness** = QY × EC / 1000 (mM⁻¹·cm⁻¹).
2. **Diversité spectrale** : clustering sur (excitation_nm, emission_nm), 1 candidat/cluster.
3. **Diversité de familles** : ≥7 familles ; Calcium ≤3.
4. **Contexte** : panachage in vivo / in cellulo si dispo.

## Pipeline (fichiers)
- `data/processed/TRAINING_TABLE_v2_2_2_extended.csv` : table étendue (ajouts QY, EC, brightness, photostability si connue).
- `deliverables/lab_v2_2_3/shortlist_top12_final.csv` : shortlist 12 finale.
- `figures/spectral_coverage_v2_2_3.png` : dispersion exc/émission des 12.

## Critères GO (v2.2.3)
- ≥150/221 systèmes avec QY mesuré ; ≥120/221 EC ; ≥100 brightness.
- Couverture spectrale (exc.) ≥300 nm ; ≥7 familles représentées.

## Note
Les QY/EC proviennent de FPbase ou littérature ; si manquants ⇒ NA. Le script gère la robustesse et la transparence (NA conservés).
