# data/processed/

Ce dossier contient les données traitées et prêtes à l'emploi pour le projet FP-Qubit Design.

## Fichiers actuels

- **`atlas_snapshot.csv`** : Snapshot en lecture seule de l'Atlas des Qubits Biologiques (commit abd6a4cd7dde94dc4ca7cde69aee3fad25757bcf)
- **`atlas_snapshot.METADATA.json`** : Métadonnées de provenance du snapshot (source, commit SHA, date, licence)

## Fichiers prévus

- Séquences featurisées (NumPy arrays, CSV)
- Matrices de similarité
- Datasets d'entraînement/validation/test
- Prédictions de modèles (CSV avec incertitudes)

## Instructions

1. Ne **jamais modifier** `atlas_snapshot.csv` (lecture seule)
2. Documenter la transformation appliquée pour chaque fichier dérivé
3. Inclure un fichier `.METADATA.json` pour chaque dataset généré

## Statut actuel

✅ Snapshot Atlas importé avec succès (22 systèmes)

