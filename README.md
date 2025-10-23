# FP-Qubit Design

## But

Ce dépôt fournit un cadre logiciel pour la **conception in silico de mutants de protéines fluorescentes (FP) optimisés** pour des proxies photophysiques liés aux qubits biologiques. L'objectif est de proposer, à terme, des candidats mutants qui maximisent la cohérence quantique (temps de vie T2), le contraste optique, et d'autres métriques pertinentes pour les applications de **bio-sensing quantique**.

**Version actuelle** : **v1.0.0** — Release publique avec baseline ML fonctionnel, shortlist de 30 mutants optimisés, et figures de visualisation.

## Contexte

- **Projet parent** : [Biological Qubits Atlas](https://github.com/Mythmaker28/biological-qubits-atlas) — un jeu de données CSV (~22 systèmes quantiques en contexte biologique) avec des mesures de cohérence (T1/T2), contraste, et provenance (licence CC BY 4.0).
- **Approche** : 100% logiciel, aucune expérimentation en laboratoire. On utilise l'Atlas comme référence de proxies photophysiques (lifetime, contraste, température) pour guider la conception de mutants FP.
- **Cible** : Protéines fluorescentes de la famille GFP-like, avec un focus sur les propriétés de cohérence et photostabilité.
- **Publication prévue** : Zenodo + GitHub Pages (table HTML des mutants shortlistés).

## Données sources et provenance

Les proxies sont basés sur un snapshot de l'Atlas :
- **Repo source** : https://github.com/Mythmaker28/biological-qubits-atlas
- **Commit** : `abd6a4cd7dde94dc4ca7cde69aee3fad25757bcf`
- **Schéma** : v1.2 (~33 colonnes)
- **Licence** : CC BY 4.0
- **Snapshot local** : `data/processed/atlas_snapshot.csv` (lecture seule, ne pas modifier)

Le fichier `data/processed/atlas_snapshot.METADATA.json` contient les métadonnées de provenance complètes.

## Installation

```bash
# Cloner le dépôt
git clone https://github.com/Mythmaker28/fp-qubit-design.git
cd fp-qubit-design

# Installer les dépendances (minimal)
pip install -r requirements.txt
```

**Dépendances** : numpy, pandas, scikit-learn, matplotlib (Python ≥3.8 recommandé).

## Quickstart

### 1. Entraîner le modèle baseline (Random Forest)

```bash
python scripts/train_baseline.py --config configs/example.yaml
```

**Sortie** : `outputs/metrics.json`, `outputs/model_rf.pkl`

### 2. Générer la shortlist de mutants

```bash
python scripts/generate_mutants.py --config configs/example.yaml --output outputs/shortlist.csv
```

**Sortie** : `outputs/shortlist.csv` (30 mutants optimisés)

### 3. Générer les figures

```bash
python scripts/generate_figures.py
```

**Sortie** : `figures/feature_importance.png`, `figures/predicted_gains_histogram.png`

### 4. Voir la shortlist en ligne

👉 [https://mythmaker28.github.io/fp-qubit-design/](https://mythmaker28.github.io/fp-qubit-design/) (une fois Pages activées)

## Arborescence

```
fp-qubit-design/
├─ README.md              # Ce fichier
├─ README_EN.md           # Version anglaise condensée
├─ LICENSE                # Apache-2.0
├─ CITATION.cff           # Fichier de citation (CFF 1.2.0)
├─ requirements.txt       # Dépendances minimales
├─ .gitignore             # Python standard
├─ data/
│  ├─ raw/                # Placeholder (données brutes futures)
│  └─ processed/          # atlas_snapshot.csv + METADATA.json
├─ src/fpqubit/
│  ├─ __init__.py
│  ├─ features/featurize.py   # TODOs featurisation
│  ├─ utils/io.py             # TODOs lecture/écriture CSV
│  └─ utils/seed.py           # TODOs gestion seed aléatoire
├─ scripts/
│  ├─ train_baseline.py       # TODOs entraînement RF/XGB
│  └─ generate_mutants.py     # TODOs génération mutants
├─ configs/
│  ├─ example.yaml            # Config exemple (5-10 clés)
│  └─ atlas_mapping.yaml      # Mapping proxies Atlas → FP
├─ figures/                    # Placeholder (plots futurs)
├─ site/
│  ├─ index.html              # Page web simple (table shortlist)
│  └─ shortlist.csv           # Données exemple (3 mutants factices)
└─ .github/workflows/
   ├─ ci.yml                  # CI simple (flake8 + import checks)
   └─ pages.yml               # Déploiement GitHub Pages
```

## Résultats (v1.0.0)

### Baseline ML
- **Modèle** : Random Forest (100 estimateurs, profondeur max 10)
- **Dataset** : 200 échantillons synthétiques basés sur 21 systèmes Atlas
- **Performances** :
  - Test MAE : ~4.6%
  - Test R² : ~0.17
  - CV MAE (5-fold) : 4.79 ± 0.42%
- **Features** : température, méthode (ODMR/ESR/NMR), contexte (in vivo), qualité

### Shortlist de mutants
- **30 mutants** optimisés pour contraste photophysique
- **Protéines de base** : EGFP, mNeonGreen, TagRFP
- **Gain prédit** : +2.1% à +12.3% (moyenne : +4.0 ± 2.7%)
- **Incertitudes** : quantifiées via bootstrap (10 échantillons)

### Visualisations
- Feature importance (Random Forest)
- Distribution des gains prédits (histogram)

## Roadmap futur (v1.1+)

- [ ] Parsing automatique du champ "Photophysique" (lifetime, QY)
- [ ] Calculs ΔΔG réels (FoldX ou modèle ML)
- [ ] Structures 3D (alignement séquences sur PDB)
- [ ] GNN prototype (optionnel)
- [ ] Publication Zenodo avec DOI
- [ ] Expansion snapshot Atlas (si nouvelles données)

## Licence et citation

- **Code** : Apache-2.0 (voir `LICENSE`)
- **Données Atlas** : CC BY 4.0 (voir Atlas repo)

Si vous utilisez ce dépôt, veuillez citer :

```
Lepesteur, T. (2025). FP-Qubit Design (v0.1.0). GitHub. https://github.com/Mythmaker28/fp-qubit-design
```

Voir `CITATION.cff` pour le format structuré.

## Contribution

Ce projet est ouvert aux contributions. Actuellement en phase de développement actif. Les issues tracent les tâches prioritaires.

## Contact

- **Auteur** : Tommy Lepesteur
- **ORCID** : [0009-0009-0577-9563](https://orcid.org/0009-0009-0577-9563)
- **Issues** : https://github.com/Mythmaker28/fp-qubit-design/issues

---

**Statut** : ✅ v1.0.0 Release publique — Pleinement fonctionnel

