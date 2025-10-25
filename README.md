﻿# FP-Qubit Design

## But

Ce dÃ©pÃ´t fournit un cadre logiciel pour la **conception in silico de mutants de protÃ©ines fluorescentes (FP) optimisÃ©s** pour des proxies photophysiques liÃ©s aux qubits biologiques. L'objectif est de proposer, Ã  terme, des candidats mutants qui maximisent la cohÃ©rence quantique (temps de vie T2), le contraste optique, et d'autres mÃ©triques pertinentes pour les applications de **bio-sensing quantique**.

**Version actuelle** : **v1.1.2** â€” Release publique avec ETL Atlas complet, **34 systÃ¨mes rÃ©els** (17 avec contraste mesurÃ©), baseline ML fonctionnel, et shortlist de mutants optimisÃ©s.

## Contexte

- **Projet parent** : [Biological Qubits Atlas](https://github.com/Mythmaker28/biological-qubits-atlas) â€” un jeu de donnÃ©es CSV (**34 systÃ¨mes quantiques** en contexte biologique, rÃ©conciliÃ©s depuis 9 sources) avec des mesures de cohÃ©rence (T1/T2), contraste (17 systÃ¨mes), et provenance (licence CC BY 4.0).
- **Approche** : 100% logiciel, aucune expÃ©rimentation en laboratoire. On utilise l'Atlas comme rÃ©fÃ©rence de proxies photophysiques (lifetime, contraste, tempÃ©rature) pour guider la conception de mutants FP.
- **Cible** : ProtÃ©ines fluorescentes de la famille GFP-like, avec un focus sur les propriÃ©tÃ©s de cohÃ©rence et photostabilitÃ©.
- **Publication prÃ©vue** : Zenodo + GitHub Pages (table HTML des mutants shortlistÃ©s).

## DonnÃ©es sources et provenance

Les proxies sont basÃ©s sur une **rÃ©conciliation exhaustive** de l'Atlas (v1.1.2) :
- **Repo source** : https://github.com/Mythmaker28/biological-qubits-atlas
- **Sources mergÃ©es** : main, v1.2.0, v1.2.1, develop, infra/pages+governance, feat/data-v1.2-extended, docs/doi-badge, chore/zenodo-metadata, chore/citation-author (9 sources)
- **SystÃ¨mes uniques** : **34** (dÃ©duplication context-aware)
- **Avec contraste mesurÃ©** : **17 / 34** (50%)
- **SchÃ©ma** : v1.2 (~33 colonnes)
- **Licence** : CC BY 4.0
- **Table finale** : `data/processed/training_table.csv` + `data/processed/TRAINING.METADATA.json`

ðŸ“Š **Statistiques contraste** : mean=8.88%, std=7.20%, range=[2.00%, 30.00%]

ðŸ“„ **Rapports** : `reports/AUDIT.md`, `reports/MISSING_REAL_SYSTEMS.md`, `reports/ATLAS_MERGE_REPORT.md`

## Installation

```bash
# Cloner le dÃ©pÃ´t
git clone https://github.com/Mythmaker28/fp-qubit-design.git
cd fp-qubit-design

# Installer les dÃ©pendances (minimal)
pip install -r requirements.txt
```

**DÃ©pendances** : numpy, pandas, scikit-learn, matplotlib (Python â‰¥3.8 recommandÃ©).

## Quickstart

### 1. EntraÃ®ner le modÃ¨le baseline (Random Forest)

```bash
python scripts/train_baseline.py --config configs/example.yaml
```

**Sortie** : `outputs/metrics.json`, `outputs/model_rf.pkl`

### 2. GÃ©nÃ©rer la shortlist de mutants

```bash
python scripts/generate_mutants.py --config configs/example.yaml --output outputs/shortlist.csv
```

**Sortie** : `outputs/shortlist.csv` (30 mutants optimisÃ©s)

### 3. GÃ©nÃ©rer les figures

```bash
python scripts/generate_figures.py
```

**Sortie** : `figures/feature_importance.png`, `figures/predicted_gains_histogram.png`

### 4. Voir la shortlist en ligne

ðŸ‘‰ [https://mythmaker28.github.io/fp-qubit-design/](https://mythmaker28.github.io/fp-qubit-design/) (une fois Pages activÃ©es)

## Arborescence

```
fp-qubit-design/
â”œâ”€ README.md              # Ce fichier
â”œâ”€ README_EN.md           # Version anglaise condensÃ©e
â”œâ”€ LICENSE                # Apache-2.0
â”œâ”€ CITATION.cff           # Fichier de citation (CFF 1.2.0)
â”œâ”€ requirements.txt       # DÃ©pendances minimales
â”œâ”€ .gitignore             # Python standard
â”œâ”€ data/
â”‚  â”œâ”€ raw/                # Placeholder (donnÃ©es brutes futures)
â”‚  â””â”€ processed/          # atlas_snapshot.csv + METADATA.json
â”œâ”€ src/fpqubit/
â”‚  â”œâ”€ __init__.py
â”‚  â”œâ”€ features/featurize.py   # TODOs featurisation
â”‚  â”œâ”€ utils/io.py             # TODOs lecture/Ã©criture CSV
â”‚  â””â”€ utils/seed.py           # TODOs gestion seed alÃ©atoire
â”œâ”€ scripts/
â”‚  â”œâ”€ train_baseline.py       # TODOs entraÃ®nement RF/XGB
â”‚  â””â”€ generate_mutants.py     # TODOs gÃ©nÃ©ration mutants
â”œâ”€ configs/
â”‚  â”œâ”€ example.yaml            # Config exemple (5-10 clÃ©s)
â”‚  â””â”€ atlas_mapping.yaml      # Mapping proxies Atlas â†’ FP
â”œâ”€ figures/                    # Placeholder (plots futurs)
â”œâ”€ site/
â”‚  â”œâ”€ index.html              # Page web simple (table shortlist)
â”‚  â””â”€ shortlist.csv           # DonnÃ©es exemple (3 mutants factices)
â””â”€ .github/workflows/
   â”œâ”€ ci.yml                  # CI simple (flake8 + import checks)
   â””â”€ pages.yml               # DÃ©ploiement GitHub Pages
```

## RÃ©sultats (v1.0.0)

### Baseline ML
- **ModÃ¨le** : Random Forest (100 estimateurs, profondeur max 10)
- **Dataset** : 200 Ã©chantillons synthÃ©tiques basÃ©s sur 21 systÃ¨mes Atlas
- **Performances** :
  - Test MAE : ~4.6%
  - Test RÂ² : ~0.17
  - CV MAE (5-fold) : 4.79 Â± 0.42%
- **Features** : tempÃ©rature, mÃ©thode (ODMR/ESR/NMR), contexte (in vivo), qualitÃ©

### Shortlist de mutants
- **30 mutants** optimisÃ©s pour contraste photophysique
- **ProtÃ©ines de base** : EGFP, mNeonGreen, TagRFP
- **Gain prÃ©dit** : +2.1% Ã  +12.3% (moyenne : +4.0 Â± 2.7%)
- **Incertitudes** : quantifiÃ©es via bootstrap (10 Ã©chantillons)

### Visualisations
- Feature importance (Random Forest)
- Distribution des gains prÃ©dits (histogram)

## Roadmap futur (v1.1+)

- [ ] Parsing automatique du champ "Photophysique" (lifetime, QY)
- [ ] Calculs Î”Î”G rÃ©els (FoldX ou modÃ¨le ML)
- [ ] Structures 3D (alignement sÃ©quences sur PDB)
- [ ] GNN prototype (optionnel)
- [ ] Publication Zenodo avec DOI
- [ ] Expansion snapshot Atlas (si nouvelles donnÃ©es)

## Licence et citation

- **Code** : Apache-2.0 (voir `LICENSE`)
- **DonnÃ©es Atlas** : CC BY 4.0 (voir Atlas repo)

Si vous utilisez ce dÃ©pÃ´t, veuillez citer :

```
Lepesteur, T. (2025). FP-Qubit Design (v0.1.0). GitHub. https://github.com/Mythmaker28/fp-qubit-design
```

Voir `CITATION.cff` pour le format structurÃ©.

## Contribution

Ce projet est ouvert aux contributions. Actuellement en phase de dÃ©veloppement actif. Les issues tracent les tÃ¢ches prioritaires.

## Contact

- **Auteur** : Tommy Lepesteur
- **ORCID** : [0009-0009-0577-9563](https://orcid.org/0009-0009-0577-9563)
- **Issues** : https://github.com/Mythmaker28/fp-qubit-design/issues

---

**Statut** : âœ… v1.0.0 Release publique â€” Pleinement fonctionnel


