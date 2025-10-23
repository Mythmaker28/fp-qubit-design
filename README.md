# FP-Qubit Design

## But

Ce dépôt fournit un cadre logiciel pour la **conception in silico de mutants de protéines fluorescentes (FP) optimisés** pour des proxies photophysiques liés aux qubits biologiques. L'objectif est de proposer, à terme, des candidats mutants qui maximisent la cohérence quantique (temps de vie T2), le contraste optique, et d'autres métriques pertinentes pour les applications de **bio-sensing quantique**.

**Attention :** Ce dépôt est un **SQUELETTE** (version 0.1.0). Il pose les bases structurelles et de reproductibilité, mais **ne contient pas encore de modèles ML entraînés**. Les fichiers contiennent des TODOs et des placeholders pour le développement futur.

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

## Quickstart (squelette)

Les scripts actuels sont des squelettes avec TODOs :

```bash
# Exemple de script de featurisation (vide, TODOs)
python scripts/generate_mutants.py --config configs/example.yaml

# Exemple de baseline (vide, TODOs)
python scripts/train_baseline.py --config configs/example.yaml
```

**Note** : Ces commandes ne produisent rien pour l'instant. Elles servent de point de départ pour le développement.

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

## Roadmap (30/60/90 jours)

### 30 jours
- [ ] Définir mapping complet Atlas → proxies FP (colonnes pertinentes)
- [ ] Implémenter featurisation de base (composition AA, propriétés physicochimiques)
- [ ] Baseline RF/XGB sur proxies synthétiques (proof-of-concept)
- [ ] Premiers mutants générés (placeholder ΔΔG)

### 60 jours
- [ ] Validation croisée des baselines
- [ ] Incertitudes (bootstrap ou GP)
- [ ] Shortlist de 10-20 mutants "qubit-friendly"
- [ ] Publication page web (GitHub Pages) avec tableau interactif

### 90 jours
- [ ] (Optionnel) Prototype GNN sur graphe protéine
- [ ] Publication Zenodo avec DOI
- [ ] Documentation complète (IMRaD)
- [ ] Ouverture aux contributions externes

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

**Statut** : 🚧 Squelette (v0.1.0) — Développement en cours

