# Release Notes - FP-Qubit Design

## v1.0.0 (2025-10-23) - Public Release

### 🎉 Première release publique

**Highlights:**
- ✅ Baseline ML fonctionnel (Random Forest)
- ✅ Shortlist de 30 mutants FP optimisés pour proxies "qubit-friendly"
- ✅ Site web interactif avec GitHub Pages
- ✅ Documentation complète (FR + EN)
- ✅ CI/CD avec GitHub Actions
- ✅ Attribution claire de l'Atlas (CC BY 4.0)

### Features

**Data & Provenance**
- Snapshot de l'Atlas des Qubits Biologiques (21 systèmes, commit `abd6a4cd`)
- Métadonnées complètes de provenance (METADATA.json)
- Attribution CC BY 4.0 dans NOTICE

**Machine Learning**
- Baseline Random Forest implémenté (`scripts/train_baseline.py`)
- Features: composition AA, propriétés physicochimiques, proxies Atlas
- Cross-validation 5-fold (MAE, R², RMSE)
- Métriques sauvegardées dans `outputs/metrics.json`

**Mutant Generation**
- Génération de mutants candidats (`scripts/generate_mutants.py`)
- 30 mutants FP (EGFP, mNeonGreen, TagRFP)
- Prédictions de gain avec incertitudes (bootstrap)
- Shortlist exportée: `outputs/shortlist.csv`

**Visualization**
- Feature importance plot (`figures/feature_importance.png`)
- Distribution des gains prédits (`figures/predicted_gains_histogram.png`)

**Documentation**
- README complet (FR + EN)
- CITATION.cff valide (CFF 1.2.0)
- NOTICE avec attribution Atlas
- Mapping proxies documenté (`configs/atlas_mapping.yaml`)

**Infrastructure**
- Site web GitHub Pages avec table interactive
- CI: lint (flake8) + test imports + dry-run scripts
- Workflows GitHub Actions (ci.yml + pages.yml)

### Requirements
- Python ≥ 3.8
- numpy, pandas, scikit-learn, matplotlib, pyyaml

### Known Limitations
- Snapshot Atlas: 21 systèmes (cible ≥34 non atteinte, limité par données disponibles)
- ΔΔG placeholder (pas de calculs FoldX/Rosetta)
- Pas de structures 3D (positions mutations approximatives)
- Baseline simple (RF uniquement, pas de GNN)

---

## v0.3.0 (2025-10-23) - Baseline & Shortlist

### Added
- Baseline Random Forest fonctionnel
- Génération de 30 mutants candidats
- Shortlist CSV réelle (20+ lignes)
- 2 figures (importance variables + histogram gains)
- Outputs: metrics.json, shortlist.csv

### Changed
- Scripts squelettes → implémentations fonctionnelles
- Site web: charge shortlist réelle (outputs/shortlist.csv)

---

## v0.2.0 (2025-10-23) - Foundation & Pages

### Added
- Snapshot Atlas (21 systèmes, commit `abd6a4cd`)
- Métadonnées de provenance (METADATA.json)
- NOTICE avec attribution CC BY 4.0
- Mapping proxies (`configs/atlas_mapping.yaml`)
- GitHub Pages activées (site web live)

### Changed
- CITATION.cff: version 0.2.0
- Documentation: mention snapshot Atlas + SHA

---

## v0.1.0 (2025-10-23) - Initial Scaffold

### Added
- Structure initiale du projet
- Scripts squelettes (TODOs)
- Documentation de base (README FR + EN)
- CITATION.cff (CFF 1.2.0)
- LICENSE (Apache-2.0)
- Workflows CI/CD (squelettes)
- Site web basique (dummy data)

---

## Roadmap

### Future (v1.1.0+)
- [ ] Parsing automatique du champ "Photophysique" (lifetime, QY, ex/em)
- [ ] Calculs ΔΔG réels (FoldX ou modèle ML)
- [ ] Structures 3D (alignement séquences sur PDB)
- [ ] GNN prototype (optionnel)
- [ ] Publication Zenodo avec DOI
- [ ] Expansion snapshot Atlas (si nouvelles données disponibles)

---

**Author**: Tommy Lepesteur (ORCID: 0009-0009-0577-9563)  
**License**: Apache-2.0 (code), CC BY 4.0 (données Atlas)  
**Repository**: https://github.com/Mythmaker28/fp-qubit-design


