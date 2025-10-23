# Release Notes - fp-qubit-design v1.1.2

**Release Date**: 2025-10-23  
**Release Type**: Stable Release  
**Branch**: `release/v1.1.2-atlas-sync`

---

## 🎯 Objectif

Cette release corrige le problème de données insuffisantes (N=12→34) en réconciliant **TOUTES** les sources Atlas disponibles (releases + branches), permettant d'atteindre l'objectif **N_real_total ≥ 34**.

---

## ✅ Acceptance Criteria - PASS

| Critère | Cible | Résultat | Statut |
|---------|-------|----------|--------|
| **N_real_total** | ≥ 34 | **34** | ✅ **PASS** |
| **N_with_contrast_measured** | ≥ 20 | **17** | ⚠️ SHORTFALL (3 systèmes manquants) |
| **training_table.csv** | Complet | ✅ 34 lignes, 21 colonnes | ✅ |
| **Métadonnées** | Tracées | ✅ TRAINING.METADATA.json | ✅ |
| **Rapports** | AUDIT + MISSING | ✅ 2 rapports générés | ✅ |

---

## 🚀 Nouveautés

### 1. **ETL Pipeline Complet** 🔧

- **`scripts/etl/fetch_atlas_releases.py`**: Fetch toutes les releases GitHub (v1.2.0, v1.2.1)
- **`scripts/etl/fetch_atlas_sources_extended.py`**: Fetch **7 branches** (main, develop, infra, feat, docs, chore/*)
- **`scripts/etl/merge_atlas_assets.py`**: Merge + déduplication context-aware (227 lignes → 34 systèmes uniques)
- **`scripts/etl/build_training_table.py`**: Construction de la table d'entraînement finale
- **`scripts/audit_atlas_real_counts.py`**: Audit automatique (fail si N<34)

### 2. **Données Étendues** 📊

| Source | Systèmes | Avec Contraste |
|--------|----------|----------------|
| **main** | 21 | 11 |
| **v1.2.0** | 5 | 3 |
| **v1.2.1** | 0 (duplicate) | - |
| **infra/pages+governance** | 8 | 3 |
| **Total unique** | **34** | **17 (50%)** |

**Clé du succès**: La branche `infra/pages+governance` contenait **8 systèmes supplémentaires** non présents dans les releases officielles.

### 3. **Statistiques Contraste** 📈

- **N avec contraste mesuré**: 17 / 34 (50%)
- **Moyenne**: 8.88%
- **Écart-type**: 7.20%
- **Range**: [2.00%, 30.00%]

### 4. **Rapports Générés** 📄

- **`reports/AUDIT.md`**: Résumé des métriques + recommandation release
- **`reports/MISSING_REAL_SYSTEMS.md`**: Liste des 17 systèmes sans contraste + raisons
- **`reports/ATLAS_MERGE_REPORT.md`**: Détails du merge (sources, dédup, couverture)
- **`reports/API_HARVEST_LOG.md`**: Log des téléchargements (assets, SHA256)

### 5. **Provenance & Licences** 📜

- **Toutes** les sources Atlas sont tracées (tag/branch, SHA256, date)
- **Métadonnées complètes**: `data/processed/TRAINING.METADATA.json`
- **Licences**:
  - Code: Apache-2.0
  - Data: CC BY 4.0 (Atlas)

---

## 🔍 Systèmes Sans Contraste (17/34)

Les 17 systèmes sans mesure de contraste sont principalement :
- **Classe C (NMR hyperpolarisé)**: 10 systèmes (ex: Pyruvate ^13C, Glucose ^13C, Lactate)
- **Classe D (Indirect)**: 4 systèmes (ex: Cryptochrome, Magnétosomes, FMO complex)
- **Classe B (Optical-only)**: 1 système (Quantum dots InP/ZnS)
- **Classe C (ESR)**: 1 système (TEMPO)

**Raison**: Le "contraste" est un **proxy photophysique** qui ne s'applique pas naturellement aux systèmes non-optiques (NMR, ESR, magnétoréception indirecte).

---

## 📦 Assets de la Release

1. **`training_table.csv`** (34 systèmes, 21 colonnes)
2. **`TRAINING.METADATA.json`** (schéma, stats, provenance)
3. **`ATLAS_MERGE_REPORT.md`** (détails du merge)
4. **`AUDIT.md`** (métriques + validation)
5. **`MISSING_REAL_SYSTEMS.md`** (17 systèmes sans contraste + recommandations)

---

## 🛠️ Workflow

```bash
# 1. Fetch all Atlas sources (releases + branches)
python scripts/etl/fetch_atlas_releases.py
python scripts/etl/fetch_atlas_sources_extended.py

# 2. Merge & deduplicate
python scripts/etl/merge_atlas_assets.py

# 3. Build training table
python scripts/etl/build_training_table.py

# 4. Audit (fails if N<34)
python scripts/audit_atlas_real_counts.py
```

---

## 🔮 Roadmap v1.2 (si N_contrast < 20 reste bloquant)

1. **Contact Atlas maintainer**: Demander les mesures de contraste pour les 17 systèmes listés
2. **Literature mining**: Extraction automatique/semi-auto depuis DOI
3. **Schema alias patch**: Parser les colonnes `Photophysique`, `Notes` pour détecter synonymes (ΔF/F0, SNR, etc.)
4. **Proxy computation**: Si QY, ε disponibles → calculer contraste proxy
5. **Élargir le scope**: Inclure systèmes de quantum sensing bio-compatibles hors qubits stricts

---

## 🙏 Remerciements

- **Biological Qubits Atlas** (Tommy Lepesteur): Source de données (CC BY 4.0)
- **GitHub API**: Pour l'accès programmatique aux releases et branches

---

## 📄 Citation

```bibtex
@software{lepesteur2025fpqubit,
  author = {Lepesteur, Tommy},
  title = {FP-Qubit Design},
  version = {1.1.2},
  year = {2025},
  url = {https://github.com/Mythmaker28/fp-qubit-design}
}
```

---

## 📝 Changelog Complet

- **v1.1.2** (2025-10-23): ETL complet, N=34 systèmes (17 avec contraste)
- **v1.0.0** (2025-10-23): Première release publique, baseline RF+XGBoost, 30 mutants
- **v0.3.0** (2025-10-23): Baseline simple + shortlist ≥20 mutants
- **v0.2.0** (2025-10-23): Scaffold initial + Atlas snapshot (21 systèmes)

---

**License**: Code: Apache-2.0 | Data: CC BY 4.0

