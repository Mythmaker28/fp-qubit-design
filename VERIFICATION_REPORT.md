# Rapport de vérification - fp-qubit-design

**Date**: 2025-10-23  
**Version**: 0.1.0 (squelette)  
**Auteur**: Tommy Lepesteur (ORCID: 0009-0009-0577-9563)

---

## ✅ Critères de réussite

### 1. Structure du projet

✅ **Arborescence complète créée** (26 fichiers)

```
fp-qubit-design/
├─ README.md                              ✅ FR complet
├─ README_EN.md                           ✅ EN condensé
├─ LICENSE                                ✅ Apache-2.0
├─ CITATION.cff                           ✅ CFF 1.2.0 valide
├─ requirements.txt                       ✅ Minimal (numpy, pandas, sklearn, matplotlib, pyyaml)
├─ .gitignore                             ✅ Python standard
├─ ISSUES.md                              ✅ Documentation des 5 issues
├─ VERIFICATION_REPORT.md                 ✅ Ce fichier
├─ data/
│  ├─ raw/                                ✅ Placeholder + README
│  └─ processed/                          ✅ atlas_snapshot.csv + METADATA.json + README
├─ src/fpqubit/
│  ├─ __init__.py                         ✅ Version 0.1.0
│  ├─ features/featurize.py               ✅ Squelette avec TODOs
│  ├─ utils/io.py                         ✅ Squelette avec TODOs
│  └─ utils/seed.py                       ✅ Squelette avec TODOs
├─ scripts/
│  ├─ train_baseline.py                   ✅ Squelette avec parser args + TODOs
│  └─ generate_mutants.py                 ✅ Squelette avec parser args + TODOs
├─ configs/
│  ├─ example.yaml                        ✅ 5-10 clés simples
│  └─ atlas_mapping.yaml                  ✅ Mapping proxies + filtres + TODOs
├─ figures/                               ✅ Placeholder + README
├─ site/
│  ├─ index.html                          ✅ Table HTML simple avec fetch cache-bust
│  └─ shortlist.csv                       ✅ 3 mutants factices
└─ .github/workflows/
   ├─ ci.yml                              ✅ Job simple: flake8 + import checks
   └─ pages.yml                           ✅ Déploiement /site via GitHub Actions
```

---

### 2. Contenu des fichiers clés

#### ✅ README.md (FR)
- [x] But clair (conception in silico FP mutants)
- [x] Contexte (Atlas des Qubits Biologiques)
- [x] Scope (100% logiciel, squelette v0.1.0)
- [x] Section "Données sources et provenance" avec URL Atlas + commit SHA
- [x] Install (3 lignes)
- [x] Quickstart (2 commandes no-op)
- [x] Roadmap 30/60/90 jours
- [x] Licence + Citation (renvoi CFF)

#### ✅ README_EN.md
- [x] Version courte (1/3 de page)
- [x] Mêmes points clés en anglais

#### ✅ CITATION.cff
- [x] CFF version 1.2.0
- [x] Auteur : Lepesteur, Tommy
- [x] ORCID : 0009-0009-0577-9563
- [x] Type : software
- [x] Version : 0.1.0
- [x] Date : 2025-10-23
- [x] Repo : https://github.com/Mythmaker28/fp-qubit-design
- [x] Licence : Apache-2.0

#### ✅ site/index.html
- [x] Page simple (titre + 3 puces but/contexte/scope)
- [x] Tableau qui charge ./shortlist.csv (fetch)
- [x] Cache-bust : `fetch('shortlist.csv?v=' + Date.now())`
- [x] CSS lisible (pas de complexité inutile)
- [x] Footer avec auteur, ORCID, liens repo/Atlas

#### ✅ site/shortlist.csv
- [x] 3 mutants factices
- [x] Colonnes : mutant_id, base_protein, mutations, proxy_target, predicted_gain, uncertainty, rationale

---

### 3. Connexion avec l'Atlas (lecture seule)

✅ **Atlas cloné en lecture seule**
- Repo : https://github.com/Mythmaker28/biological-qubits-atlas
- Branch : main
- Commit SHA : `abd6a4cd7dde94dc4ca7cde69aee3fad25757bcf`
- Licence : CC BY 4.0

✅ **Snapshot créé**
- Fichier : `data/processed/atlas_snapshot.csv`
- Nombre de systèmes : 22 (ligne 2 à 23 du CSV)
- Métadonnées : `data/processed/atlas_snapshot.METADATA.json`
  - source_repo ✅
  - branch ✅
  - commit ✅
  - schema (v1.2) ✅
  - rows (22) ✅
  - date_cloned ✅
  - license ✅

✅ **Mapping proxies créé**
- Fichier : `configs/atlas_mapping.yaml`
- Proxies définis : lifetime, contrast, temperature, method, context
- Filtres définis : only_room_temperature, exclude_indirect, min_verification, exclude_toxic, min_quality
- Colonnes manquantes identifiées : Quantum_yield, ISC_rate, Photostability
- TODOs documentés

✅ **Vérification colonnes Atlas**

Colonnes présentes dans `atlas_snapshot.csv` (33 colonnes) :
1. Systeme
2. Classe
3. Hote_contexte
4. Methode_lecture
5. Frequence
6. B0_Tesla
7. Spin_type
8. Defaut
9. Polytype_Site
10. T1_s
11. T2_us
12. Contraste_%
13. Temperature_K
14. Taille_objet_nm
15. Source_T2
16. Source_T1
17. Source_Contraste
18. T2_us_err
19. T1_s_err
20. Contraste_err
21. Hyperpol_flag
22. Cytotox_flag
23. Toxicity_note
24. Temp_controlled
25. Photophysique
26. Conditions
27. Limitations
28. In_vivo_flag
29. DOI
30. Annee
31. Qualite
32. Verification_statut
33. Notes

**Colonnes clés vérifiées** :
- ✅ T1_s, T2_us (cohérence)
- ✅ Contraste_% (contraste optique)
- ✅ Temperature_K (température)
- ✅ Methode_lecture (méthode)
- ✅ Hote_contexte (contexte biologique)
- ✅ Photophysique (champ texte avec lifetime, QY, ex/em)
- ✅ Cytotox_flag (toxicité)
- ✅ Verification_statut (qualité)

**Colonnes manquantes** (documentées dans `atlas_mapping.yaml`) :
- ⚠️ Quantum_yield (pas de colonne dédiée, dans Photophysique)
- ⚠️ ISC_rate (taux de croisement intersystème, absent)
- ⚠️ Photostability (mentionné en texte seulement)

---

### 4. GitHub Workflows

✅ **ci.yml (CI simple)**
- [x] Trigger sur push/PR (main/master)
- [x] Setup Python 3.9
- [x] Install requirements.txt
- [x] Lint avec flake8 (syntax errors E9,F63,F7,F82)
- [x] Test imports (fpqubit, seed, io)
- [x] Dry-run scripts (train_baseline.py, generate_mutants.py)

✅ **pages.yml (GitHub Pages)**
- [x] Trigger sur push main/master + workflow_dispatch
- [x] Permissions : contents:read, pages:write, id-token:write
- [x] Upload artifact depuis ./site
- [x] Deploy to GitHub Pages

**Note** : GitHub Pages doit être activé manuellement dans Settings → Pages → Source = "GitHub Actions"

---

### 5. Issues initiales (documentées)

✅ **5 issues documentées dans ISSUES.md** :

1. **[Data] Connect Atlas → Define proxy mapping**
   - Labels : data, priority-high, good-first-issue
   - Tâches : Vérifier colonnes, parser Photophysique, compléter mapping, créer load_atlas_proxies()

2. **[ML] Implement baseline models (Random Forest, XGBoost)**
   - Labels : ml, priority-high
   - Tâches : Définir features, créer dataset, splitter, entraîner RF/XGB, CV, sauvegarder modèles

3. **[Pipeline] Define mutant shortlist selection pipeline**
   - Labels : pipeline, priority-medium
   - Tâches : Générer mutations, ΔΔG placeholder, prédire proxies, score multi-objectif, shortlist

4. **[Docs] Create IMRaD template + Zenodo publication plan**
   - Labels : docs, priority-low, publication
   - Tâches : Template IMRaD, rédiger sections, plan Zenodo, checklist pré-publication

5. **[Infra] Setup GitHub badges, topics, and Pages**
   - Labels : infra, priority-medium, github-pages
   - Tâches : Badges, topics, activer Pages, tester déploiement

**Note** : Ces issues doivent être créées manuellement sur GitHub ou via GitHub CLI (commandes fournies).

---

### 6. Git repository

✅ **Repo Git initialisé**
- Commit initial : `f2bd675` (26 fichiers, 1525 insertions)
- Message : "Initial commit: fp-qubit-design scaffold (v0.1.0)"

✅ **Fichiers commités** : 26 fichiers
- .github/workflows/ (2 fichiers)
- configs/ (2 fichiers)
- data/ (4 fichiers)
- figures/ (1 fichier)
- scripts/ (2 fichiers)
- site/ (2 fichiers)
- src/fpqubit/ (6 fichiers)
- Racine (7 fichiers : README, LICENSE, CITATION, requirements, .gitignore, ISSUES, VERIFICATION_REPORT)

---

## 🚀 Prochaines étapes

### Étape 1 : Publier sur GitHub
```bash
cd "C:\Users\tommy\Documents\atlas suite\fp-qubit-design"
git remote add origin https://github.com/Mythmaker28/fp-qubit-design.git
git branch -M main
git push -u origin main
```

### Étape 2 : Activer GitHub Pages
1. Aller sur https://github.com/Mythmaker28/fp-qubit-design/settings/pages
2. Source : "GitHub Actions"
3. Sauvegarder
4. Attendre le déploiement (~2 min)
5. Accéder à https://mythmaker28.github.io/fp-qubit-design/

### Étape 3 : Créer les 5 issues
Option A : Manuellement (copier-coller depuis `ISSUES.md`)
Option B : GitHub CLI (commandes dans `ISSUES.md`)

### Étape 4 : Ajouter topics GitHub
Repo → Settings → About → Topics :
- `quantum-sensing`
- `biophysics`
- `fluorescent-proteins`
- `protein-design`
- `machine-learning`
- `dataset`
- `biological-qubits`

### Étape 5 : Vérifier CI
1. Premier push déclenche CI
2. Vérifier badge vert dans Actions
3. Si échec, corriger selon logs

---

## ✅ Checklist finale

- [x] Arborescence complète (26 fichiers)
- [x] README.md (FR) complet et clair
- [x] README_EN.md (EN) condensé
- [x] CITATION.cff valide (CFF 1.2.0)
- [x] LICENSE Apache-2.0
- [x] Atlas snapshot + métadonnées (commit abd6a4cd)
- [x] Mapping proxies (atlas_mapping.yaml)
- [x] Scripts squelettes avec TODOs (train_baseline.py, generate_mutants.py)
- [x] Site web (index.html + shortlist.csv factice)
- [x] GitHub workflows (ci.yml + pages.yml)
- [x] 5 issues documentées (ISSUES.md)
- [x] Git repo initialisé + commit initial
- [x] Dossier temp_atlas nettoyé
- [x] Rapport de vérification créé (ce fichier)

---

## 📊 Statistiques

- **Fichiers créés** : 26
- **Lignes de code** : 1525
- **Dépendances** : 5 (numpy, pandas, scikit-learn, matplotlib, pyyaml)
- **Systèmes Atlas** : 22 (snapshot)
- **Mutants factices** : 3 (site/shortlist.csv)
- **Issues documentées** : 5
- **Workflows CI/CD** : 2

---

## 🎯 Statut final

**✅ TOUS LES CRITÈRES DE RÉUSSITE SONT REMPLIS**

Le projet `fp-qubit-design` est prêt à être publié sur GitHub.

Le squelette (v0.1.0) est complet, propre, minimal, et reproductible.

Les prochaines étapes (développement des modèles, entraînement, shortlist réelle) sont documentées dans la roadmap et les issues.

---

**Fin du rapport**

