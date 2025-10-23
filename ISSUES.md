# Issues initiales pour fp-qubit-design

Ce fichier documente les 5 issues prioritaires à créer sur GitHub une fois le repo publié.

## Issue #1: Connecter Atlas → Définir mapping de proxies

**Titre**: `[Data] Connect Atlas → Define proxy mapping`

**Description**:
Définir le mapping complet entre les colonnes de l'Atlas des Qubits Biologiques et les proxies pertinents pour les protéines fluorescentes.

**Tâches**:
- [ ] Vérifier la présence des colonnes clés dans `atlas_snapshot.csv` (T1, T2, Contraste, Température, Méthode, Contexte)
- [ ] Parser le champ `Photophysique` pour extraire : lifetime, quantum yield (QY), longueurs d'onde d'excitation/émission
- [ ] Identifier les colonnes manquantes (ISC rate, photostabilité quantitative)
- [ ] Compléter `configs/atlas_mapping.yaml` avec les colonnes validées
- [ ] Créer une fonction `load_atlas_proxies()` dans `src/fpqubit/utils/io.py`
- [ ] Tester le chargement sur 5-10 systèmes Atlas classe A/B

**Labels**: `data`, `priority-high`, `good-first-issue`

---

## Issue #2: Implémenter baselines ML (RF/XGB)

**Titre**: `[ML] Implement baseline models (Random Forest, XGBoost)`

**Description**:
Implémenter les modèles baseline Random Forest et XGBoost pour prédire les proxies FP (lifetime, contrast, temperature stability).

**Tâches**:
- [ ] Définir les variables d'entrée (features) : composition AA, propriétés physicochimiques, position des mutations
- [ ] Créer un dataset synthétique ou semi-synthétique (si données réelles insuffisantes)
- [ ] Implémenter splitter train/validation/test (stratifié)
- [ ] Compléter `scripts/train_baseline.py` avec entraînement RF/XGB
- [ ] Cross-validation (5-fold) + métriques (MAE, R², RMSE)
- [ ] Sauvegarder modèles entraînés (pickle ou joblib)
- [ ] Générer plots de performance (feature importance, prédictions vs. ground truth)

**Labels**: `ml`, `priority-high`

---

## Issue #3: Pipeline de sélection shortlist (ΔΔG + incertitudes)

**Titre**: `[Pipeline] Define mutant shortlist selection pipeline`

**Description**:
Créer un pipeline automatisé pour générer et sélectionner les meilleurs mutants candidats avec incertitudes quantifiées.

**Tâches**:
- [ ] Générer mutations aléatoires ou guidées (règles heuristiques : positions proches chromophore)
- [ ] Calculer ΔΔG placeholder (modèle simple ou valeurs aléatoires contrôlées)
- [ ] Prédire proxies (lifetime, contrast) avec modèles baseline + incertitudes (bootstrap ou GP)
- [ ] Définir fonction de score multi-objectif (ex: weighted sum ou Pareto front)
- [ ] Shortlist top 10-20 mutants
- [ ] Écrire `site/shortlist.csv` avec colonnes : mutant_id, base_protein, mutations, proxy_target, predicted_gain, uncertainty, rationale
- [ ] Valider que `site/index.html` charge correctement la shortlist

**Labels**: `pipeline`, `priority-medium`

---

## Issue #4: Documentation IMRaD + Plan Zenodo

**Titre**: `[Docs] Create IMRaD template + Zenodo publication plan`

**Description**:
Préparer la documentation scientifique (format IMRaD) et le plan de publication Zenodo.

**Tâches**:
- [ ] Créer template IMRaD (Introduction, Methods, Results, Discussion) dans `docs/paper_template.md`
- [ ] Rédiger section **Introduction** : contexte qubits biologiques, objectifs, scope
- [ ] Rédiger section **Methods** : featurisation, baselines ML, sélection mutants
- [ ] Préparer section **Results** : placeholder pour performances modèles, shortlist mutants
- [ ] Définir plan de publication Zenodo : titre, auteurs, abstract, keywords, related identifiers (Atlas DOI)
- [ ] Ajouter checklist pré-publication : vérification CITATION.cff, LICENSE, README, workflows CI/Pages

**Labels**: `docs`, `priority-low`, `publication`

---

## Issue #5: Infra GitHub (badges, topics, Pages)

**Titre**: `[Infra] Setup GitHub badges, topics, and Pages`

**Description**:
Configurer l'infrastructure GitHub pour améliorer la visibilité et l'accès au projet.

**Tâches**:
- [ ] Ajouter badges au README : CI status, Pages deployment status, License, DOI (Zenodo, placeholder)
- [ ] Configurer topics GitHub : `quantum-sensing`, `biophysics`, `fluorescent-proteins`, `protein-design`, `machine-learning`, `dataset`, `biological-qubits`
- [ ] Activer GitHub Pages depuis Settings → Pages → Source = "GitHub Actions"
- [ ] Vérifier déploiement Pages : accès à `https://mythmaker28.github.io/fp-qubit-design/`
- [ ] Tester chargement de `shortlist.csv` sur Pages (cache-bust fonctionnel)
- [ ] Ajouter lien Pages dans README

**Labels**: `infra`, `priority-medium`, `github-pages`

---

## Instructions de création

Une fois le repo publié sur GitHub :

1. Aller dans l'onglet **Issues**
2. Cliquer sur **New Issue** pour chaque issue ci-dessus
3. Copier-coller le titre et la description
4. Ajouter les labels suggérés
5. Assigner à soi-même si nécessaire

Les issues peuvent être créées automatiquement via GitHub CLI :

```bash
gh issue create --title "[Data] Connect Atlas → Define proxy mapping" --body "$(cat ISSUES.md | sed -n '/Issue #1/,/^---$/p')" --label "data,priority-high,good-first-issue"
# Répéter pour les issues #2 à #5
```

