# v1.1.4 Reprise - Verdict Final

**Date**: 2025-10-24  
**Status**: ❌ **VALIDATION FAILED - BLOCKED**

---

## Impressions Obligatoires

```
V114_STATUS=RESUMED_AND_BLOCKED
SOURCE=local_fallback
SHA256=0c79b6c5fa523fb8f4da0ae512f1bc32b270e4677602b53e85cd24d74330738c
N_total=2 (expected 66, delta: -64)
N_measured_AB=0 (expected 54, delta: -54)
families>=3=0 (expected 7, delta: -7)
GAP=-97.0% of expected data
ECE=NA (cannot train with N=2)
R2_OOF=NA (cannot train with N=2)
MAE_OOF=NA (cannot train with N=2)
SHORTLIST_COUNT=NA (cannot generate with N=2)
PAGES=NA (blocked on data)
```

---

## Ce qui a été accompli

### ✅ Phase A : Ingestion
- Chemin A (GitHub Release v1.2.1) : **FAIL** (asset not found)
- Chemin B (Fallback Local) : **SUCCESS**
  - File: `data/processed/atlas_fp_optical.csv`
  - SHA256: `0c79b6c5fa523fb8f4da0ae512f1bc32b270e4677602b53e85cd24d74330738c`
  - Size: 689 bytes

### ❌ Phase B : Validation
- **FAILED** : Counts do not match v1.2.1 specification
- Expected: N=66 total, 54 measured A/B, ≥7 families
- Actual: N=2 total, 0 measured A/B, 0 families (≥3)
- **Gap**: -64 FP systems (-97%)

### ⏸️ Phases C-J : BLOCKED
- Cannot proceed with ML pipeline (minimum N≥40 required)
- Insufficient data for:
  - Nested-CV family-stratified (need ≥7 families)
  - UQ calibration (need ≥20 samples)
  - SHAP/explainability (need diverse feature space)
  - Shortlist generation (need robust predictions)

---

## Root Cause

Le fichier `atlas_fp_optical.csv` avec **66 FP** et **54 measured A/B** **n'existe pas** dans les sources accessibles :

1. ❌ GitHub Release v1.2.1 : asset absent
2. ❌ GitHub Tags : pas d'asset
3. ❌ GitHub Branches : fichier introuvable (25 locations searched)
4. ❌ Local fallback : contient seulement 2 systèmes

**Conclusion** : Le dataset v1.2.1 spécifié (66 FP optical) **n'a pas encore été publié** par le mainteneur de `biological-qubits-atlas`.

---

## Artifacts Générés

### Scripts (3)
1. `scripts/consume/fetch_atlas_fp_optical_v1_2_1_canonical.py` - Chemin A
2. `scripts/consume/fetch_atlas_fp_optical_fallback.py` - Chemin B
3. `scripts/consume/validate_atlas_counts.py` - Validation

### Rapports (2)
4. `reports/ATLAS_MISMATCH.md` - Diff détaillé
5. `reports/V114_RESUME_VERDICT.md` - Ce rapport
6. `data/external/atlas/PROVENANCE.md` - Provenance

### Metadata (1)
7. `data/processed/TRAINING.METADATA.json` - Source info + SHA256

---

## Options pour Débloquer v1.1.4

### Option 1: Attendre Publication Atlas ⏳
- **Action** : Contacter maintainer de `biological-qubits-atlas`
- **Issue** : Request publication of `atlas_fp_optical.csv` v1.2.1
- **Timeline** : Incertaine

### Option 2: Intégrer FPbase 🔬
- **Source** : https://www.fpbase.org (API publique)
- **Expected** : ≥50 FP avec propriétés photophysiques mesurées
- **Timeline** : 2-4 semaines de développement
- **Advantages** :
  - Source communautaire de référence
  - Données curées et validées
  - Propriétés photophysiques complètes

### Option 3: Literature Mining 📚
- **Sources** : PubMed, bioRxiv, tables supplémentaires
- **Expected** : +10-20 FP
- **Timeline** : 2-3 semaines
- **Challenges** : Extraction manuelle, hétérogénéité

### Option 4: Mode Démo (N=2) ⚠️
- **Proceed** : Continuer avec les 2 systèmes disponibles
- **Limitations** :
  - Pas de CV robuste
  - Pas de UQ calibration
  - Pas de généralisation
  - **Documentation claire** des limites
- **Use case** : Démonstration du pipeline uniquement

---

## Recommandation

**🥇 Priorité 1** : **Intégrer FPbase** (Option 2)

**Raisons** :
- Source fiable et communautaire
- Données structurées et accessibles via API
- Couverture large (>200 FP documentées)
- Propriétés photophysiques mesurées
- Permet d'atteindre N≥50 (target: 40)

**Action immédiate** :
1. Explorer FPbase API/export
2. Mapper propriétés FPbase → schéma `fp-qubit-design`
3. Implémenter `scripts/etl/fetch_fpbase.py`
4. Merger avec Atlas (2 systèmes) pour diversité

---

## Suggestions/Insights

### 🔍 Découvertes Intéressantes

1. **Gap Atlas-FP** : L'Atlas `biological-qubits-atlas` est **majoritairement** composé de :
   - Centres de couleur (NV, SiV dans diamant/SiC) : ~15 systèmes
   - Systèmes NMR (^13C hyperpolarisé) : ~10 systèmes
   - Quantum Dots non-FP : quelques systèmes
   - **FP optiques** : seulement ~2 systèmes
   
   → L'Atlas n'est **pas focalisé sur les FP** mais sur les systèmes quantiques bio-intrinsèques **tous types confondus**.

2. **Scope Mismatch** : Le projet `fp-qubit-design` (FP optical uniquement) et l'Atlas (broad quantum bio-systems) ont des scopes **différents**. C'était prévisible dès v1.1.2/v1.1.3.

3. **FPbase = Source Naturelle** : Pour un projet centré sur les FP, FPbase est la **source canonique évidente**. L'Atlas devrait être une source **complémentaire** (contextes biologiques, readouts ODMR/ESR) mais pas la source principale.

4. **Lesson Learned** : Pour des projets ML sur FP, partir de **FPbase + littérature** (N≥50) puis enrichir avec Atlas pour les **proxies quantum** (T1/T2, coherence).

### 💡 Phénomènes Intéressants

- **Contraste photophysique** : Les 2 systèmes trouvés ont des contrastes très différents (12% vs 3%), montrant la **large variabilité** des propriétés quantum des FP.
- **Température** : Lecture à 295K (room temp) vs 77K (cryogénique) → impact massif sur coherence.

---

## Next Actions

**Choix requis** : Quelle option choisir pour débloquer v1.1.4 ?

1. **Attendre** publication Atlas
2. **Intégrer FPbase** (recommandé)
3. **Mine literature**
4. **Démo mode** (N=2)

---

**License**: Code Apache-2.0, Data CC BY 4.0  
**Author**: Tommy Lepesteur (ORCID: 0009-0009-0577-9563)

