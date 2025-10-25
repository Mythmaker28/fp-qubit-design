# SUGGESTIONS - fp-qubit-design v1.1.4 (BLOCKED)

**Generated**: 2025-10-23  
**Status**: ⚠️ **Pipeline bloqué** - Données canoniques introuvables

---

## 🤔 Avez-vous des suggestions, idées, phénomènes intéressants ou intuitions ?

### 📊 Ce que nous avons découvert

**Problème principal** : Le fichier `atlas_fp_optical.csv` v1.2.1 (66 systèmes FP, 54 mesurés) **n'existe pas** dans le dépôt public Atlas.

**Réalité des données** :
- Atlas v1.2.1 : 26 systèmes total
  - 10 centres de couleur (NV, SiV, ODMR)
  - 10 systèmes NMR (noyaux hyperpolarisés)
  - 4 systèmes ESR/EPR
  - **2 FP optical** (1 protéine fluor + 1 quantum dot)

**Gap** : 64 systèmes FP manquants (-97%)

---

## 💡 Suggestions pour Débloquer v1.1.4

### Suggestion 1 : **Intégrer FPbase** ⭐⭐⭐ (Recommandé)

**FPbase** (https://www.fpbase.org/) est une base publique de ~1000 protéines fluorescentes avec propriétés photophysiques.

**API disponible** :
```bash
# Liste de toutes les FP
curl https://www.fpbase.org/api/proteins/

# Détails d'une FP
curl https://www.fpbase.org/api/proteins/egfp/
```

**Données disponibles** :
- Brightness (QY × epsilon)
- Quantum Yield (QY)
- Lifetime (τ)
- **ΔF/F₀** pour sensors (calcium, voltage, pH)
- Excitation/Emission spectra
- Photostability

**Workflow proposé** :
1. `scripts/consume/fetch_fpbase.py` → télécharge API FPbase
2. Filtre : `is_sensor=True` ou `has_delta_f=True`
3. Normalise → `contrast_normalized = ΔF/F₀`
4. Merge avec Atlas (2 systèmes) → **N≥50** total

**Avantages** :
- ✅ Données peer-reviewed (publications liées)
- ✅ Licence CC BY 4.0 (compatible)
- ✅ Couvre toutes les familles FP (GFP, RFP, calcium, voltage, pH)
- ✅ Permet training robuste (N≥50)

**Inconvénients** :
- ⚠️ Pas toutes les FP ont ΔF/F₀ (seulement sensors)
- ⚠️ Dépendance externe (risque API rate-limit)

---

### Suggestion 2 : **Parser Literature (DOI)** ⭐⭐

Les 2 systèmes FP de l'Atlas ont des DOI :
- FP ODMR : `10.1038/s41586-024-08300-4`
- QD CdSe : `10.1103/PhysRevLett.104.067405`

**Workflow** :
1. Fetch PDF/HTML via DOI
2. Parse avec LLM (GPT-4, Claude) ou regex
3. Extraire : contrast, QY, lifetime, T°, pH
4. Valider manuellement

**Avantages** :
- ✅ Haute qualité (peer-reviewed)
- ✅ Contexte expérimental complet

**Inconvénients** :
- ⚠️ Lent (manual/semi-auto)
- ⚠️ Risque parsing errors
- ⚠️ Paywall pour PDFs

---

### Suggestion 3 : **Contacter Maintainer Atlas** ⭐⭐

**Action** : Ouvrir une issue dans `biological-qubits-atlas` :

> **Titre** : "Request: atlas_fp_optical.csv filtered subset for FP design"
>
> **Message** :
> "Hi @Mythmaker28, I'm working on fp-qubit-design which uses Atlas as a data source.
>
> I'm looking for a filtered subset of FP optical systems (biosensors, fluorescent proteins) with photophysical properties.
>
> Current Atlas v1.2.1 has only 2 FP systems (vs 10 color centers, 10 NMR).
>
> Would you consider:
> 1. Creating an `atlas_fp_optical.csv` subset?
> 2. Expanding Atlas with more FP data (FPbase integration)?
> 3. Collaborating on FP-focused extension?
>
> See gap analysis: [link to DATA_REALITY_v1.1.4.md]
>
> Thanks!"

**Avantages** :
- ✅ Source canonique unique (pas de fragmentation)
- ✅ Provenance Atlas (déjà cité)

**Inconvénients** :
- ⚠️ Dépend du maintainer (délai inconnu)
- ⚠️ Peut refuser (hors scope Atlas)

---

### Suggestion 4 : **Élargir le Scope** ⭐ (Dernière option)

**Option** : Inclure les **colour centers avec readout optical** (ODMR).

**Justification** :
- Les centres NV/SiV ont un **readout optical** (ODMR = Optically Detected Magnetic Resonance)
- Propriétés photophysiques similaires (excitation, cohérence)
- N=12 avec contraste (vs N=2 FP only)

**Nouveau scope** : "Bio-Quantum Sensors" (FP + Color Centers + ODMR)

**Avantages** :
- ✅ Débloque immédiatement (N=12)
- ✅ Reste "optical" (ODMR)

**Inconvénients** :
- ❌ Viole spécification user ("FP optical ONLY, pas de NV/SiV")
- ❌ Color centers ≠ protéines biologiques
- ❌ Scope mismatch avec nom "fp-qubit-design"

---

## 🎯 Recommandation Finale

### Plan Pragmatique (v1.2)

**Phase 1** : Intégrer FPbase (Suggestion 1) ⭐⭐⭐
- Timeline : 1-2 semaines
- Résultat : N≥50 FP optical
- Débloquer v1.1.4 pipeline

**Phase 2** : Literature mining (Suggestion 2) ⭐⭐
- Timeline : 2-3 semaines
- Résultat : +10-20 FP high-quality
- Améliorer training

**Phase 3** : Contact maintainer (Suggestion 3) ⭐⭐
- Timeline : variable (dépend réponse)
- Résultat : Atlas FP-focused release (idéal)

### v1.1.4-pre (Interim Release)

**Livrables** :
- ✅ Discovery log (`WHERE_I_LOOKED.md`)
- ✅ Data reality report (`DATA_REALITY_v1.1.4.md`)
- ✅ Suggestions (ce fichier)
- ✅ Robust fetch script (`resolve_atlas_v1_2_1.py`)
- ❌ Training pipeline (BLOCKED, N=2 insufficient)

**Status** : **PRE-RELEASE** (blocked, waiting for data enrichment)

---

## 📝 Autres Idées

### Idée 1 : **Synthetic Augmentation (Controlled)**

Si N reste faible (<30), générer **FP variants synthétiques** basés sur :
- Mutations single-point (AAindex-guided)
- Contraintes physico-chimiques (BLOSUM, hydrophobicity)
- Distributions matching real data

**Label clairement** : `is_synthetic=True` dans metadata.

**Avantages** :
- Augmente N pour training
- Contrôlé (pas random)

**Inconvénients** :
- ⚠️ Pas "measured-only" (viole spec v1.1.4)
- ⚠️ Risque overfitting

---

### Idée 2 : **Transfer Learning from Color Centers**

**Workflow** :
1. Train model sur color centers (N=10, optical ODMR)
2. Apprendre relation structure → contrast optical
3. Fine-tune sur FP (N=2)
4. Domain adaptation (CycleGAN, DANN)

**Hypothèse** : Propriétés optiques (excitation, cohérence, contraste) transférables entre color centers et FP.

**Avantages** :
- Utilise données existantes (N=12 optical)
- Proof-of-concept for transfer learning

**Inconvénients** :
- ⚠️ Domain shift (semiconductor vs protein)
- ⚠️ Incertain (besoin validation)

---

### Idée 3 : **Active Learning Loop**

**Si** données FPbase intégrées (N≥50) :

1. Train initial model (N=50)
2. Predict sur space FP (mutations, variants)
3. **Sélectionner top-K uncertain** (UQ-guided)
4. → Recherche literature/expérimental pour ces K
5. → Ajouter au training set
6. → Retrain
7. Repeat jusqu'à convergence

**Avantages** :
- Optimise data collection (focus high-value)
- Améliore model itérativement

**Inconvénients** :
- ⚠️ Nécessite literature access ou expérimental
- ⚠️ Lent (itératif)

---

## ❓ Questions Ouvertes

1. **Pourquoi atlas_fp_optical.csv (66 entrées) était attendu ?**
   - Source de cette spécification ?
   - Confusion avec un autre projet ?
   - Dataset interne non publié ?

2. **Priorité user : Training rapide vs Data quality ?**
   - Si rapide → élargir scope (color centers)
   - Si quality → attendre FPbase integration

3. **Budget pour external data ?**
   - FPbase API : gratuit (CC BY 4.0)
   - Literature mining : temps humain (manual validation)
   - Expérimental : hors scope (zero wet-lab)

---

## 🏁 Conclusion

**v1.1.4 bloquée** par manque de données FP (N=2 vs N=54 attendu).

**Solution recommandée** : **v1.2 avec intégration FPbase** (N≥50).

**Timeline** : 2-4 semaines.

**Alternative rapide** : Élargir à optical ODMR (N=12) mais viole spec.

---

**Avez-vous d'autres suggestions, intuitions, ou phénomènes intéressants à partager ?** 💭

---

**License** : Code: Apache-2.0 | Data: CC BY 4.0


