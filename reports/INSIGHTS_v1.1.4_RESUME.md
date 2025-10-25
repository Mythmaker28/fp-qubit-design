# Suggestions & Insights - v1.1.4 Reprise

**Date**: 2025-10-24  
**Context**: Validation failed - N=2/66 FP systems found

---

## 🔍 Découvertes & Phénomènes Intéressants

### 1. **Gap Structurel Atlas-FP** 🎯

L'Atlas `biological-qubits-atlas` est **majoritairement composé** de systèmes quantiques **non-FP** :

| Type de Système | Count (v1.2.1) | % |
|-----------------|----------------|---|
| **Centres de couleur** (NV/SiV/GeV diamant/SiC) | ~15 | 58% |
| **NMR** (^13C hyperpolarisé, métabolites) | ~10 | 38% |
| **Quantum Dots** non-bio | ~2 | 8% |
| **FP optiques** (cible de ce projet) | **~2** | **8%** |

**Insight** : L'Atlas est **broad-spectrum quantum biosystems**, pas "FP-focused". Le projet `fp-qubit-design` (scope: FP optical uniquement) et l'Atlas ont des **scopes orthogonaux**.

**Lesson Learned** : Pour un projet ML sur FP, **ne pas partir de l'Atlas** comme source primaire. Partir de **FPbase** (source canonique FP) et utiliser l'Atlas comme source **complémentaire** pour enrichir avec proxies quantum (T1/T2, coherence).

---

### 2. **Contraste Photophysique : Large Variabilité** 📊

Les 2 FP trouvés montrent des contrastes **très différents** :

- **FP+ODMR** (295K, HeLa) : **12%** contrast
- **QD CdSe** (77K, cryogénique) : **3%** contrast

**Ratio** : 4x différence !

**Factors** :
- **Température** : 295K vs 77K → impact massif sur phonon coupling, dephasing
- **Environnement** : in_cellulo (crowded) vs in_vitro (clean)
- **Architecture moléculaire** : β-barrel (FP) vs inorganic shell (QD)

**Implication ML** : Le contraste est **fortement context-dependent**. Un modèle robuste nécessite :
- Diversité de familles (≥7)
- Diversité de contextes (T, pH, hôte)
- Features contextuelles explicites

→ **N=2 est insuffisant** pour capturer cette variance.

---

### 3. **FPbase = Source Naturelle pour FP ML** 🔬

**FPbase** (https://www.fpbase.org) est la **base de données communautaire de référence** pour les protéines fluorescentes :

**Statistiques** :
- **>200 FP** documentées (GFP, RFP, biosensors, photoconvertible, etc.)
- **Propriétés mesurées** : brightness, QY, lifetime, photostability, maturation, pH stability
- **Séquences** : alignements, mutations, familles structurales
- **Spectres** : excitation/emission (raw data)
- **Contextes** : host, tags, fusion constructs
- **Curation** : community-validated, peer-reviewed

**API/Export** : JSON, CSV, API REST

**Mapping FPbase → `fp-qubit-design`** :

| Propriété FPbase | Proxy Quantum | Relation |
|------------------|---------------|----------|
| **Quantum Yield (QY)** | ISC rate, triplet | ↑ QY → ↓ triplet → ↑ coherence |
| **Lifetime** | T2*, dephasing | ↑ lifetime → state stability |
| **Photostability** | T1, decay | ↑ stability → ↑ readout window |
| **Brightness** | SNR, contrast | ↑ brightness → ↑ readout fidelity |
| **Maturation** | Folding kinetics | Fast maturation → ↑ yield in vivo |

**Action recommandée** : Implémenter `scripts/etl/fetch_fpbase.py` pour :
1. Télécharger export FPbase (JSON/CSV)
2. Filtrer FP avec propriétés complètes (N≥50)
3. Mapper propriétés → proxies quantum
4. Merger avec Atlas (2 systèmes) pour **diversité cross-platform**

**Timeline estimée** : 1-2 semaines de développement

---

### 4. **"Measured-Only" Philosophy : Trade-off Data/Quality** ⚖️

Le projet v1.1.4 adopte une philosophie **"measured-only"** (pas de synthétiques, pas de proxies computés).

**Avantages** ✅ :
- Traçabilité scientifique
- Reproductibilité
- Crédibilité pour publication

**Challenges** ⚠️ :
- **Sparse data** : Les mesures photophysiques complètes sont **rares** dans la littérature
- **Publication bias** : Seuls les FP "performants" sont publiés avec mesures détaillées
- **Hétérogénéité** : Protocoles/conditions variables entre labs

**Alternative pragmatique** : Mode **"hybrid"** (v1.2 ?) :
- **Tier A** : Measured (high confidence) → 50%
- **Tier B** : Computed from related FP (medium confidence) → 30%
- **Tier C** : Physics-based proxies (low confidence, flagged) → 20%

→ Permet d'atteindre N≥100 tout en **taggant explicitement** la provenance/confiance.

---

### 5. **Readout Multimodal : Opportunity** 🌟

Les 2 systèmes trouvés illustrent un **phénomène intéressant** :

**FP+ODMR** = **double readout** :
- Optical (fluorescence) → ΔF/F0
- Magnetic (ODMR) → spin state

**Advantages** :
- Orthogonal information channels
- Cross-validation readout
- Richer feature space for ML

**Implication design** : Favoriser les FP avec **multi-modal readout** (optical + ODMR/NMR) pour des proxies quantum **directs** (pas seulement photophysiques).

**Candidats** :
- FP + paramagnetic tags (spin labels)
- FP + hyperpolarizable nuclei (^13C, ^15N)
- FP in proximity to NV centers (hybrid systems)

→ **Future direction** pour v1.2+

---

### 6. **Temperature as Critical Feature** 🌡️

**Observation** : Contraste 12% @ 295K vs 3% @ 77K (QD)

**Physical basis** :
- ↓ T → ↓ phonon coupling → ↑ coherence (T2)
- ↓ T → ↓ vibrational modes → narrower linewidths
- ↓ T → changes in ISC rates (triplet formation)

**ML implication** : **Temperature MUST be an explicit feature** in any FP quantum model. Sans ça, le modèle mélange des régimes physiques incomparables.

**Feature engineering recommendation** :
```python
features = [
    'temperature_K',  # explicit
    'T_normalized',   # T / T_room (295K)
    'thermal_regime', # categorical: cryogenic / room / physiological
    'kT_eV'           # thermal energy (physical scaling)
]
```

---

### 7. **In Cellulo Context : The "Real World"** 🧬

Le FP+ODMR est mesuré **in cellulo** (HeLa cells) → **contexte biologique réel** ✅

**Challenges in cellulo** vs in vitro :
- **Crowding** : high protein concentration (300-400 g/L) → viscosity, interactions
- **Ionic strength** : variable [salt], pH buffering
- **Oxidative stress** : ROS, oxidation states
- **Autofluorescence** : background from other biomolecules
- **Photodamage** : phototoxicity limits illumination power

**Impact on quantum properties** :
- ↑ dephasing (crowding)
- ↓ contrast (background)
- ↓ photostability (ROS)

**ML recommendation** : Si l'objectif est **in vivo sensing**, prioriser les mesures **in cellulo/in vivo** (même si N plus petit) sur les mesures **in vitro** (N large mais non-représentatif).

**Trade-off** : Quality (biological relevance) vs Quantity (N)

---

## 🚀 Recommandations Actionnables

### Court Terme (1-2 semaines)

1. **Intégrer FPbase** ⭐ (priorité #1)
   - Script `fetch_fpbase.py`
   - Target : N≥50 FP avec brightness/QY/lifetime
   - Merger avec Atlas (N=2) pour diversité

2. **Créer Issue sur Atlas**
   - Demander publication de `atlas_fp_optical.csv` v1.2.1 (66 FP)
   - Lien vers `reports/ATLAS_MISMATCH.md`

### Moyen Terme (3-4 semaines)

3. **Literature Mining**
   - PubMed query : "(fluorescent protein) AND (quantum yield OR lifetime OR photostability)"
   - Extraction tables supplémentaires
   - Target : +10-20 FP

4. **Hybrid Mode** (v1.2)
   - Tier A/B/C avec tagging explicite
   - Permet N≥100 avec traçabilité

### Long Terme (2-3 mois)

5. **Multi-Modal Readout**
   - Focus FP + ODMR/NMR
   - Collaboration expérimentale ?

6. **In Vivo Priority**
   - Prioriser mesures biologiques sur mesures in vitro

---

## 📊 Metrics Recap

| Metric | v1.1.4 Actual | v1.1.4 Target | Gap |
|--------|---------------|---------------|-----|
| **N_total** | 2 | 66 | -64 (-97%) |
| **N_measured_AB** | 0 | 54 | -54 |
| **Families** | 0 (≥3) | 7 | -7 |
| **Pipeline** | BLOCKED | RUNNING | - |

**Blocker** : Données insuffisantes (N=2 << 40 minimum)

**Solution recommandée** : FPbase integration (≥50 FP) + Atlas (2) → N≥52 ✅

---

## 🎯 Final Thought

**L'échec de v1.1.4** n'est **pas un échec technique** mais un **scope mismatch structurel** :

- Atlas = broad quantum bio (NV/NMR/ESR/FP)
- fp-qubit-design = narrow FP optical

**La bonne stratégie** : **FPbase first, Atlas second** (complementary).

**Le vrai insight** : Pour des projets ML domaine-spécifiques, **partir de la source canonique du domaine** (ici FPbase), pas d'une source adjacente (Atlas).

---

**Next Step** : Choix utilisateur requis (Option 1/2/3/4 dans `V114_RESUME_VERDICT.md`)

**License**: Code Apache-2.0, Data CC BY 4.0  
**Author**: Tommy Lepesteur (ORCID: 0009-0009-0577-9563)


