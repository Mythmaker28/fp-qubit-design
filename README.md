FP-Qubit Design
But

Ce dépôt fournit un cadre logiciel pour la conception in silico de mutants de protéines fluorescentes (FP) optimisés pour des proxys photophysiques liés aux qubits biologiques. L’objectif : proposer des candidats mutants qui maximisent la cohérence quantique (p. ex. temps de vie T2), le contraste optique, et d’autres métriques pertinentes pour des applications de bio-sensing quantique.

État du projet

Version actuelle (données) : Atlas v2.2.2 (balanced)
221 systèmes utiles, 30 familles, Calcium ≈ 22.6 %.
Kit labo prêt : shortlist, layouts 96/24 puits, protocole squelette, SHA256.

Modélisation (v2.2.2) : baseline ML testée (RF/ExtraTrees/Huber, variantes).
Résultat : NO-GO sur critères stricts (R² ≥ 0.20, MAE < 7.81).
→ Utiliser la shortlist top-20 / top-12 pour la validation expérimentale.

Pages : https://mythmaker28.github.io/fp-qubit-design/

(compteurs dynamiques chargés depuis deliverables/lab_v2_2_2/status.json)

Données & provenance

Source primaire : Biological Qubits Atlas

Version intégrée : v2.2.2 (balanced), couverture optique 100 %, provenance/licences complètes.

Fichiers livrables (répertoire du repo) :

deliverables/lab_v2_2_2/shortlist_lab_sheet.csv

deliverables/lab_v2_2_2/shortlist_top12_final.csv

deliverables/lab_v2_2_2/plate_layout_96.csv

deliverables/lab_v2_2_2/plate_layout_24.csv

deliverables/lab_v2_2_2/protocol_skeleton.md

deliverables/lab_v2_2_2/filters_recommendations.md

deliverables/lab_v2_2_2/SHA256SUMS.txt

deliverables/lab_v2_2_2/status.json (atlas_version, n_useful, …)

Archive : deliverables/lab_v2_2_2.zip

Licences :
Code Apache-2.0. Données/Docs CC BY 4.0 (se référer aux en-têtes des fichiers et à l’Atlas).

Installation rapide
# Cloner
git clone https://github.com/Mythmaker28/fp-qubit-design.git
cd fp-qubit-design

# (Optionnel) créer un venv Python ≥ 3.11
python -m venv .venv && . .venv/Scripts/activate      # Windows PowerShell
# source .venv/bin/activate                           # Linux/Mac

# Dépendances minimales (si vous utilisez les scripts d’évaluation)
pip install -r requirements.txt  # sinon, ignorer


Remarque : pour l’usage “labo”, le kit est autonome (CSV/MD/ZIP). Les scripts ne sont pas nécessaires pour consulter/utiliser les livrables.

Utilisation (mode labo)

Télécharger deliverables/lab_v2_2_2.zip ou utiliser les fichiers dans deliverables/lab_v2_2_2/.

Ouvrir :

shortlist_top12_final.csv : sélection finale pour tests.

plate_layout_96.csv / plate_layout_24.csv : planches prêtes.

protocol_skeleton.md : paramètres spectraux + pas de mesure.

filters_recommendations.md : fenêtres exc/émission recommandées.

Vérifier l’intégrité avec SHA256SUMS.txt (optionnel).

Utilisation (mode évaluation – optionnel)

Des scripts minimaux peuvent être fournis pour refaire une évaluation sur TRAINING_TABLE_v2_2_2_balanced.csv (provenant de l’Atlas).
Les variantes testées (RF/ExtraTrees/Huber, CQR simple) n’ont pas franchi les seuils R²/MAE mais calibrent correctement l’UQ.

Si vous ne trouvez pas les scripts, utilisez uniquement le mode labo ci-dessus : c’est l’usage recommandé à ce stade.

Arborescence (extrait)
fp-qubit-design/
├─ README.md
├─ LICENSE
├─ requirements.txt
├─ deliverables/
│  └─ lab_v2_2_2/
│     ├─ shortlist_lab_sheet.csv
│     ├─ shortlist_top12_final.csv
│     ├─ plate_layout_96.csv
│     ├─ plate_layout_24.csv
│     ├─ protocol_skeleton.md
│     ├─ filters_recommendations.md
│     ├─ SHA256SUMS.txt
│     └─ status.json
├─ index.html            # Page d’accueil (compteurs dynamiques)
└─ .nojekyll             # Pages GitHub sans Jekyll

Roadmap (v2.3)

Features : quantum_yield, extinction_coefficient, brightness = QE×EC, photostabilité.

Données : enrichir Voltage / neurotransmetteurs (diversifier vs Calcium).

Modèles : routeurs par famille/spectral, stacking simple, CQR calibré par fold.

CI : tests schéma + métriques de calibration (ECE/coverage) sur échelle originale.

Citation

Si vous utilisez ce dépôt :

Lepesteur, T. (2025). FP-Qubit Design (v2.2.2). GitHub. https://github.com/Mythmaker28/fp-qubit-design

Contribution & contact

Contributions bienvenues via Issues/PR (UTF-8 only 🙃).

Auteur : Tommy Lepesteur — ORCID 0009-0009-0577-9563

Issues : https://github.com/Mythmaker28/fp-qubit-design/issues

Statut : kit labo v2.2.2 livré (shortlist & protocoles). Modélisation NO-GO sur critères stricts → prochaine étape : v2.3 (features & données) + validation expérimentale de la shortlist.


