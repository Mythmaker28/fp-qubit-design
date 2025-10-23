# figures/

Ce dossier contient les figures et graphiques générés pour le projet FP-Qubit Design.

## Types de figures prévues

- **Performances des modèles** : courbes ROC, matrices de confusion, importance des features
- **Analyses exploratoires** : distributions des proxies, corrélations, PCA
- **Mutants** : heatmaps de prédictions, diagrammes de Pareto (gain vs. incertitude)
- **Structures** : visualisations 3D des mutants (si structures disponibles)

## Format recommandé

- Format vectoriel (PDF, SVG) pour publications
- Format raster haute résolution (PNG 300 DPI) pour présentations
- Nommer les fichiers avec des noms descriptifs : `feature_importance_RF.pdf`, `mutants_shortlist_heatmap.png`

## Instructions

1. Toujours inclure les scripts de génération dans `scripts/` ou notebooks Jupyter
2. Documenter les figures dans le README principal ou dans un notebook
3. Ne pas commiter les fichiers PNG/PDF si > 1 MB (utiliser `.gitignore`)

## Statut actuel

🚧 Dossier vide — figures à générer lors du développement futur

