# Plan : évaluation des segmentations binaires

Conception approuvée : postprocessing/evaluate.py, métriques par patient et résumé JSON dans le répertoire de validation choisi. Surface Dice à 2 mm et HD95 physiques. Ne pas modifier les masques ni summary.json de nnU-Net.

- Tests synthétiques : masques identiques, décalage anisotrope, prédiction vide, faux positif isolé, composantes fusionnées, géométrie invalide.
- Implémentation : lecture NIfTI stricte, surfaces pondérées surface-distance 0.1, composantes 26-connexes et appariement maximal un-à-un avec recouvrement non nul.
- Résumé macro par patient, moyenne/médiane/quartiles/écart-type et nombre de valeurs définies. Les HD95 indéfinis deviennent null avec statut explicite, jamais zéro.
- Validation réelle des 13 patients du fold 0, checkpoint final identifié comme tel. Comparaison du Dice avec summary.json existant.
