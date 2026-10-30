# Plan et conception du visualiseur

Demande approuvée : script Python VTK dédié, coupes et vue 3D, données externes au dépôt.

1. Charger les NIfTI avec NiBabel et vérifier la concordance image/masque.
2. Afficher trois coupes orthogonales et le masque, plus une surface tumorale 3D avec les plans.
3. Ajouter navigation au clavier, sélection du patient, capture hors écran pour vérification.
4. Vérifier géométrie, rejet des masques décalés, rendu réel et documentation.

Choix : repère RAS canonique, rééchantillonnage uniquement si affine oblique pour respecter la géométrie.
