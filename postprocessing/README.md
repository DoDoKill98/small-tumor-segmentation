# Évaluation des segmentations

Ce dossier contient du code d'évaluation ; il n'applique aucun nettoyage aux prédictions.

## Lancer

```powershell
cd 'C:\Users\pchx\small tumor'
python -m pip install -r postprocessing\requirements.txt
python postprocessing\evaluate.py --checkpoint-label checkpoint_final.safetensors
```

Par défaut : les 13 patients de validation du fold 0, dans le run baseline_100epochs_fold0_20261029. Le script écrit **evaluation_metrics.json dans le dossier validation évalué**. Il ne modifie ni les masques, ni le summary.json natif. Pour recalculer ce seul rapport : `--overwrite`.

Le dossier validation automatique existant est issu du checkpoint **final**. Les 32 patients test ont été prédits avec **best** et ne disposent pas d'annotations publiques. Ne pas confondre les deux. `--checkpoint-label` est une déclaration de provenance, pas une preuve automatique du checkpoint utilisé. Pour évaluer best, générer d'abord ses prédictions sur les 13 patients de validation dans un dossier distinct puis utiliser :

```powershell
python postprocessing\evaluate.py --prediction-dir 'CHEMIN_VALIDATION_BEST' --checkpoint-label checkpoint_best.safetensors
```

Les 13 identifiants doivent correspondre exactement à la partition choisie. La racine des labels se change avec `--reference-dir`, les partitions avec `--splits` et le fold avec `--fold`.

## Définitions et unités

- Dice = 2 TP / (2 TP + FP + FN).
- Rappel voxel = TP/(TP+FN), précision voxel = TP/(TP+FP).
- Surface Dice : somme des aires des éléments de surface à distance <= 2 mm de l'autre surface, divisée par l'aire totale des deux surfaces. Implémentation surface-distance 0.1 (Google DeepMind), pondérée par les aires physiques. Tolérance configurable par `--surface-tolerance-mm`.
- HD95 : maximum des deux percentiles 95 dirigés, pondérés par les aires, en mm. Cette définition est explicitement choisie ; d'autres bibliothèques utilisent un percentile des distances concaténées et peuvent donner des résultats différents.
- Volumes : nombre de voxels multiplié par le volume physique du voxel, divisé par 1000 pour obtenir des mL. Erreur signée = prédiction - annotation ; erreur absolue = valeur absolue.
- Régions : composantes 26-connexes. Détection = appariement maximal un-à-un entre composantes ayant au moins un voxel commun. C'est un critère permissif, pas une garantie de bonne segmentation. Une fusion ne compte qu'une détection ; `reference_regions_with_any_overlap` et `regions_completely_missed` distinguent ce cas d'une région entièrement oubliée. Les composantes prédites non appariées sont comptées par rapport aux annotations disponibles, sans affirmation clinique.

Le script vérifie dimensions, affines, unités mm et labels binaires ; il refuse les géométries différentes ou cisaillées. Les orientations obliques orthogonales sont acceptées : la rotation conserve les distances. Pas de rééchantillonnage automatique ni de redimensionnement pour faire correspondre les masques.

## Cas vides et statistiques

Deux masques vides : Dice et Surface Dice = 1, HD95 = 0 (convention explicite). Un seul vide : Dice et Surface Dice = 0, HD95 = null avec statut explicite. Précision/rappel sont null si leur dénominateur est nul.

Le résumé donne moyenne macro par patient, médiane, quartiles, écart-type échantillonnal et nombres de valeurs définies/non définies. Les totaux de régions sont fournis séparément. Bootstrap percentile de la moyenne : 2000 rééchantillonnages de patients, graine 12345 ; réglages `--bootstrap-samples` et `--seed`. Pour les métriques non définies, l'intervalle est conditionnel aux valeurs définies : consulter les dénominateurs et les échecs, pas seulement la moyenne.

Avec seulement 13 patients, les intervalles restent incertains. Ces patients servent aussi à la sélection du modèle : il s'agit de résultats de validation, pas d'un test indépendant. Ne pas utiliser les coupes ou les voxels comme observations statistiques indépendantes.

## Tests

```powershell
python -m unittest discover -s postprocessing -p test_evaluate.py -v
```

Référence : https://github.com/google-deepmind/surface-distance

## Regroupement uniforme des fragments annotés

```powershell
python postprocessing\group_annotation_fragments.py --max-distance-mm 2
```

Remplace l'ancien audit des composantes brutes. Un même groupe rassemble les fragments dont les distances physiques sont sous le seuil, avec regroupement transitif. La distance est celle entre centres des voxels de frontière, dans le repère physique du NIfTI. Le rapport conserve les distances et les volumes de tous les fragments : rien n'est effacé ni ajouté aux masques.

Le seuil est obligatoire et expérimental : 2 mm est un premier essai, pas une définition clinique ni la conséquence de la tolérance du Surface Dice. Plusieurs groupes appellent une inspection visuelle ; ils ne prouvent pas plusieurs tumeurs. Un groupe unique ne prouve pas non plus une seule lésion anatomique. Le script ne force jamais le résultat à une seule masse.

Rapport : `postprocessing/reports/grouped_annotations.json`. Options : `--labels-dir`, `--output`, `--max-distance-mm`. Une exécution remplace seulement son propre rapport.

## Expérience : garder la plus grande région prédite

```powershell
python postprocessing\keep_largest_region.py
```

Le défaut traite uniquement les 13 prédictions du répertoire de validation du checkpoint final. Les résultats vont dans `validation/largest_component`. Le script ne consulte aucune annotation : sélection du plus grand volume 26-connexe, égalités tranchées par l'ordre déterministe des labels SciPy. Les prédictions vides restent vides. Il refuse tout dossier de sortie déjà existant ; options `--input-dir` et `--output-dir`. Il conserve les géométries NIfTI et crée `postprocessing_manifest.json` avec les volumes conservés/supprimés et les empreintes des fichiers sources.

Évaluation et comparaison (remplacer le chemin par le dossier de validation) :

```powershell
$validation = 'C:\Users\pchx\Datasets\nnUNet\nnUNet_results\baseline_100epochs_fold0_20261029\Dataset006_Lung\nnUNetTrainer_100epochs__nnUNetPlans_6GB__3d_fullres\fold_0\validation'
python postprocessing\evaluate.py --prediction-dir "$validation\largest_component" --checkpoint-label checkpoint_final.safetensors
python postprocessing\compare_evaluations.py --before "$validation\evaluation_metrics.json" --after "$validation\largest_component\evaluation_metrics.json"
python visualization\view_comparison.py --case lung_006 --prediction-dir "$validation\largest_component"
```

La comparaison vérifie les identifiants, partitions, références et définitions des métriques. Elle donne les différences appariées, les nombres de cas améliorés/dégradés et un bootstrap des différences moyennes.

### Résultat exploratoire du fold 0, modèle final

- Dice moyen : 0,58584 brut → 0,54283 après filtrage.
- Surface Dice 2 mm moyen : 0,59280 → 0,56605.
- HD95 moyen : 141,80 → 55,70 mm.
- Dice amélioré sur 9 patients, dégradé sur 4 : lung_006, lung_034, lung_042, lung_079. Sur ces quatre patients, le Dice filtré est nul : la plus grande région conservée ne recouvre pas l'annotation.

Conclusion : conserver la plus grande région est une référence expérimentale, pas un nettoyage validé. La baisse du HD95 ne compense pas la suppression de vraies cibles. Cette règle n'a pas été appliquée aux 32 patients test. Les masques bruts restent intacts. Le réglage sur ce fold est exploratoire et n'est pas une performance indépendante.


## Oracle : sélectionner la composante qui recouvre le mieux l'annotation

```powershell
python postprocessing\oracle_select_region.py
```

Expérience diagnostique uniquement : l'oracle sélectionne, parmi les composantes prédites 26-connexes, celle ayant le **Dice maximal avec le masque réel complet**. Cela diffère d'une maximisation du simple nombre de voxels communs. La composante est conservée entière : les débordements restent, aucun contour n'est découpé avec la vérité terrain. Égalité : première composante dans l'ordre déterministe du label SciPy. Aucun recouvrement : masque vide et statut explicite. Aucun tissu tumoral absent de la prédiction ne peut être récupéré.

Entrée par défaut : 13 patients du fold 0, modèle final. Sortie séparée : `validation/oracle_best_overlap`. Les masques bruts et les résultats largest_component restent intacts. Le script refuse un dossier existant. Son manifeste contient les empreintes des prédictions et des annotations, la composante retenue et les scores de chaque candidate.

```powershell
$validation = 'C:\Users\pchx\Datasets\nnUNet\nnUNet_results\baseline_100epochs_fold0_20261029\Dataset006_Lung\nnUNetTrainer_100epochs__nnUNetPlans_6GB__3d_fullres\fold_0\validation'
python postprocessing\evaluate.py --prediction-dir "$validation\oracle_best_overlap" --checkpoint-label checkpoint_final.safetensors
python postprocessing\compare_evaluations.py --before "$validation\evaluation_metrics.json" --after "$validation\oracle_best_overlap\evaluation_metrics.json"
python visualization\view_comparison.py --case lung_006 --prediction-dir "$validation\oracle_best_overlap"
```

**ORACLE : utilise les annotations pour choisir la sortie.** Les scores ne sont pas ceux d'un nettoyage automatique applicable aux 32 patients test. Ils donnent le meilleur Dice possible par sélection d'une seule composante existante, pas un maximum parmi toutes les unions possibles ni une borne supérieure de toutes les autres métriques. L'oracle aide à séparer erreurs de sélection et erreurs de segmentation. L'évaluation et la comparaison reprennent explicitement cette provenance.

Résultats oracle, fold 0 modèle final : Dice moyen 0,76229, Surface Dice 2 mm 0,78737, HD95 moyen 6,544 mm (bruts : 0,58584 / 0,59280 / 141,803 mm). Dice amélioré sur 13/13 patients pour ce jeu, résultat oracle uniquement. Pour lung_006 : Dice 0,74145 et HD95 4,923 mm. La sélection par ground truth est la cause de l'avantage ; il reste à apprendre ou concevoir une sélection sans annotation. Les prédictions et annotations sources ont été vérifiées inchangées par SHA256.

## Sélection automatique CT : première version expérimentale

```powershell
python postprocessing\automatic_ct_selection.py
```

Configuration figée avant évaluation : `automatic_selection_v1.json`. Le script lit seulement les scanners CT, les prédictions et les identifiants de la partition ; il ne lit aucune annotation pour choisir la région.

Méthode : orientation canonique RAS, grille CT grossière proche de 3 mm ; masque corporel par coupe, air interne sous -450 HU, conservation de jusqu'à deux composantes dominantes puis remplissage des trous par coupe. Le volume de cette ROI est contrôlé pour détecter les échecs manifestes. Ce contrôle ne garantit pas que la ROI corresponde aux poumons.

Classement : 70 % de fraction située à moins de 10 mm de la ROI, 20 % de fraction dans la ROI et 10 % de volume relatif. Une région est sélectionnée uniquement si le score est au moins 0,5 et dépasse le second de 0,15. Sinon le masque original entier est conservé. Estimation anatomique défaillante ou grille oblique : même conservation intégrale. Une seule région déjà présente est conservée. Les contours de la région choisie ne sont pas retouchés.

Sortie séparée : `validation/automatic_ct_v1`. Ce dossier contient les masques, les ROI grossières dans `roi/`, les scores et décisions dans `postprocessing_manifest.json`, les évaluations et comparaisons appariées et le verdict expérimental. La géométrie d'origine est restaurée et vérifiée lors de l'export. La configuration et les empreintes des prédictions sources sont enregistrées. Le programme refuse une sortie existante.

### Résultat du premier essai (13 patients, checkpoint final)

| Variante | Dice moyen | Surface Dice 2 mm | HD95 moyen (mm) |
| --- | --- | --- | --- |
| Brut | 0,58584 | 0,59280 | 141,80 |
| Plus grande composante | 0,54283 | 0,56605 | 55,70 |
| Automatique CT v1 | 0,57406 | 0,57473 | 120,11 |
| Oracle (utilise les annotations) | 0,76229 | 0,78737 | 6,54 |

La v1 sélectionne sur 2 patients et préserve les 11 autres par ambiguïté. Elle améliore lung_046, mais supprime la cible annotée chez lung_006 (Dice nul). **Cette version n'est donc pas retenue comme nettoyage automatique fiable et n'a pas été appliquée aux 32 patients test.** Les paramètres n'ont pas été ajustés après observation des résultats. Les moyennes de HD95 plus faibles ne compensent pas une cible supprimée.

La revue de code n'a pas trouvé de problème important de fuite d'annotations ou de coordonnées ; l'échec est celui de la règle heuristique. Le volume et la proximité à une ROI grossière ne suffisent pas ici à reconnaître la bonne composante. Une ROI fiable d'organes et/ou les probabilités du réseau seraient des informations supplémentaires à étudier dans une prochaine expérience distincte.

Pour inspecter ce résultat sans modifier les autres affichages :

```powershell
$validation = 'C:\Users\pchx\Datasets\nnUNet\nnUNet_results\baseline_100epochs_fold0_20261029\Dataset006_Lung\nnUNetTrainer_100epochs__nnUNetPlans_6GB__3d_fullres\fold_0\validation'
python visualization\view_comparison.py --case lung_006 --prediction-dir "$validation\automatic_ct_v1"
```

## Statistiques des probabilités par région prédite

```powershell
python postprocessing\region_probability_stats.py
```

Entrée par défaut : `fold_0/validation_best_probabilities`, checkpoint **best**. Sortie : `region_probability_stats.json` dans ce même dossier. Le script ne consulte aucune annotation et ne sélectionne/supprime aucune composante. Options `--prediction-dir`, `--output` et `--overwrite` ; l'écrasement est limité aux rapports de ce type, jamais aux exports d'inférence ni à leurs métadonnées.

Le script lit les `.npz` avec `allow_pickle=False`, vérifie le lecteur SimpleITKIO et les canaux du dataset, puis transforme explicitement la probabilité tumorale `[Z,Y,X]` en `[X,Y,Z]`. Il vérifie les dimensions, les valeurs finies dans [0,1], la somme des probabilités fond+tumeur et la concordance avec le masque issu de l'argmax. Les rares différences acceptables de précision numérique sont comptées séparément. Il n'exécute pas les `.pkl`.

### Où lire les valeurs dans le JSON

`patients` contient une entrée par patient. Chaque entrée contient `regions`, puis pour chaque composante :

- `component_id`, `volume_ml`, boîte en indices XYZ et centre physique en mm.
- `tumor_probability.mean`, `.median`, `.std_population`, `.minimum`, `.maximum`.
- `tumor_probability.quantiles` : p05, p10, p25, p50, p75, p90, p95.
- `tumor_probability.fraction_ge_0_75`, `.fraction_ge_0_90`, `.fraction_ge_0_95`.
- `tumor_probability.entropy_bits_mean` et `.entropy_bits_p95` : entropie binaire en bits (0 = probabilité extrême, 1 = ambiguïté maximale).
- `rank_by_mean_probability` et `rank_by_volume` : classements descriptifs uniquement, pas une décision de sélection.

`all_predicted_voxels` résume l'ensemble des voxels prédits du patient, avec pondération par voxel, et non une moyenne donnant le même poids à chaque composante. `sum_tumor_probabilities_within_region_ml` est une somme restreinte à la composante prédite, pas le volume tumoral attendu sur tout le scanner.

Les identifiants de composantes sont propres au masque et à son ordre de voxels : ne pas les rapprocher directement des identifiants d'une autre expérience/orientation. Utiliser la géométrie et le recouvrement.

### Limites

Ces valeurs décrivent la probabilité de classe par voxel, conditionnellement à une région déjà prédite. Elles ne constituent pas une probabilité calibrée que la région soit une vraie tumeur. Une région parasite peut être très confiante, et l'entropie n'identifie pas toutes les erreurs. Le script ne mesure pas les lésions manquées.

Premier résultat : 13 patients best, 64 régions, zéro désaccord entre masques et argmax des probabilités. Les données brutes restent intactes. Aucune nouvelle règle de nettoyage n'est appliquée à cette étape.

## Sélecteur appris avec cibles oracle

```powershell
python postprocessing\train_region_selector.py
```

Le premier entraînement est terminé. Le programme refuse de remplacer une expérience existante ; pour reproduire : `--output-dir` avec un nouveau dossier. Les paramètres sont fixés dans region_selector.py : régression logistique L2 C=0,1, StandardScaler, solveur lbfgs, classes équilibrées et graine 12345. Les poids de base donnent le même total à chaque patient ; l'équilibrage de classes est ensuite appliqué et peut changer les totaux effectifs.

Les cibles oracle sont recalculées sur **les masques best**, jamais reprises des masques final. La liste explicite de caractéristiques contient seulement volume, volume relatif, position RAS normalisée au champ de vue, moyenne et quantile p10 de probabilité, entropie, fraction >=0,9 et remplissage de boîte. Aucun Dice, recouvrement, identifiant de patient ou résultat oracle n'entre dans les caractéristiques. Les empreintes des prédictions sont vérifiées entre les statistiques et les cibles.

### Validation

Leave-one-patient-out : 13 ajustements, chacun sur 12 patients, avec ajustement du scaler et des pondérations uniquement sur ces patients. On sélectionne ensuite une composante entière pour le patient exclu. La liste des patients utilisés pour chaque ajustement est enregistrée.

Résultats : choix oracle retrouvé pour 11/13 patients ; Dice moyen 0,66641 contre 0,58585 brut ; Surface Dice 2 mm 0,69619 ; HD95 moyen 31,134 mm ; erreur absolue de volume moyenne 4,934 mL contre 8,499 mL brute. Échecs persistants : lung_006 et lung_042, Dice nul. Le gain de Dice reste incertain (IC bootstrap apparié du delta contenant zéro).

Limites : seulement 13 patients, pas 64 observations indépendantes ; le checkpoint best du réseau principal a déjà été sélectionné avec ces 13 patients. La validation du sélecteur ne supprime pas ce biais amont. Il s'agit d'un prototype de développement, pas d'une performance de test indépendante. Aucune nouvelle règle d'abstention n'a été ajustée après ce résultat.

### Fichiers

`C:\Users\pchx\Datasets\nnUNet\experiments\region_selector_v1` contient :

- `selector.joblib` : modèle final entraîné sur les 13 patients, jamais utilisé pour les prédictions leave-one-out.
- `training_summary.json` : paramètres, versions et bilan.
- `training_candidates.json` : caractéristiques et cibles d'apprentissage.
- `loo_predictions/` : 13 masques produits sans entraîner le sélecteur sur le patient concerné, manifeste des ajustements, évaluation et comparaison au brut.

Pour voir une sortie de validation :

```powershell
python visualization\view_comparison.py --case lung_034 --prediction-dir 'C:\Users\pchx\Datasets\nnUNet\experiments\region_selector_v1\loo_predictions'
```

L'application du modèle final à de nouveaux patients est dans `apply_region_selector.py --stats CHEMIN_STATS --output-dir NOUVEAU_DOSSIER`. Elle refuse les patients ayant entraîné le modèle final et un checkpoint principal différent. Elle ne consulte aucune annotation. Les scores du sélecteur ne sont pas des probabilités cliniques calibrées. Les lots entièrement vides restent vides. Charger uniquement les modèles joblib locaux et de confiance.
