# Visualisation des scanners et segmentations avec VTK

Quatre afficheurs pour explorer les scanners CT, leurs annotations et les prédictions nnU-Net. Ils ne modifient ni les volumes ni les masques. Le réseau produit une segmentation : le CT présenté avec la prédiction reste le scanner original.

## Installation

Depuis PowerShell, à la racine du projet :

```powershell
cd "C:\Users\pchx\small tumor"
python -m pip install -r visualization\requirements.txt
```

Les données et résultats restent hors du dépôt Git, sous `C:\Users\pchx\Datasets`.

## Démarrage rapide

Exécuter ce bloc une fois dans le terminal. Modifier `$patient` pour changer de cas.

```powershell
$project = Join-Path $env:USERPROFILE "small tumor"
$storage = Join-Path $env:USERPROFILE "Datasets\nnUNet"
$fold = Join-Path $storage "nnUNet_results\baseline_100epochs_fold0_20261029\Dataset006_Lung\nnUNetTrainer_100epochs__nnUNetPlans_6GB__3d_fullres\fold_0"
$best = Join-Path $fold "validation_best_probabilities"
$patient = "lung_006"
Set-Location $project
```

### 1. Scanner original et annotation — `view_scan.py`

```powershell
python visualization\view_scan.py --case $patient
```

Trois coupes orthogonales et une vue 3D. Le masque réel est superposé lorsqu'il existe. Sans annotation, le scanner seul reste affichable :

```powershell
python visualization\view_scan.py --case lung_002
```

Cet outil utilise par défaut les données MSD extraites :

```text
C:\Users\pchx\Datasets\MedicalSegmentationDecathlon\Task06_Lung\extracted\Task06_Lung
├── imagesTr\    scanners des 63 patients annotés
├── labelsTr\    annotations correspondantes
└── imagesTs\    32 scanners sans annotations publiques
```

Dans ce format, scanner et annotation portent le même nom, par exemple `lung_006.nii.gz`. Pour un autre emplacement : `--data-root "CHEMIN_TASK06_LUNG"`.

### 2. Scanner original et prédiction brute best — `view_prediction.py`

```powershell
python visualization\view_prediction.py --case $patient --prediction-dir "$best"
```

En haut : CT seul. En bas : mêmes coupes avec la prédiction orange. À droite : surface prédite et plans 3D. Cet outil n'affiche pas de ground truth.

Pour un patient du test officiel :

```powershell
python visualization\view_prediction.py --case lung_002 --prediction-dir "$storage\predictions\baseline_100epochs_fold0_best"
```

Les scanners peuvent être au format nnU-Net (`lung_006_0000.nii.gz`) ou MSD (`lung_006.nii.gz`), sous `imagesTr` ou `imagesTs`. La racine se choisit avec `--data-root`.

### 3. Annotation réelle contre prédiction brute — `view_comparison.py`

```powershell
python visualization\view_comparison.py --prediction-file "$best\${patient}.nii.gz"
```

Le nom du patient est déduit du fichier. Variante équivalente :

```powershell
python visualization\view_comparison.py --case $patient --prediction-dir "$best"
```

`--prediction-file` et `--prediction-dir` sont mutuellement exclusifs. Si `--case` est aussi fourni avec un fichier, les identifiants doivent correspondre.

| Couleur dans les coupes | Signification |
| --- | --- |
| Vert | Annotation réelle seule : voxels manqués par la prédiction |
| Rouge | Prédiction seule : voxels prédits hors annotation |
| Jaune | Intersection de l'annotation et de la prédiction |

En 3D, les deux surfaces entières sont vertes et rouges, translucides. Le jaune des coupes n'est pas un troisième masque.

Une annotation réelle est obligatoire pour cet outil. Les 32 patients du test officiel ne peuvent donc pas être comparés à une ground truth publique.

### 4. Annotation réelle contre masque oracle best

```powershell
python visualization\view_comparison.py --prediction-file "$best\oracle_best_overlap\${patient}.nii.gz"
```

L'oracle conserve la composante prédite ayant le meilleur Dice avec l'annotation. Il ne redessine pas ses contours. Le titre de la fenêtre indique `ORACLE`.

**L'oracle utilise la ground truth pour sélectionner la région : ce n'est pas une méthode de prédiction autonome.**

Pour comparer à une autre expérience, passer son dossier avec `--prediction-dir`, ou son fichier avec `--prediction-file`. Par exemple, pour inspecter une sortie du sélecteur où le patient était exclu de l'entraînement :

```powershell
python visualization\view_comparison.py --case $patient --prediction-dir "$storage\experiments\region_selector_v1\loo_predictions"
```

### 5. Toutes les régions, scores et volumes — `view_region_scores.py`

```powershell
python visualization\view_region_scores.py --case $patient
```

Le scanner original apparaît en haut, les coupes superposées en bas, et les surfaces 3D avec un tableau à droite. **Toutes les régions brutes sont conservées, même celles de faible score.**

| Couleur | Signification |
| --- | --- |
| Orange | Région au score maximal, sans garantie de justesse |
| Rouge | Autres régions prédites |
| Cyan | Région actuellement parcourue, si différente de la première |
| Vert | Annotation réelle, si disponible |
| Jaune | Recouvrement avec l'annotation dans les coupes |

Le tableau affiche l'identifiant, le `selector_score` et le volume en mL. Le score est une note de classement du sélecteur, **pas un Dice ni une probabilité clinique calibrée**. Il diffère de la probabilité voxel moyenne de nnU-Net.

Par défaut, les scores viennent de la validation leave-one-patient-out : le patient affiché était exclu de l'ajustement de son sélecteur. Le checkpoint nnU-Net lui-même a cependant été choisi sur ces 13 patients ; cela reste une expérience de développement.

Paramètres explicites :

```powershell
$scores = Join-Path $storage "experiments\region_selector_v1\loo_predictions\postprocessing_manifest.json"
python visualization\view_region_scores.py --case $patient --prediction-dir "$best" --score-manifest "$scores"
```

`--prediction-dir` doit contenir les **masques bruts** et leur `region_probability_stats.json`, pas seulement la région sélectionnée. Les empreintes des fichiers sont vérifiées pour éviter de mélanger les identifiants ou les expériences. Une annotation absente n'empêche pas cet affichage, mais les scores correspondants doivent déjà avoir été calculés.

## Commandes interactives

| Commande | Action | Outils |
| --- | --- | --- |
| Molette sur une coupe | Parcourir les coupes ; vues comparatives synchronisées | Tous |
| Molette dans la 3D | Zoom | Tous |
| Glisser dans la 3D | Tourner la caméra | Tous |
| Bouton central dans la 3D | Déplacer la caméra | Tous |
| `M` | Masquer/afficher le masque ou toutes les prédictions | Tous |
| `G` | Masquer/afficher l'annotation réelle | Comparison, scores |
| `P` | Masquer/afficher les plans dans la 3D | Tous |
| `R` | Réinitialiser coupes et caméras | Tous |
| `Q` ou Échap | Quitter | Tous |
| `N` / `B` | Région suivante/précédente dans le classement | Scores |
| `Home` | Centrer sur la région au score maximal | Scores |
| `T` | Centrer sur l'annotation, si elle est disponible | Scores |

Ces touches ne suppriment aucune région des fichiers. Dans le visualiseur de scores, le centrage initial se fait sur la région au score maximal, qui peut être incorrecte : utiliser `T` pour retrouver l'annotation ou `N/B` pour examiner les autres régions.

## Choisir les bons fichiers

| Dossier sous `fold_0` | Contenu |
| --- | --- |
| `validation` | Prédictions du modèle **final** |
| `validation_best_probabilities` | Prédictions **best**, `.npz` de probabilités et leurs statistiques |
| `validation_best_probabilities\oracle_best_overlap` | Régions sélectionnées avec la ground truth, depuis les prédictions best |

Les exemples ci-dessus sélectionnent explicitement **best**. Attention : sans argument de prédiction, `view_comparison.py` conserve son défaut historique `validation` (**final**), tandis que `view_prediction.py` utilise les prédictions des 32 patients test.

Les 63 cas annotés sont répartis en 50 patients d'entraînement et 13 de validation dans le fold 0. Les commandes oracle et scores actuels concernent les 13 cas de validation, notamment `lung_006`, `lung_010`, `lung_033`, `lung_034`, `lung_041`, `lung_042`, `lung_046`, `lung_048`, `lung_059`, `lung_065`, `lung_066`, `lung_070`, `lung_079`.

## Géométrie et captures

Les vues utilisent le repère physique **RAS**, en millimètres, avec des coupes axiales, coronales et sagittales. Ce n'est pas une convention radiologique d'affichage. Les fichiers lus par NiBabel ont des axes de tableau XYZ ; les cartes de probabilités nnU-Net SimpleITKIO sont stockées CZYX et sont traitées séparément par le script de statistiques.

Les afficheurs vérifient la géométrie des masques par rapport au scanner. Une acquisition oblique peut être rééchantillonnée **en mémoire** pour l'affichage ; les sources restent intactes. Le visualiseur de scores conserve les identifiants avant réorientation et refuse un rééchantillonnage qui ferait disparaître une région.

Le centrage depuis une annotation ou une prédiction sert uniquement à l'exploration visuelle. Il ne doit pas fournir la localisation de la cible à une évaluation automatique.

Tous les outils acceptent `--screenshot` pour produire un PNG hors écran, sans ouvrir une fenêtre interactive :

```powershell
python visualization\view_comparison.py --prediction-file "$best\oracle_best_overlap\${patient}.nii.gz" --screenshot "$env:TEMP\oracle_${patient}.png"
```

Les PNG sont des aperçus ; conserver les NIfTI pour l'entraînement et les mesures.

## Vérifications et dépannage

```powershell
python -m unittest discover -s visualization -v
python visualization\check_region_scores_render.py
```

Le premier contrôle les entrées, couleurs, axes et correspondances des scores. Le second est un petit test de rendu VTK : navigation synchronisée, étendue complète des textures et maintien des régions de faible score.

- **Fichier introuvable** : vérifier le patient, le dossier de prédiction et la distinction MSD/nnU-Net.
- **Annotation absente** : les patients `imagesTs` n'ont pas de labels publics ; utiliser `view_scan.py` ou `view_prediction.py`.
- **Scores incompatibles** : utiliser le masque brut et le manifeste issus de la même expérience, sans modifier les JSON à la main.
- **Petite région peu visible en 3D** : masquer les plans avec `P`, tourner/zoomer, ou parcourir les régions avec `N/B` dans l'outil de scores.

Aide détaillée : `python visualization\NOM_DU_SCRIPT.py --help`.
