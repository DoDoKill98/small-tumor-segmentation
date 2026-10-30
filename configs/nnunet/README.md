# Configuration nnU-Net : Task06 Lung

## Stockage (hors Git et hors OneDrive)

`C:\Users\pchx\Datasets\nnUNet\nnUNet_raw\Dataset006_Lung` : 63 imagesTr, 63 labelsTr, 32 imagesTs.

Les scanners portent le suffixe `_0000.nii.gz` (canal CT) ; les masques gardent leur nom original. Les données sources MSD sont conservées. La conversion officielle a été exécutée avec :

```powershell
. .\configs\nnunet\environment.ps1
nnUNetv2_convert_MSD_dataset -i "$env:USERPROFILE\Datasets\MedicalSegmentationDecathlon\Task06_Lung\extracted\Task06_Lung" -np 2
```

Ne pas relancer la conversion sur un ID déjà existant. Le dataset brut est converti et ses comptes/noms/métadonnées ont été vérifiés. Le prétraitement 3d_fullres de la première expérience est terminé ; ne pas le refaire pour chaque entraînement utilisant les mêmes plans.

## Prétraitement et planification

```powershell
cd 'C:\Users\pchx\small tumor'
python configs\nnunet\prepare.py
```

Ce script vérifie l'intégrité avec nnU-Net et prépare seulement `3d_fullres`. Cible GPU : 6 Go sur la carte de 8 Go pour garder une marge ; deux processus pour limiter l'utilisation des 16 Go de RAM. Cela modifie les tailles de patch/batch par rapport au planning par défaut : utiliser le nom distinct `nnUNetPlans_6GB` et conserver exactement le même plan entre comparaisons. Contrôler l'espace disque et la RAM pendant cette étape.

## Entraînement (après préparation réussie)

```powershell
. .\configs\nnunet\environment.ps1
# Choisir un nom inédit pour chaque répétition afin de séparer les checkpoints.
$env:nnUNet_results = Join-Path $env:USERPROFILE 'Datasets\nnUNet\nnUNet_results\baseline_run01'
nnUNetv2_train 6 3d_fullres 0 -p nnUNetPlans_6GB --npz
```

Répéter les folds 0 à 4. Ne pas confondre folds de validation et répétitions indépendantes. Les 32 cas imagesTs restent hors des partitions, leurs annotations n'étant pas publiques.

## Reproductibilité

`splits_final.json` contient les 5 partitions communes (graine 12345, identifiants triés), également copiées dans nnUNet_preprocessed/Dataset006_Lung. Chaque patient est en validation exactement une fois.

`environment_snapshot.json` identifie les versions principales et le commit nnU-Net ; `installed-packages.txt` est un inventaire de la session, pas un environnement isolé validé ni un verrou d'installation portable. `baseline.json` est le manifeste documentaire de la référence.

Le lanceur train_100epochs.py fixe les graines Python, NumPy et PyTorch à 12345 et enregistre les paramètres de l'essai. Le calcul GPU et les augmentations ne sont pas garantis strictement déterministes. Un entraînement de 100 époques du fold 0, sa validation et l'inférence best ont été exécutés ; les autres folds restent à entraîner. Avant une campagne de répétitions, adapter les identifiants d'essais et documenter les graines et budgets.

Vérification seule, sans prétraitement : python configs\nnunet\prepare.py --check. Le lanceur Python ne nécessite aucun changement de politique PowerShell et utilise le même interpréteur pour nnU-Net. Le fichier prepare.ps1 est conservé comme alternative pour les sessions qui autorisent les scripts.


## Validation BEST avec probabilités

```powershell
python configs\nnunet\predict_validation_best.py --check
python configs\nnunet\predict_validation_best.py
```

Inférence uniquement sur les 13 patients de validation du fold 0 avec `checkpoint_best.safetensors`. Aucun réentraînement et aucune inférence des 32 patients test. Sortie distincte : `fold_0/validation_best_probabilities`. Le lanceur peut reprendre avec `--resume` et refuse d'écraser des prédictions existantes sans cette option. Il vérifie les identifiants et l'empreinte du checkpoint.

Pour chaque patient : masque `.nii.gz`, probabilités `.npz` (clé `probabilities`), propriétés `.pkl`. Le JSON `prediction_manifest.json` contient la provenance et les réglages, pas les cartes volumineuses. `probability_shapes.json` contrôle les dimensions à la fin. Avec ce modèle SimpleITKIO, les cartes sont ordonnées CZYX, canaux 0=fond et 1=tumeur ; les masques lus par NiBabel sont XYZ. Il faudra effectuer une permutation explicite des cartes pour des statistiques régionales en XYZ.

À la fin, le script lance l'évaluation des 13 masques best et produit son propre `evaluation_metrics.json`. Le statut et les journaux sont dans le même dossier. Ce lanceur ne calcule pas les statistiques régionales. Le script postprocessing/region_probability_stats.py les produit dans region_probability_stats.json ; elles ont été calculées pour les 13 patients best de cette expérience.

