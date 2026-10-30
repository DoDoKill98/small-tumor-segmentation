# Dot-source this file before invoking nnU-Net: . .\configs\nnunet\environment.ps1
$nnunetStorageRoot = Join-Path $env:USERPROFILE 'Datasets\nnUNet'
$env:nnUNet_raw = Join-Path $nnunetStorageRoot 'nnUNet_raw'
$env:nnUNet_preprocessed = Join-Path $nnunetStorageRoot 'nnUNet_preprocessed'
$env:nnUNet_results = Join-Path $nnunetStorageRoot 'nnUNet_results'
# Windows baseline: avoid compilation dependencies and limit augmentation RAM usage.
$env:nnUNet_compile = 'false'
$env:nnUNet_n_proc_DA = '2'
