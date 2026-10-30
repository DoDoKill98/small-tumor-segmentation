# Launch from any working directory. This script does not start training.
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'environment.ps1')
$datasetRoot = Join-Path $env:nnUNet_raw 'Dataset006_Lung'
if (-not (Test-Path -LiteralPath (Join-Path $datasetRoot 'dataset.json'))) { throw 'Convert the MSD dataset first; see README.md.' }
python -c "import nnunetv2" 
if ($LASTEXITCODE -ne 0) { throw 'nnU-Net import failed' }
nnUNetv2_plan_and_preprocess -d 6 --verify_dataset_integrity -npfp 2 -np 2 -c 3d_fullres -gpu_memory_target 6 -overwrite_plans_name nnUNetPlans_6GB
if ($LASTEXITCODE -ne 0) { throw 'Planning/preprocessing failed' }
$destination = Join-Path $env:nnUNet_preprocessed 'Dataset006_Lung\splits_final.json'
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'splits_final.json') -Destination $destination -Force
Write-Output 'Preprocessing completed. Training has not been started.'
