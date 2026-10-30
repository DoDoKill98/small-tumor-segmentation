"""Prepare nnU-Net without relying on PowerShell execution policies."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

CONFIG_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Validate inputs and show command without preprocessing')
    parser.add_argument('--storage-root', type=Path, default=Path.home()/'Datasets/nnUNet')
    args = parser.parse_args()
    config = json.loads((CONFIG_DIR/'baseline.json').read_text(encoding='utf-8-sig'))
    env = os.environ.copy()
    for key in ('nnUNet_raw','nnUNet_preprocessed','nnUNet_results'):
        env[key] = str(args.storage_root.resolve()/key)
    env.update(nnUNet_compile='false', nnUNet_n_proc_DA='2')
    raw = Path(env['nnUNet_raw'])/config['dataset_name']
    if not (raw/'dataset.json').is_file():
        parser.error(f'Dataset not found: {raw}')
    data = json.loads((raw/'dataset.json').read_text())
    ids = {p.name.removesuffix('_0000.nii.gz') for p in (raw/'imagesTr').glob('*_0000.nii.gz')}
    labels = {p.name.removesuffix('.nii.gz') for p in (raw/'labelsTr').glob('*.nii.gz')}
    if not ids or ids != labels or len(ids) != data['numTraining']:
        parser.error('Training images, masks and numTraining do not agree')
    splits_text = (CONFIG_DIR/'splits_final.json').read_text(encoding='utf-8-sig')
    splits = json.loads(splits_text)
    validations = []
    for split in splits:
        train,val = set(split['train']),set(split['val'])
        if train & val or train | val != ids or len(train)!=len(split['train']) or len(val)!=len(split['val']):
            parser.error('Invalid patient splits')
        validations.extend(split['val'])
    if len(splits)!=config['cross_validation_folds'] or sorted(validations)!=sorted(ids):
        parser.error('Every patient must appear in validation exactly once')
    version = importlib.metadata.version('nnunetv2')
    entry = ('from nnunetv2.experiment_planning.plan_and_preprocess_entrypoints '
             'import plan_and_preprocess_entry; plan_and_preprocess_entry()')
    command = [sys.executable,'-c',entry,'-d',str(config['dataset_id']),
               '--verify_dataset_integrity','-npfp',str(config['preprocessing_processes']),
               '-np',str(config['preprocessing_processes']),'-c',config['configuration'],
               '-pl',config['planner'],'-gpu_memory_target',str(config['gpu_memory_target_gb']),
               '-overwrite_plans_name',config['plans_name']]
    print(f'Python: {sys.executable}\nnnU-Net: {version}\nDataset: {raw}',flush=True)
    print(f'Patients: {len(ids)} train/val; {len(list((raw/"imagesTs").glob("*_0000.nii.gz")))} official test',flush=True)
    print('Arguments:',subprocess.list2cmdline(command[3:]),flush=True)
    if args.check:
        print('Checks passed. No preprocessing or training started.')
        return
    preprocessed = Path(env['nnUNet_preprocessed'])/config['dataset_name']
    preprocessed.mkdir(parents=True,exist_ok=True)
    destination = preprocessed/'splits_final.json'
    if destination.exists() and json.loads(destination.read_text(encoding='utf-8-sig')) != splits:
        parser.error('Existing splits differ; refusing to overwrite them')
    destination.write_text(splits_text,encoding='utf-8')
    subprocess.run(command,env=env,check=True)
    print('Preprocessing completed. No training started.')


if __name__ == '__main__':
    main()
