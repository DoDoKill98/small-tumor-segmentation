"""Predict all official test cases using the 100-epoch fold-0 BEST checkpoint."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from train_100epochs import wait_for_process


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-pid',type=int)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    storage=Path.home()/'Datasets/nnUNet'
    run=storage/'nnUNet_results/baseline_100epochs_fold0_20261029'
    model=run/'Dataset006_Lung/nnUNetTrainer_100epochs__nnUNetPlans_6GB__3d_fullres'
    source=storage/'nnUNet_raw/Dataset006_Lung/imagesTs'
    output=storage/'predictions/baseline_100epochs_fold0_best'
    checkpoint=model/'fold_0/checkpoint_best.safetensors'
    images=sorted(source.glob('*_0000.nii.gz'))
    if len(images)!=32: raise RuntimeError(f'Expected 32 test scans, found {len(images)}')
    if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
    command=['-i',str(source),'-o',str(output),'-m',str(model),'-f','0',
             '-chk',checkpoint.name,'-npp','1','-nps','1','-device','cuda','--disable_progress_bar']
    if args.resume: command.append('--continue_prediction')
    print('Checkpoint:',checkpoint,flush=True)
    print('Patients:',len(images),'Output:',output,flush=True)
    if args.check: return
    output.mkdir(parents=True,exist_ok=True)
    if list(output.glob('*.nii.gz')) and not args.resume: raise RuntimeError('Predictions exist; use --resume to preserve completed files')
    status=output/'status.json'
    status.write_text(json.dumps({'status':'waiting_for_validation' if args.wait_pid else 'starting'}))
    if args.wait_pid: wait_for_process(args.wait_pid)
    previous_summary=model/'fold_0/validation/summary.json'
    if not previous_summary.is_file():
        raise RuntimeError('Training validation not complete; inspect previous training logs')
    os.environ['nnUNet_compile']='false'
    for key in ('nnUNet_raw','nnUNet_preprocessed'): os.environ[key]=str(storage/key)
    os.environ['nnUNet_results']=str(run)
    manifest={'checkpoint':str(checkpoint),'checkpoint_sha256':hashlib.file_digest(checkpoint.open('rb'),'sha256').hexdigest(),
              'arguments':command,'patients':[p.name.removesuffix('_0000.nii.gz') for p in images],
              'tta':True,'sliding_window_step':0.5,'save_probabilities':False,
              'nnunetv2':importlib.metadata.version('nnunetv2'),'note':'Official test set: no public labels; no Dice claim'}
    (output/'prediction_manifest.json').write_text(json.dumps(manifest,indent=2))
    status.write_text(json.dumps({'status':'running','pid':os.getpid()}))
    sys.argv=['nnUNetv2_predict_from_modelfolder']+command
    try:
        entry=next(e for e in importlib.metadata.distribution('nnunetv2').entry_points if e.name=='nnUNetv2_predict_from_modelfolder')
        entry.load()()
        missing=[p.name.removesuffix('_0000.nii.gz') for p in images if not (output/(p.name.removesuffix('_0000.nii.gz')+'.nii.gz')).is_file()]
        if missing: raise RuntimeError('Missing predictions: '+str(missing))
        status.write_text(json.dumps({'status':'completed','predictions':32}))
    except BaseException as exc:
        status.write_text(json.dumps({'status':'failed','error':str(exc)}))
        raise


if __name__=='__main__': main()
