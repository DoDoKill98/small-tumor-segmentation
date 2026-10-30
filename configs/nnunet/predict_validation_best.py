"""Infer fold-0 validation with BEST and save full voxelwise probabilities."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
import nibabel as nib
import numpy as np

HERE=Path(__file__).resolve().parent
STORAGE=Path.home()/'Datasets/nnUNet'
MODEL=STORAGE/'nnUNet_results/baseline_100epochs_fold0_20261029/Dataset006_Lung/nnUNetTrainer_100epochs__nnUNetPlans_6GB__3d_fullres'
OUTPUT=MODEL/'fold_0/validation_best_probabilities'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    split_path=HERE/'splits_final.json'; split_bytes=split_path.read_bytes()
    split=json.loads(split_bytes.decode('utf-8-sig'))[0]
    patients=sorted(split['val'])
    if len(patients)!=13 or len(set(patients))!=13 or set(patients)&set(split['train']): raise ValueError('Invalid fold-0 validation split')
    ct_dir=STORAGE/'nnUNet_raw/Dataset006_Lung/imagesTr'
    paths=[ct_dir/(patient+'_0000.nii.gz') for patient in patients]
    for path in paths:
        if not path.is_file(): raise FileNotFoundError(path)
    checkpoint=MODEL/'fold_0/checkpoint_best.safetensors'
    if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
    worst_bytes=sum(int(np.prod(nib.load(p).shape))*2*4 for p in paths)
    free=shutil.disk_usage(STORAGE).free
    print('Checkpoint:',checkpoint,'\nPatients:',len(patients),'\nOutput:',OUTPUT,flush=True)
    print('Free GiB:',round(free/2**30,1),'Conservative uncompressed float32 probability GiB:',round(worst_bytes/2**30,1),flush=True)
    if free<worst_bytes+3*2**30: raise RuntimeError('Insufficient free disk margin for probabilities')
    if args.check: return
    OUTPUT.mkdir(parents=True,exist_ok=True)
    status=OUTPUT/'status.json'
    if list(OUTPUT.glob('*.nii.gz')) and not args.resume: raise RuntimeError('Existing masks; use --resume to continue without overwriting')
    manifest_path=OUTPUT/'prediction_manifest.json'
    with checkpoint.open('rb') as stream: digest=hashlib.file_digest(stream,'sha256').hexdigest()
    if args.resume and manifest_path.exists():
        previous=json.loads(manifest_path.read_text())
        if previous['checkpoint_sha256']!=digest or previous['patients']!=patients: raise RuntimeError('Resume provenance mismatch')
    manifest={'checkpoint':str(checkpoint),'checkpoint_sha256':digest,'patients':patients,'fold':0,
              'split_sha256':hashlib.sha256(split_bytes).hexdigest(),'save_probabilities':True,
              'probability_storage':'.npz key probabilities, channels background=0 and cancer=1; spatial order follows nnU-Net reader (SimpleITKIO: ZYX), not NIfTI XYZ',
              'tta':True,'sliding_window_step':0.5,'preprocessing_workers':1,'export_workers':1,
              'nnunetv2':importlib.metadata.version('nnunetv2'),'started_utc':datetime.now(timezone.utc).isoformat(),
              'scope':'13 annotated validation patients only; labels not used for inference'}
    manifest_path.write_text(json.dumps(manifest,indent=2))
    status.write_text(json.dumps({'status':'starting','pid':os.getpid(),'expected_patients':13}))
    try:
        os.environ['nnUNet_compile']='false'
        for key in ('nnUNet_raw','nnUNet_preprocessed'): os.environ[key]=str(STORAGE/key)
        os.environ['nnUNet_results']=str(STORAGE/'nnUNet_results/baseline_100epochs_fold0_20261029')
        import torch
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
        predictor=nnUNetPredictor(tile_step_size=.5,use_gaussian=True,use_mirroring=True,
                                  perform_everything_on_device=True,device=torch.device('cuda'),
                                  verbose=False,verbose_preprocessing=False,allow_tqdm=False)
        predictor.initialize_from_trained_model_folder(str(MODEL),use_folds=(0,),checkpoint_name=checkpoint.name)
        if predictor.plans_manager.image_reader_writer_class.__name__!='SimpleITKIO':
            raise RuntimeError('Probability axis checks require SimpleITKIO; inspect reader before proceeding')
        status.write_text(json.dumps({'status':'running_inference','pid':os.getpid(),'expected_patients':13}))
        predictor.predict_from_files([[str(path)] for path in paths],str(OUTPUT),save_probabilities=True,
                                     overwrite=not args.resume,num_processes_preprocessing=1,num_processes_segmentation_export=1)
        shapes=[]
        for patient,source in zip(patients,paths):
            for ending in ('.nii.gz','.npz','.pkl'):
                if not (OUTPUT/(patient+ending)).is_file(): raise RuntimeError(f'Missing export: {patient}{ending}')
            with zipfile.ZipFile(OUTPUT/(patient+'.npz')) as archive:
                with archive.open('probabilities.npy') as stream:
                    version=np.lib.format.read_magic(stream)
                    if version==(1,0): shape,_,dtype=np.lib.format.read_array_header_1_0(stream)
                    elif version==(2,0): shape,_,dtype=np.lib.format.read_array_header_2_0(stream)
                    else: raise RuntimeError(f'Unsupported NPY header {version}')
            expected=(2,*nib.load(source).shape[::-1])
            if tuple(shape)!=expected: raise RuntimeError(f'{patient}: unexpected probability shape {shape}, expected {expected}')
            shapes.append({'patient':patient,'shape_CZYX':list(shape),'dtype':str(dtype)})
        (OUTPUT/'probability_shapes.json').write_text(json.dumps(shapes,indent=2))
        status.write_text(json.dumps({'status':'evaluating','pid':os.getpid(),'completed_predictions':13}))
        evaluator=HERE.parents[1]/'postprocessing/evaluate.py'
        command=[sys.executable,'-u',str(evaluator),'--prediction-dir',str(OUTPUT),'--checkpoint-label','checkpoint_best.safetensors']
        if (OUTPUT/'evaluation_metrics.json').exists(): command.append('--overwrite')
        subprocess.run(command,check=True)
        status.write_text(json.dumps({'status':'completed','predictions':13,'probability_files':13,'evaluation_completed':True}))
        print('Completed: BEST validation masks, probabilities, and evaluation.',flush=True)
    except BaseException as exc:
        status.write_text(json.dumps({'status':'failed','error':str(exc)}))
        raise


if __name__=='__main__': main()
