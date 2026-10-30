"""Launch the single-fold 100-epoch baseline with a recorded configuration."""
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent
RUN_NAME = 'baseline_100epochs_fold0_20261029'


def wait_for_process(pid):
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE,wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x100000,False,pid)
    if not handle:
        if ctypes.get_last_error()==87: return
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        print(f'Waiting for previous validation process {pid}...',flush=True)
        if kernel.WaitForSingleObject(handle,0xFFFFFFFF) != 0:
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel.CloseHandle(handle)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-pid',type=int)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    storage=Path.home()/'Datasets/nnUNet'
    config=json.loads((HERE/'baseline.json').read_text(encoding='utf-8-sig'))
    for key in ('nnUNet_raw','nnUNet_preprocessed'):
        os.environ[key]=str(storage/key)
    root=storage/'nnUNet_results'/RUN_NAME
    os.environ.update(nnUNet_results=str(root),nnUNet_compile='false',nnUNet_n_proc_DA='1')
    dataset=config['dataset_name']
    pp=storage/'nnUNet_preprocessed'/dataset
    splits=json.loads((HERE/'splits_final.json').read_text())
    assert json.loads((pp/'splits_final.json').read_text())==splits, 'Splits differ'
    case_ids=set(splits[0]['train'])|set(splits[0]['val'])
    data_dir=pp/'nnUNetPlans_6GB_3d_fullres'
    for case in case_ids:
        for suffix in ('.b2nd','_seg.b2nd','.pkl'):
            if not (data_dir/(case+suffix)).is_file(): raise FileNotFoundError(case+suffix)
    trainer='nnUNetTrainer_100epochs'
    fold=root/dataset/f'{trainer}__{config["plans_name"]}__3d_fullres'/'fold_0'
    if fold.exists() and not args.resume: raise RuntimeError('Run already exists; use --resume only for an interrupted run')
    command=['6','3d_fullres','0','-p',config['plans_name'],'-tr',trainer,'-device','cuda']
    if args.resume: command.append('--c')
    print('Configuration checked:',command,flush=True)
    if args.check: return
    root.mkdir(parents=True,exist_ok=True)
    if args.wait_pid:
        (root/'queue_status.json').write_text(json.dumps({'status':'waiting_for_validation','pid':args.wait_pid}))
        wait_for_process(args.wait_pid)
        previous=storage/'nnUNet_results/smoke_5epochs_20261029'/dataset/'nnUNetTrainer_5epochs__nnUNetPlans_6GB__3d_fullres/fold_0/validation/summary.json'
        if not previous.is_file(): raise RuntimeError('Previous run exited without validation summary; inspect its log before starting')
    seed=12345
    random.seed(seed)
    import numpy as np
    import torch
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    assert torch.cuda.is_available(), 'CUDA unavailable'
    manifest={'arguments':command,'epochs':100,'fold':0,'seed':seed,'augmentation_workers':1,
              'train_patients':splits[0]['train'],'validation_patients':splits[0]['val'],
              'started':datetime.now().isoformat(),'torch':torch.__version__,
              'nnunetv2':importlib.metadata.version('nnunetv2'),'initialization':'fresh unless --resume',
              'scope':'single-fold development baseline; no independent test performance claim'}
    (root/('resume_manifest.json' if args.resume else 'run_manifest.json')).write_text(json.dumps(manifest,indent=2))
    (root/'queue_status.json').write_text(json.dumps({'status':'training','pid':os.getpid()}))
    sys.argv=['nnUNetv2_train']+command
    entry=next(e for e in importlib.metadata.distribution('nnunetv2').entry_points if e.name=='nnUNetv2_train')
    entry.load()()
    (root/'queue_status.json').write_text(json.dumps({'status':'completed'}))


if __name__=='__main__': main()
