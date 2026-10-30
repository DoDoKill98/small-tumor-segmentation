"""ORACLE diagnostic: keep the whole predicted component with best ground-truth Dice."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import nibabel as nib
import numpy as np
from scipy import ndimage
from evaluate import validate_geometry
from keep_largest_region import DEFAULT_INPUT


def select_oracle_region(prediction,reference):
    if prediction.shape!=reference.shape or prediction.ndim!=3: raise ValueError('Expected matching 3D masks')
    if not np.all((prediction==0)|(prediction==1)) or not np.all((reference==0)|(reference==1)):
        raise ValueError('Expected finite binary labels 0 and 1')
    pred=prediction>0; gt=reference>0; result=np.zeros(pred.shape,np.uint8)
    nr=int(gt.sum())
    stats={'input_regions':0,'reference_voxels':nr,'input_voxels':int(pred.sum()),
           'selected_component_id':None,'selected_dice':0.,'kept_voxels':0,
           'removed_voxels':int(pred.sum()),'status':'no_overlapping_component','components':[]}
    if not pred.any():
        stats['status']='empty_prediction' if nr else 'both_empty'
        return result,stats
    bounds=[]
    for axis in range(3):
        indices=np.flatnonzero(pred.any(axis=tuple(i for i in range(3) if i!=axis)))
        bounds.append(slice(int(indices[0]),int(indices[-1])+1))
    labels,count=ndimage.label(pred[tuple(bounds)],ndimage.generate_binary_structure(3,3))
    sizes=np.bincount(labels.ravel(),minlength=count+1)
    overlap=np.bincount(labels[gt[tuple(bounds)]],minlength=count+1)
    sizes[0]=0; overlap[0]=0
    scores=2*overlap[1:]/(sizes[1:]+nr)
    selected=int(np.argmax(scores))+1 if nr and np.any(overlap[1:]) else None
    stats['input_regions']=int(count)
    stats['components']=[{'component_id':i,'voxels':int(sizes[i]),'overlap_voxels':int(overlap[i]),
                          'dice_with_full_reference':float(scores[i-1]),'selected':i==selected} for i in range(1,count+1)]
    if selected is not None:
        result[tuple(bounds)]=labels==selected
        stats.update(selected_component_id=selected,selected_dice=float(scores[selected-1]),
                     kept_voxels=int(sizes[selected]),removed_voxels=int(pred.sum()-sizes[selected]),status='selected_by_ground_truth')
    elif not nr: stats['status']='empty_reference'
    return result,stats


def file_hash(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    project=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,default=DEFAULT_INPUT)
    parser.add_argument('--reference-dir',type=Path,default=Path.home()/'Datasets/nnUNet/nnUNet_raw/Dataset006_Lung/labelsTr')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--splits',type=Path,default=project/'configs/nnunet/splits_final.json')
    parser.add_argument('--fold',type=int,default=0)
    args=parser.parse_args(); output=args.output_dir or args.input_dir/'oracle_best_overlap'
    split_bytes=args.splits.read_bytes(); splits=json.loads(split_bytes.decode('utf-8-sig'))
    if args.fold<0 or args.fold>=len(splits): parser.error('Invalid fold')
    split=splits[args.fold]; expected=set(split['val']); files=sorted(args.input_dir.glob('*.nii.gz'))
    if not expected or expected & set(split['train']) or expected!={p.name.removesuffix('.nii.gz') for p in files}:
        parser.error('Input must match exactly the annotated validation patients of the selected fold')
    for path in files:
        if not (args.reference_dir/path.name).is_file(): parser.error(f'Missing ground truth: {path.name}')
    if output.exists(): parser.error('Output already exists; choose a new directory')
    output.mkdir(parents=True); records=[]
    print('ORACLE: selection uses ground truth. Not an automatic deployable method.',flush=True)
    for path in files:
        reference_path=args.reference_dir/path.name
        image,ref=nib.load(path),nib.load(reference_path)
        validate_geometry(ref.shape,ref.affine,image.shape,image.affine)
        if image.header.get_xyzt_units()[0]!='mm' or ref.header.get_xyzt_units()[0]!='mm':
            raise ValueError('Expected explicit spatial units mm')
        cleaned,stats=select_oracle_region(np.asarray(image.dataobj),np.asarray(ref.dataobj))
        header=image.header.copy(); header.set_data_dtype(np.uint8)
        out=nib.Nifti1Image(cleaned,image.affine,header)
        for getter,setter in [(image.get_qform,out.set_qform),(image.get_sform,out.set_sform)]:
            matrix,code=getter(coded=True)
            if matrix is not None: setter(matrix,int(code))
        destination=output/path.name; nib.save(out,destination)
        saved=nib.load(destination)
        validate_geometry(image.shape,image.affine,saved.shape,saved.affine)
        voxel_ml=abs(float(np.linalg.det(image.affine[:3,:3])))/1000
        records.append({'patient':path.name.removesuffix('.nii.gz'),'input_sha256':file_hash(path),
                        'reference_sha256':file_hash(reference_path),'kept_volume_ml':stats['kept_voxels']*voxel_ml,**stats})
        print(path.name,':',stats['status'],'; selected Dice =',round(stats['selected_dice'],4),flush=True)
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'method':'oracle_max_dice_whole_component',
              'uses_ground_truth_for_selection':True,'deployable':False,
              'input_dir':str(args.input_dir.resolve()),'reference_dir':str(args.reference_dir.resolve()),
              'fold':args.fold,'split_sha256':hashlib.sha256(split_bytes).hexdigest(),
              'selection':'Maximum Dice of one entire predicted 26-connected component against the full reference mask',
              'no_overlap_policy':'Empty output with explicit failure status; no invented segmentation',
              'tie_rule':'First component in deterministic scipy raster labeling',
              'limitations':['Oracle uses the answer to select a region; not independent performance',
                             'Best possible Dice among single predicted components, not an upper bound over arbitrary unions',
                             'Cannot correct retained contours or recover unpredicted tumor tissue'],
              'patients':records}
    temporary=output/'postprocessing_manifest.json.tmp'; temporary.write_text(json.dumps(manifest,indent=2,allow_nan=False)); os.replace(temporary,output/'postprocessing_manifest.json')
    print('Oracle outputs:',output,flush=True)


if __name__=='__main__': main()
