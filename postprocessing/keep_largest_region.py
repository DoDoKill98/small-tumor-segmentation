"""Keep the largest 26-connected predicted region; no reference labels are read."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import nibabel as nib
import numpy as np
from scipy import ndimage

DEFAULT_INPUT=Path.home()/'Datasets/nnUNet/nnUNet_results/baseline_100epochs_fold0_20261029/Dataset006_Lung/nnUNetTrainer_100epochs__nnUNetPlans_6GB__3d_fullres/fold_0/validation'


def keep_largest_region(values,voxel_volume_mm3):
    if values.ndim!=3 or not np.all((values==0)|(values==1)): raise ValueError('Expected finite binary 3D prediction')
    if not np.isfinite(voxel_volume_mm3) or voxel_volume_mm3<=0: raise ValueError('Invalid voxel volume')
    mask=values>0; result=np.zeros(values.shape,np.uint8)
    total=int(mask.sum())
    summary={'input_voxels':total,'input_volume_ml':total*voxel_volume_mm3/1000,
             'input_regions':0,'kept_voxels':0,'removed_voxels':0,'kept_volume_ml':0.,'removed_volume_ml':0.,
             'removed_regions':0,'largest_region_tie_count':0,'components':[]}
    if not total: return result,summary
    bounds=[]
    for axis in range(3):
        indices=np.flatnonzero(mask.any(axis=tuple(i for i in range(3) if i!=axis)))
        bounds.append(slice(int(indices[0]),int(indices[-1])+1))
    labels,count=ndimage.label(mask[tuple(bounds)],ndimage.generate_binary_structure(3,3))
    sizes=np.bincount(labels.ravel(),minlength=count+1); sizes[0]=0
    chosen=int(np.argmax(sizes)); kept=int(sizes[chosen])
    result[tuple(bounds)]=labels==chosen
    summary.update(input_regions=int(count),kept_voxels=kept,removed_voxels=total-kept,
                   kept_volume_ml=kept*voxel_volume_mm3/1000,removed_volume_ml=(total-kept)*voxel_volume_mm3/1000,
                   removed_regions=int(count-1),largest_region_tie_count=int(np.count_nonzero(sizes[1:]==kept)),
                   components=[{'component_id':int(i),'voxels':int(sizes[i]),'volume_ml':float(sizes[i]*voxel_volume_mm3/1000),'kept':bool(i==chosen)} for i in range(1,count+1)])
    return result,summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,default=DEFAULT_INPUT)
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args(); output=args.output_dir or args.input_dir/'largest_component'
    files=sorted(args.input_dir.glob('*.nii.gz'))
    if not files: parser.error('No predicted NIfTI masks found')
    if output.exists(): parser.error('Output already exists; choose a new directory to avoid overwriting masks')
    output.mkdir(parents=True)
    records=[]
    for path in files:
        image=nib.load(path)
        if image.header.get_xyzt_units()[0]!='mm': parser.error(f'{path.name}: expected mm units')
        volume=abs(float(np.linalg.det(image.affine[:3,:3])))
        try: cleaned,stats=keep_largest_region(np.asarray(image.dataobj),volume)
        except ValueError as exc: parser.error(f'{path.name}: {exc}')
        header=image.header.copy(); header.set_data_dtype(np.uint8)
        new_image=nib.Nifti1Image(cleaned,image.affine,header)
        for getter,setter in [(image.get_qform,new_image.set_qform),(image.get_sform,new_image.set_sform)]:
            matrix,code=getter(coded=True)
            if matrix is not None: setter(matrix,int(code))
        destination=output/path.name; nib.save(new_image,destination)
        check=nib.load(destination)
        if check.shape!=image.shape or not np.allclose(check.affine,image.affine,rtol=0,atol=1e-4): raise RuntimeError('Geometry changed during export')
        with path.open('rb') as stream: digest=hashlib.file_digest(stream,'sha256').hexdigest()
        records.append({'patient':path.name.removesuffix('.nii.gz'),'input_sha256':digest,**stats})
        print(path.name,':',stats['input_regions'],'regions ->',int(stats['kept_voxels']>0),'; removed',stats['removed_voxels'],'voxels',flush=True)
    report={'created_utc':datetime.now(timezone.utc).isoformat(),'method':'largest_26_connected_component',
            'input_dir':str(args.input_dir.resolve()),'output_dir':str(output.resolve()),
            'uses_ground_truth_for_selection':False,'tie_rule':'first component in deterministic scipy raster labeling',
            'limitations':['Keeps only one connected region; may remove real targets or retain a large false positive',
                           'Does not refine the retained boundary or recover missed tumors'],
            'patients':records}
    temporary=output/'postprocessing_manifest.json.tmp'; temporary.write_text(json.dumps(report,indent=2,allow_nan=False)); os.replace(temporary,output/'postprocessing_manifest.json')
    print('Output:',output,flush=True)


if __name__=='__main__': main()
