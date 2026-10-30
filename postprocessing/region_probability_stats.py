"""Describe tumor probabilities for each predicted region; no ground truth or selection."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
import tempfile
import nibabel as nib
import numpy as np
from scipy import ndimage

DEFAULT_INPUT=Path.home()/'Datasets/nnUNet/nnUNet_results/baseline_100epochs_fold0_20261029/Dataset006_Lung/nnUNetTrainer_100epochs__nnUNetPlans_6GB__3d_fullres/fold_0/validation_best_probabilities'


def probabilities_to_xyz(probabilities,mask_shape):
    if probabilities.ndim!=4 or probabilities.shape[0]!=2 or probabilities.shape[1:]!=tuple(mask_shape)[::-1]:
        raise ValueError(f'Expected probabilities [2,Z,Y,X] matching mask {mask_shape}; got {probabilities.shape}')
    return probabilities[1].transpose(2,1,0)


def validate_probability_alignment(probabilities,mask):
    probabilities_to_xyz(probabilities,mask.shape)
    if not np.issubdtype(probabilities.dtype,np.floating): raise ValueError('Expected floating point probabilities')
    if not np.all((mask==0)|(mask==1)): raise ValueError('Expected binary predicted mask')
    epsilon=max(1e-6,float(np.finfo(probabilities.dtype).eps)*4)
    mismatches=0; ties=0; normalization_error=0.
    # Bounded-memory blocks avoid allocating a full-volume argmax/int64 array.
    for z in range(0,probabilities.shape[1],8):
        block=probabilities[:,z:z+8]
        if not np.isfinite(block).all() or block.min()<0 or block.max()>1:
            raise ValueError('Probabilities must be finite and in [0,1]')
        p0,p1=block[0],block[1]
        error=float(np.max(np.abs(p0.astype(np.float32)+p1.astype(np.float32)-1)))
        normalization_error=max(normalization_error,error)
        if error>max(.002,epsilon*2): raise ValueError('Background/tumor probabilities do not sum to one')
        exported=mask[:,:,z:z+8].transpose(2,1,0)>0
        different=exported!=(p1>p0)
        n=int(different.sum()); mismatches+=n
        if n:
            close=np.abs(p1[different].astype(float)-p0[different].astype(float))<=epsilon
            if not close.all(): raise ValueError('Mask/probability labels disagree away from numerical ties: possible axis or source mismatch')
            ties+=n
    return {'probability_axes':'CZYX','mask_axes':'XYZ','disagreement_voxels':mismatches,
            'disagreements_explained_by_numerical_ties':ties,'maximum_normalization_error':normalization_error,
            'tie_epsilon':epsilon}


def probability_statistics(values):
    values=np.asarray(values,dtype=np.float64)
    if values.size==0: return {'voxel_count':0,'mean':None,'median':None,'entropy_bits_mean':None}
    if not np.isfinite(values).all() or values.min()<0 or values.max()>1: raise ValueError('Invalid probabilities')
    clipped=np.clip(values,1e-15,1-1e-15)
    entropy=-(clipped*np.log2(clipped)+(1-clipped)*np.log2(1-clipped))
    quantiles=np.percentile(values,[5,10,25,50,75,90,95])
    return {'voxel_count':int(values.size),'mean':float(values.mean()),'median':float(quantiles[3]),
            'std_population':float(values.std()),'minimum':float(values.min()),'maximum':float(values.max()),
            'quantiles':{key:float(v) for key,v in zip(['p05','p10','p25','p50','p75','p90','p95'],quantiles)},
            'fraction_ge_0_75':float(np.mean(values>=.75)),
            'fraction_ge_0_90':float(np.mean(values>=.9)),
            'fraction_ge_0_95':float(np.mean(values>=.95)),
            'entropy_bits_mean':float(entropy.mean()),'entropy_bits_p95':float(np.percentile(entropy,95))}


def summarize_regions(mask,probabilities_xyz,affine):
    foreground=mask>0
    if not foreground.any():
        return {'region_count':0,'all_predicted_voxels':probability_statistics(np.array([])),'regions':[]}
    bounds=[]
    for axis in range(3):
        indices=np.flatnonzero(foreground.any(axis=tuple(i for i in range(3) if i!=axis)))
        bounds.append(slice(int(indices[0]),int(indices[-1])+1))
    offset=np.array([s.start for s in bounds]); cropped=foreground[tuple(bounds)]
    labels,n=ndimage.label(cropped,ndimage.generate_binary_structure(3,3))
    probs=probabilities_xyz[tuple(bounds)]
    boxes=ndimage.find_objects(labels)
    voxel_ml=abs(float(np.linalg.det(affine[:3,:3])))/1000
    if not np.isfinite(affine).all() or voxel_ml<=0: raise ValueError('Invalid affine geometry')
    rows=[]; all_values=[]
    for i,box in enumerate(boxes,1):
        component=labels[box]==i
        values=probs[box][component]
        stats=probability_statistics(values); all_values.append(values)
        coords=np.argwhere(component)+offset+np.array([s.start for s in box])
        centroid=coords.mean(axis=0)
        rows.append({'component_id':i,'volume_ml':stats['voxel_count']*voxel_ml,
                     'centroid_voxel_xyz':centroid.tolist(),
                     'centroid_world_mm':nib.affines.apply_affine(affine,centroid).tolist(),
                     'bounding_box_xyz_exclusive':[[int(s.start+offset[k]),int(s.stop+offset[k])] for k,s in enumerate(box)],
                     'tumor_probability':stats,
                     'sum_tumor_probabilities_within_region_ml':float(np.asarray(values,dtype=float).sum()*voxel_ml)})
    for rank,row in enumerate(sorted(rows,key=lambda r:(-r['tumor_probability']['mean'],r['component_id'])),1):
        row['rank_by_mean_probability']=rank
    for rank,row in enumerate(sorted(rows,key=lambda r:(-r['volume_ml'],r['component_id'])),1): row['rank_by_volume']=rank
    return {'region_count':int(n),'all_predicted_voxels':probability_statistics(np.concatenate(all_values)),
            'regions':rows}


def file_hash(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def validate_output_path(output,overwrite):
    output=Path(output)
    if output.suffix.lower()!='.json' or output.resolve().suffix.lower()!='.json':
        raise ValueError('Output must be a JSON report, never a mask or probability export')
    if output.exists():
        if not overwrite: raise ValueError('Statistics output exists; use --overwrite to replace only this JSON report')
        try: previous=json.loads(output.read_text())
        except (ValueError,OSError,UnicodeError) as exc:
            raise ValueError('Refusing to overwrite a file that is not a probability statistics report') from exc
        if not isinstance(previous,dict) or previous.get('report_type')!='region_probability_statistics':
            raise ValueError('Refusing to overwrite inference metadata or another kind of report')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prediction-dir',type=Path,default=DEFAULT_INPUT)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args(); root=args.prediction_dir; output=args.output or root/'region_probability_stats.json'
    try: validate_output_path(output,args.overwrite)
    except ValueError as exc: parser.error(str(exc))
    manifest_path=root/'prediction_manifest.json'; manifest=json.loads(manifest_path.read_text())
    if not manifest.get('save_probabilities'): parser.error('Inference manifest does not declare saved probabilities')
    plans=json.loads((root/'plans.json').read_text()); dataset=json.loads((root/'dataset.json').read_text())
    if plans.get('image_reader_writer')!='SimpleITKIO': parser.error('Only SimpleITKIO CZYX probability exports are supported')
    if dataset.get('labels')!={'background':0,'cancer':1}: parser.error('Expected binary channels background=0, cancer=1')
    patients=manifest['patients']
    if not patients or len(set(patients))!=len(patients): parser.error('Invalid manifest patient list')
    if set(patients)!={p.name.removesuffix('.nii.gz') for p in root.glob('*.nii.gz')}: parser.error('Patient masks differ from inference manifest')
    if set(patients)!={p.name.removesuffix('.npz') for p in root.glob('*.npz')}: parser.error('Probability files differ from inference manifest')
    for patient in patients:
        if Path(patient).name!=patient: parser.error('Invalid patient ID')
    cases=[]
    for i,patient in enumerate(patients,1):
        started=time.perf_counter(); mask_path=root/(patient+'.nii.gz'); prob_path=root/(patient+'.npz')
        image=nib.load(mask_path)
        if len(image.shape)!=3 or image.header.get_xyzt_units()[0]!='mm': parser.error(f'{patient}: expected 3D mask in mm')
        mask=np.asarray(image.dataobj)
        with np.load(prob_path,allow_pickle=False) as archive:
            if 'probabilities' not in archive.files: parser.error(f'{patient}: missing probabilities key')
            probabilities=archive['probabilities']
        try:
            checks=validate_probability_alignment(probabilities,mask)
            values_xyz=probabilities_to_xyz(probabilities,mask.shape)
            regions=summarize_regions(mask,values_xyz,image.affine)
        except ValueError as exc: parser.error(f'{patient}: {exc}')
        case={'patient':patient,'shape_xyz':list(mask.shape),'probability_shape_czyx':list(probabilities.shape),
              'probability_dtype':str(probabilities.dtype),'alignment_checks':checks,
              'mask_sha256':file_hash(mask_path),'probabilities_sha256':file_hash(prob_path),**regions,
              'runtime_seconds':time.perf_counter()-started}
        cases.append(case)
        print(f'{i}/{len(patients)} {patient}: {regions["region_count"]} regions; alignment OK',flush=True)
        for row in regions['regions']:
            print(f'  component {row["component_id"]}: {row["volume_ml"]:.3f} mL; mean P(tumor)={row["tumor_probability"]["mean"]:.4f}; median={row["tumor_probability"]["median"]:.4f}',flush=True)
        del probabilities,values_xyz,mask
    report={'report_type':'region_probability_statistics','schema_version':1,'created_utc':datetime.now(timezone.utc).isoformat(),
            'prediction_dir':str(root.resolve()),'checkpoint':manifest['checkpoint'],
            'checkpoint_sha256':manifest['checkpoint_sha256'],'inference_manifest_sha256':file_hash(manifest_path),
            'uses_ground_truth':False,'automatic_selection_applied':False,
            'definitions':{'probability':'Class-1 voxelwise network probability, not a calibrated probability that a region is a tumor',
                           'regions':'26-connected components of the exported binary prediction',
                           'axes':'NPZ CZYX -> explicit transpose (2,1,0) of class 1 -> mask XYZ; alignment verified against argmax, with numerical ties reported',
                           'entropy':'Binary Shannon entropy in bits, averaged within each component',
                           'sum_tumor_probabilities_within_region_ml':'Sum of class-1 probabilities restricted to this predicted component times voxel volume; not total expected tumor volume',
                           'population':'Only voxels inside predicted components; not a measure of missed lesions'},
            'limitations':['High-confidence false positives may remain; probability alone is not a validated selection rule',
                           'No ground-truth correctness labels or clinical interpretation are assigned to these regions'],
            'versions':{name:importlib.metadata.version(name) for name in ['numpy','scipy','nibabel']},
            'summary':{'patients':len(cases),'predicted_regions':sum(c['region_count'] for c in cases),
                       'empty_predictions':sum(c['region_count']==0 for c in cases),
                       'alignment_disagreement_voxels':sum(c['alignment_checks']['disagreement_voxels'] for c in cases)},
            'patients':cases}
    output.parent.mkdir(parents=True,exist_ok=True)
    validate_output_path(output,args.overwrite)
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=output.parent,suffix='.json.tmp',delete=False) as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
        temporary=Path(stream.name)
    os.replace(temporary,output)
    print('Report:',output,flush=True); print('Summary:',report['summary'],flush=True)


if __name__=='__main__': main()
