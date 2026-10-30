"""Experimental CT-guided selection of predicted regions; never opens reference labels."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import time
import nibabel as nib
import numpy as np
from scipy import ndimage
from evaluate import validate_geometry
from keep_largest_region import DEFAULT_INPUT

HERE=Path(__file__).resolve().parent


def largest_2d(mask):
    labels,n=ndimage.label(mask)
    if not n: return np.zeros_like(mask,bool)
    counts=np.bincount(labels.ravel()); counts[0]=0
    return labels==int(np.argmax(counts))


def estimate_lung_roi(ct,spacing,config):
    spacing=np.asarray(spacing,float)
    if ct.ndim!=3 or spacing.shape!=(3,) or not np.isfinite(ct).all() or np.any(spacing<=0):
        raise ValueError('Invalid CT/spacing')
    steps=np.maximum(1,np.rint(config['coarse_spacing_mm']/spacing).astype(int))
    coarse=ct[tuple(slice(None,None,int(s)) for s in steps)]
    coarse_spacing=spacing*steps
    air=np.zeros(coarse.shape,bool)
    for z in range(coarse.shape[2]):
        tissue=coarse[:,:,z]>config['body_threshold_hu']
        closed=ndimage.binary_closing(tissue,iterations=config['body_closing_iterations'])
        body=ndimage.binary_fill_holes(largest_2d(closed))
        air[:,:,z]=(coarse[:,:,z]<config['lung_threshold_hu']) & body
    labels,n=ndimage.label(air,ndimage.generate_binary_structure(3,3))
    counts=np.bincount(labels.ravel()); counts[0]=0
    chosen=[]
    if n:
        largest=int(counts.max())
        chosen=[int(i) for i in np.argsort(-counts) if i>0 and counts[i]>=largest*config['lung_component_min_fraction']][:config['max_lung_components']]
    roi=np.isin(labels,chosen) if chosen else np.zeros_like(air)
    # Fill dense internal islands (e.g. vessels) on each coarse axial slice.
    for z in range(roi.shape[2]): roi[:,:,z]=ndimage.binary_fill_holes(roi[:,:,z])
    volume=float(roi.sum()*np.prod(coarse_spacing)/1000.)
    reliable=bool(chosen and config['minimum_lung_volume_ml']<=volume<=config['maximum_lung_volume_ml'])
    quality={'reliable':reliable,'roi_volume_ml':volume,'air_components_kept':len(chosen),
             'coarse_spacing_mm':coarse_spacing.tolist(),
             'reason':'plausible_volume_heuristic' if reliable else 'empty_or_implausible_roi_volume'}
    return roi,steps,quality


def select_with_roi(prediction,spacing,roi,steps,config,roi_reliable=True):
    if prediction.ndim!=3 or not np.all((prediction==0)|(prediction==1)): raise ValueError('Expected binary 3D prediction')
    spacing=np.asarray(spacing,float); steps=np.asarray(steps,int)
    if spacing.shape!=(3,) or np.any(spacing<=0) or not np.isfinite(spacing).all() or steps.shape!=(3,) or np.any(steps<1):
        raise ValueError('Invalid grid spacing/steps')
    original=prediction.astype(np.uint8)
    if not original.any(): return original,{'status':'empty_prediction','input_regions':0,'candidates':[],'selected_component_id':None}
    bounds=[]
    for axis in range(3):
        coords=np.flatnonzero(original.any(axis=tuple(i for i in range(3) if i!=axis)))
        bounds.append(slice(int(coords[0]),int(coords[-1])+1))
    offset=np.array([s.start for s in bounds])
    labels,n=ndimage.label(original[tuple(bounds)],ndimage.generate_binary_structure(3,3))
    decision={'input_regions':int(n),'selected_component_id':None,'candidates':[]}
    if not roi_reliable or not roi.any():
        decision['status']='unreliable_roi_preserve_all'; return original,decision
    distance=ndimage.distance_transform_edt(~roi,sampling=spacing*steps)
    sizes=np.bincount(labels.ravel()); sizes[0]=0
    boxes=ndimage.find_objects(labels); weights=config['score_weights']
    for i,box in enumerate(boxes,1):
        points=np.argwhere(labels[box]==i)+offset+np.array([s.start for s in box])
        distances=ndimage.map_coordinates(distance,(points/steps).T,order=1,mode='nearest')
        within_margin=float(np.mean(distances<=config['lung_margin_mm']))
        within_lung=float(np.mean(distances<=1e-6))
        relative_volume=float(sizes[i]/sizes.max())
        score=weights['within_margin']*within_margin+weights['within_lung']*within_lung+weights['relative_volume']*relative_volume
        decision['candidates'].append({'component_id':i,'voxels':int(sizes[i]),'volume_ml':float(sizes[i]*np.prod(spacing)/1000),
                                       'within_margin_fraction':within_margin,'within_lung_fraction':within_lung,
                                       'median_distance_to_roi_mm':float(np.median(distances)),
                                       'relative_volume':relative_volume,'score':float(score)})
    ranked=sorted(decision['candidates'],key=lambda r:(-r['score'],r['component_id']))
    best=ranked[0]; gap=best['score']-ranked[1]['score'] if len(ranked)>1 else None
    decision.update(best_score=best['score'],score_gap=gap)
    if n==1: decision['status']='single_region_preserved'; return original,decision
    if best['score']<config['minimum_score']:
        decision['status']='no_credible_candidate_preserve_all'; return original,decision
    if gap<config['minimum_score_gap']:
        decision['status']='ambiguous_preserve_all'; return original,decision
    out=np.zeros(original.shape,np.uint8); out[tuple(bounds)]=labels==best['component_id']
    decision.update(status='selected',selected_component_id=best['component_id'])
    return out,decision


def restore_original_orientation(values,canonical_affine,original_affine):
    transform=nib.orientations.ornt_transform(nib.io_orientation(canonical_affine),nib.io_orientation(original_affine))
    return nib.orientations.apply_orientation(values,transform)


def hash_file(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    project=HERE.parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,default=DEFAULT_INPUT)
    parser.add_argument('--ct-dir',type=Path,default=Path.home()/'Datasets/nnUNet/nnUNet_raw/Dataset006_Lung/imagesTr')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--config',type=Path,default=HERE/'automatic_selection_v1.json')
    parser.add_argument('--splits',type=Path,default=project/'configs/nnunet/splits_final.json')
    parser.add_argument('--fold',type=int,default=0)
    args=parser.parse_args()
    config_bytes=args.config.read_bytes(); config=json.loads(config_bytes.decode('utf-8-sig'))
    splits=json.loads(args.splits.read_text(encoding='utf-8-sig'))
    if args.fold<0 or args.fold>=len(splits): parser.error('Invalid fold')
    expected=set(splits[args.fold]['val']); files=sorted(args.input_dir.glob('*.nii.gz'))
    if {p.name.removesuffix('.nii.gz') for p in files}!=expected or not expected: parser.error('Expected exactly the validation patient IDs')
    output=args.output_dir or args.input_dir/'automatic_ct_v1'
    if output.exists(): parser.error('Output exists; choose a fresh directory')
    ct_paths={}
    for p in files:
        case=p.name.removesuffix('.nii.gz')
        options=[args.ct_dir/(case+s) for s in ('_0000.nii.gz','.nii.gz')]
        found=[p for p in options if p.is_file()]
        if len(found)!=1: parser.error(f'Expected one CT for {case}, found {len(found)}')
        ct_paths[case]=found[0]
    output.mkdir(parents=True); (output/'roi').mkdir(); records=[]
    for path in files:
        started=time.perf_counter(); case=path.name.removesuffix('.nii.gz'); ct_path=ct_paths[case]
        image=nib.load(path); ct_image=nib.load(ct_path)
        validate_geometry(ct_image.shape,ct_image.affine,image.shape,image.affine)
        if image.header.get_xyzt_units()[0]!='mm' or ct_image.header.get_xyzt_units()[0]!='mm': raise ValueError('Expected explicit units mm')
        canonical_ct=nib.as_closest_canonical(ct_image); canonical_pred=nib.as_closest_canonical(image)
        scan=canonical_ct.get_fdata(dtype=np.float32); pred=np.asarray(canonical_pred.dataobj)
        spacing=np.linalg.norm(canonical_ct.affine[:3,:3],axis=0)
        aligned=np.allclose(canonical_ct.affine[:3,:3],np.diag(np.diag(canonical_ct.affine[:3,:3])),atol=1e-5)
        if aligned:
            roi,steps,quality=estimate_lung_roi(scan,spacing,config)
        else:
            steps=np.ones(3,int); roi=np.zeros((1,1,1),bool); quality={'reliable':False,'reason':'oblique_grid_preserve_all'}
        cleaned,decision=select_with_roi(pred,spacing,roi,steps,config,quality['reliable'])
        original_order=restore_original_orientation(cleaned,canonical_pred.affine,image.affine)
        header=image.header.copy(); header.set_data_dtype(np.uint8)
        export=nib.Nifti1Image(original_order,image.affine,header)
        for getter,setter in [(image.get_qform,export.set_qform),(image.get_sform,export.set_sform)]:
            matrix,code=getter(coded=True)
            if matrix is not None: setter(matrix,int(code))
        nib.save(export,output/path.name)
        saved=nib.load(output/path.name); validate_geometry(image.shape,image.affine,saved.shape,saved.affine)
        if aligned:
            roi_affine=canonical_ct.affine.copy(); roi_affine[:3,:3]=roi_affine[:3,:3]@np.diag(steps)
            roi_image=nib.Nifti1Image(roi.astype(np.uint8),roi_affine); roi_image.header.set_xyzt_units('mm')
            nib.save(roi_image,output/'roi'/path.name)
        records.append({'patient':case,'ct_path':str(ct_path),'input_sha256':hash_file(path),
                        'runtime_seconds':time.perf_counter()-started,'roi_quality':quality,**decision})
        print(case,decision['status'],'scores',[(x['component_id'],round(x['score'],3)) for x in decision['candidates']],flush=True)
        del scan,pred,cleaned,original_order,export
    manifest={'method':'automatic_ct_roi_v1','uses_ground_truth_for_selection':False,'experimental':True,
              'created_utc':datetime.now(timezone.utc).isoformat(),'input_dir':str(args.input_dir.resolve()),
              'config':config,'config_sha256':hashlib.sha256(config_bytes).hexdigest(),'patients':records,
              'limitations':['Heuristic lung ROI, not a validated organ segmentation','Ambiguity preserves all components, not necessarily correct ones',
                             'A confident score can still remove a real target','Cannot recover missed tissue or refine retained contours']}
    temporary=output/'postprocessing_manifest.json.tmp'; temporary.write_text(json.dumps(manifest,indent=2,allow_nan=False)); os.replace(temporary,output/'postprocessing_manifest.json')
    print('Output:',output,flush=True)


if __name__=='__main__': main()
