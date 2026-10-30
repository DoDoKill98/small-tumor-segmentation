"""Evaluate binary NIfTI predictions on annotated validation patients (physical mm)."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from surface_distance import metrics as surface_metrics

STORAGE=Path.home()/'Datasets/nnUNet'
DEFAULT_VALIDATION=STORAGE/'nnUNet_results/baseline_100epochs_fold0_20261029/Dataset006_Lung/nnUNetTrainer_100epochs__nnUNetPlans_6GB__3d_fullres/fold_0/validation'


def validate_geometry(ref_shape,ref_affine,pred_shape,pred_affine):
    if len(ref_shape)!=3 or ref_shape!=pred_shape:
        raise ValueError('Prediction/reference must have identical 3D shapes')
    if not np.allclose(ref_affine,pred_affine,rtol=0,atol=1e-4):
        raise ValueError('Prediction/reference affine mismatch; no automatic resampling')
    basis=np.asarray(ref_affine[:3,:3],dtype=float)
    spacing=np.linalg.norm(basis,axis=0)
    if not np.isfinite(ref_affine).all() or np.any(spacing<=0):
        raise ValueError('Invalid geometry')
    directions=basis/spacing
    if not np.allclose(directions.T@directions,np.eye(3),atol=1e-5):
        raise ValueError('Sheared geometry cannot be evaluated with axis spacings alone')
    return spacing


def crop_union(a,b):
    union=a|b
    if not union.any(): return a,b
    slices=[]
    for axis in range(3):
        coordinates=np.flatnonzero(union.any(axis=tuple(i for i in range(3) if i!=axis)))
        slices.append(slice(int(coordinates[0]),int(coordinates[-1])+1))
    return a[tuple(slices)],b[tuple(slices)]


def region_metrics(reference,prediction):
    structure=ndimage.generate_binary_structure(3,3)
    a,na=ndimage.label(reference,structure)
    b,nb=ndimage.label(prediction,structure)
    overlap=(a>0)&(b>0)
    detected=any_overlap=0
    if overlap.any():
        pairs=np.unique(np.column_stack((a[overlap]-1,b[overlap]-1)),axis=0)
        graph=csr_matrix((np.ones(len(pairs),dtype=np.int8),(pairs[:,0],pairs[:,1])),shape=(na,nb))
        detected=int(np.count_nonzero(maximum_bipartite_matching(graph,perm_type='column')>=0))
        any_overlap=len(np.unique(pairs[:,0]))
    return {'reference_regions':int(na),'predicted_regions':int(nb),
            'regions_detected':detected,'regions_missed':int(na-detected),
            'regions_unmatched_prediction':int(nb-detected),
            'reference_regions_with_any_overlap':int(any_overlap),
            'regions_completely_missed':int(na-any_overlap),
            'region_recall':detected/na if na else None,
            'region_precision':detected/nb if nb else None}


def compute_metrics(reference,prediction,spacing,tolerance_mm=2.):
    if reference.shape!=prediction.shape or reference.ndim!=3:
        raise ValueError('Expected matching 3D masks')
    spacing=np.asarray(spacing,float)
    if spacing.shape!=(3,) or not np.isfinite(spacing).all() or np.any(spacing<=0):
        raise ValueError('Expected 3 positive finite spacings in mm')
    if not np.isfinite(tolerance_mm) or tolerance_mm<0:
        raise ValueError('Invalid surface tolerance')
    a,b=crop_union(reference.astype(bool),prediction.astype(bool))
    nr,npred=int(a.sum()),int(b.sum())
    tp=int(np.count_nonzero(a&b)); fp=npred-tp; fn=nr-tp
    voxel_ml=float(np.prod(spacing)/1000.)
    result={'reference_voxels':nr,'predicted_voxels':npred,'tp_voxels':tp,'fp_voxels':fp,'fn_voxels':fn,
            'dice':2*tp/(nr+npred) if nr+npred else 1.,
            'voxel_recall':tp/nr if nr else None,'voxel_precision':tp/npred if npred else None,
            'reference_volume_ml':nr*voxel_ml,'predicted_volume_ml':npred*voxel_ml,
            'signed_volume_error_ml':(npred-nr)*voxel_ml,
            'absolute_volume_error_ml':abs(npred-nr)*voxel_ml,
            'relative_volume_error_percent':100*(npred-nr)/nr if nr else None,
            'empty_reference':nr==0,'empty_prediction':npred==0}
    if nr and npred:
        distances=surface_metrics.compute_surface_distances(a,b,spacing)
        result.update(surface_dice=float(surface_metrics.compute_surface_dice_at_tolerance(distances,tolerance_mm)),
                      hd95_mm=float(surface_metrics.compute_robust_hausdorff(distances,95)),surface_status='defined')
    elif nr+npred:
        result.update(surface_dice=0.,hd95_mm=None,surface_status='undefined_one_empty')
    else:
        result.update(surface_dice=1.,hd95_mm=0.,surface_status='both_empty')
    result.update(region_metrics(a,b))
    return result


def read_case(reference_path,prediction_path,tolerance_mm):
    ref,pred=nib.load(reference_path),nib.load(prediction_path)
    spacing=validate_geometry(ref.shape,ref.affine,pred.shape,pred.affine)
    for name,image in [('reference',ref),('prediction',pred)]:
        if image.header.get_xyzt_units()[0]!='mm':
            raise ValueError(f'{name} spatial unit must be explicitly mm')
    a,b=np.asarray(ref.dataobj),np.asarray(pred.dataobj)
    if not np.all((a==0)|(a==1)) or not np.all((b==0)|(b==1)):
        raise ValueError('Only binary labels 0 and 1 are supported; invalid/nonfinite values found')
    result=compute_metrics(a,b,spacing,tolerance_mm)
    result['spacing_mm']=spacing.tolist()
    result['shape']=list(ref.shape)
    return result


def summarize(cases,bootstrap_samples,seed):
    metrics=['dice','surface_dice','hd95_mm','voxel_recall','voxel_precision',
             'reference_volume_ml','predicted_volume_ml','signed_volume_error_ml',
             'absolute_volume_error_ml','relative_volume_error_percent','region_recall','region_precision']
    rng=np.random.default_rng(seed); summary={}
    for name in metrics:
        values=np.array([c[name] for c in cases if c[name] is not None],dtype=float)
        row={'n_total':len(cases),'n_defined':len(values),'n_undefined':len(cases)-len(values)}
        if len(values):
            row.update(mean=float(values.mean()),median=float(np.median(values)),
                       q25=float(np.percentile(values,25)),q75=float(np.percentile(values,75)),
                       std_sample=float(values.std(ddof=1)) if len(values)>1 else None)
            if bootstrap_samples and len(values)>1:
                means=rng.choice(values,size=(bootstrap_samples,len(values)),replace=True).mean(axis=1)
                row['bootstrap_mean_ci95']=np.percentile(means,[2.5,97.5]).tolist()
        summary[name]=row
    count_keys=['reference_regions','predicted_regions','regions_detected','regions_missed',
                'regions_completely_missed','regions_unmatched_prediction']
    return {'macro_by_patient':summary,'region_totals':{k:sum(c[k] for c in cases) for k in count_keys},
            'empty_predictions':sum(c['empty_prediction'] for c in cases),
            'empty_references':sum(c['empty_reference'] for c in cases)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prediction-dir',type=Path,default=DEFAULT_VALIDATION)
    parser.add_argument('--reference-dir',type=Path,default=STORAGE/'nnUNet_raw/Dataset006_Lung/labelsTr')
    parser.add_argument('--splits',type=Path,default=Path(__file__).resolve().parents[1]/'configs/nnunet/splits_final.json')
    parser.add_argument('--fold',type=int,default=0)
    parser.add_argument('--surface-tolerance-mm',type=float,default=2.)
    parser.add_argument('--checkpoint-label',default='unknown',help='Declared provenance; does not verify how predictions were generated')
    parser.add_argument('--bootstrap-samples',type=int,default=2000)
    parser.add_argument('--seed',type=int,default=12345)
    parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args()
    output=args.prediction_dir/'evaluation_metrics.json'
    if output.exists() and not args.overwrite: parser.error(f'{output} exists; use --overwrite to replace this report only')
    if args.bootstrap_samples<0 or not np.isfinite(args.surface_tolerance_mm) or args.surface_tolerance_mm<0:
        parser.error('Invalid bootstrap count or tolerance')
    split_bytes=args.splits.read_bytes(); splits=json.loads(split_bytes.decode('utf-8-sig'))
    if args.fold<0 or args.fold>=len(splits): parser.error('Fold out of range')
    split=splits[args.fold]; expected=set(split['val'])
    if not expected or len(expected)!=len(split['val']) or expected & set(split['train']): parser.error('Invalid validation split')
    actual={p.name.removesuffix('.nii.gz') for p in args.prediction_dir.glob('*.nii.gz')}
    if actual!=expected:
        parser.error(f'Validation patient mismatch. Missing={sorted(expected-actual)}, unexpected={sorted(actual-expected)}')
    cases=[]
    for patient in sorted(expected):
        ref=args.reference_dir/(patient+'.nii.gz'); pred=args.prediction_dir/(patient+'.nii.gz')
        if not ref.is_file(): parser.error(f'No public annotation for {patient}: {ref}')
        try: result=read_case(ref,pred,args.surface_tolerance_mm)
        except (ValueError,OSError) as exc: parser.error(f'{patient}: {exc}')
        cases.append({'patient':patient,**result})
        hd=f'{result["hd95_mm"]:.2f}' if result['hd95_mm'] is not None else 'undefined'
        print(f'{patient}: Dice={result["dice"]:.4f} SurfaceDice={result["surface_dice"]:.4f} HD95={hd} mm',flush=True)
    report={'schema_version':1,'created_utc':datetime.now(timezone.utc).isoformat(),
            'prediction_dir':str(args.prediction_dir.resolve()),'reference_dir':str(args.reference_dir.resolve()),
            'checkpoint_label':args.checkpoint_label,'checkpoint_provenance':'user_declared_not_verified_against_masks',
            'fold':args.fold,'split_sha256':hashlib.sha256(split_bytes).hexdigest(),
            'surface_tolerance_mm':args.surface_tolerance_mm,
            'definitions':{'surface_dice':'Surfel-area weighted, symmetric, distances <= tolerance',
                           'hd95':'Maximum of the two directed area-weighted 95th percentile surface distances, mm',
                           'regions':'26-connectivity; maximum-cardinality one-to-one matching; any nonzero voxel overlap',
                           'regions_missed':'Unmatched reference components (including merges); completely_missed reports zero overlap separately',
                           'volume_error':'prediction minus reference, mL',
                           'empty_masks':'Both empty: Dice/SurfaceDice=1, HD95=0. One empty: Dice/SurfaceDice=0, HD95=null. Precision/recall=null when denominator zero',
                           'summary':'Macro average: equal patient weights. Undefined values excluded with explicit denominator. Bootstrap CIs conditional on defined values'},
            'bootstrap':{'samples':args.bootstrap_samples,'seed':args.seed,'unit':'patient','method':'percentile bootstrap of mean'},
            'limitations':['Validation used for model/checkpoint selection; not an independent test estimate',
                           'Connected components are operational regions, not confirmed distinct clinical lesions',
                           'Any-overlap region matching is permissive; additional regions are relative to available labels'],
            'versions':{n:importlib.metadata.version(n) for n in ['numpy','scipy','nibabel','surface-distance']},
            'patients':cases,'summary':summarize(cases,args.bootstrap_samples,args.seed)}
    manifest_path=args.prediction_dir/'postprocessing_manifest.json'
    if manifest_path.is_file():
        selection_manifest=json.loads(manifest_path.read_text())
        uses_gt=selection_manifest.get('uses_ground_truth_for_selection',False)
        report['postprocessing']={'method':selection_manifest.get('method'),
                                 'uses_ground_truth_for_selection':uses_gt,
                                 'manifest_path':str(manifest_path.resolve())}
        if uses_gt:
            report['limitations'].append('ORACLE: ground truth used to select outputs; diagnostic only, not deployable performance')
    temporary=output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    os.replace(temporary,output)
    print('Report:',output,flush=True)
    for name in ['dice','surface_dice','hd95_mm']:
        print(name,report['summary']['macro_by_patient'][name],flush=True)


if __name__=='__main__': main()
