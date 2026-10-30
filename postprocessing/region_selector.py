"""Feature-only oracle-supervised region selector. No reference-mask input here."""
from pathlib import Path
import hashlib
import json
import nibabel as nib
import numpy as np
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES=['log1p_volume_ml','relative_volume','center_R_fraction','center_A_fraction','center_S_fraction',
               'probability_mean','probability_p10','entropy_bits_mean','fraction_probability_ge_090','bbox_fill_fraction']
MODEL_CONFIG={'model':'L2 logistic regression','C':0.1,'class_weight':'balanced','solver':'lbfgs','max_iter':2000,'seed':12345,
              'patient_weighting':'equal base patient sample weights, then sklearn class balancing; final effective patient totals may differ','hyperparameter_search':False}


def file_hash(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def candidate_features(candidate,patient_candidates,shape,affine):
    corners=np.array([[i,j,k] for i in (0,shape[0]-1) for j in (0,shape[1]-1) for k in (0,shape[2]-1)])
    world=nib.affines.apply_affine(affine,corners)
    low,high=world.min(axis=0),world.max(axis=0)
    fraction=(np.asarray(candidate['centroid_world_mm'])-low)/np.maximum(high-low,1e-8)
    if np.any(fraction < -1e-5) or np.any(fraction>1+1e-5): raise ValueError('Candidate centroid outside CT field of view')
    maximum=max(c['volume_ml'] for c in patient_candidates)
    box=candidate['bounding_box_xyz_exclusive']; box_volume=np.prod([b-a for a,b in box])
    probability=candidate['tumor_probability']
    values=[np.log1p(candidate['volume_ml']),candidate['volume_ml']/maximum,*fraction.tolist(),
            probability['mean'],probability['quantiles']['p10'],probability['entropy_bits_mean'],
            probability['fraction_ge_0_90'],probability['voxel_count']/box_volume]
    result=np.asarray(values,float)
    if not np.isfinite(result).all(): raise ValueError('Nonfinite model features')
    return result


def load_feature_rows(stats_path):
    stats_path=Path(stats_path); stats=json.loads(stats_path.read_text()); root=stats_path.parent
    if stats.get('uses_ground_truth') or stats.get('automatic_selection_applied'): raise ValueError('Expected raw probability features without annotation selection')
    patients=stats['patients']; names=[c['patient'] for c in patients]
    if len(names)!=len(set(names)): raise ValueError('Duplicate patient records')
    rows=[]; groups=[]; matrix=[]
    for patient in patients:
        name=patient['patient']; path=root/(name+'.nii.gz')
        if Path(name).name!=name or not path.is_file(): raise ValueError('Missing/invalid predicted mask')
        if file_hash(path)!=patient['mask_sha256']: raise ValueError(f'Prediction changed since statistics: {name}')
        image=nib.load(path)
        if list(image.shape)!=patient['shape_xyz']: raise ValueError('Shape differs from statistics')
        ids=[r['component_id'] for r in patient['regions']]
        if len(ids)!=len(set(ids)): raise ValueError('Duplicate component IDs')
        for candidate in patient['regions']:
            values=candidate_features(candidate,patient['regions'],image.shape,image.affine)
            matrix.append(values); groups.append(name)
            rows.append({'patient':name,'component_id':candidate['component_id'],
                         'features':dict(zip(FEATURE_NAMES,values.tolist()))})
    features=np.stack(matrix) if matrix else np.empty((0,len(FEATURE_NAMES)),float)
    return features,np.asarray(groups),rows,stats


def score_candidate_rows(pipeline,X):
    return pipeline.predict_proba(X)[:,1] if len(X) else np.empty(0,float)


def fit_selector(X,y,groups):
    if set(np.unique(y))!={0,1}: raise ValueError('Training requires positive and negative oracle targets')
    counts={g:int(np.sum(groups==g)) for g in np.unique(groups)}
    weights=np.asarray([1/counts[g] for g in groups])
    model=Pipeline([('scaler',StandardScaler()),('classifier',LogisticRegression(
        C=MODEL_CONFIG['C'],class_weight='balanced',solver='lbfgs',max_iter=2000,random_state=12345))])
    model.fit(X,y,scaler__sample_weight=weights,classifier__sample_weight=weights)
    return model


def leave_one_patient_out(X,y,groups):
    X=np.asarray(X,float); y=np.asarray(y); groups=np.asarray(groups)
    scores=np.full(len(y),np.nan); audit=[]
    for held_out in sorted(np.unique(groups)):
        training=groups!=held_out; testing=~training
        model=fit_selector(X[training],y[training],groups[training])
        scores[testing]=model.predict_proba(X[testing])[:,1]
        audit.append({'held_out_patient':str(held_out),'training_patients':sorted(np.unique(groups[training]).tolist()),
                      'training_rows':int(training.sum()),'held_out_rows':int(testing.sum()),
                      'scaler_training_mean':model.named_steps['scaler'].mean_.tolist()})
    if not np.isfinite(scores).all(): raise ValueError('Incomplete held-out scores')
    return scores,audit


def export_component(source,component_id,destination):
    image=nib.load(source); values=np.asarray(image.dataobj)
    if values.ndim!=3 or not np.all((values==0)|(values==1)): raise ValueError('Expected binary 3D mask')
    mask=values>0; output=np.zeros(values.shape,np.uint8)
    if component_id is not None:
        bounds=[]
        for axis in range(3):
            idx=np.flatnonzero(mask.any(axis=tuple(i for i in range(3) if i!=axis)))
            if not len(idx): raise ValueError('Selected component in an empty mask')
            bounds.append(slice(int(idx[0]),int(idx[-1])+1))
        labels,n=ndimage.label(mask[tuple(bounds)],ndimage.generate_binary_structure(3,3))
        if component_id<1 or component_id>n: raise ValueError('Selected component ID out of range')
        output[tuple(bounds)]=labels==component_id
    header=image.header.copy(); header.set_data_dtype(np.uint8)
    exported=nib.Nifti1Image(output,image.affine,header)
    for getter,setter in [(image.get_qform,exported.set_qform),(image.get_sform,exported.set_sform)]:
        matrix,code=getter(coded=True)
        if matrix is not None: setter(matrix,int(code))
    nib.save(exported,destination)
    saved=nib.load(destination)
    if saved.shape!=image.shape or not np.allclose(saved.affine,image.affine,rtol=0,atol=1e-4): raise ValueError('Export changed geometry')
