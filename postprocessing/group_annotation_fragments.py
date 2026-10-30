"""Group nearby annotation fragments in physical mm without editing the masks."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


def group_fragments(values,affine,max_distance_mm):
    if values.ndim!=3 or not np.all((values==0)|(values==1)): raise ValueError('Expected binary finite 3D annotation')
    if not np.isfinite(max_distance_mm) or max_distance_mm<0: raise ValueError('Invalid distance threshold')
    affine=np.asarray(affine,float)
    if affine.shape!=(4,4) or not np.isfinite(affine).all(): raise ValueError('Invalid affine')
    voxel_mm3=abs(float(np.linalg.det(affine[:3,:3])))
    if voxel_mm3<=0: raise ValueError('Singular affine')
    mask=values>0
    if not mask.any(): return {'fragment_count':0,'group_count':0,'groups':[],'fragment_distances':[]}
    slices=[]
    for axis in range(3):
        indices=np.flatnonzero(mask.any(axis=tuple(i for i in range(3) if i!=axis)))
        slices.append(slice(int(indices[0]),int(indices[-1])+1))
    offset=np.array([s.start for s in slices]); mask=mask[tuple(slices)]
    labels,n=ndimage.label(mask,ndimage.generate_binary_structure(3,3))
    sizes=np.bincount(labels.ravel(),minlength=n+1)[1:]
    boxes=ndimage.find_objects(labels)
    trees=[]; points=[]
    for i,box in enumerate(boxes):
        fragment=labels[box]==i+1
        boundary=fragment & ~ndimage.binary_erosion(fragment,border_value=0)
        coordinates=np.argwhere(boundary)+offset+np.array([s.start for s in box])
        physical=nib.affines.apply_affine(affine,coordinates)
        points.append(physical); trees.append(cKDTree(physical))
    parent=list(range(n))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]; i=parent[i]
        return i
    distances=[]
    for i in range(n):
        for j in range(i+1,n):
            distance=float(np.min(trees[i].query(points[j],k=1)[0]))
            linked=distance<=max_distance_mm+1e-8
            if linked: parent[find(j)]=find(i)
            distances.append({'fragment_a':i+1,'fragment_b':j+1,'distance_mm':distance,'linked':linked})
    members={}
    for i in range(n): members.setdefault(find(i),[]).append(i)
    groups=[]
    for member_ids in members.values():
        count=int(sum(sizes[i] for i in member_ids))
        groups.append({'fragment_ids':[i+1 for i in member_ids],'voxel_count':count,
                       'volume_ml':count*voxel_mm3/1000.,'fraction_of_annotation':count/int(mask.sum())})
    groups.sort(key=lambda g:-g['voxel_count'])
    for rank,g in enumerate(groups,1): g['rank_by_volume']=rank
    return {'fragment_count':int(n),'group_count':len(groups),'groups':groups,'fragment_distances':distances}


def main():
    project=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels-dir',type=Path,default=Path.home()/'Datasets/nnUNet/nnUNet_raw/Dataset006_Lung/labelsTr')
    parser.add_argument('--max-distance-mm',type=float,required=True,help='Experimental grouping threshold, not a clinical lesion definition')
    parser.add_argument('--output',type=Path,default=project/'postprocessing/reports/grouped_annotations.json')
    args=parser.parse_args()
    if not np.isfinite(args.max_distance_mm) or args.max_distance_mm<0: parser.error('Distance must be finite and nonnegative')
    files=sorted(args.labels_dir.glob('*.nii.gz'))
    if not files: parser.error('No annotation files found')
    description=args.labels_dir.parent/'dataset.json'
    if description.exists() and json.loads(description.read_text()).get('numTraining',len(files))!=len(files):
        parser.error('Annotation count does not match dataset.json')
    cases=[]
    for i,path in enumerate(files):
        image=nib.load(path)
        if image.header.get_xyzt_units()[0]!='mm': parser.error(f'{path.name}: spatial unit must be mm')
        try: result=group_fragments(np.asarray(image.dataobj),image.affine,args.max_distance_mm)
        except ValueError as exc: parser.error(f'{path.name}: {exc}')
        cases.append({'patient':path.name.removesuffix('.nii.gz'),**result})
        if (i+1)%10==0: print('Checked',i+1,'/',len(files),flush=True)
    summary={'patients_checked':len(cases),'single_group_patients':sum(c['group_count']==1 for c in cases),
             'group_count_histogram':dict(Counter(c['group_count'] for c in cases)),
             'patients_requiring_visual_review':[c['patient'] for c in cases if c['group_count']!=1]}
    report={'created_utc':datetime.now(timezone.utc).isoformat(),'labels_dir':str(args.labels_dir.resolve()),
            'method':'26-connected fragments grouped by transitive physical proximity; no mask edits and no fragment removal',
            'max_distance_mm':args.max_distance_mm,
            'distance_definition':'Minimum Euclidean distance between centers of boundary voxels, transformed by full NIfTI affine, in mm',
            'limitations':['Geometric groups are not confirmed anatomical lesions',
                           'The chosen proximity threshold is experimental and independent of Surface Dice tolerance',
                           'Transitive links may connect endpoints farther apart than the threshold'],
            'summary':summary,'patients':cases}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.output.with_suffix('.json.tmp'); temporary.write_text(json.dumps(report,indent=2,allow_nan=False)); os.replace(temporary,args.output)
    print(json.dumps(summary),flush=True); print('Report:',args.output,flush=True)


if __name__=='__main__': main()
