"""VTK ranking viewer: keep every predicted region and inspect selector scores."""
import argparse
import hashlib
import json
import textwrap
from pathlib import Path
import nibabel as nib
import numpy as np
from scipy import ndimage
import vtk
from view_comparison import ComparisonViewer, check_binary
from view_prediction import find_scan, STORAGE
from view_scan import load_pair, to_vtk

DEFAULT_PREDICTIONS=STORAGE/'nnUNet_results/baseline_100epochs_fold0_20261029/Dataset006_Lung/nnUNetTrainer_100epochs__nnUNetPlans_6GB__3d_fullres/fold_0/validation_best_probabilities'
DEFAULT_SCORES=STORAGE/'experiments/region_selector_v1/loo_predictions/postprocessing_manifest.json'


def label_prediction(mask):
    if mask.ndim!=3 or not np.all((mask==0)|(mask==1)): raise ValueError('Expected binary 3D prediction')
    labels,n=ndimage.label(mask,ndimage.generate_binary_structure(3,3))
    dtype=np.uint8 if 2*n+2<=255 else np.uint16 if 2*n+2<=65535 else np.uint32
    return labels.astype(dtype),int(n)


def ranked_rows(candidates,statistics,count):
    scores={r['component_id']:r['selector_score'] for r in candidates}
    stats={r['component_id']:r for r in statistics}
    expected=set(range(1,count+1))
    if len(scores)!=len(candidates) or len(stats)!=len(statistics) or set(scores)!=expected or set(stats)!=expected:
        raise ValueError('Scores/statistics do not match every raw predicted component')
    if not all(np.isfinite(s) and 0<=s<=1 for s in scores.values()): raise ValueError('Invalid selector score')
    return sorted([{'component_id':i,'selector_score':float(scores[i]),**{k:v for k,v in stats[i].items() if k!='component_id'}} for i in expected],
                  key=lambda r:(-r['selector_score'],r['component_id']))


def display_codes(labels,ground_truth,show_predictions,show_ground_truth,offset):
    result=labels.copy() if show_predictions else np.zeros_like(labels)
    if show_ground_truth: result=result+(ground_truth>0).astype(labels.dtype)*offset
    return result


def file_hash(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


class ScoreViewer(ComparisonViewer):
    def __init__(self,scan,ground_truth,prediction,region_labels,spacing,origin,rows,title):
        self.has_reference=ground_truth is not None
        reference=ground_truth if ground_truth is not None else np.zeros(prediction.shape,np.uint8)
        super().__init__(scan,reference,prediction,spacing,origin,title)
        self.region_labels=region_labels; self.rows=rows; self.active_rank=0
        self.offset=len(rows)+1
        self.gt_visible=self.has_reference
        self.score_lut=vtk.vtkLookupTable(); self.score_lut.SetNumberOfTableValues(2*self.offset)
        self.score_lut.SetRange(0,2*self.offset-1)
        self.colors.SetLookupTable(self.score_lut)
        self.renderers[3].RemoveActor(self.surface)
        label_image=to_vtk(region_labels,spacing,origin)
        self.all_region_surface=vtk.vtkActor()
        if rows:
            contour=vtk.vtkDiscreteMarchingCubes(); contour.SetInputData(label_image)
            contour.GenerateValues(len(rows),1,len(rows)); contour.Update()
            mapper=vtk.vtkPolyDataMapper(); mapper.SetInputConnection(contour.GetOutputPort())
            mapper.SetLookupTable(self.score_lut); mapper.SetScalarRange(0,2*self.offset-1)
            mapper.SetScalarModeToUseCellData(); mapper.ScalarVisibilityOn()
            self.all_region_surface.SetMapper(mapper); self.all_region_surface.GetProperty().SetOpacity(.85)
            self.renderers[3].AddActor(self.all_region_surface)
        self.window.SetSize(1800,1000)
        self.refresh_colors()
        if rows:
            self.focus_active(reset=True)
            self.initial=self.indices.copy()
        self.update(); self.reset_cameras()

    def update_palette(self):
        n=len(self.rows)
        best=self.rows[0]['component_id'] if n else None
        active=self.rows[self.active_rank]['component_id'] if n else None
        self.score_lut.SetTableValue(0,0,0,0,0)
        self.score_lut.SetTableValue(self.offset,0,1,.15,.6)
        for i in range(1,n+1):
            color=(1.,.65,0.) if i==best else (0.,.8,1.) if i==active else (1.,.08,.12)
            self.score_lut.SetTableValue(i,*color,.65)
            self.score_lut.SetTableValue(self.offset+i,1.,1.,0.,.65)
        self.score_lut.Build()

    def refresh_colors(self):
        if not hasattr(self,'region_labels'): return super().refresh_colors()
        self.update_palette()
        codes=display_codes(self.region_labels,self.ground_truth,self.visible,self.gt_visible,self.offset)
        self.comparison_image=to_vtk(codes,self.spacing,self.origin)
        self.colors.SetInputData(self.comparison_image); self.colors.UpdateWholeExtent()
        # Freeze RGBA output: three slice mappers must not negotiate conflicting
        # update extents through the same vtkImageMapToColors pipeline.
        self.rgba_image=vtk.vtkImageData(); self.rgba_image.DeepCopy(self.colors.GetOutput())
        for overlay in self.overlays: overlay.GetMapper().SetInputData(self.rgba_image)

    def update(self):
        super().update()
        if not hasattr(self,'region_labels'): return
        self.gt_surface.SetVisibility(self.has_reference and self.gt_visible)
        self.all_region_surface.SetVisibility(self.visible and bool(self.rows))
        for i,axis in enumerate(self.axes):
            self.texts[i].SetInput(f'{self.names[i]} | {self.indices[axis]+1}/{self.shape[axis]}\nOrange: meilleur score | Rouge: autres\nCyan: region parcourue | Jaune: recouvrement GT')
        start=max(0,self.active_rank-5); shown=self.rows[start:start+12]
        table=['REGIONS - TOUTES CONSERVEES','  ID   Score   Volume mL']
        for row in shown:
            index=self.rows.index(row)
            marker='>' if index==self.active_rank else ' '
            star='*' if index==0 else ' '
            table.append(f'{marker}{star}{row["component_id"]:2d}   {row["selector_score"]:.3f}   {row["volume_ml"]:.3f}')
        if not self.rows: table.append('Aucune region predite')
        table.extend(['','* = score maximal, pas verite terrain','N / B : parcourir les regions','Home : meilleure region | T : GT',
                      'M : predictions | G : annotation','P : plans | R : reset | Q : quitter','Glisser : tourner | molette : coupe/zoom',
                      'Vert : annotation | Jaune : recouvrement','Scores de classement non calibres'])
        if not self.has_reference: table.append('Annotation non disponible')
        wrapped=[part for line in table for part in (textwrap.wrap(line,width=32,break_long_words=False,replace_whitespace=False) if len(line)>32 else [line])]
        self.texts[3].SetInput('\n'.join(wrapped)); self.texts[3].GetTextProperty().SetFontSize(15)

    def focus_active(self,reset=False):
        if not self.rows: return
        center=np.asarray(self.rows[self.active_rank]['centroid_world_mm'])
        self.indices=[int(np.clip(round((center[k]-self.origin[k])/self.spacing[k]),0,self.shape[k]-1)) for k in range(3)]
        self.refresh_colors(); self.update()
        if reset: self.reset_cameras()
        for ren in self.renderers+self.original_renderers: ren.ResetCameraClippingRange()
        self.window.Render()

    def key(self,obj,event):
        key=self.interactor.GetKeySym().lower()
        if key in ('n','b','home') and self.rows:
            self.active_rank=0 if key=='home' else (self.active_rank+(1 if key=='n' else -1))%len(self.rows)
            self.focus_active(); return
        if key=='t' and self.has_reference:
            points=np.argwhere(self.ground_truth>0)
            if len(points): self.indices=[int(v) for v in np.median(points,axis=0)]
            self.update(); self.window.Render(); return
        super().key(obj,event)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',default='lung_006')
    parser.add_argument('--prediction-dir',type=Path,default=DEFAULT_PREDICTIONS,help='RAW masks, not one-region selector outputs')
    parser.add_argument('--score-manifest',type=Path,default=DEFAULT_SCORES)
    parser.add_argument('--data-root',type=Path,default=STORAGE/'nnUNet_raw/Dataset006_Lung')
    parser.add_argument('--ground-truth-dir',type=Path)
    parser.add_argument('--screenshot',type=Path)
    args=parser.parse_args(); case=args.case.removesuffix('.nii.gz').removesuffix('_0000')
    if not case or Path(case).name!=case: parser.error('Invalid patient ID')
    try:
        prediction=args.prediction_dir/(case+'.nii.gz'); check_binary(prediction)
        stats_path=args.prediction_dir/'region_probability_stats.json'
        statistics=json.loads(stats_path.read_text()); scores=json.loads(args.score_manifest.read_text())
        source_hash=scores.get('source_probability_stats_sha256',scores.get('source_statistics_sha256'))
        if not source_hash or source_hash!=file_hash(stats_path): raise ValueError('Score manifest comes from different probability statistics')
        matches=[p for p in statistics['patients'] if p['patient']==case]
        decisions=[p for p in scores['patients'] if p['patient']==case]
        if len(matches)!=1 or len(decisions)!=1: raise ValueError('Expected exactly one score/statistics entry for this patient')
        patient=matches[0]
        if patient['mask_sha256']!=file_hash(prediction): raise ValueError('Raw prediction differs from the score source')
        raw=nib.load(prediction); labels,count=label_prediction(np.asarray(raw.dataobj))
        rows=ranked_rows(decisions[0]['candidates'],patient['regions'],count)
        image=find_scan(args.data_root,case)
        scan,pred,spacing,origin=load_pair(image,prediction)
        labeled=nib.as_closest_canonical(nib.Nifti1Image(labels,raw.affine))
        del labels
        target_affine=np.diag([*spacing,1.]); target_affine[:3,3]=origin
        if labeled.shape!=scan.shape or not np.allclose(labeled.affine,target_affine,atol=1e-4):
            from nibabel.processing import resample_from_to
            labeled=resample_from_to(labeled,(scan.shape,target_affine),order=0)
        display_labels=np.asarray(labeled.dataobj).astype(np.uint8 if 2*count+2<=255 else np.uint16 if 2*count+2<=65535 else np.uint32)
        if set(np.unique(display_labels))-{0}!=set(range(1,count+1)):
            raise ValueError('Display resampling loses a region; refusing to silently hide it')
        gt_path=(args.ground_truth_dir or args.data_root/'labelsTr')/(case+'.nii.gz')
        gt=None
        if gt_path.is_file():
            check_binary(gt_path); second_scan,gt,s2,o2=load_pair(image,gt_path); del second_scan
            if gt.shape!=pred.shape or not np.allclose(spacing,s2) or not np.allclose(origin,o2): raise ValueError('Ground-truth display grid differs')
    except (ValueError,OSError) as exc: parser.error(str(exc))
    print('Raw prediction:',prediction,'\nScore source:',args.score_manifest,flush=True)
    print('All',count,'regions retained. Highest selector score does not guarantee a correct target.',flush=True)
    for row in rows: print('ID',row['component_id'],'score',row['selector_score'],'volume mL',row['volume_ml'],flush=True)
    viewer=ScoreViewer(scan,gt,pred,display_labels,spacing,origin,rows,f'Small tumor | {case} | ALL regions + selector scores')
    if args.screenshot:
        viewer.window.SetOffScreenRendering(1); viewer.window.Render()
        capture=vtk.vtkWindowToImageFilter(); capture.SetInput(viewer.window); capture.ReadFrontBufferOff(); capture.Update()
        args.screenshot.parent.mkdir(parents=True,exist_ok=True)
        writer=vtk.vtkPNGWriter(); writer.SetFileName(str(args.screenshot)); writer.SetInputConnection(capture.GetOutputPort()); writer.Write()
        viewer.window.Finalize(); print('Saved:',args.screenshot)
    else:
        viewer.interactor.Initialize(); viewer.window.Render(); viewer.interactor.Start(); viewer.window.Finalize()


if __name__=='__main__': main()
