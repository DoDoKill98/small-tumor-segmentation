"""VTK comparison: ground truth green, prediction red, overlapping voxels yellow."""
import argparse
from pathlib import Path
import nibabel as nib
import numpy as np
import vtk
from view_prediction import PredictionViewer, find_scan, STORAGE
from view_scan import load_pair, to_vtk

DEFAULT_VALIDATION=STORAGE/'nnUNet_results/baseline_100epochs_fold0_20261029/Dataset006_Lung/nnUNetTrainer_100epochs__nnUNetPlans_6GB__3d_fullres/fold_0/validation'


def comparison_labels(ground_truth,prediction,show_gt=True,show_prediction=True):
    if ground_truth.shape!=prediction.shape:
        raise ValueError('Mask shapes differ')
    return ((ground_truth>0).astype(np.uint8)*int(show_gt)
            +2*(prediction>0).astype(np.uint8)*int(show_prediction))


class ComparisonViewer(PredictionViewer):
    def __init__(self,scan,ground_truth,prediction,spacing,origin,title):
        super().__init__(scan,prediction,spacing,origin,title)
        self.ground_truth=ground_truth
        self.prediction=prediction
        self.gt_visible=True
        self.comparison_lut=vtk.vtkLookupTable()
        self.comparison_lut.SetNumberOfTableValues(4); self.comparison_lut.SetRange(0,3)
        for i,color in enumerate([(0,0,0,0),(0,1,.15,.6),(1,.05,.05,.6),(1,1,0,.65)]):
            self.comparison_lut.SetTableValue(i,*color)
        self.comparison_lut.Build()
        self.colors.SetLookupTable(self.comparison_lut)
        self.surface.GetProperty().SetColor(1,.05,.05)
        self.surface.GetProperty().SetOpacity(.6)
        self.gt_image=to_vtk(ground_truth.astype(np.uint8),spacing,origin)
        contour=vtk.vtkDiscreteMarchingCubes(); contour.SetInputData(self.gt_image)
        contour.SetValue(0,1); contour.Update()
        mapper=vtk.vtkPolyDataMapper(); mapper.SetInputConnection(contour.GetOutputPort()); mapper.ScalarVisibilityOff()
        self.gt_surface=vtk.vtkActor(); self.gt_surface.SetMapper(mapper)
        self.gt_surface.GetProperty().SetColor(0,1,.15)
        self.gt_surface.GetProperty().SetOpacity(.6)
        self.renderers[3].AddActor(self.gt_surface)
        # Prefer the reference location for initial inspection, also when prediction misses it.
        coordinates=np.argwhere(ground_truth>0)
        if len(coordinates):
            self.initial=[int(v) for v in np.median(coordinates,axis=0)]
            self.indices=self.initial.copy()
        self.refresh_colors(); self.update(); self.reset_cameras()

    def refresh_colors(self):
        labels=comparison_labels(self.ground_truth,self.prediction,self.gt_visible,self.visible)
        self.comparison_image=to_vtk(labels,self.spacing,self.origin)
        self.colors.SetInputData(self.comparison_image); self.colors.Update()

    def update(self):
        super().update()
        if not hasattr(self,'ground_truth'): return
        for i,axis in enumerate(self.axes):
            self.overlays[i].SetVisibility(self.visible or self.gt_visible)
            self.texts[i].SetInput(f'{self.names[i]} | {self.indices[axis]+1}/{self.shape[axis]}\nVert: GT seule | Rouge: prediction seule\nJaune: recouvrement | G: GT | M: prediction')
            self.texts[i].GetTextProperty().SetFontSize(13)
        self.gt_surface.SetVisibility(self.gt_visible)
        self.texts[3].SetInput('COMPARAISON 3D\nVert: annotation reelle\nRouge: prediction\nG / M: masquer les surfaces\nP: plans | R: reset\nGlisser: tourner | molette: zoom\nRepere RAS, unites mm')

    def key(self,obj,event):
        key=self.interactor.GetKeySym().lower()
        if key in ('g','m'):
            if key=='g': self.gt_visible=not self.gt_visible
            else: self.visible=not self.visible
            self.refresh_colors(); self.update(); self.window.Render()
        else:
            super().key(obj,event)


def check_binary(path):
    values=np.asarray(nib.load(path).dataobj)
    if not np.all((values==0)|(values==1)):
        raise ValueError(f'Expected binary finite labels 0 and 1: {path}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',help='Patient ID; inferred from --prediction-file, otherwise lung_006')
    parser.add_argument('--data-root',type=Path,default=STORAGE/'nnUNet_raw/Dataset006_Lung')
    parser.add_argument('--ground-truth-dir',type=Path)
    prediction_source=parser.add_mutually_exclusive_group()
    prediction_source.add_argument('--prediction-dir',type=Path,default=DEFAULT_VALIDATION)
    prediction_source.add_argument('--prediction-file',type=Path,help='Direct path to a predicted NIfTI mask, including oracle output')
    parser.add_argument('--screenshot',type=Path)
    args=parser.parse_args()
    if args.prediction_file and not args.prediction_file.name.endswith('.nii.gz'):
        parser.error('--prediction-file must be a .nii.gz mask')
    file_case=args.prediction_file.name.removesuffix('.nii.gz') if args.prediction_file else None
    case=(args.case or file_case or 'lung_006').removesuffix('.nii.gz').removesuffix('_0000')
    if file_case is not None and file_case!=case:
        parser.error('--case does not match the patient name in --prediction-file')
    if not case or Path(case).name!=case: parser.error('--case must be a patient ID')
    try:
        image=find_scan(args.data_root,case)
        ground_truth=(args.ground_truth_dir or args.data_root/'labelsTr')/(case+'.nii.gz')
        prediction=args.prediction_file or args.prediction_dir/(case+'.nii.gz')
        for path in (ground_truth,prediction):
            if not path.is_file(): raise ValueError(f'Mask not found: {path}. Official test patients have no public ground truth.')
            check_binary(path)
        scan,pred,spacing,origin=load_pair(image,prediction)
        second_scan,gt,gt_spacing,gt_origin=load_pair(image,ground_truth)
        del second_scan
        if gt.shape!=pred.shape or not np.allclose(spacing,gt_spacing) or not np.allclose(origin,gt_origin):
            raise ValueError('Display grids differ')
    except (ValueError,OSError) as exc:
        parser.error(str(exc))
    print('Original CT:',image,'\nGround truth (green):',ground_truth,'\nPrediction (red):',prediction,flush=True)
    source_label='ORACLE' if prediction.parent.name=='oracle_best_overlap' else prediction.parent.name
    viewer=ComparisonViewer(scan,gt,pred,spacing,origin,f'Small tumor | {case} | GT vs prediction | {source_label}')
    if args.screenshot:
        viewer.window.SetOffScreenRendering(1); viewer.window.Render()
        capture=vtk.vtkWindowToImageFilter(); capture.SetInput(viewer.window); capture.ReadFrontBufferOff(); capture.Update()
        args.screenshot.parent.mkdir(parents=True,exist_ok=True)
        writer=vtk.vtkPNGWriter(); writer.SetFileName(str(args.screenshot)); writer.SetInputConnection(capture.GetOutputPort()); writer.Write()
        viewer.window.Finalize(); print('Saved:',args.screenshot)
    else:
        viewer.interactor.Initialize(); viewer.window.Render(); viewer.interactor.Start(); viewer.window.Finalize()


if __name__=='__main__': main()
