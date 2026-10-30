"""Compare an unchanged CT with its predicted mask in synchronized VTK views."""
import argparse
from pathlib import Path
import nibabel as nib
import numpy as np
import vtk
from view_scan import Viewer, load_pair

STORAGE = Path.home()/'Datasets/nnUNet'


def find_scan(root, case):
    candidates=[root/split/(case+suffix) for split in ('imagesTs','imagesTr')
                for suffix in ('_0000.nii.gz','.nii.gz')]
    found=[p for p in candidates if p.is_file()]
    if len(found)!=1:
        raise ValueError(f'Expected one original CT for {case}, found {len(found)} in {root}')
    return found[0]


class PredictionViewer(Viewer):
    def __init__(self, scan, prediction, spacing, origin, title):
        super().__init__(scan,prediction,spacing,origin,title)
        self.window.SetSize(1600,900)
        self.renderers[3].SetViewport(.75,0,1,1)
        self.original_actors=[]
        self.original_texts=[]
        self.original_renderers=[]
        for i in range(3):
            self.renderers[i].SetViewport(i*.25,0,(i+1)*.25,.5)
            ren=vtk.vtkRenderer(); ren.SetViewport(i*.25,.5,(i+1)*.25,1)
            ren.SetBackground(.045,.055,.075)
            self.window.AddRenderer(ren); self.original_renderers.append(ren)
            actor=vtk.vtkImageActor(); actor.SetInputData(self.scan)
            actor.GetProperty().SetColorWindow(1400); actor.GetProperty().SetColorLevel(-300)
            actor.InterpolateOff(); ren.AddActor(actor); self.original_actors.append(actor)
            text=vtk.vtkTextActor(); text.SetPosition(8,10)
            text.GetTextProperty().SetFontSize(14); ren.AddViewProp(text)
            self.original_texts.append(text)
            # Shared camera guarantees the same zoom and orientation for each pair.
            ren.SetActiveCamera(self.renderers[i].GetActiveCamera())
            self.texts[i].GetTextProperty().SetFontSize(14)
        self.update(); self.reset_cameras()

    def update(self):
        super().update()
        if not hasattr(self,'original_actors'): return
        for i,actor in enumerate(self.original_actors):
            actor.SetDisplayExtent(*self.actors[i].GetDisplayExtent())
            axis=self.axes[i]
            section=f'{self.names[i]} | {self.indices[axis]+1}/{self.shape[axis]}'
            self.original_texts[i].SetInput(f'CT ORIGINAL | {section}\nMolette: coupe | R: reset | Q: quitter')
            self.texts[i].SetInput(f'CT + PREDICTION | {section}\nOrange: prediction | M: masquer')
        self.texts[3].SetInput('PREDICTION 3D\nOrange: masque du reseau\nGlisser: tourner\nMolette: zoom | P: plans\nRepere RAS, distances en mm\nPas de verite terrain test')
        self.texts[3].GetTextProperty().SetFontSize(14)

    def scroll(self, delta):
        x,_=self.interactor.GetEventPosition(); w,_=self.window.GetSize()
        panel=min(3,int(4*x/max(w,1)))
        if panel==3:
            self.renderers[3].GetActiveCamera().Dolly(1.1 if delta>0 else 1/1.1)
        else:
            axis=self.axes[panel]
            self.indices[axis]=max(0,min(self.shape[axis]-1,self.indices[axis]+delta))
            self.update()
        for ren in self.renderers+self.original_renderers: ren.ResetCameraClippingRange()
        self.window.Render()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',default='lung_002')
    parser.add_argument('--data-root',type=Path,default=STORAGE/'nnUNet_raw/Dataset006_Lung')
    parser.add_argument('--prediction-dir',type=Path,default=STORAGE/'predictions/baseline_100epochs_fold0_best')
    parser.add_argument('--screenshot',type=Path,help='Render a PNG offscreen instead of opening the interactive window')
    args=parser.parse_args()
    case=args.case.removesuffix('.nii.gz').removesuffix('_0000')
    if not case or Path(case).name!=case: parser.error('--case must be a patient ID')
    try:
        image=find_scan(args.data_root,case)
        prediction=args.prediction_dir/(case+'.nii.gz')
        if not prediction.is_file(): raise ValueError(f'Prediction not found: {prediction}')
        raw=np.asarray(nib.load(prediction).dataobj)
        if not np.all((raw==0)|(raw==1)): raise ValueError('Expected binary predicted labels (0 and 1)')
        del raw
        scan,mask,spacing,origin=load_pair(image,prediction)
    except (ValueError,OSError) as exc:
        parser.error(str(exc))
    print('Original CT:',image,'\nPrediction:',prediction,flush=True)
    print('No test ground truth is loaded. Orange is the prediction, not an annotation.',flush=True)
    viewer=PredictionViewer(scan,mask,spacing,origin,f'Small tumor | {case} | original vs prediction')
    if args.screenshot:
        viewer.window.SetOffScreenRendering(1); viewer.window.Render()
        capture=vtk.vtkWindowToImageFilter(); capture.SetInput(viewer.window)
        capture.ReadFrontBufferOff(); capture.Update()
        args.screenshot.parent.mkdir(parents=True,exist_ok=True)
        writer=vtk.vtkPNGWriter(); writer.SetFileName(str(args.screenshot))
        writer.SetInputConnection(capture.GetOutputPort()); writer.Write()
        viewer.window.Finalize(); print('Saved:',args.screenshot)
    else:
        viewer.interactor.Initialize(); viewer.window.Render(); viewer.interactor.Start(); viewer.window.Finalize()


if __name__=='__main__': main()
