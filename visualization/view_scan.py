"""Local NIfTI viewer: three orthogonal slices and a 3D tumor surface (RAS, mm)."""
import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
import vtk
from vtk.util.numpy_support import numpy_to_vtk

DEFAULT_DATA = Path.home() / 'Datasets/MedicalSegmentationDecathlon/Task06_Lung/extracted/Task06_Lung'


def validate_pair(image, mask):
    if len(image.shape) != 3 or image.shape != mask.shape:
        raise ValueError('Image and mask must be matching 3D volumes.')
    if not np.allclose(image.affine, mask.affine, atol=1e-4):
        raise ValueError('Image and mask geometry differs; refusing a misleading overlay.')


def to_vtk(array, spacing, origin):
    data = vtk.vtkImageData()
    data.SetDimensions(*array.shape)
    data.SetSpacing(*spacing)
    data.SetOrigin(*origin)
    data.GetPointData().SetScalars(numpy_to_vtk(array.ravel(order='F'), deep=True))
    return data


def resolve_case(root, case):
    candidates = [root / split / f'{case}.nii.gz' for split in ('imagesTr', 'imagesTs')]
    found = [p for p in candidates if p.is_file()]
    if not found:
        raise ValueError(f'Scanner not found in imagesTr or imagesTs: {case}')
    if len(found) > 1:
        raise ValueError(f'Ambiguous patient ID in imagesTr and imagesTs: {case}')
    mask = root / 'labelsTr' / f'{case}.nii.gz'
    return found[0], mask if found[0].parent.name == 'imagesTr' and mask.is_file() else None


def load_pair(image_path, mask_path=None):
    image = nib.load(image_path)
    mask = nib.load(mask_path) if mask_path is not None else None
    if len(image.shape) != 3:
        raise ValueError('Scanner must be a 3D volume.')
    if mask is not None:
        validate_pair(image, mask)
    image = nib.as_closest_canonical(image)
    mask = nib.as_closest_canonical(mask) if mask is not None else None
    # Axis-aligned vtkImageActor needs an orthogonal physical grid.
    if not np.allclose(image.affine[:3,:3], np.diag(np.diag(image.affine[:3,:3])), atol=1e-5):
        from nibabel.processing import resample_to_output, resample_from_to
        image = resample_to_output(image, voxel_sizes=image.header.get_zooms()[:3], order=1, cval=-1024)
        if mask is not None: mask = resample_from_to(mask, image, order=0)
    scan = image.get_fdata(dtype=np.float32)
    labels = mask.get_fdata(dtype=np.float32) if mask is not None else None
    if not np.isfinite(scan).all() or (labels is not None and not np.isfinite(labels).all()):
        raise ValueError('Volume contains non-finite values.')
    tumor = (labels > 0).astype(np.uint8) if labels is not None else None
    return scan, tumor, image.header.get_zooms()[:3], image.affine[:3,3]


class Viewer:
    def __init__(self, scan, mask, spacing, origin, title):
        self.has_annotation = mask is not None
        if mask is None: mask = np.zeros(scan.shape, dtype=np.uint8)
        self.scan = to_vtk(scan, spacing, origin)
        self.mask = to_vtk(mask, spacing, origin)
        self.shape, self.spacing, self.origin = scan.shape, spacing, origin
        locations = np.argwhere(mask > 0)
        self.initial = [int(v) for v in (np.median(locations,axis=0) if len(locations) else np.array(scan.shape)//2)]
        self.indices = self.initial.copy()
        self.visible = self.has_annotation
        self.window = vtk.vtkRenderWindow()
        self.window.SetWindowName(title)
        self.window.SetSize(1280,900)
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.window)
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(style)
        self.renderers, self.actors, self.overlays, self.planes, self.texts = [], [], [], [], []
        self.axes = [2,1,0]
        names = ['Axiale (R/A)', 'Coronale (R/S)', 'Sagittale (A/S)']
        viewports = [(0,.5,.5,1),(.5,.5,1,1),(0,0,.5,.5),(.5,0,1,.5)]
        lut = vtk.vtkLookupTable(); lut.SetNumberOfTableValues(2); lut.SetRange(0,1)
        lut.SetTableValue(0,0,0,0,0); lut.SetTableValue(1,1,.25,.05,.55); lut.Build()
        self.colors = vtk.vtkImageMapToColors(); self.colors.SetInputData(self.mask)
        self.colors.SetLookupTable(lut); self.colors.SetOutputFormatToRGBA(); self.colors.Update()
        for i, viewport in enumerate(viewports):
            ren = vtk.vtkRenderer(); ren.SetViewport(*viewport); ren.SetBackground(.06,.075,.10)
            self.window.AddRenderer(ren); self.renderers.append(ren)
            text = vtk.vtkTextActor(); text.SetPosition(12,12)
            text.GetTextProperty().SetFontSize(15); text.GetTextProperty().SetColor(.95,.95,.95)
            ren.AddViewProp(text); self.texts.append(text)
            if i == 3: continue
            actor = vtk.vtkImageActor(); actor.SetInputData(self.scan)
            actor.GetProperty().SetColorWindow(1400); actor.GetProperty().SetColorLevel(-300)
            actor.InterpolateOff(); ren.AddActor(actor); self.actors.append(actor)
            overlay = vtk.vtkImageActor(); overlay.GetMapper().SetInputConnection(self.colors.GetOutputPort())
            overlay.InterpolateOff(); ren.AddActor(overlay); self.overlays.append(overlay)
            plane = vtk.vtkImageActor(); plane.SetInputData(self.scan)
            plane.GetProperty().SetColorWindow(1400); plane.GetProperty().SetColorLevel(-300)
            plane.GetProperty().SetOpacity(.45)
            self.planes.append(plane)
        for plane in self.planes: self.renderers[3].AddActor(plane)
        surface = vtk.vtkDiscreteMarchingCubes(); surface.SetInputData(self.mask); surface.SetValue(0,1); surface.Update()
        mapper = vtk.vtkPolyDataMapper(); mapper.SetInputConnection(surface.GetOutputPort()); mapper.ScalarVisibilityOff()
        self.surface = vtk.vtkActor(); self.surface.SetMapper(mapper)
        self.surface.GetProperty().SetColor(1,.3,.08); self.renderers[3].AddActor(self.surface)
        outline = vtk.vtkOutlineFilter(); outline.SetInputData(self.scan)
        om = vtk.vtkPolyDataMapper(); om.SetInputConnection(outline.GetOutputPort())
        oa = vtk.vtkActor(); oa.SetMapper(om); oa.GetProperty().SetColor(.5,.6,.7); self.renderers[3].AddActor(oa)
        self.names = names
        self.update()
        self.reset_cameras()
        # High priority consumes wheel events, preventing accidental camera zoom.
        style.AddObserver('MouseWheelForwardEvent', lambda o,e:self.scroll(1), 1)
        style.AddObserver('MouseWheelBackwardEvent', lambda o,e:self.scroll(-1), 1)
        self.interactor.AddObserver('KeyPressEvent', self.key)
        print('Tumor surface triangles:', surface.GetOutput().GetNumberOfCells())

    def update(self):
        for i,axis in enumerate(self.axes):
            extent = [0,self.shape[0]-1,0,self.shape[1]-1,0,self.shape[2]-1]
            extent[2*axis:2*axis+2] = [self.indices[axis]]*2
            for actor in (self.actors[i],self.overlays[i],self.planes[i]): actor.SetDisplayExtent(*extent)
            shift = [0.,0.,0.]; shift[axis] = .02
            self.overlays[i].SetPosition(*shift)
            self.overlays[i].SetVisibility(self.visible)
            self.texts[i].SetInput(f'{self.names[i]} | coupe {self.indices[axis]+1}/{self.shape[axis]}\nMolette: coupe | M: masque | R: reset | Q: quitter')
        self.surface.SetVisibility(self.visible)
        status = 'orange: annotation' if self.has_annotation else 'sans annotation disponible'
        self.texts[3].SetInput(f'3D RAS (mm) | {status}\nGlisser: tourner | molette: zoom | P: plans')

    def reset_cameras(self):
        center = [self.origin[k]+self.spacing[k]*self.indices[k] for k in range(3)]
        for i,axis in enumerate(self.axes):
            camera = self.renderers[i].GetActiveCamera(); position = center.copy(); position[axis] += 1000
            camera.SetFocalPoint(*center); camera.SetPosition(*position)
            camera.SetViewUp(*( (0,1,0) if axis==2 else (0,0,1) ))
            camera.ParallelProjectionOn(); self.renderers[i].ResetCamera()
        self.renderers[3].ResetCamera()
        self.renderers[3].GetActiveCamera().Azimuth(30); self.renderers[3].GetActiveCamera().Elevation(20)
        self.renderers[3].ResetCameraClippingRange()

    def scroll(self, delta):
        x,y = self.interactor.GetEventPosition(); w,h = self.window.GetSize()
        panel = (0 if x<w/2 else 1) if y>=h/2 else (2 if x<w/2 else 3)
        if panel == 3:
            self.renderers[3].GetActiveCamera().Dolly(1.1 if delta > 0 else 1/1.1)
            self.renderers[3].ResetCameraClippingRange()
            self.window.Render()
            return
        axis = self.axes[panel]
        self.indices[axis] = max(0,min(self.shape[axis]-1,self.indices[axis]+delta))
        self.update()
        for ren in self.renderers: ren.ResetCameraClippingRange()
        self.window.Render()

    def key(self, obj, event):
        key = self.interactor.GetKeySym().lower()
        if key == 'm' and self.has_annotation: self.visible = not self.visible; self.update()
        elif key == 'p':
            for plane in self.planes: plane.SetVisibility(not plane.GetVisibility())
        elif key == 'r': self.indices=self.initial.copy(); self.update(); self.reset_cameras()
        elif key in ('q','escape'): self.interactor.TerminateApp(); return
        self.window.Render()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=DEFAULT_DATA)
    parser.add_argument('--case',default='lung_001',help='Patient ID, e.g. lung_001')
    parser.add_argument('--screenshot',type=Path,help='Save PNG offscreen instead of opening the viewer')
    args = parser.parse_args()
    case = args.case.removesuffix('.nii.gz')
    if Path(case).name != case: parser.error('--case must be a patient ID, not a path')
    try:
        image,mask = resolve_case(args.data_root,case)
    except ValueError as exc:
        parser.error(str(exc))
    scan,tumor,spacing,origin = load_pair(image,mask)
    print('Image:',image,'\nAnnotation:',mask if mask is not None else 'absente (scanner seul)','\nShape:',scan.shape,'Spacing (mm):',spacing)
    viewer = Viewer(scan,tumor,spacing,origin,f'Small tumor | {case}')
    if args.screenshot:
        viewer.window.SetOffScreenRendering(1); viewer.window.Render()
        capture = vtk.vtkWindowToImageFilter(); capture.SetInput(viewer.window); capture.ReadFrontBufferOff(); capture.Update()
        args.screenshot.parent.mkdir(parents=True,exist_ok=True)
        writer = vtk.vtkPNGWriter(); writer.SetFileName(str(args.screenshot)); writer.SetInputConnection(capture.GetOutputPort()); writer.Write()
        viewer.window.Finalize(); print('Saved:',args.screenshot)
    else:
        viewer.interactor.Initialize(); viewer.window.Render(); viewer.interactor.Start(); viewer.window.Finalize()

if __name__ == '__main__': main()


