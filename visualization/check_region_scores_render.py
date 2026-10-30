"""Small offscreen integration check for the VTK score viewer (requires VTK rendering)."""
import numpy as np
from view_region_scores import ScoreViewer,label_prediction


def main():
    scan=np.zeros((12,14,16),np.float32)
    mask=np.zeros(scan.shape,np.uint8); mask[2:4,2:4,2:4]=1; mask[8:10,8:10,8:10]=1
    labels,_=label_prediction(mask)
    rows=[{'component_id':1,'selector_score':.8,'volume_ml':.008,'centroid_world_mm':[2.5,2.5,2.5]},
          {'component_id':2,'selector_score':.02,'volume_ml':.008,'centroid_world_mm':[8.5,8.5,8.5]}]
    viewer=ScoreViewer(scan,mask,mask,labels,(1,1,1),(0,0,0),rows,'VTK score viewer check')
    viewer.window.SetOffScreenRendering(1)
    try:
        viewer.window.Render()
        viewer.interactor.SetKeyEventInformation(0,0,'n',0,'n'); viewer.key(None,None)
        assert viewer.active_rank==1
        assert np.count_nonzero(viewer.region_labels)==16
        assert viewer.all_region_surface.GetVisibility()
        assert viewer.original_actors[0].GetDisplayExtent()==viewer.actors[0].GetDisplayExtent()
        full_extent=(0,11,0,13,0,15)
        for overlay in viewer.overlays:
            assert overlay.GetInput().GetExtent()==full_extent
        viewer.interactor.SetKeyEventInformation(0,0,'m',0,'m'); viewer.key(None,None)
        assert not viewer.all_region_surface.GetVisibility()
        viewer.interactor.SetKeyEventInformation(0,0,'m',0,'m'); viewer.key(None,None)
        assert viewer.all_region_surface.GetVisibility()
        assert np.count_nonzero(viewer.region_labels)==16
        print('PASS: synchronized navigation, full RGBA extents, independent display toggles, no region removed')
    finally:
        viewer.window.Finalize()


if __name__=='__main__': main()
