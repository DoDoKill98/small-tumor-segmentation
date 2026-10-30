import json
from pathlib import Path
import unittest
import numpy as np
import nibabel as nib
from automatic_ct_selection import select_with_roi, estimate_lung_roi, restore_original_orientation

CONFIG=json.loads((Path(__file__).parent/'automatic_selection_v1.json').read_text(encoding='utf-8-sig'))

class AutomaticSelectionTests(unittest.TestCase):
    def test_small_pulmonary_component_beats_large_remote_component(self):
        roi=np.zeros((40,20,20),bool); roi[2:12,2:12,2:12]=True
        pred=np.zeros(roi.shape,np.uint8); pred[4:6,4:6,4:6]=1; pred[30:38,4:12,4:12]=1
        out,decision=select_with_roi(pred,(1,1,1),roi,(1,1,1),CONFIG,True)
        self.assertEqual(decision['status'],'selected')
        self.assertEqual(int(out.sum()),8); self.assertTrue(np.all(out<=pred))

    def test_ambiguous_candidates_preserved(self):
        roi=np.ones((20,20,20),bool)
        pred=np.zeros(roi.shape,np.uint8); pred[2:5,2:5,2:5]=1; pred[12:15,12:15,12:15]=1
        out,decision=select_with_roi(pred,(1,1,1),roi,(1,1,1),CONFIG,True)
        self.assertEqual(decision['status'],'ambiguous_preserve_all')
        np.testing.assert_array_equal(out,pred)

    def test_unreliable_roi_preserves_mask(self):
        pred=np.zeros((8,8,8),np.uint8); pred[1,1,1]=1; pred[6,6,6]=1
        out,decision=select_with_roi(pred,(1,1,1),np.zeros_like(pred,bool),(1,1,1),CONFIG,False)
        self.assertEqual(decision['status'],'unreliable_roi_preserve_all')
        np.testing.assert_array_equal(out,pred)

    def test_distances_use_spacing(self):
        roi=np.zeros((30,12,12),bool); roi[:5,:,:]=1
        pred=np.zeros(roi.shape,np.uint8); pred[8,4,4]=1; pred[22,4,4]=1
        _,a=select_with_roi(pred,(1,1,1),roi,(1,1,1),CONFIG,True)
        _,b=select_with_roi(pred,(4,1,1),roi,(1,1,1),CONFIG,True)
        self.assertGreater(a['candidates'][0]['within_margin_fraction'],b['candidates'][0]['within_margin_fraction'])

    def test_synthetic_ct_enclosed_air(self):
        ct=np.full((48,48,24),-1000,np.float32)
        ct[5:43,5:43,:]=30
        ct[12:22,12:36,3:21]=-800; ct[27:37,12:36,3:21]=-800
        roi,steps,quality=estimate_lung_roi(ct,(3,3,3),CONFIG)
        self.assertTrue(roi[15,20,10]); self.assertFalse(roi[0,0,10])
        self.assertEqual(tuple(steps),(1,1,1))

class OrientationTests(unittest.TestCase):
    def test_permuted_flipped_nifti_roundtrip(self):
        values=np.arange(120,dtype=np.int16).reshape(4,5,6)
        affine=np.array([[0,0,2,10],[0,-3,0,20],[4,0,0,-5],[0,0,0,1]],float)
        original=nib.Nifti1Image(values,affine)
        canonical=nib.as_closest_canonical(original)
        restored=restore_original_orientation(np.asarray(canonical.dataobj),canonical.affine,original.affine)
        np.testing.assert_array_equal(restored,values)

if __name__=='__main__': unittest.main()
