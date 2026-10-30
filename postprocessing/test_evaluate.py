import unittest
import numpy as np
from evaluate import compute_metrics, validate_geometry

class MetricsTests(unittest.TestCase):
    def cube(self):
        a=np.zeros((24,24,24),bool); a[3:7,3:7,3:7]=True
        return a

    def test_identical(self):
        a=self.cube(); m=compute_metrics(a,a,(1,1,2),2)
        self.assertEqual(m['dice'],1); self.assertEqual(m['surface_dice'],1)
        self.assertEqual(m['hd95_mm'],0); self.assertAlmostEqual(m['reference_volume_ml'],.128)

    def test_physical_distance(self):
        a=np.zeros((16,16,16),bool); b=a.copy(); a[3,3,3]=True; b[3,3,7]=True
        m=compute_metrics(a,b,(1,1,3),2)
        self.assertEqual(m['surface_dice'],0)
        self.assertGreater(m['hd95_mm'],8)
        self.assertEqual(m['regions_missed'],1)

    def test_empty_prediction(self):
        a=self.cube(); m=compute_metrics(a,np.zeros_like(a),(1,1,1),2)
        self.assertEqual(m['dice'],0); self.assertEqual(m['surface_dice'],0)
        self.assertIsNone(m['hd95_mm']); self.assertIsNone(m['voxel_precision'])
        self.assertEqual(m['regions_missed'],1)

    def test_extra_region(self):
        a=self.cube(); b=a.copy(); b[15:18,15:18,15:18]=True
        m=compute_metrics(a,b,(1,1,1),2)
        self.assertEqual(m['regions_detected'],1)
        self.assertEqual(m['regions_unmatched_prediction'],1)

    def test_merged_regions_match_once(self):
        a=self.cube(); a[12:15,3:7,3:7]=True
        b=a.copy(); b[7:12,3:7,3:7]=True
        m=compute_metrics(a,b,(1,1,1),2)
        self.assertEqual(m['reference_regions'],2)
        self.assertEqual(m['regions_detected'],1)
        self.assertEqual(m['regions_missed'],1)
        self.assertEqual(m['reference_regions_with_any_overlap'],2)

    def test_empty_both(self):
        a=np.zeros((3,3,3),bool); m=compute_metrics(a,a,(1,1,1),2)
        self.assertEqual(m['dice'],1); self.assertEqual(m['hd95_mm'],0)
        self.assertEqual(m['surface_status'],'both_empty')

    def test_shear_rejected(self):
        a=np.eye(4); a[0,1]=.4
        with self.assertRaises(ValueError): validate_geometry((3,3,3),a,(3,3,3),a)

if __name__=='__main__': unittest.main()
