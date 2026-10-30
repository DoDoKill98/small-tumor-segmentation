import unittest
import numpy as np
from keep_largest_region import keep_largest_region

class LargestRegionTests(unittest.TestCase):
    def test_keeps_largest_without_reference(self):
        a=np.zeros((12,12,12),np.uint8); a[1:4,1:4,1:4]=1; a[9,9,9]=1
        out,report=keep_largest_region(a,1.)
        self.assertEqual(int(out.sum()),27)
        self.assertEqual(report['removed_voxels'],1)
        self.assertEqual(report['input_regions'],2)
        self.assertFalse(out[9,9,9]); self.assertTrue(np.all(out<=a))

    def test_empty_remains_empty(self):
        out,report=keep_largest_region(np.zeros((4,4,4),np.uint8),1.)
        self.assertEqual(int(out.sum()),0); self.assertEqual(report['input_regions'],0)

    def test_tie_is_deterministic(self):
        a=np.zeros((5,5,5),np.uint8); a[0,0,0]=1; a[4,4,4]=1
        out,report=keep_largest_region(a,2.)
        self.assertEqual(int(out.sum()),1); self.assertTrue(out[0,0,0])
        self.assertEqual(report['largest_region_tie_count'],2)

    def test_diagonal_connection_preserved(self):
        a=np.zeros((4,4,4),np.uint8); a[0,0,0]=1; a[1,1,1]=1
        out,report=keep_largest_region(a,1.)
        self.assertEqual(int(out.sum()),2); self.assertEqual(report['input_regions'],1)

    def test_invalid_labels_rejected(self):
        with self.assertRaises(ValueError): keep_largest_region(np.full((3,3,3),2),1.)

if __name__=='__main__': unittest.main()
