import unittest
import numpy as np
from oracle_select_region import select_oracle_region

class OracleTests(unittest.TestCase):
    def test_selects_smaller_correct_component(self):
        gt=np.zeros((14,14,14),np.uint8); gt[1:3,1:3,1:3]=1
        pred=gt.copy(); pred[8:13,8:13,8:13]=1
        out,stats=select_oracle_region(pred,gt)
        np.testing.assert_array_equal(out,gt)
        self.assertEqual(stats['selected_dice'],1)

    def test_keeps_whole_component_including_oversegmentation(self):
        gt=np.zeros((10,10,10),np.uint8); gt[1:3,1:3,1:3]=1
        pred=np.zeros_like(gt); pred[1:5,1:5,1:5]=1
        out,stats=select_oracle_region(pred,gt)
        np.testing.assert_array_equal(out,pred)
        self.assertGreater(int(out.sum()),int(gt.sum()))

    def test_no_overlap_produces_empty_and_failure_status(self):
        gt=np.zeros((10,10,10),np.uint8); gt[1,1,1]=1
        pred=np.zeros_like(gt); pred[8,8,8]=1
        out,stats=select_oracle_region(pred,gt)
        self.assertEqual(int(out.sum()),0)
        self.assertEqual(stats['status'],'no_overlapping_component')

    def test_maximum_dice_not_maximum_overlap_count(self):
        gt=np.zeros((20,20,20),np.uint8); gt[1:3,1:3,1:3]=1; gt[12:15,12:15,12:15]=1
        pred=np.zeros_like(gt); pred[1:3,1:3,1:3]=1; pred[10:19,10:19,10:19]=1
        out,stats=select_oracle_region(pred,gt)
        self.assertEqual(int(out.sum()),8)
        self.assertTrue(np.all(out<=pred))

    def test_empty_prediction(self):
        gt=np.zeros((4,4,4),np.uint8); gt[1,1,1]=1
        out,stats=select_oracle_region(np.zeros_like(gt),gt)
        self.assertEqual(stats['input_regions'],0); self.assertEqual(int(out.sum()),0)

if __name__=='__main__': unittest.main()
