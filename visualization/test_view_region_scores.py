import unittest
import numpy as np
from view_region_scores import label_prediction, ranked_rows, display_codes

class RegionScoreViewerTests(unittest.TestCase):
    def test_low_score_region_is_not_removed(self):
        mask=np.zeros((5,5,5),np.uint8); mask[1,1,1]=1; mask[4,4,4]=1
        labels,n=label_prediction(mask)
        candidates=[{'component_id':1,'selector_score':.9},{'component_id':2,'selector_score':.01}]
        volumes=[{'component_id':1,'volume_ml':1},{'component_id':2,'volume_ml':2}]
        rows=ranked_rows(candidates,volumes,n)
        self.assertEqual(len(rows),2); self.assertEqual(rows[1]['selector_score'],.01)
        codes=display_codes(labels,np.zeros_like(mask),True,False,n+1)
        self.assertEqual(int(np.count_nonzero(codes)),2)

    def test_scores_must_match_all_component_ids(self):
        with self.assertRaises(ValueError): ranked_rows([{'component_id':1,'selector_score':.5}],[],2)

    def test_overlay_ground_truth_does_not_change_ranking(self):
        labels=np.array([0,1,2],np.uint8); gt=np.array([1,1,0],np.uint8)
        np.testing.assert_array_equal(display_codes(labels,gt,True,True,3),[3,4,2])
        np.testing.assert_array_equal(display_codes(labels,gt,False,True,3),[3,3,0])

    def test_invalid_scores_rejected(self):
        with self.assertRaises(ValueError): ranked_rows([{'component_id':1,'selector_score':float('nan')}],[{'component_id':1,'volume_ml':1}],1)

if __name__=='__main__': unittest.main()
