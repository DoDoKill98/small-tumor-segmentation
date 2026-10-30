import unittest
import numpy as np
from view_comparison import comparison_labels

class ComparisonTests(unittest.TestCase):
    def test_colors_distinguish_errors_and_overlap(self):
        gt=np.array([0,1,0,1],dtype=np.uint8)
        pred=np.array([0,0,1,1],dtype=np.uint8)
        np.testing.assert_array_equal(comparison_labels(gt,pred),[0,1,2,3])

    def test_toggles_preserve_other_mask(self):
        gt=np.array([1,1,0],dtype=np.uint8); pred=np.array([0,1,1],dtype=np.uint8)
        np.testing.assert_array_equal(comparison_labels(gt,pred,True,False),[1,1,0])
        np.testing.assert_array_equal(comparison_labels(gt,pred,False,True),[0,2,2])
        np.testing.assert_array_equal(comparison_labels(gt,pred,False,False),[0,0,0])

if __name__=='__main__': unittest.main()
