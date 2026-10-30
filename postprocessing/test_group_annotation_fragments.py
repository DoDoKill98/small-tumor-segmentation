import unittest
import numpy as np
from group_annotation_fragments import group_fragments

class FragmentGroupingTests(unittest.TestCase):
    def test_close_fragments_grouped_far_region_preserved(self):
        a=np.zeros((20,8,8),np.uint8)
        a[1:4,2:5,2:5]=1; a[5,3,3]=1; a[15,3,3]=1
        r=group_fragments(a,np.eye(4),2.)
        self.assertEqual(r['fragment_count'],3)
        self.assertEqual(r['group_count'],2)
        self.assertEqual(r['groups'][0]['voxel_count'],28)
        self.assertEqual(sum(g['voxel_count'] for g in r['groups']),int(a.sum()))

    def test_spacing_changes_grouping(self):
        a=np.zeros((8,8,8),np.uint8); a[1,2,2]=1; a[3,2,2]=1
        self.assertEqual(group_fragments(a,np.eye(4),2.)['group_count'],1)
        self.assertEqual(group_fragments(a,np.diag([3.,1.,1.,1.]),2.)['group_count'],2)

    def test_empty(self):
        self.assertEqual(group_fragments(np.zeros((3,3,3),np.uint8),np.eye(4),2.)['group_count'],0)

if __name__=='__main__': unittest.main()
