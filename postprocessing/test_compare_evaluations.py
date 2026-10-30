import unittest
from compare_evaluations import compare_reports

class PairedComparisonTests(unittest.TestCase):
    def report(self,patient='p',dice=.5):
        row={'patient':patient,'dice':dice,'surface_dice':.6,'hd95_mm':None,
             'voxel_recall':.7,'voxel_precision':.8,'absolute_volume_error_ml':1.,
             'regions_completely_missed':0,'regions_unmatched_prediction':1}
        return {'fold':0,'split_sha256':'same','surface_tolerance_mm':2,'checkpoint_label':'final',
                'definitions':{'same':True},'reference_dir':'.','patients':[row]}

    def test_paired_direction_and_undefined_distance(self):
        r=compare_reports(self.report(dice=.5),self.report(dice=.7))
        self.assertEqual(r['summary']['dice']['improved'],1)
        self.assertEqual(r['summary']['hd95_mm']['n_undefined_pairs'],1)

    def test_different_patients_rejected(self):
        with self.assertRaises(ValueError): compare_reports(self.report(),self.report(patient='other'))

    def test_different_tolerances_rejected(self):
        after=self.report(); after['surface_tolerance_mm']=5
        with self.assertRaises(ValueError): compare_reports(self.report(),after)

if __name__=='__main__': unittest.main()
