import unittest
import json
import tempfile
from pathlib import Path
import numpy as np
from region_probability_stats import probabilities_to_xyz, probability_statistics, validate_probability_alignment, validate_output_path

class ProbabilityStatsTests(unittest.TestCase):
    def test_zyx_to_xyz_is_explicit(self):
        p=np.zeros((2,5,4,3),np.float32)
        p[1,4,2,1]=.9
        xyz=probabilities_to_xyz(p,(3,4,5))
        self.assertEqual(xyz.shape,(3,4,5))
        self.assertAlmostEqual(float(xyz[1,2,4]),.9,places=6)

    def test_summary_and_entropy(self):
        s=probability_statistics(np.array([.5,1.]))
        self.assertAlmostEqual(s['mean'],.75)
        self.assertAlmostEqual(s['median'],.75)
        self.assertAlmostEqual(s['entropy_bits_mean'],.5)
        self.assertEqual(s['fraction_ge_0_90'],.5)

    def test_empty_statistics_are_explicit(self):
        s=probability_statistics(np.array([]))
        self.assertEqual(s['voxel_count'],0); self.assertIsNone(s['mean'])

    def test_wrong_orientation_detected_by_mask(self):
        m=np.zeros((3,4,5),np.uint8); m[1,2,4]=1
        p=np.zeros((2,5,4,3),np.float32); p[0]=1
        p[0,1,2,1]=.1; p[1,1,2,1]=.9
        with self.assertRaises(ValueError): validate_probability_alignment(p,m)

    def test_valid_mask_alignment(self):
        m=np.zeros((3,4,5),np.uint8); m[1,2,4]=1
        p=np.zeros((2,5,4,3),np.float32); p[0]=1
        p[0,4,2,1]=.1; p[1,4,2,1]=.9
        self.assertEqual(validate_probability_alignment(p,m)['disagreement_voxels'],0)

class OutputSafetyTests(unittest.TestCase):
    def test_probability_export_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'lung_006.npz'; source.write_bytes(b'original')
            with self.assertRaises(ValueError): validate_output_path(source,True)
            self.assertEqual(source.read_bytes(),b'original')

    def test_inference_metadata_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'prediction_manifest.json'; source.write_text('{}')
            with self.assertRaises(ValueError): validate_output_path(source,True)

    def test_only_existing_stats_report_can_be_replaced(self):
        with tempfile.TemporaryDirectory() as folder:
            report=Path(folder)/'stats.json'
            report.write_text(json.dumps({'report_type':'region_probability_statistics'}))
            validate_output_path(report,True)
            with self.assertRaises(ValueError): validate_output_path(report,False)

if __name__=='__main__': unittest.main()
