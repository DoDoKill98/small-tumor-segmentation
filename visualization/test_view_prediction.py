import tempfile
from pathlib import Path
import unittest
from view_prediction import find_scan

class PredictionInputTests(unittest.TestCase):
    def test_nnunet_test_scan(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); (root/'imagesTs').mkdir()
            scan=root/'imagesTs/lung_002_0000.nii.gz'; scan.touch()
            self.assertEqual(find_scan(root,'lung_002'),scan)

    def test_original_msd_scan(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); (root/'imagesTs').mkdir()
            scan=root/'imagesTs/lung_002.nii.gz'; scan.touch()
            self.assertEqual(find_scan(root,'lung_002'),scan)

    def test_missing_scan_is_explicit(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError): find_scan(Path(folder),'lung_999')

if __name__=='__main__': unittest.main()
