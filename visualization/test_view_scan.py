import unittest
import tempfile
from pathlib import Path
import numpy as np
import nibabel as nib
from view_scan import validate_pair, to_vtk, resolve_case, load_pair

class GeometryTests(unittest.TestCase):
    def test_misaligned_mask_rejected(self):
        a = nib.Nifti1Image(np.zeros((3,4,5), dtype=np.int16), np.eye(4))
        shifted = np.eye(4); shifted[0,3] = 10
        b = nib.Nifti1Image(np.zeros((3,4,5), dtype=np.uint8), shifted)
        with self.assertRaises(ValueError): validate_pair(a,b)

    def test_vtk_preserves_values_and_spacing(self):
        a = np.arange(24, dtype=np.float32).reshape((2,3,4))
        v = to_vtk(a, (0.7,0.8,2.0), (10,20,30))
        self.assertEqual(v.GetDimensions(), (2,3,4))
        self.assertEqual(v.GetSpacing(), (0.7,0.8,2.0))
        self.assertEqual(v.GetOrigin(), (10,20,30))
        self.assertEqual(v.GetScalarComponentAsDouble(1,2,3,0), float(a[1,2,3]))

class UnannotatedTests(unittest.TestCase):
    def test_test_case_without_annotation(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); (root/'imagesTs').mkdir()
            path=root/'imagesTs/lung_002.nii.gz'
            nib.save(nib.Nifti1Image(np.zeros((3,4,5),dtype=np.int16),np.eye(4)),path)
            image,mask=resolve_case(root,'lung_002')
            self.assertEqual(image,path)
            self.assertIsNone(mask)
            scan,tumor,spacing,origin=load_pair(image,mask)
            self.assertEqual(scan.shape,(3,4,5))
            self.assertIsNone(tumor)

    def test_training_case_still_finds_mask(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for name in ('imagesTr','labelsTr'): (root/name).mkdir()
            for name in ('imagesTr','labelsTr'): (root/name/'lung_001.nii.gz').touch()
            image,mask=resolve_case(root,'lung_001')
            self.assertEqual(mask,root/'labelsTr/lung_001.nii.gz')

if __name__ == '__main__': unittest.main()
