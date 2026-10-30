import unittest
import tempfile
import json
from pathlib import Path
import nibabel as nib
import numpy as np
from apply_region_selector import validate_application
from region_selector import FEATURE_NAMES, candidate_features, leave_one_patient_out, load_feature_rows, score_candidate_rows, export_component

class SelectorTests(unittest.TestCase):
    def candidate(self):
        return {'component_id':1,'volume_ml':2.,'centroid_world_mm':[2.,3.,4.],
                'bounding_box_xyz_exclusive':[[1,3],[1,3],[1,3]],
                'tumor_probability':{'voxel_count':8,'mean':.8,'quantiles':{'p10':.6},
                                     'entropy_bits_mean':.4,'fraction_ge_0_90':.3}}

    def test_features_exclude_oracle_and_ids(self):
        c=self.candidate(); first=candidate_features(c,[c],(10,10,10),np.eye(4))
        c.update(patient='leak',oracle_dice=1.,selected=True,component_id=999)
        second=candidate_features(c,[c],(10,10,10),np.eye(4))
        np.testing.assert_array_equal(first,second)
        self.assertEqual(len(first),len(FEATURE_NAMES))

    def test_world_position_handles_flipped_axes(self):
        c=self.candidate(); affine=np.diag([-1.,1.,2.,1.]); affine[0,3]=9
        values=candidate_features(c,[c],(10,10,10),affine)
        self.assertAlmostEqual(values[2],2/9)
        self.assertAlmostEqual(values[4],4/18)

    def test_leave_one_patient_out_has_no_shared_patient(self):
        X=np.array([[0.,0.],[1.,1.],[0.,.1],[1.,1.1],[.1,0.],[1.1,1.]])
        y=np.array([0,1,0,1,0,1]); groups=np.array(['a','a','b','b','c','c'])
        scores,audit=leave_one_patient_out(X,y,groups)
        self.assertEqual(len(scores),6)
        for row in audit:
            self.assertNotIn(row['held_out_patient'],row['training_patients'])
            self.assertEqual(len(row['training_patients']),2)
        scores2,_=leave_one_patient_out(X,y,groups)
        np.testing.assert_allclose(scores,scores2)

class DeploymentGuardsTests(unittest.TestCase):
    def test_known_training_patient_rejected(self):
        bundle={'feature_names':FEATURE_NAMES,'backbone_checkpoint_sha256':'best','training_patients':['a']}
        with self.assertRaises(ValueError): validate_application(bundle,{'checkpoint_sha256':'best'},['a'])

    def test_different_backbone_rejected(self):
        bundle={'feature_names':FEATURE_NAMES,'backbone_checkpoint_sha256':'best','training_patients':['a']}
        with self.assertRaises(ValueError): validate_application(bundle,{'checkpoint_sha256':'final'},['b'])

class EmptyBatchAndExportTests(unittest.TestCase):
    def test_entirely_empty_batch_is_supported(self):
        import hashlib
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); path=root/'empty.nii.gz'
            nib.save(nib.Nifti1Image(np.zeros((3,4,5),np.uint8),np.eye(4)),path)
            stats={'uses_ground_truth':False,'automatic_selection_applied':False,
                   'patients':[{'patient':'empty','mask_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                                'shape_xyz':[3,4,5],'regions':[]}]}
            source=root/'stats.json'; source.write_text(json.dumps(stats))
            X,groups,rows,_=load_feature_rows(source)
            self.assertEqual(X.shape,(0,len(FEATURE_NAMES)))
            self.assertEqual(score_candidate_rows(None,X).size,0)
            self.assertEqual(rows,[])

    def test_empty_known_training_patient_still_rejected(self):
        bundle={'feature_names':FEATURE_NAMES,'backbone_checkpoint_sha256':'best','training_patients':['a']}
        stats={'checkpoint_sha256':'best','patients':[{'patient':'a','regions':[]}]}
        with self.assertRaises(ValueError): validate_application(bundle,stats,[])

    def test_export_preserves_flipped_geometry_and_whole_component(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); values=np.zeros((8,9,10),np.uint8)
            values[1:3,1:3,1:3]=1; values[6,7,8]=1
            affine=np.diag([-2.,3.,1.,1.]); affine[0,3]=50
            source=root/'input.nii.gz'; destination=root/'output.nii.gz'
            nib.save(nib.Nifti1Image(values,affine),source)
            export_component(source,1,destination)
            exported=nib.load(destination)
            np.testing.assert_allclose(exported.affine,affine)
            self.assertEqual(int(np.asarray(exported.dataobj).sum()),8)
            self.assertTrue(np.all(np.asarray(exported.dataobj)<=values))

if __name__=='__main__': unittest.main()
