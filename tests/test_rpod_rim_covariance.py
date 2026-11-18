import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from fsw.rpod import BellSurvey


def cloud(rotation=None,center=(0.,0.,2.)):
    a=np.linspace(0,2*np.pi,48,endpoint=False)
    points=np.asarray(center)+.24*np.c_[np.cos(a),np.sin(a),np.zeros(48)]
    return points if rotation is None else points@rotation.T


class RimCovarianceTests(unittest.TestCase):
    def test_shared_bias_floor_enters_gate_after_many_samples(self):
        survey=BellSurvey()
        for i in range(100):survey.observe(cloud(),i*.5,np.eye(3)*1e-12)
        # A 10 mm coherent pose-induced displacement is inside the declared
        # 5 mm shared one-sigma floor at the existing NIS=25 gate.
        survey.observe(cloud(center=(.01,0.,2.)),50.,np.eye(3)*1e-12)
        self.assertEqual(survey.rejected,0)
        self.assertLess(survey.last_nis,25.)
        self.assertTrue(survey.valid(50.))

    def test_noisy_pose_accumulates_geometry_within_original_accuracy(self):
        rng=np.random.default_rng(8);survey=BellSurvey();sigma=np.radians(.45)
        for i in range(80):
            r=Rotation.from_rotvec(rng.normal(0,sigma,3)).as_matrix()
            survey.observe(cloud(r),i*.5,np.eye(3)*sigma**2)
        self.assertTrue(survey.valid(39.5))
        self.assertLess(np.linalg.norm(survey.center-[0,0,2]),.025)
        self.assertGreaterEqual(survey.accepted,20)

    def test_high_uncertainty_does_not_qualify_from_count_alone(self):
        survey=BellSurvey()
        for i in range(25):survey.observe(cloud(),i*.5,np.eye(3))
        self.assertFalse(survey.valid(12.))

    def test_inconsistent_geometry_rejected_without_overwriting_fit(self):
        survey=BellSurvey()
        for i in range(25):survey.observe(cloud(),i*.5,np.eye(3)*1e-6)
        before=survey.center.copy()
        survey.observe(cloud(center=(1.,0.,2.)),12.5,np.eye(3)*1e-6)
        self.assertEqual(survey.rejected,1)
        np.testing.assert_array_equal(survey.center,before)

    def test_freshness_still_required(self):
        survey=BellSurvey()
        for i in range(25):survey.observe(cloud(),i*.5,np.eye(3)*1e-6)
        self.assertTrue(survey.valid(12.))
        self.assertFalse(survey.valid(15.))

    def test_bad_angular_covariance_rejected(self):
        with self.assertRaises(ValueError):BellSurvey().observe(cloud(),0.,-np.eye(3))
