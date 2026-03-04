"""
Magnetometer with hard iron and soft iron distortion model.

Pre-calibration distortion (raw sensor):
    Hard iron : ±100-300 nT from magnetorquer rods
    Soft iron : ~1-2% distortion from aluminium structure

Post-calibration residual (after on-orbit scalar calibration):
    Hard iron : ±10-20 nT
    Soft iron : ~0.1-0.3% residual distortion

This model uses POST-CALIBRATION values. The CALIBRATION FSW mode
(commissioning phase) performs the Merayo scalar calibration before
fine pointing begins, reducing distortion to these residual levels.
The 6-state MEKF then handles the residual naturally within its
100 nT measurement noise budget.

Reference:
    Merayo et al., "Scalar calibration of vector magnetometers", 2000.
    Springmann & Cutler, JGCD 2012.
"""

import numpy as np
from utils.quaternion import rot_matrix


class Magnetometer:

    def __init__(self, sigma_nT=100.0, hard_iron_nT=None, soft_iron=None):
        self.sigma = sigma_nT * 1e-9

        # Post-calibration hard iron residual: ±15 nT
        if hard_iron_nT is None:
            self.b_hi = np.random.uniform(-15e-9, 15e-9, 3)
        else:
            self.b_hi = np.array(hard_iron_nT) * 1e-9

        # Post-calibration soft iron residual: ~0.2% distortion
        if soft_iron is None:
            perturb   = np.random.uniform(-0.002, 0.002, (3, 3))
            self.A_si = np.eye(3) + perturb
        else:
            self.A_si = np.array(soft_iron)

    def measure(self, q, B_inertial):
        R      = rot_matrix(q)
        B_body = R @ B_inertial
        B_dist = self.A_si @ B_body + self.b_hi
        noise  = np.random.normal(0, self.sigma, 3)
        return B_dist + noise
