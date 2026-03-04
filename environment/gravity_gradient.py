import numpy as np
from utils.quaternion import rot_matrix

class GravityGradient:

    def __init__(self, inertia):
        self.I  = inertia
        self.mu = 3.986004418e14   # m³/s²

    def compute(self, pos_km, q):
        """
        Gravity gradient torque in body frame.
        T_gg = 3μ/r³ * r̂_body × (I · r̂_body)

        Parameters:
            pos_km : ECI position vector [km]
            q      : spacecraft quaternion [w, x, y, z]

        Returns:
            T_gg   : torque vector in body frame [N·m]
        """
        r_inertial = pos_km * 1000.0        # km → m
        r_mag      = np.linalg.norm(r_inertial)
        r_hat_I    = r_inertial / r_mag

        # Rotate nadir vector to body frame
        R       = rot_matrix(q)
        r_hat_b = R @ r_hat_I

        coeff = 3.0 * self.mu / r_mag**3
        T_gg  = coeff * np.cross(r_hat_b, self.I @ r_hat_b)

        return T_gg