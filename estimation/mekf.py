import numpy as np
from utils.quaternion import quat_multiply, normalize, rot_matrix


class MEKF:
    """
    Multiplicative Extended Kalman Filter — 6-state formulation.

    State vector (error state):
        dx = [dtheta (3)  — attitude error [rad]
              dbias  (3)] — gyro bias error [rad/s]

    Reference:
        Markley & Crassidis, "Fundamentals of Spacecraft Attitude
        Determination and Control", §7.3
    """

    def __init__(self, dt):
        self.dt = dt

        self.q    = np.array([1., 0., 0., 0.])
        self.bias = np.zeros(3)

        # Attitude uncertainty: ~5° initial
        self.P = np.eye(6) * 0.01
        self.P[3:6, 3:6] = np.eye(3) * (np.radians(1)/3600)**2

        self.Q = np.diag([1e-6, 1e-6, 1e-6,
                          1e-12, 1e-12, 1e-12])

        self.R_mag = np.eye(3) * 1e-4
        self.R_sun = np.eye(3) * 1e-4

    def predict(self, omega_m):
        omega      = omega_m - self.bias
        wx, wy, wz = omega

        Omega = np.array([
            [ 0,   -wx, -wy, -wz],
            [ wx,   0,   wz, -wy],
            [ wy,  -wz,  0,   wx],
            [ wz,   wy, -wx,  0 ]
        ])

        self.q += 0.5 * self.dt * Omega @ self.q
        self.q  = normalize(self.q)

        F           = np.zeros((6, 6))
        F[0:3, 3:6] = -np.eye(3)
        Phi         = np.eye(6) + F * self.dt
        self.P      = Phi @ self.P @ Phi.T + self.Q

    def update_vector(self, z_body, v_inertial, R):
        # Normalise to unit vectors — filter works in direction space
        v_inertial = v_inertial / np.linalg.norm(v_inertial)
        z_body     = z_body     / np.linalg.norm(z_body)

        Rb     = rot_matrix(self.q)
        z_pred = Rb @ v_inertial

        vx, vy, vz = z_pred
        skew = np.array([
            [ 0,  -vz,  vy],
            [ vz,  0,  -vx],
            [-vy,  vx,  0 ]
        ])

        H         = np.zeros((3, 6))
        H[:, 0:3] = -skew

        y     = z_body - z_pred
        S     = H @ self.P @ H.T + R
        mahal = float(y @ np.linalg.inv(S) @ y)
        if mahal > 25.0:   # 4-sigma gate
            return

        K  = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ y

        # Joseph form — numerically stable for any K
        IKH    = np.eye(6) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T

        # Enforce symmetry and positive definiteness
        self.P = 0.5 * (self.P + self.P.T)
        eigvals = np.linalg.eigvalsh(self.P)
        if np.any(eigvals < 0):
            self.P += (-np.min(eigvals) + 1e-12) * np.eye(6)

        dq        = np.hstack([1., 0.5 * dx[0:3]])
        self.q    = normalize(quat_multiply(dq, self.q))
        self.bias += dx[3:6]