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

        # ── Aule Space 50kg GEO servicer noise parameters ─────────────
        # P_init: 0.1° attitude uncertainty (star tracker acquired)
        #         0.5 deg/hr bias uncertainty (Sensonor STIM300 class)
        att_init_rad = np.radians(0.1)             # 0.1° initial pointing uncertainty
        bias_init_rads = np.radians(0.5) / 3600.0  # 0.5 deg/hr bias uncertainty
        self.P = np.eye(6) * att_init_rad**2
        self.P[3:6, 3:6] = np.eye(3) * bias_init_rads**2

        # Q: process noise
        # Q_att: Sensonor STIM300 ARW ~ 0.15 deg/√hr = 7.3e-7 rad/√s → Q_att = ARW² = 5e-8 rad²/s
        #        (much better than 3U CubeSat's 3.44 deg/√hr equivalent from old Q=1e-6)
        # Q_bias: rate random walk ~ 0.1 deg/hr/√hr = 8.1e-10 rad/s/√s → Q_bias = 6.5e-13
        self.Q = np.diag([5e-8, 5e-8, 5e-8,
                          1e-12, 1e-12, 1e-12])

        # R_mag: magnetometer noise at GEO
        #   GEO field ~100-200 nT vs 30,000 nT at LEO → 300x weaker signal
        #   Measurement is dominated by field error, not sensor noise
        #   sigma_unit_vector ~ 0.7 rad → R_mag = 0.5
        #   (effectively tells filter: don't trust magnetometer at GEO)
        self.R_mag = np.eye(3) * 0.5

        # R_sun: sun sensor noise
        #   50kg servicer quality sun sensor: sigma ~ 0.1° = 1.7e-3 rad
        #   R_sun = sigma² = 3e-6 (10x better than 3U coarse sun sensor)
        self.R_sun = np.eye(3) * 3e-6

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
    def update_star_tracker(self, q_meas: np.ndarray, R_st: np.ndarray):
        """
        MEKF update from star tracker full-quaternion measurement.

        Unlike update_vector() which takes a single body vector,
        the star tracker gives a complete attitude quaternion directly.
        The innovation is the 3-component error quaternion vector part
        between the measured and estimated attitude.

        Parameters
        ----------
        q_meas : measured quaternion [w,x,y,z] from star tracker
        R_st   : 3x3 noise covariance (from StarTracker.R_st)

        Derivation:
            Error quaternion: dq = q_meas ⊗ q_hat*
            For small errors: dq ≈ [1, delta_theta/2]
            Innovation: y = 2 * dq[1:4]  (attitude error 3-vector)
            H = [I_3x3 | 0_3x3]  (measurement depends only on attitude, not bias)
        """
        if q_meas is None:
            return

        # Sign consistency — choose the hemisphere closest to current estimate
        if np.dot(q_meas, self.q) < 0:
            q_meas = -q_meas

        # Error quaternion: dq = q_meas ⊗ q_hat_conj
        q_hat_conj = np.array([self.q[0], -self.q[1], -self.q[2], -self.q[3]])
        dq = quat_multiply(q_meas, q_hat_conj)
        if dq[0] < 0:
            dq = -dq

        # Innovation: 3-vector attitude error
        # dq ≈ [1, delta_theta/2] for small errors → y = 2*dq[1:4]
        y = 2.0 * dq[1:4]

        # Measurement Jacobian H (3x6): attitude states only, not bias
        H = np.hstack([np.eye(3), np.zeros((3, 3))])

        # Innovation covariance
        S = H @ self.P @ H.T + R_st

        # Mahalanobis gate — reject if error > 5-sigma
        try:
            S_inv = np.linalg.inv(S)
            mahal = float(y @ S_inv @ y)
        except np.linalg.LinAlgError:
            return
        if mahal > 25.0:   # 5-sigma gate
            return

        # Kalman gain
        K  = self.P @ H.T @ S_inv

        # State update
        dx = K @ y

        # Joseph-form covariance update (numerically stable)
        IKH    = np.eye(6) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R_st @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        eigvals = np.linalg.eigvalsh(self.P)
        if np.any(eigvals < 0):
            self.P += (-np.min(eigvals) + 1e-12) * np.eye(6)

        # Apply correction and reset
        dq_corr    = np.hstack([1.0, 0.5 * dx[0:3]])
        self.q     = normalize(quat_multiply(dq_corr, self.q))
        self.bias += dx[3:6]