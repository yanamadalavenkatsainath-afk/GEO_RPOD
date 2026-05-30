"""
Chief Pose Estimator — 6-State Attitude EKF
============================================
Replaces the frame-differencing omega estimator with a proper
multiplicative EKF on the chief's [q(4), omega(3)] state.

State:   x = [q_w, q_x, q_y, q_z,  omega_x, omega_y, omega_z]
Measurement: orientation R_body2lvlh from PnP (Wahba/SVD) each frame

Advantages over frame-differencing
-----------------------------------
- Continuous estimate between PnP frames (gyro-free integration)
- Explicit uncertainty propagation (P matrix)
- Filter rejects outlier PnP solutions via Mahalanobis gate
- omega estimate is smooth — no window latency or LP filter artifact
- Valid flag reflects filter convergence, not window accumulation

Accuracy at GEO tumble (0.116 deg/s)
--------------------------------------
Frame-diff: SNR~5 at 5s window, ~20% accuracy, 5s latency
Pose EKF:   SNR driven by PnP noise (~2 deg per frame), but
            averaged over ALL frames → better long-run accuracy
            and zero latency (estimate valid every step)

The _estimate_orientation() PnP step is unchanged from v1.
Only the state propagation and update are new.

Reference:
    Markley & Crassidis, "Fundamentals of Spacecraft Attitude
    Determination and Control", Springer 2014, Ch. 4
    Opromolla et al., IEEE TAES 2017
"""

import numpy as np


def _rot_matrix(q):
    """Quaternion [w,x,y,z] → 3x3 rotation matrix (body → frame)."""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1-2*(y*y+z*z),  2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),    1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),    2*(y*z+w*x),   1-2*(x*x+y*y)],
    ])


def _quat_normalize(q):
    return q / np.linalg.norm(q)


def _quat_to_rvec(q):
    """Quaternion error to rotation vector (small angle)."""
    return 2.0 * q[1:4] / max(q[0], 1e-6)


def _rot_matrix_to_quat(R):
    """Rotation matrix → quaternion [w,x,y,z], numerically stable."""
    trace = R[0,0] + R[1,1] + R[2,2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        return np.array([0.25/s,
                         (R[2,1]-R[1,2])*s,
                         (R[0,2]-R[2,0])*s,
                         (R[1,0]-R[0,1])*s])
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        return np.array([(R[2,1]-R[1,2])/s, 0.25*s,
                         (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s])
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        return np.array([(R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s,
                         0.25*s, (R[1,2]+R[2,1])/s])
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        return np.array([(R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s,
                         (R[1,2]+R[2,1])/s, 0.25*s])


class ChiefPoseEstimator:
    """
    6-state EKF for chief angular rate estimation.

    State: [q(4), omega(3)] — quaternion + angular rate in body frame.
    Propagation: free Euler integration (no torque model needed for
    short timescales; gravity gradient changes omega slowly).
    Update: PnP-derived R_body2lvlh converted to quaternion measurement.

    Parameters
    ----------
    cam_sensor    : CameraSensor instance
    dt            : timestep [s]
    sigma_omega_process : process noise on omega [rad/s/sqrt(s)]
    sigma_pnp_deg : PnP orientation noise 1-sigma [deg]
    gate_k        : Mahalanobis gate (reject if chi2 > gate_k^2)
    """

    def __init__(self,
                 cam_sensor,
                 dt:                   float = 0.1,
                 sigma_omega_process:  float = 0.001,   # rad/s/sqrt(s)
                 sigma_pnp_deg:        float = 3.0,     # deg per PnP solve
                 gate_k:               float = 5.0,
                 N_avg:                int   = 50,      # kept for API compat
                 alpha_filter:         float = 0.3,     # kept for API compat
                 sigma_omega:          float = 0.002):  # kept for API compat

        self.cam   = cam_sensor
        self.dt    = dt
        self.gate_k = gate_k

        # ── EKF state ──────────────────────────────────────────────
        # Quaternion: random init (unknown chief attitude)
        q0 = np.random.randn(4)
        self._q = q0 / np.linalg.norm(q0)
        self._omega = np.zeros(3)          # angular rate estimate [rad/s]

        # Covariance: 3-element attitude error + 3-element omega
        self._P = np.diag([np.radians(30.0)**2]*3 +   # 30 deg init uncertainty
                          [np.radians(5.0)**2]*3)      # 5 deg/s omega uncertainty

        # Process noise covariance (6x6 error state)
        sig_q = np.radians(0.01) * np.sqrt(dt)        # tiny attitude diffusion
        sig_w = sigma_omega_process * np.sqrt(dt)
        self._Q = np.diag([sig_q**2]*3 + [sig_w**2]*3)

        # Measurement noise covariance (3x3 orientation error)
        sig_pnp = np.radians(sigma_pnp_deg)
        self._R_meas = np.eye(3) * sig_pnp**2

        # Validity
        self._valid         = False
        self._update_count  = 0
        self._frame_count   = 0   # kept for API compat

        # Last successful PnP R_body2lvlh — exposed directly to avoid the
        # q_est frame ambiguity. This rotation matrix is in a well-defined
        # frame (LVLH) without any ECI->LVLH conversion step.
        # None until the first successful PnP orientation estimate.
        self._last_R_b2l    = None

        print(f"  ChiefPoseEstimator (EKF): sigma_pnp={sigma_pnp_deg:.1f}deg  "
              f"sigma_omega_proc={np.degrees(sigma_omega_process):.4f}deg/s/sqrt(s)  "
              f"gate={gate_k}sigma")

    # ─────────────────────────────────────────────────────────────────
    # Public API — same signature as v1 for drop-in replacement
    # ─────────────────────────────────────────────────────────────────

    def update(self, dr_lvlh, q_chief, R_l2e_inv=None):
        """
        Run one EKF predict+update step.

        Close-range handling (<2 m):
          EPnP degenerates and the truth+noise fallback produces i.i.d.
          random orientation jumps each step.  Those jumps look like
          ~30 deg/s to the EKF, inflating the omega estimate and driving
          a spurious ~45 mm/s port velocity.  Below 2 m we skip the EKF
          update entirely and coast on the last valid omega from > 2 m;
          we only refresh _last_R_b2l from truth so port geometry stays
          accurate.  Chief tumbles at ~0.12 deg/s, so omega and R_b2l
          both drift negligibly over a 20-30 s capture window.

        Returns
        -------
        omega_est : estimated angular rate in chief body frame [rad/s]
        valid     : True after filter has converged (>= 10 updates)
        """
        # ── Predict ──────────────────────────────────────────────
        self._predict()

        true_range = float(np.linalg.norm(dr_lvlh))

        # ── Close-range coast: skip EKF update, refresh port geometry ─
        if true_range < 2.0:
            # Use EKF's propagated attitude as the best available estimate.
            # The EKF integrates the coasted omega, giving a smooth, noise-free
            # orientation without any truth dependency.
            self._last_R_b2l = _rot_matrix(self._q)
            if self._update_count >= 10:
                self._valid = True
            return self._omega.copy(), self._valid

        # ── Range-dependent R gain scheduling (2 m – far field) ──────
        if true_range < 5.0:
            r_scale = 0.25 + 0.75 * (true_range - 2.0) / 3.0
        elif true_range < 20.0:
            r_scale = 0.5
        else:
            r_scale = 1.0
        R_use = self._R_meas * r_scale

        # ── Measurement from PnP ────────────────────────────────
        R_meas = self._estimate_orientation(dr_lvlh, q_chief)
        if R_meas is not None:
            self._last_R_b2l = R_meas.copy()
            self._update(R_meas, R_override=R_use)

        # Mark valid after 10 successful updates (~1s of data)
        if self._update_count >= 10:
            self._valid = True

        return self._omega.copy(), self._valid

    # ─────────────────────────────────────────────────────────────────
    # EKF internals
    # ─────────────────────────────────────────────────────────────────

    def _predict(self):
        """Propagate state and covariance by dt."""
        dt  = self.dt
        w   = self._omega
        wx, wy, wz = w
        wmag = np.linalg.norm(w)

        # Quaternion kinematics: q_dot = 0.5 * Omega(w) * q
        Omega = 0.5 * np.array([
            [ 0,  -wx, -wy, -wz],
            [ wx,  0,   wz, -wy],
            [ wy, -wz,  0,   wx],
            [ wz,  wy, -wx,  0 ],
        ])
        self._q = _quat_normalize(self._q + dt * (Omega @ self._q))

        # omega held constant (no torque model) — free precession assumption
        # This is valid because gravity-gradient timescale >> sim step

        # ── Error-state covariance propagation ───────────────────
        # F is the 6x6 linearised dynamics for [dtheta, domega]
        # dtheta_dot = -[w x] dtheta + domega
        # domega_dot = 0 (constant omega model)
        skew_w = np.array([
            [ 0,   -wz,  wy],
            [ wz,   0,  -wx],
            [-wy,  wx,   0 ],
        ])
        F = np.zeros((6, 6))
        F[0:3, 0:3] = -skew_w
        F[0:3, 3:6] = np.eye(3)
        # F[3:6, :] = 0  (omega not driven)

        Phi = np.eye(6) + F * dt   # first-order STM
        P_new = Phi @ self._P @ Phi.T + self._Q
        # Guard against overflow: if any diagonal element exceeds 1e6,
        # the filter has lost tracking — reset to initial uncertainty.
        if np.any(np.diag(P_new) > 1e6) or not np.all(np.isfinite(P_new)):
            self._P = np.diag([np.radians(30.0)**2]*3 + [np.radians(5.0)**2]*3)
            self._omega = np.zeros(3)   # reset stale omega on divergence
            self._valid = False
            self._update_count = 0
        else:
            self._P = P_new

    def _update(self, R_meas, R_override=None):
        """EKF update from PnP rotation matrix. R_override replaces _R_meas when set."""
        R_noise = R_override if R_override is not None else self._R_meas

        R_est  = _rot_matrix(self._q)

        # Innovation: rotation error between measured and predicted
        R_err  = R_meas @ R_est.T
        # Convert rotation matrix error to rotation vector (small angle)
        cos_a  = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
        angle  = np.arccos(cos_a)
        if angle < 1e-8:
            z_err = np.zeros(3)
        else:
            axis  = np.array([R_err[2,1]-R_err[1,2],
                               R_err[0,2]-R_err[2,0],
                               R_err[1,0]-R_err[0,1]]) / (2.0*np.sin(angle)+1e-12)
            z_err = axis * angle

        # H = [I | 0]: attitude error observable, omega not directly
        H = np.hstack([np.eye(3), np.zeros((3, 3))])

        S = H @ self._P @ H.T + R_noise

        # Mahalanobis gate
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return
        mah2 = float(z_err @ S_inv @ z_err)
        if mah2 > self.gate_k**2:
            return   # reject outlier PnP solution

        # Kalman gain and Joseph-form update
        K = self._P @ H.T @ S_inv
        dx = K @ z_err   # [dtheta(3), domega(3)]

        # Apply attitude correction multiplicatively
        dq = np.array([1.0, 0.5*dx[0], 0.5*dx[1], 0.5*dx[2]])
        self._q = _quat_normalize(np.array([
            dq[0]*self._q[0] - dq[1]*self._q[1] - dq[2]*self._q[2] - dq[3]*self._q[3],
            dq[0]*self._q[1] + dq[1]*self._q[0] + dq[2]*self._q[3] - dq[3]*self._q[2],
            dq[0]*self._q[2] - dq[1]*self._q[3] + dq[2]*self._q[0] + dq[3]*self._q[1],
            dq[0]*self._q[3] + dq[1]*self._q[2] - dq[2]*self._q[1] + dq[3]*self._q[0],
        ]))

        # Apply omega correction
        self._omega += dx[3:6]

        # Joseph-form covariance update
        IKH = np.eye(6) - K @ H
        self._P = IKH @ self._P @ IKH.T + K @ R_noise @ K.T

        self._update_count += 1

    # ─────────────────────────────────────────────────────────────────
    # PnP orientation estimator — unchanged from v1
    # ─────────────────────────────────────────────────────────────────

    def _estimate_orientation(self, dr_lvlh, q_chief):
        """
        Estimate chief body orientation via EPnP from camera pixels.

        Only called for range >= 2 m (update() handles close-range separately).
        Returns R_body2lvlh or None if < 4 points visible.
        """
        true_range = float(np.linalg.norm(dr_lvlh))
        if true_range < self.cam.min_range or true_range > self.cam.max_range:
            return None

        # ── Build camera frame ─────────────────────────────────────
        r_hat    = dr_lvlh / true_range
        world_up = np.array([0., 0., 1.])
        if abs(np.dot(r_hat, world_up)) > 0.99:
            world_up = np.array([0., 1., 0.])
        cam_X = np.cross(world_up, r_hat);  cam_X /= np.linalg.norm(cam_X)
        cam_Y = np.cross(r_hat, cam_X)
        R_l2c = np.vstack([cam_X, cam_Y, r_hat])

        # ── Project model points ───────────────────────────────────
        R_body2lvlh = _rot_matrix(q_chief)
        pts_cam = (R_l2c @ R_body2lvlh @ self.cam.model_pts.T).T
        pts_cam += (R_l2c @ dr_lvlh).reshape(1, 3)

        # Collect visible points with pixel noise
        px_obs   = []   # noisy pixel observations (M, 2)
        pts_body = []   # corresponding body-frame 3D points (M, 3)
        for i, P_c in enumerate(pts_cam):
            Z = P_c[2]
            if Z <= 0.01: continue
            u = self.cam.f * P_c[0] / Z + self.cam.cx
            v = self.cam.f * P_c[1] / Z + self.cam.cy
            if not (0 <= u < self.cam.W and 0 <= v < self.cam.H): continue
            u_n = u + np.random.normal(0, self.cam.sigma_px)
            v_n = v + np.random.normal(0, self.cam.sigma_px)
            px_obs.append([u_n, v_n])
            pts_body.append(self.cam.model_pts[i])

        if len(px_obs) < 4:
            return None

        px_obs   = np.array(px_obs)    # (M, 2)
        pts_body = np.array(pts_body)  # (M, 3)
        M        = len(px_obs)

        # ── EPnP: control-point parameterisation ──────────────────
        # Reference: Lepetit, Moreno-Noguer, Fua, IJCV 2009
        #
        # Step 1: choose 4 control points in body frame.
        #   c0 = centroid, c1-c3 = principal axes (weighted PCA)
        c0   = np.mean(pts_body, axis=0)
        dpts = pts_body - c0                    # (M, 3) centred
        U, S, Vt = np.linalg.svd(dpts, full_matrices=False)
        # Control points along principal axes, scaled by singular values
        c1 = c0 + Vt[0] * S[0] / np.sqrt(M)
        c2 = c0 + Vt[1] * S[1] / np.sqrt(M)
        c3 = c0 + Vt[2] * S[2] / np.sqrt(M)
        ctrl_body = np.array([c0, c1, c2, c3])  # (4, 3)

        # Step 2: homogeneous barycentric coordinates of each 3D point
        # p_i = sum_j alpha_ij * c_j  →  alpha_i = M_ctrl^{-1} @ (p_i, 1)
        ctrl_aug = np.hstack([ctrl_body, np.ones((4, 1))])   # (4, 4)
        pts_aug  = np.hstack([pts_body,  np.ones((M, 1))])   # (M, 4)
        try:
            alphas = np.linalg.solve(ctrl_aug.T, pts_aug.T).T  # (M, 4)
        except np.linalg.LinAlgError:
            return None

        # Step 3: build the 2M × 12 linear system M @ x = 0
        # where x = [c0_cam; c1_cam; c2_cam; c3_cam] (12 unknowns)
        f  = self.cam.f
        cx = self.cam.cx; cy = self.cam.cy
        L  = np.zeros((2 * M, 12))
        for i in range(M):
            a  = alphas[i]           # (4,) barycentric coords
            ui = px_obs[i, 0]
            vi = px_obs[i, 1]
            for j in range(4):
                L[2*i,   3*j    ] =  f * a[j]
                L[2*i,   3*j + 2] = (cx - ui) * a[j]
                L[2*i+1, 3*j + 1] =  f * a[j]
                L[2*i+1, 3*j + 2] = (cy - vi) * a[j]

        # Step 4: solve via null-space of L (N=1 approximation)
        try:
            _, _, Vt_L = np.linalg.svd(L)
        except np.linalg.LinAlgError:
            return None
        x_est = Vt_L[-1]   # last right singular vector = null-space

        ctrl_cam = x_est.reshape(4, 3)  # (4, 3) control points in camera frame

        # Enforce positive depth (flip sign if needed)
        if ctrl_cam[0, 2] < 0:
            ctrl_cam = -ctrl_cam

        # Step 5: recover R from control points (Procrustes)
        # ctrl_body → ctrl_cam via rigid transform R, t
        # Subtract centroids
        c_body = np.mean(ctrl_body, axis=0)
        c_cam  = np.mean(ctrl_cam,  axis=0)
        A = (ctrl_cam  - c_cam).T  @ (ctrl_body - c_body)   # (3, 3)
        try:
            U_p, _, Vt_p = np.linalg.svd(A)
        except np.linalg.LinAlgError:
            return None
        d_p = np.linalg.det(U_p @ Vt_p)
        R_body2cam = U_p @ np.diag([1., 1., d_p]) @ Vt_p

        # R_body2cam: body → camera frame
        # R_body2lvlh = R_l2c.T @ R_body2cam
        R_body2lvlh_est = R_l2c.T @ R_body2cam
        return R_body2lvlh_est

    # ─────────────────────────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────────────────────────

    @property
    def q_est(self):
        """Estimated chief quaternion [w,x,y,z] from pose EKF.

        WARNING: This quaternion lives in the camera/measurement frame,
        not ECI. Do NOT use as R_e2l @ rot_matrix(q_est) to get R_body2lvlh.
        Use R_body2lvlh property instead, which returns the PnP result directly.
        """
        return self._q.copy()

    @property
    def R_body2lvlh(self):
        """
        Last successful PnP-derived rotation matrix: body -> LVLH frame.

        Use this directly for port offset reconstruction:
            port_lvlh = R_body2lvlh @ DOCK_PORT_BODY

        Returns None if no successful PnP has been computed yet.
        Unlike q_est, this is in a well-defined frame with no ambiguity.
        """
        return self._last_R_b2l.copy() if self._last_R_b2l is not None else None

    @property
    def omega_estimate(self):
        return self._omega.copy()

    @property
    def omega_uncertainty_rad_s(self):
        return float(np.sqrt(np.mean(np.diag(self._P)[3:6])))

    @property
    def is_valid(self):
        return self._valid