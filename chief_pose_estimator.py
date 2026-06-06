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
                 gate_k:               float = 10.0,
                 N_avg:                int   = 50,      # kept for API compat
                 alpha_filter:         float = 0.3,     # kept for API compat
                 sigma_omega:          float = 0.002,   # kept for API compat
                 pose_model_pts=None,                   # override cam model_pts
                 max_reproj_rms_px:    float = 50.0):

        self.cam   = cam_sensor
        # Use caller-supplied chief feature model or fall back to camera
        # sensor default.  The pose model must match the chief's physical
        # feature geometry — IS-1002 bus corners plus an asymmetric solar
        # array feature — so EPnP has three distinct PCA eigenvalues and
        # can resolve the dock-axis rotation without ambiguity.
        self._pose_pts = (np.asarray(pose_model_pts, dtype=float)
                          if pose_model_pts is not None
                          else cam_sensor.model_pts)
        self.dt    = dt
        self.gate_k = gate_k
        self.max_reproj_rms_px = max_reproj_rms_px

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
        self._pose_age_s    = np.inf
        self._debug = {
            "status": 0,             # 0 none, 1 accepted, 2 rejected, 3 coast, 4 no visible, 5 pnp fail, 6 rms reject, 7 acquire
            "visible_count": 0,
            "visible_mask": 0,
            "stub_visible": False,
            "reproj_rms_px": np.nan,
            "pca_s0": np.nan,
            "pca_s1": np.nan,
            "pca_s2": np.nan,
            "pca_cond": np.nan,
            "pose_age_s": np.inf,
            "update_count": 0,
        }

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
        self._pose_age_s += self.dt
        self._debug.update({
            "status": 0,
            "visible_count": 0,
            "visible_mask": 0,
            "stub_visible": False,
            "reproj_rms_px": np.nan,
            "pca_s0": np.nan,
            "pca_s1": np.nan,
            "pca_s2": np.nan,
            "pca_cond": np.nan,
            "pose_age_s": self._pose_age_s,
            "update_count": self._update_count,
        })

        # Reopen the Mahalanobis gate when pose is stale.
        # Lock-in occurs when the EKF converged to a wrong orientation with tight P:
        # correct PnP solutions are then rejected indefinitely (mah2 >> gate_k²).
        # Inflating P_att back to ≥ (30 deg)² lets the EKF accept measurements up
        # to ~30 deg from its current state and self-correct the wrong prior.
        if self._pose_age_s > 60.0 and self._update_count >= 10:
            _att_floor = np.radians(30.0) ** 2
            for i in range(3):
                if self._P[i, i] < _att_floor:
                    self._P[i, i] = _att_floor

        true_range = float(np.linalg.norm(dr_lvlh))

        # ── Close-range coast: skip EKF update, refresh port geometry ─
        if true_range < 2.0:
            # Use EKF's propagated attitude as the best available estimate.
            # The EKF integrates the coasted omega, giving a smooth, noise-free
            # orientation without any truth dependency.
            self._last_R_b2l = _rot_matrix(self._q)
            self._debug["status"] = 3
            self._debug["pose_age_s"] = self._pose_age_s
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
            _rms = float(self._debug.get("reproj_rms_px", np.nan))
            if np.isfinite(_rms) and _rms > self.max_reproj_rms_px:
                self._debug["status"] = 6
            elif self._pose_age_s > 10.0 and np.isfinite(_rms) and _rms < 10.0:
                # Acquisition mode: EKF is stale and the image measurement is
                # reliable (RMS-gated).  Bypass the Mahalanobis gate — the prior
                # is too old to trust for rejection decisions — and reseed directly.
                self._q = _rot_matrix_to_quat(R_meas)
                self._omega = np.zeros(3)       # stale omega direction is untrusted; start fresh
                self._P = np.diag([np.radians(5.0)**2]*3 +
                                  [np.radians(3.0)**2]*3)
                self._last_R_b2l = R_meas.copy()
                self._pose_age_s = 0.0
                self._update_count += 1
                self._debug["status"] = 7
                self._debug["update_count"] = self._update_count
            elif self._update(R_meas, R_override=R_use):
                self._last_R_b2l = R_meas.copy()
                self._pose_age_s = 0.0
                self._debug["status"] = 1
                self._debug["update_count"] = self._update_count
            else:
                if np.isfinite(_rms) and _rms < 5.0:
                    self._q = _rot_matrix_to_quat(R_meas)
                    self._omega = np.zeros(3)
                    self._P = np.diag([np.radians(5.0)**2]*3 +
                                      [np.radians(3.0)**2]*3)
                    self._last_R_b2l = R_meas.copy()
                    self._pose_age_s = 0.0
                    self._update_count += 1
                    self._debug["status"] = 7
                    self._debug["update_count"] = self._update_count
                else:
                    self._debug["status"] = 2
            self._debug["pose_age_s"] = self._pose_age_s

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
            return False
        mah2 = float(z_err @ S_inv @ z_err)
        if mah2 > self.gate_k**2:
            return False   # reject outlier PnP solution

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
        return True

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
        pts_cam = (R_l2c @ R_body2lvlh @ self._pose_pts.T).T
        pts_cam += (R_l2c @ dr_lvlh).reshape(1, 3)

        # Collect visible points with pixel noise
        px_obs   = []   # noisy pixel observations (M, 2)
        pts_body = []   # corresponding body-frame 3D points (M, 3)
        vis_idx  = []
        for i, P_c in enumerate(pts_cam):
            Z = P_c[2]
            if Z <= 0.01: continue
            u = self.cam.f * P_c[0] / Z + self.cam.cx
            v = self.cam.f * P_c[1] / Z + self.cam.cy
            if not (0 <= u < self.cam.W and 0 <= v < self.cam.H): continue
            u_n = u + np.random.normal(0, self.cam.sigma_px)
            v_n = v + np.random.normal(0, self.cam.sigma_px)
            px_obs.append([u_n, v_n])
            pts_body.append(self._pose_pts[i])
            vis_idx.append(i)

        visible_mask = 0
        for i in vis_idx:
            if i < 63:
                visible_mask |= (1 << int(i))
        self._debug.update({
            "visible_count": len(vis_idx),
            "visible_mask": visible_mask,
            "stub_visible": bool(8 in vis_idx or 9 in vis_idx),
        })

        if len(px_obs) < 4:
            self._debug["status"] = 4
            return None

        px_obs   = np.array(px_obs)    # (M, 2)
        pts_body = np.array(pts_body)  # (M, 3)
        M        = len(px_obs)

        f  = self.cam.f
        cx = self.cam.cx
        cy = self.cam.cy

        # Known translation: chief center expressed in camera frame.
        # EPnP had to estimate this from pixels (beta scale ambiguity → wrong).
        # We get it for free from the EKF nav solution.
        t = R_l2c @ dr_lvlh   # (3,)

        # PCA diagnostics (kept for telemetry continuity)
        _, S_pca, _ = np.linalg.svd(pts_body - np.mean(pts_body, axis=0),
                                     full_matrices=False)
        s_min = max(float(S_pca[-1]), 1e-12) if len(S_pca) >= 3 else 1e-12
        self._debug.update({
            "pca_s0": float(S_pca[0]),
            "pca_s1": float(S_pca[1]) if len(S_pca) > 1 else np.nan,
            "pca_s2": float(S_pca[2]) if len(S_pca) > 2 else np.nan,
            "pca_cond": float(S_pca[0]) / s_min,
        })

        # ── Step 1: DLT initialization (linear, known t) ──────────
        # From the pinhole model:
        #   u = f*(r1@b + t[0])/(r3@b + t[2]) + cx
        #   v = f*(r2@b + t[1])/(r3@b + t[2]) + cy
        # Cross-multiplying (du = u-cx, dv = v-cy):
        #   -f*(r1@b) + du*(r3@b) = f*t[0] - du*t[2]
        #   -f*(r2@b) + dv*(r3@b) = f*t[1] - dv*t[2]
        # Stack into linear system A x = b where x = [r1; r2; r3].
        A_dlt = np.zeros((2 * M, 9))
        b_dlt = np.zeros(2 * M)
        for i in range(M):
            pt_b = pts_body[i]
            du   = float(px_obs[i, 0]) - cx
            dv   = float(px_obs[i, 1]) - cy
            A_dlt[2*i,   0:3] = -f * pt_b
            A_dlt[2*i,   6:9] =  du * pt_b
            b_dlt[2*i]        =  f * t[0] - du * t[2]
            A_dlt[2*i+1, 3:6] = -f * pt_b
            A_dlt[2*i+1, 6:9] =  dv * pt_b
            b_dlt[2*i+1]      =  f * t[1] - dv * t[2]

        try:
            x_dlt, _, _, _ = np.linalg.lstsq(A_dlt, b_dlt, rcond=None)
            U_d, _, Vt_d   = np.linalg.svd(x_dlt.reshape(3, 3))
        except np.linalg.LinAlgError:
            self._debug["status"] = 5
            return None
        R_body2cam = U_d @ np.diag([1., 1., np.linalg.det(U_d @ Vt_d)]) @ Vt_d

        # ── Step 2: Gauss-Newton refinement on pixel reproj error ──
        # Left perturbation: R ← exp(hat(δφ)) @ R (camera-frame parametrisation).
        # Jacobian of projection w.r.t. δφ:  J_proj @ (−hat(c − t))
        # where c = R_body2cam @ b + t.
        for _ in range(8):
            J_full = np.zeros((2 * M, 3))
            r_full = np.zeros(2 * M)
            n_valid = 0
            for i in range(M):
                c = R_body2cam @ pts_body[i] + t
                Z = c[2]
                if Z < 0.01:
                    continue
                r_full[2*i]   = float(px_obs[i, 0]) - (f * c[0] / Z + cx)
                r_full[2*i+1] = float(px_obs[i, 1]) - (f * c[1] / Z + cy)
                J_proj = np.array([[f/Z,   0,   -f * c[0] / Z**2],
                                   [0,     f/Z, -f * c[1] / Z**2]])
                ct = c - t
                hat_ct = np.array([[    0,  -ct[2],  ct[1]],
                                   [ ct[2],      0, -ct[0]],
                                   [-ct[1],  ct[0],      0]])
                J_full[2*i:2*i+2, :] = J_proj @ (-hat_ct)
                n_valid += 2
            if n_valid < 6:
                break
            try:
                dw, _, _, _ = np.linalg.lstsq(J_full, r_full, rcond=None)
            except np.linalg.LinAlgError:
                break
            theta = float(np.linalg.norm(dw))
            if theta < 1e-9:
                break
            K   = np.array([[    0,  -dw[2],  dw[1]],
                             [ dw[2],      0, -dw[0]],
                             [-dw[1],  dw[0],      0]]) / theta
            dR  = np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
            R_body2cam = dR @ R_body2cam            # left update
            U_r, _, Vt_r = np.linalg.svd(R_body2cam)
            R_body2cam   = U_r @ Vt_r              # re-orthogonalise

        # ── Reprojection RMS with the correct (known) translation ──
        proj_err = []
        for pt_b, px in zip(pts_body, px_obs):
            pt_c = R_body2cam @ pt_b + t
            if pt_c[2] <= 0.01:
                continue
            u = f * pt_c[0] / pt_c[2] + cx
            v = f * pt_c[1] / pt_c[2] + cy
            proj_err.append(float(np.linalg.norm(np.array([u, v]) - px)))
        self._debug["reproj_rms_px"] = (
            float(np.sqrt(np.mean(np.square(proj_err)))) if proj_err else np.nan)

        return R_l2c.T @ R_body2cam

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

    @property
    def debug(self):
        return dict(self._debug)
