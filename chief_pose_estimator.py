"""
Chief Pose Estimator — Vision-Based Tumble Rate Estimation
==========================================================
Estimates the chief's angular rate (omega) from consecutive camera
measurements, removing the last truth dependency from guidance.

Pipeline
--------
1. Camera projects known 3D model points (chief body frame) → 2D pixels
   accounting for chief attitude (passed in from chief_att.q).
   NOTE: chief_att.q is still truth for the projection step — this is
   physically correct because the camera sees the actual tumbling target.
   What we're estimating is omega, not the attitude itself.

2. PnP-lite recovers the chief body orientation in the camera frame
   from the 2D-3D correspondences.

3. Finite-difference between consecutive orientation estimates gives
   the angular rate: omega_est ≈ delta_angle_vector / dt

4. A first-order low-pass filter smooths the estimate:
   omega_filt = (1 - alpha) * omega_filt + alpha * omega_raw

Output
------
omega_est_body : estimated angular rate in chief body frame [rad/s]
                 Replaces chief_att.omega_body in:
                   - port velocity feedforward to TERMINAL guidance
                   - port velocity in docking capture check

Accuracy
--------
At 0.116 deg/s tumble rate and camera dt=0.1s:
  delta_angle ~ 0.0116 deg = 0.202 mrad per frame
  pixel noise ~ 1.5px on 800px focal length ~ 1.9 mrad pointing error
  SNR ~ 0.1 — single-frame estimate is very noisy

Solution: average over N_avg frames before differencing, giving
  effective dt = N_avg * 0.1s
  At N_avg=50 (5s): delta_angle ~ 0.58 deg >> noise → SNR ~ 5

This gives ~20% accuracy on omega — sufficient for port velocity
feedforward (which needs ~1mm/s accuracy on a 0.5m port at 0.1 deg/s).

Reference:
    Opromolla et al., "Pose estimation for spacecraft relative navigation
    using model-based algorithms", IEEE Aerosp. Electron. Syst. 2017
    Sharma et al., "Pose estimation for non-cooperative spacecraft
    rendezvous using CNN", Acta Astronautica 2018
"""

import numpy as np


def _rot_matrix(q):
    """Quaternion [w,x,y,z] → 3x3 rotation matrix (body→frame)."""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1-2*(y*y+z*z),  2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),    1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),    2*(y*z+w*x),   1-2*(x*x+y*y)],
    ])


def _rotation_to_axis_angle(R):
    """
    Extract rotation vector (axis * angle) from rotation matrix.
    Returns 3-vector with magnitude = rotation angle [rad].
    Rodrigues formula, numerically stable via atan2.
    """
    # Rotation angle from trace
    cos_angle = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos_angle)

    if angle < 1e-8:
        return np.zeros(3)

    # Rotation axis from skew-symmetric part
    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ]) / (2.0 * np.sin(angle) + 1e-12)

    return axis * angle


class ChiefPoseEstimator:
    """
    Estimates chief body angular rate from camera feature tracking.

    Parameters
    ----------
    cam_sensor    : CameraSensor instance (used for feature projection)
    dt            : simulation timestep [s]
    N_avg         : number of frames to average before differencing.
                    Higher = less noise, more latency.
                    Default 50 → 5s effective window at dt=0.1s.
    alpha_filter  : low-pass filter coefficient (0=static, 1=no filter).
                    Default 0.3 — moderate smoothing.
    sigma_omega   : omega measurement noise 1-sigma [rad/s].
                    Used to set uncertainty on estimate.
                    Default 0.002 rad/s (~0.11 deg/s).
    """

    def __init__(self,
                 cam_sensor,
                 dt:          float = 0.1,
                 N_avg:       int   = 50,
                 alpha_filter: float = 0.3,
                 sigma_omega: float = 0.002):

        self.cam    = cam_sensor
        self.dt     = dt
        self.N_avg  = N_avg
        self.alpha  = alpha_filter
        self.sigma  = sigma_omega

        # State
        self._R_prev      = None    # orientation estimate at last window
        self._frame_count = 0       # frames since last window update
        self._omega_filt  = np.zeros(3)   # filtered omega estimate [rad/s]
        self._omega_raw   = np.zeros(3)   # unfiltered omega [rad/s]
        self._valid       = False   # True once first estimate is available
        self._stale_frames = 0   # consecutive frames with no camera measurement

        # Accumulate rotation within window using incremental updates
        self._R_accum     = np.eye(3)   # accumulated rotation this window

        print(f"  ChiefPoseEstimator: N_avg={N_avg} frames  "
              f"effective_window={N_avg*dt:.1f}s  "
              f"sigma_omega={np.degrees(sigma_omega):.3f} deg/s")

    def update(self,
               dr_lvlh:   np.ndarray,
               q_chief:   np.ndarray,
               R_l2e_inv: np.ndarray = None
               ) -> tuple:
        """
        Update omega estimate from one camera frame.

        Parameters
        ----------
        dr_lvlh   : true relative position of chief in LVLH [m]
                    (used to pass to camera sensor for projection)
        q_chief   : chief attitude quaternion [w,x,y,z] — truth orientation
                    used by camera to project tumbling features correctly.
                    This is NOT a truth dependency for omega: we are using
                    the actual visual appearance of the tumbling target.
        R_l2e_inv : unused, kept for API compatibility

        Returns
        -------
        omega_est : estimated angular rate in chief body frame [rad/s]
        valid     : True if estimate has converged (>= 2 windows seen)
        """
        # Get current orientation estimate from PnP
        R_curr = self._estimate_orientation(dr_lvlh, q_chief)

        # ── Omega fallback ───────────────────────────────────────────
        # MC analysis: omega_err=3.8deg/s mean driven by stale estimates
        # in cam_ok<5% runs. When camera is unavailable, returning zero
        # is safer than returning a stale estimate that drifts unboundedly.
        FALLBACK_FRAMES = 100   # 10s at dt=0.1s — after this, go to zero
        if R_curr is None:
            # No valid measurement this frame.
            self._frame_count += 1
            # If window has stalled badly, return zero instead of stale
            if self._frame_count > FALLBACK_FRAMES * 2:
                return np.zeros(3), False
            return self._omega_filt.copy(), self._valid

        if not self._valid:
            return np.zeros(3), False

        # Accumulate rotation within the current window
        if self._frame_count == 0:
            self._R_window_start = R_curr.copy()

        # Incremental rotation: R_rel = R_curr @ R_prev_frame^T
        # We track the rotation from window start to now
        self._frame_count += 1

        if self._frame_count >= self.N_avg:
            # Window complete — compute angular rate from total rotation
            R_delta = R_curr @ self._R_window_start.T
            angle_vec = _rotation_to_axis_angle(R_delta)

            # omega = angle_vector / window_time
            window_time = self.N_avg * self.dt
            self._omega_raw = angle_vec / window_time

            # Low-pass filter
            self._omega_filt = ((1.0 - self.alpha) * self._omega_filt
                                + self.alpha * self._omega_raw)

            # Reset for next window
            self._frame_count = 0
            self._valid = True

            angle_deg = np.degrees(np.linalg.norm(angle_vec))
            omega_deg = np.degrees(np.linalg.norm(self._omega_filt))

        # ── Omega fallback ──────────────────────────────────────────
        # If camera has been unavailable for > FALLBACK_FRAMES consecutive
        # steps, the pose estimator has no data to work with. In that case,
        # returning a stale filtered estimate is worse than returning zero
        # (stale estimate drifts unboundedly; zero gives no feedforward but
        # also no wrong feedforward). MC analysis showed omega_err=3.8deg/s
        # mean, driven by stale estimates in cam_ok<5% runs.
        FALLBACK_FRAMES = 100   # 10 seconds at dt=0.1s
        if not self._valid or self._frame_count > FALLBACK_FRAMES * 2:
            # No valid estimate yet, or window has stalled — return zero
            return np.zeros(3), False
        return self._omega_filt.copy(), self._valid

    def _estimate_orientation(self,
                               dr_lvlh: np.ndarray,
                               q_chief:  np.ndarray
                               ) -> np.ndarray:
        """
        Estimate chief body orientation in LVLH from camera projection.

        Uses PnP-lite: match projected 2D points back to 3D model to
        recover the rotation R_body2lvlh.

        Steps:
        1. Get projected pixel coordinates (from camera sensor, which uses
           the truth q_chief to project the tumbling model correctly)
        2. Build camera-frame unit rays from pixels
        3. Match rays to known 3D model points
        4. Solve for rotation via SVD (Wahba's problem)

        Returns
        -------
        R_body2lvlh : 3x3 rotation matrix or None if < 4 points visible
        """
        true_range = float(np.linalg.norm(dr_lvlh))
        if true_range < self.cam.min_range or true_range > self.cam.max_range:
            return None

        # Build camera frame (same as in CameraSensor.measure)
        r_hat    = dr_lvlh / true_range
        world_up = np.array([0., 0., 1.])
        if abs(np.dot(r_hat, world_up)) > 0.99:
            world_up = np.array([0., 1., 0.])
        cam_X = np.cross(world_up, r_hat)
        cam_X /= np.linalg.norm(cam_X)
        cam_Y  = np.cross(r_hat, cam_X)
        # R_l2c: transforms LVLH → camera frame
        R_l2c  = np.vstack([cam_X, cam_Y, r_hat])   # (3,3)

        # Get truth rotation for projection
        R_body2lvlh = _rot_matrix(q_chief)

        # Project model points
        pts_cam = (R_l2c @ R_body2lvlh @ self.cam.model_pts.T).T
        pts_cam += (R_l2c @ dr_lvlh).reshape(1, 3)

        # Collect visible points and add pixel noise
        rays_cam  = []   # back-projected unit rays in camera frame
        pts_body  = []   # corresponding 3D model points in body frame

        for i, P_c in enumerate(pts_cam):
            Z = P_c[2]
            if Z <= 0.01:
                continue
            u = self.cam.f * P_c[0] / Z + self.cam.cx
            v = self.cam.f * P_c[1] / Z + self.cam.cy
            if not (0 <= u < self.cam.W and 0 <= v < self.cam.H):
                continue

            # Add pixel noise
            u_n = u + np.random.normal(0, self.cam.sigma_px)
            v_n = v + np.random.normal(0, self.cam.sigma_px)

            # Back-project to unit ray in camera frame
            ray = np.array([(u_n - self.cam.cx) / self.cam.f,
                             (v_n - self.cam.cy) / self.cam.f,
                             1.0])
            ray /= np.linalg.norm(ray)
            rays_cam.append(ray)
            pts_body.append(self.cam.model_pts[i])

        if len(rays_cam) < 4:
            return None

        rays_cam = np.array(rays_cam)   # (M, 3)
        pts_body = np.array(pts_body)   # (M, 3)

        # Wahba's problem: find R that best aligns body vectors to camera rays
        # body vectors: model points normalized from centroid
        # camera vectors: back-projected rays (approximate directions)
        #
        # Estimate depth of each point from range + z-offset
        centroid_body = np.mean(pts_body, axis=0)
        centroid_cam  = np.mean(rays_cam, axis=0)
        centroid_cam /= np.linalg.norm(centroid_cam)

        # Demeaned model points (shape vectors)
        b_vecs = pts_body - centroid_body   # (M, 3) in body frame

        # Demeaned ray directions (approximate shape vectors in camera frame)
        # Scale by range to get approximate metric positions
        pts_cam_metric = rays_cam * true_range   # rough scale
        c_vecs = pts_cam_metric - np.mean(pts_cam_metric, axis=0)   # (M, 3)

        # SVD solution to Wahba: R = V @ diag(1,1,det(VU^T)) @ U^T
        # where A = sum(w_i * b_i @ c_i^T) = U @ S @ V^T
        W = np.zeros((3, 3))
        for bv, cv in zip(b_vecs, c_vecs):
            W += np.outer(bv, cv)

        try:
            U, S, Vt = np.linalg.svd(W)
        except np.linalg.LinAlgError:
            return None

        d = np.linalg.det(Vt.T @ U.T)
        R_est = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T

        # R_est maps body → camera frame
        # Convert to body → LVLH: R_body2lvlh = R_l2c^T @ R_est
        R_body2lvlh_est = R_l2c.T @ R_est

        return R_body2lvlh_est

    @property
    def omega_estimate(self) -> np.ndarray:
        """Current best omega estimate in chief body frame [rad/s]."""
        return self._omega_filt.copy()

    @property
    def omega_uncertainty_rad_s(self) -> float:
        """1-sigma uncertainty on omega magnitude [rad/s]."""
        if not self._valid:
            return 1.0   # large uncertainty before first estimate
        return self.sigma

    @property
    def is_valid(self) -> bool:
        return self._valid