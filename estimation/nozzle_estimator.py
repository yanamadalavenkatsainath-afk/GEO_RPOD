"""
estimation/nozzle_estimator.py

Non-cooperative target feature estimator: detects a ring-shaped protruding
feature (engine bell / LAE nozzle) from an accumulated point cloud.

No geometry prior — works from raw XYZ hits only.

Algorithm
---------
1. Accumulate point cloud (rolling window) in chief-CoM-centred LVLH.
2. SVD → 3 principal axes of the cloud.
3. At each of 6 axis-extreme pairs, collect nearby points.
4. RANSAC circle fit in the plane perpendicular to that axis.
5. Accept fits with radius in search band and inliers ≥ minimum.
6. Best fit (most inliers, lowest residual) → feature position, axis, radius.
7. False-positive score: fraction of competing valid fits across all 6 probes.
8. Temporal stability: estimate must hold for N seconds before GNC commits.

GNC-readable interface
----------------------
  estimate          (3,) LVLH [m]   — feature centre position
  axis              (3,) unit       — approach axis (points FROM exit TOWARD deputy)
  confidence        float [0,1]
  radius            float [m]       — estimated feature radius from RANSAC
  radius_residual   float [m]       — mean inlier residual; lower = better fit
  inlier_count      int             — RANSAC inlier count
  false_positive_score float [0,1]  — competing valid fits / total probes; 0 = unambiguous
  stable_for_s      float [s]       — consecutive seconds above conf threshold
  estimate_drift_m  float [m]       — position drift over stability window
"""

import numpy as np


# ── Algorithm constants ────────────────────────────────────────────────────────

# Feature radius search band [m] — covers common GEO LAE bell diameters
_R_MIN = 0.10
_R_MAX = 0.45

# RANSAC
_RANSAC_ITERS = 80
_RANSAC_TOL_M = 0.04   # inlier band around fitted circle [m]
_MIN_INLIERS  = 6

# Rolling point cloud window (steps)
_WINDOW       = 400

# Steps to retain for drift estimation (at DT_OUTER = 0.1 s → 6 s window)
_HISTORY_LEN  = 60


# ── Geometry helpers ───────────────────────────────────────────────────────────

def _circle_from_3pts(p1, p2, p3):
    """Circumscribed circle of 3 2-D points. Returns (cx, cy, r) or None."""
    ax, ay = p1;  bx, by = p2;  cx, cy = p3
    d = 2.0 * (ax*(by - cy) + bx*(cy - ay) + cx*(ay - by))
    if abs(d) < 1e-10:
        return None
    ux = ((ax*ax + ay*ay)*(by - cy) + (bx*bx + by*by)*(cy - ay)
          + (cx*cx + cy*cy)*(ay - by)) / d
    uy = ((ax*ax + ay*ay)*(cx - bx) + (bx*bx + by*by)*(ax - cx)
          + (cx*cx + cy*cy)*(bx - ax)) / d
    r  = np.sqrt((ax - ux)**2 + (ay - uy)**2)
    return ux, uy, r


def _ransac_circle(pts2d, n_iter=_RANSAC_ITERS, tol=_RANSAC_TOL_M):
    """
    RANSAC circle fit on 2-D points.
    Returns (centre2d, radius, inlier_mask, mean_residual).
    """
    if len(pts2d) < 3:
        return None, 0.0, np.zeros(len(pts2d), bool), np.inf

    best_inliers  = 0
    best_result   = None
    best_residual = np.inf

    for _ in range(n_iter):
        idx = np.random.choice(len(pts2d), 3, replace=False)
        res = _circle_from_3pts(*pts2d[idx])
        if res is None:
            continue
        cx, cy, r = res
        if not (_R_MIN < r < _R_MAX):
            continue
        dists   = np.sqrt((pts2d[:, 0] - cx)**2 + (pts2d[:, 1] - cy)**2)
        inliers = np.abs(dists - r) < tol
        n_in    = inliers.sum()
        residual = float(np.mean(np.abs(dists[inliers] - r))) if n_in else np.inf
        if n_in > best_inliers or (n_in == best_inliers and residual < best_residual):
            best_inliers  = n_in
            best_residual = residual
            best_result   = (np.array([cx, cy]), r, inliers, residual)

    if best_result is None:
        return None, 0.0, np.zeros(len(pts2d), bool), np.inf
    return best_result


# ── Estimator class ────────────────────────────────────────────────────────────

class NozzleEstimator:
    """
    Online ring-feature detector from lidar point cloud.

    Parameters
    ----------
    min_pts      : minimum cloud size before attempting detection
    conf_decay   : confidence decay per update when no fit found
    conf_threshold: threshold for stable_for_s counter
    """

    def __init__(self,
                 min_pts:        int   = 40,
                 conf_decay:     float = 0.02,
                 conf_threshold: float = 0.60):
        # Core state
        self._cloud           = np.zeros((0, 3))
        self._estimate        = np.zeros(3)
        self._axis            = np.array([0., 0., 1.])  # approach axis (from exit toward deputy)
        self._radius          = 0.0
        self._radius_residual = np.inf
        self._inlier_count    = 0
        self._fp_score        = 0.0
        self._confidence      = 0.0
        self._initialized     = False

        # Temporal stability
        self._conf_threshold    = conf_threshold
        self._stable_steps      = 0
        self._stable_s          = 0.0
        self._estimate_history  = []   # list of (3,) arrays
        self._drift_m           = 0.0

        self._min_pts    = min_pts
        self._conf_decay = conf_decay

    # ── Public interface ──────────────────────────────────────────────────────

    def update(self,
               new_pts:        np.ndarray,
               chief_com_lvlh: np.ndarray,
               dt:             float = 0.1):
        """
        Ingest new lidar hits and refresh all estimates.

        Parameters
        ----------
        new_pts        : (K, 3) hit points in LVLH
        chief_com_lvlh : (3,) chief CoM position in LVLH
        dt             : time step [s] (for stable_for_s counter)
        """
        if len(new_pts) == 0:
            self._confidence  = max(0.0, self._confidence - self._conf_decay)
            if self._confidence < self._conf_threshold:
                self._stable_steps = 0
                self._stable_s     = 0.0
                self._estimate_history.clear()
                self._drift_m = 0.0
            return

        centred      = new_pts - chief_com_lvlh
        self._cloud  = np.vstack([self._cloud, centred])[-_WINDOW:]

        if len(self._cloud) < self._min_pts:
            return

        self._detect(chief_com_lvlh)

        # Temporal stability tracking
        if self._initialized and self._confidence >= self._conf_threshold:
            self._stable_steps += 1
            self._stable_s      = self._stable_steps * dt
            self._estimate_history.append(self._estimate.copy())
            if len(self._estimate_history) > _HISTORY_LEN:
                self._estimate_history.pop(0)
            if len(self._estimate_history) >= 2:
                self._drift_m = float(np.linalg.norm(
                    self._estimate_history[-1] - self._estimate_history[0]))
        else:
            self._stable_steps = 0
            self._stable_s     = 0.0
            self._estimate_history.clear()
            self._drift_m      = 0.0

    @property
    def estimate(self) -> np.ndarray:
        """Estimated feature centre in LVLH [m]."""
        return self._estimate.copy()

    @property
    def axis(self) -> np.ndarray:
        """
        Approach axis unit vector in LVLH.
        Points FROM the feature exit plane TOWARD the approaching vehicle.
        The deputy should approach along this direction.
        """
        return self._axis.copy()

    @property
    def confidence(self) -> float:
        """Detection confidence [0, 1]."""
        return self._confidence

    @property
    def radius(self) -> float:
        """Estimated feature radius from RANSAC [m]."""
        return self._radius

    @property
    def radius_residual(self) -> float:
        """Mean inlier distance from fitted circle [m]. Lower = better fit."""
        return self._radius_residual

    @property
    def inlier_count(self) -> int:
        """Number of RANSAC inliers in best fit."""
        return self._inlier_count

    @property
    def false_positive_score(self) -> float:
        """
        Fraction of competing valid circle fits across all probed extremes.
        0.0  = exactly one valid fit (unambiguous).
        0.5+ = multiple competing fits (scene is geometrically ambiguous).
        """
        return self._fp_score

    @property
    def stable_for_s(self) -> float:
        """Seconds the estimate has been continuously above conf_threshold."""
        return self._stable_s

    @property
    def estimate_drift_m(self) -> float:
        """Estimate position drift over the stability window [m]."""
        return self._drift_m

    @property
    def is_valid(self) -> bool:
        return self._initialized and self._confidence > 0.30

    def reset(self):
        self._cloud           = np.zeros((0, 3))
        self._estimate        = np.zeros(3)
        self._axis            = np.array([0., 0., 1.])
        self._radius          = 0.0
        self._radius_residual = np.inf
        self._inlier_count    = 0
        self._fp_score        = 0.0
        self._confidence      = 0.0
        self._initialized     = False
        self._stable_steps    = 0
        self._stable_s        = 0.0
        self._estimate_history.clear()
        self._drift_m         = 0.0

    # ── Internals ─────────────────────────────────────────────────────────────

    def _detect(self, chief_com_lvlh: np.ndarray):
        cloud = self._cloud

        # PCA: three principal axes of the accumulated cloud
        _, _, Vt = np.linalg.svd(cloud - cloud.mean(axis=0), full_matrices=False)
        axes = Vt  # rows are principal axes, descending variance

        best_inliers  = _MIN_INLIERS - 1
        best_residual = np.inf
        best_centre   = None
        best_axis_dir = None
        best_radius   = 0.0

        n_attempts  = 0   # total axis-extreme probes
        n_valid     = 0   # probes that yielded a valid circle fit

        for axis in axes:
            proj = cloud @ axis
            p05  = np.percentile(proj, 5)
            p95  = np.percentile(proj, 95)

            for sign, extreme_val in [(+1, p95), (-1, p05)]:
                n_attempts += 1

                # Cluster points near this axis extreme
                mask = np.abs(proj - extreme_val) < 0.20
                if mask.sum() < 3:
                    continue

                cluster = cloud[mask]

                # Local 2-D basis perpendicular to axis
                u = np.cross(axis, [0., 0., 1.])
                if np.linalg.norm(u) < 1e-6:
                    u = np.cross(axis, [1., 0., 0.])
                u /= np.linalg.norm(u)
                v  = np.cross(axis, u)

                pts2d = np.column_stack([cluster @ u, cluster @ v])
                centre2d, radius, inliers, residual = _ransac_circle(pts2d)

                if centre2d is None or inliers.sum() < _MIN_INLIERS:
                    continue

                n_in = inliers.sum()
                n_valid += 1

                if (n_in > best_inliers
                        or (n_in == best_inliers and residual < best_residual)):
                    best_inliers  = n_in
                    best_residual = residual
                    best_radius   = radius
                    # 3-D centre in CoM-centred frame
                    c3d = (centre2d[0] * u + centre2d[1] * v
                           + extreme_val * axis)
                    best_centre = c3d
                    # Approach axis: sign * axis points FROM body center TO feature.
                    # Negate so it points FROM exit TOWARD the approaching vehicle.
                    best_axis_dir = -float(sign) * axis

        if best_centre is not None:
            nozzle_lvlh = best_centre + chief_com_lvlh
            alpha = 0.30 if self._initialized else 1.0
            self._estimate        = (1.0 - alpha) * self._estimate + alpha * nozzle_lvlh
            ax_norm = np.linalg.norm(best_axis_dir)
            self._axis            = best_axis_dir / max(ax_norm, 1e-12)
            self._radius          = best_radius
            self._radius_residual = best_residual
            self._inlier_count    = best_inliers
            inlier_conf           = min(1.0, best_inliers / 20.0)
            self._confidence      = min(1.0, self._confidence + 0.08 * inlier_conf)
            self._initialized     = True
            # False-positive score: fraction of OTHER valid fits (excluding the best)
            self._fp_score = max(0.0, (n_valid - 1) / max(1, n_attempts))
        else:
            self._confidence = max(0.0, self._confidence - self._conf_decay)
            # Keep fp_score from last successful detection
