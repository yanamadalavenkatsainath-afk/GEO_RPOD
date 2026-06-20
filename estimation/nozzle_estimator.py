"""
estimation/nozzle_estimator.py

Detects the LAE nozzle location from an accumulated point cloud.
No geometry prior — works only from raw XYZ hits.

Algorithm
---------
1. Accumulate point cloud (rolling window) in chief-CoM-centred LVLH.
2. SVD → 3 principal axes of the cloud (body axes approximation).
3. At each of 6 axis extremes, collect nearby points.
4. RANSAC circle fit in the plane perpendicular to that axis.
5. Accept fits whose radius falls in the LAE nozzle diameter range.
6. Best fit (most inliers) → nozzle centre estimate + confidence [0, 1].

The GNC reads `estimate` (LVLH position) and `confidence` (0–1).
It enters TERMINAL only once confidence > NOZZLE_CONF_THRESHOLD.
"""

import numpy as np


# Nozzle radius search band [m] — covers common GEO LAE bell sizes
_R_MIN = 0.10
_R_MAX = 0.45

# RANSAC parameters
_RANSAC_ITERS    = 80
_RANSAC_TOL_M    = 0.04   # inlier distance from fitted circle [m]
_MIN_INLIERS     = 6

# Rolling window size
_WINDOW          = 400


def _circle_from_3pts(p1, p2, p3):
    """Circumscribed circle of 3 2D points. Returns (cx, cy, r) or None."""
    ax, ay = p1
    bx, by = p2
    cx, cy = p3
    d = 2.0 * (ax*(by - cy) + bx*(cy - ay) + cx*(ay - by))
    if abs(d) < 1e-10:
        return None
    ux = ((ax*ax + ay*ay)*(by - cy) + (bx*bx + by*by)*(cy - ay)
          + (cx*cx + cy*cy)*(ay - by)) / d
    uy = ((ax*ax + ay*ay)*(cx - bx) + (bx*bx + by*by)*(ax - cx)
          + (cx*cx + cy*cy)*(bx - ax)) / d
    r = np.sqrt((ax - ux)**2 + (ay - uy)**2)
    return ux, uy, r


def _ransac_circle(pts2d, n_iter=_RANSAC_ITERS, tol=_RANSAC_TOL_M):
    """RANSAC circle fit on 2D points. Returns (centre2d, radius, inlier_mask)."""
    if len(pts2d) < 3:
        return None, 0.0, np.zeros(len(pts2d), bool)

    best_inliers = 0
    best_result  = None

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
        if n_in > best_inliers:
            best_inliers = n_in
            best_result  = (np.array([cx, cy]), r, inliers)

    if best_result is None:
        return None, 0.0, np.zeros(len(pts2d), bool)
    return best_result


class NozzleEstimator:
    """
    Online LAE nozzle detector from lidar point cloud.

    Parameters
    ----------
    min_pts      : minimum cloud size before attempting detection
    conf_decay   : confidence decay per step when no good fit found
    """

    def __init__(self, min_pts: int = 40, conf_decay: float = 0.02):
        self._cloud      = np.zeros((0, 3))
        self._estimate   = np.zeros(3)
        self._confidence = 0.0
        self._min_pts    = min_pts
        self._conf_decay = conf_decay
        self._initialized = False

    # ── public interface ──────────────────────────────────────────────

    def update(self, new_pts: np.ndarray, chief_com_lvlh: np.ndarray):
        """
        Ingest new point cloud hits and refresh estimate.

        Parameters
        ----------
        new_pts        : (K, 3) new hit points in LVLH
        chief_com_lvlh : (3,) chief CoM position in LVLH (from flash lidar)
        """
        if len(new_pts) == 0:
            self._confidence = max(0.0, self._confidence - self._conf_decay)
            return

        # Accumulate in CoM-centred frame
        centred = new_pts - chief_com_lvlh
        self._cloud = np.vstack([self._cloud, centred])[-_WINDOW:]

        if len(self._cloud) < self._min_pts:
            return

        self._detect(chief_com_lvlh)

    @property
    def estimate(self) -> np.ndarray:
        """Estimated nozzle centre in LVLH [m]."""
        return self._estimate.copy()

    @property
    def confidence(self) -> float:
        """Confidence in [0, 1]. GNC uses > 0.6 to enter TERMINAL."""
        return self._confidence

    @property
    def is_valid(self) -> bool:
        return self._initialized and self._confidence > 0.3

    def reset(self):
        self._cloud       = np.zeros((0, 3))
        self._estimate    = np.zeros(3)
        self._confidence  = 0.0
        self._initialized = False

    # ── internals ─────────────────────────────────────────────────────

    def _detect(self, chief_com_lvlh: np.ndarray):
        cloud = self._cloud

        # PCA: find 3 principal axes of the cloud
        _, _, Vt = np.linalg.svd(cloud - cloud.mean(axis=0), full_matrices=False)
        axes = Vt  # rows are principal axes, descending variance

        best_inliers = _MIN_INLIERS - 1
        best_centre  = None
        best_axis    = None
        best_sign    = None

        for axis in axes:
            proj = cloud @ axis          # projection along this axis
            p05  = np.percentile(proj, 5)
            p95  = np.percentile(proj, 95)

            for sign, extreme_val in [(+1, p95), (-1, p05)]:
                # Cluster points near this extreme
                mask = np.abs(proj - extreme_val) < 0.20
                if mask.sum() < 3:
                    continue

                cluster = cloud[mask]

                # Project cluster onto plane perpendicular to axis
                u = np.cross(axis, [0, 0, 1])
                if np.linalg.norm(u) < 1e-6:
                    u = np.cross(axis, [1, 0, 0])
                u /= np.linalg.norm(u)
                v  = np.cross(axis, u)

                pts2d = np.column_stack([cluster @ u, cluster @ v])
                centre2d, radius, inliers = _ransac_circle(pts2d)

                if centre2d is None or inliers.sum() < _MIN_INLIERS:
                    continue

                n_in = inliers.sum()
                if n_in > best_inliers:
                    best_inliers = n_in
                    # Reconstruct 3D centre
                    c3d = (centre2d[0] * u + centre2d[1] * v
                           + extreme_val * axis)
                    best_centre = c3d
                    best_axis   = axis
                    best_sign   = sign

        if best_centre is not None:
            nozzle_lvlh = best_centre + chief_com_lvlh
            # EMA update
            alpha = 0.3 if self._initialized else 1.0
            self._estimate    = (1 - alpha) * self._estimate + alpha * nozzle_lvlh
            inlier_conf       = min(1.0, best_inliers / 20.0)
            self._confidence  = min(1.0, self._confidence + 0.08 * inlier_conf)
            self._initialized = True
        else:
            self._confidence = max(0.0, self._confidence - self._conf_decay)
