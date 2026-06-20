"""
sensors/lidar_pointcloud_sensor.py

Simulates a flash lidar returning N 3D point hits off the chief surface.

The sensor shoots N rays from the deputy origin into the chief's bounding
sphere, finds the nearest triangle intersection via Möller–Trumbore, and
returns noisy LVLH-frame hit points. The estimator receives only XYZ
coordinates — no material, geometry, or face identity is passed through.

This is the ONLY file that touches chief geometry. Everything downstream
(nozzle_estimator.py) works from raw point clouds with no geometry prior.
"""

import numpy as np

from render.chief_renderer import _VERTS_BODY, _TRIS
from sim_config import CHIEF_BODY_HALF_EXTENTS_M, LAE_NOZZLE_LENGTH_M


# Chief bounding sphere radius (circumscribes body + nozzle protrusion)
_HX, _HY, _HZ = CHIEF_BODY_HALF_EXTENTS_M
_BSPHERE_R = np.sqrt(_HX**2 + _HY**2 + (_HZ + LAE_NOZZLE_LENGTH_M)**2) + 0.05


def _rot_matrix(q):
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)],
    ])


def _moller_trumbore(ray_o, ray_d, v0, v1, v2, eps=1e-9):
    """Return t along ray to triangle intersection, or None if no hit."""
    e1 = v1 - v0
    e2 = v2 - v0
    h  = np.cross(ray_d, e2)
    a  = np.dot(e1, h)
    if abs(a) < eps:
        return None
    f = 1.0 / a
    s = ray_o - v0
    u = f * np.dot(s, h)
    if u < 0.0 or u > 1.0:
        return None
    q = np.cross(s, e1)
    v = f * np.dot(ray_d, q)
    if v < 0.0 or u + v > 1.0:
        return None
    t = f * np.dot(e2, q)
    return t if t > eps else None


class LidarPointCloudSensor:
    """
    Flash lidar point cloud: N rays per shot, returns hit points in LVLH.

    Parameters
    ----------
    n_rays        : rays per update (spread over chief face)
    noise_sigma_m : 1-sigma range noise [m]
    """

    def __init__(self, n_rays: int = 60, noise_sigma_m: float = 0.02):
        self._n_rays       = n_rays
        self._noise_sigma  = noise_sigma_m

    def measure(self, dr_lvlh: np.ndarray, q_chief: np.ndarray,
                 rng=None) -> np.ndarray:
        """
        Shoot rays at the chief and return hit points in LVLH.

        Parameters
        ----------
        dr_lvlh  : (3,) chief CoM position relative to deputy [m], LVLH
        q_chief  : (4,) chief body→LVLH quaternion [w,x,y,z]
        rng      : numpy Generator

        Returns
        -------
        pts : (K, 3) hit points in LVLH frame, K ≤ n_rays
              Returns empty array if no hits.
        """
        rng = rng if rng is not None else np.random.default_rng()
        R_b2l = _rot_matrix(q_chief)

        # Transform chief vertices to LVLH
        verts_lvlh = (R_b2l @ _VERTS_BODY.T).T + dr_lvlh

        # Deputy is at origin; chief CoM at dr_lvlh
        # Build random rays that pass through chief bounding sphere
        r_hat = dr_lvlh / np.linalg.norm(dr_lvlh)

        # Orthonormal basis perpendicular to boresight
        up  = np.array([0., 0., 1.]) if abs(r_hat[2]) < 0.9 else np.array([1., 0., 0.])
        u   = np.cross(r_hat, up);  u /= np.linalg.norm(u)
        v   = np.cross(r_hat, u)

        hits = []
        for _ in range(self._n_rays):
            # Random offset within bounding sphere disk
            while True:
                dx, dy = rng.uniform(-_BSPHERE_R, _BSPHERE_R, 2)
                if dx*dx + dy*dy < _BSPHERE_R**2:
                    break
            ray_o = np.zeros(3)
            ray_d = dr_lvlh + dx*u + dy*v
            ray_d = ray_d / np.linalg.norm(ray_d)

            t_min = np.inf
            for tri in _TRIS:
                v0, v1, v2 = verts_lvlh[tri[0]], verts_lvlh[tri[1]], verts_lvlh[tri[2]]
                t = _moller_trumbore(ray_o, ray_d, v0, v1, v2)
                if t is not None and t < t_min:
                    t_min = t

            if t_min < np.inf:
                hit = ray_o + t_min * ray_d
                noise = rng.normal(0.0, self._noise_sigma, 3)
                hits.append(hit + noise)

        return np.array(hits) if hits else np.zeros((0, 3))
