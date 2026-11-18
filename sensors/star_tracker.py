"""
Star Tracker Sensor Model
=========================
Models a miniature star tracker for a 3U CubeSat at GEO.
Representative hardware: NST-1 (Sinclair), MAI-SS, Nano Star Camera.

Output: full attitude quaternion from star pattern matching.
Noise:  Gaussian per axis. Cross-boresight ~5 arcsec, roll ~20 arcsec.

Limitations modelled:
    - Sun exclusion cone (30 deg) — detector saturation
    - Earth exclusion cone (20 deg) — stray light
    - Acquisition delay: 30s after power-on
    - Update rate: 4 Hz

Why critical at GEO:
    B field is ~100 nT at GEO (vs 30000 nT LEO).
    Magnetometer gives zero attitude info at GEO.
    Without star tracker, MEKF dead-reckons on gyro alone
    over 9hr Lambert coast → 10-100 deg pointing error.
    With star tracker: <0.01 deg maintained throughout.

Reference:
    Markley & Crassidis, FSADC, §7.4
"""

import numpy as np
from utils.random_streams import random_generator
from utils.quaternion import normalize, quat_from_rotvec, quat_multiply, rot_matrix
from utils.validation import finite_scalar, vector


class StarTracker:
    """
    Miniature star tracker sensor.

    Parameters
    ----------
    sigma_cross_arcsec : cross-boresight noise 1-sigma [arcsec]
    sigma_roll_arcsec  : about-boresight noise 1-sigma [arcsec]
    sun_excl_deg       : sun exclusion half-angle [deg]
    earth_excl_deg     : Earth exclusion half-angle [deg]
    update_rate_hz     : measurement output rate [Hz]
    acquisition_s      : time to first fix [s]
    boresight_body     : boresight unit vector in body frame (default +z)
    """

    def __init__(self,
                 sigma_cross_arcsec: float = 5.0,
                 sigma_roll_arcsec:  float = 20.0,
                 sun_excl_deg:       float = 30.0,
                 earth_excl_deg:     float = 20.0,
                 update_rate_hz:     float = 4.0,
                 acquisition_s:      float = 30.0,
                 boresight_body:     np.ndarray = None,
                 occulting_radius_m: float = 6.3781e6,
                 rng=None):

        self.rng = random_generator(rng=rng)
        self.occulting_radius_m=finite_scalar(occulting_radius_m,'occulting_radius_m',strictly_positive=True)
        self.sigma_cross    = np.radians(finite_scalar(sigma_cross_arcsec, "sigma_cross_arcsec", minimum=0) / 3600.0)
        self.sigma_roll     = np.radians(finite_scalar(sigma_roll_arcsec, "sigma_roll_arcsec", minimum=0) / 3600.0)
        sun_excl_deg = finite_scalar(sun_excl_deg, "sun_excl_deg", minimum=0)
        earth_excl_deg = finite_scalar(earth_excl_deg, "earth_excl_deg", minimum=0)
        if max(sun_excl_deg, earth_excl_deg) > 180:
            raise ValueError("Exclusion angles must be <= 180 degrees")
        self.cos_sun_excl   = np.cos(np.radians(sun_excl_deg))
        self.cos_earth_excl = np.cos(np.radians(earth_excl_deg))
        self.earth_excl_rad = np.radians(earth_excl_deg)
        self.dt_meas        = 1.0 / finite_scalar(update_rate_hz, "update_rate_hz", strictly_positive=True)
        self.acq_delay      = finite_scalar(acquisition_s, "acquisition_s", minimum=0)

        self.boresight = normalize(vector([0., 0., 1.] if boresight_body is None
                                         else boresight_body, "boresight_body", size=3))

        self._t_start     = None
        self._t_last_meas = -999.0

        # 3x3 noise covariance for MEKF quaternion-vector update
        # [cross, cross, roll] mapped to body x,y,z axes
        self.error_covariance_body = (np.eye(3) * self.sigma_cross**2
            + (self.sigma_roll**2 - self.sigma_cross**2)
            * np.outer(self.boresight, self.boresight))
        self.R_st = self.error_covariance_body / 4.0  # legacy quaternion-vector covariance

    def measure(self,
                q_true:      np.ndarray,
                sun_vec_eci: np.ndarray,
                pos_eci_m:   np.ndarray,
                t:           float,
                *, scheduled=False) -> tuple:
        """
        Generate star tracker quaternion measurement.

        Parameters
        ----------
        q_true      : true quaternion [w,x,y,z] body→ECI
        sun_vec_eci : unit vector toward Sun in ECI
        pos_eci_m   : spacecraft ECI position [m]
        t           : simulation time [s]

        Returns
        -------
        q_meas : noisy quaternion [w,x,y,z] or None
        R_st   : 3×3 noise covariance for MEKF
        valid  : True if measurement available
        """
        if not isinstance(scheduled, bool):
            raise ValueError("scheduled must be boolean")
        # The common Sensor adapter already owns a sampling grid. Legacy
        # direct calls retain this model's own minimum-interval gate.
        # Acquisition delay
        if self._t_start is None:
            self._t_start = t
        if t - self._t_start < self.acq_delay:
            return None, self.R_st, False

        # Update rate gate
        if not scheduled and t - self._t_last_meas < self.dt_meas:
            return None, self.R_st, False

        # Boresight in ECI
        q_true = normalize(vector(q_true, "q_true", size=4))
        sun_vec_eci = normalize(vector(sun_vec_eci, "sun_vec_eci", size=3))
        pos_eci_m = vector(pos_eci_m, "pos_eci_m", size=3)
        if np.linalg.norm(pos_eci_m) <= self.occulting_radius_m:
            return None, self.R_st, False
        R_b2i    = rot_matrix(q_true)
        bore_eci = R_b2i @ self.boresight

        # Sun exclusion
        if np.dot(bore_eci, sun_vec_eci) > self.cos_sun_excl:
            return None, self.R_st, False

        # Earth exclusion
        r_sat       = np.linalg.norm(pos_eci_m)
        earth_dir   = -pos_eci_m / r_sat
        alpha_earth = np.arcsin(np.clip(self.occulting_radius_m / r_sat, 0, 1))
        cos_e       = np.dot(bore_eci, earth_dir)
        earth_angle = np.arccos(np.clip(cos_e, -1, 1))
        if earth_angle - alpha_earth < self.earth_excl_rad:
            return None, self.R_st, False

        # Add noise: different sigma on cross vs roll axes
        white = self.rng.normal(0, 1, 3)
        axial = np.dot(white, self.boresight) * self.boresight
        noise = self.sigma_cross * (white - axial) + self.sigma_roll * axial
        q_meas = quat_multiply(q_true, quat_from_rotvec(noise))
        q_meas = q_meas / np.linalg.norm(q_meas)
        if q_meas[0] < 0:
            q_meas = -q_meas

        self._t_last_meas = t
        return q_meas, self.R_st, True

    @staticmethod
    def _q2R(q):
        w, x, y, z = q
        return np.array([
            [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
            [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]
        ])

    @staticmethod
    def _qmult(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
