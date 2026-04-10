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
                 boresight_body:     np.ndarray = None):

        self.sigma_cross    = np.radians(sigma_cross_arcsec / 3600.0)
        self.sigma_roll     = np.radians(sigma_roll_arcsec  / 3600.0)
        self.cos_sun_excl   = np.cos(np.radians(sun_excl_deg))
        self.cos_earth_excl = np.cos(np.radians(earth_excl_deg))
        self.dt_meas        = 1.0 / update_rate_hz
        self.acq_delay      = acquisition_s

        self.boresight = (np.array(boresight_body) / np.linalg.norm(boresight_body)
                          if boresight_body is not None
                          else np.array([0., 0., 1.]))

        self._t_start     = None
        self._t_last_meas = -999.0

        # 3x3 noise covariance for MEKF quaternion-vector update
        # [cross, cross, roll] mapped to body x,y,z axes
        s_c = self.sigma_cross / 2.0
        s_r = self.sigma_roll  / 2.0
        self.R_st = np.diag([s_c**2, s_c**2, s_r**2])

    def measure(self,
                q_true:      np.ndarray,
                sun_vec_eci: np.ndarray,
                pos_eci_m:   np.ndarray,
                t:           float) -> tuple:
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
        # Acquisition delay
        if self._t_start is None:
            self._t_start = t
        if t - self._t_start < self.acq_delay:
            return None, self.R_st, False

        # Update rate gate
        if t - self._t_last_meas < self.dt_meas:
            return None, self.R_st, False

        # Boresight in ECI
        R_b2i    = self._q2R(q_true)
        bore_eci = R_b2i @ self.boresight

        # Sun exclusion
        if np.dot(bore_eci, sun_vec_eci) > self.cos_sun_excl:
            return None, self.R_st, False

        # Earth exclusion
        r_sat       = np.linalg.norm(pos_eci_m)
        earth_dir   = -pos_eci_m / r_sat
        alpha_earth = np.arcsin(np.clip(6.3781e6 / r_sat, 0, 1))
        cos_e       = np.dot(bore_eci, earth_dir)
        earth_angle = np.arccos(np.clip(cos_e, -1, 1))
        if earth_angle - alpha_earth < np.radians(20.0):
            return None, self.R_st, False

        # Add noise: different sigma on cross vs roll axes
        noise = np.array([
            np.random.normal(0, self.sigma_cross),
            np.random.normal(0, self.sigma_cross),
            np.random.normal(0, self.sigma_roll)
        ])
        dq = np.array([1.0, noise[0]/2, noise[1]/2, noise[2]/2])
        dq = dq / np.linalg.norm(dq)

        q_meas = self._qmult(dq, q_true)
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