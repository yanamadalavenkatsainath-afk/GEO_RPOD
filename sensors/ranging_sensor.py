"""
Relative Navigation Sensor — Range + Bearing + Doppler
=======================================================
Simulates a proximity sensor that measures:
    - Range      : scalar distance to deputy [m]
    - Azimuth    : bearing angle in LVLH x-y plane [rad]
    - Elevation  : out-of-plane bearing angle [rad]
    - Range-rate : radial closing speed via Doppler [m/s]

The Doppler channel replaces truth velocity injection in the TH-EKF.
Previously the code set th_ekf.x[3:6] = true_cw_vel directly (truth
injection). Now measure_doppler() provides a physically realistic noisy
velocity estimate with sigma_v = 5 mm/s on the radial component.

Doppler noise model:
    sigma_doppler = 0.005 m/s  (5 mm/s 1-sigma on radial velocity).
    Representative of commercial VBS / short-range lidar: PRISMA, ATV, Dragon.
    Lateral velocity from consecutive EKF position difference:
        sigma_lateral = sigma_pos / dt  (~200 mm/s coast, ~2 mm/s prox)

Reference:
    Fehse, Automated Rendezvous and Docking of Spacecraft, Sec 6.3, 2003.
    D'Amico & Montenbruck, JGCD 29(3), 2006.
"""

import numpy as np
from utils.random_streams import random_generator
from utils.quaternion import normalize
from utils.validation import finite_scalar, vector


class RangingBearingSensor:
    """
    Range + bearing + Doppler sensor for relative navigation.

    Primary measurement: z = [range_m, az_rad, el_rad]
    Doppler channel: measure_doppler() returns full 3-D velocity estimate.

    Parameters
    ----------
    sigma_range_m    : absolute range noise 1-sigma [m]. Default 2 m.
    sigma_range_frac : fractional range noise. Default 0.5 %.
    sigma_angle_rad  : angular noise 1-sigma [rad]. Default 0.5 deg.
    sigma_doppler_ms : Doppler radial velocity noise [m/s]. Default 5 mm/s.
    fov_half_deg     : sensor half-angle FOV [deg]. Default 45 deg.
    min_range_m      : minimum detectable range [m]. Default 1 m.
    max_range_m      : maximum detectable range [m]. Default 10 km.
    """

    def __init__(self,
                 sigma_range_m:    float = 2.0,
                 sigma_range_frac: float = 0.005,
                 sigma_angle_rad:  float = np.radians(0.5),
                 sigma_doppler_ms: float = 0.005,
                 fov_half_deg:     float = 45.0,
                 min_range_m:      float = 1.0,
                 max_range_m:      float = 10_000.0,
                 rng=None):

        self.rng = random_generator(rng=rng)
        self.sigma_range_abs  = finite_scalar(sigma_range_m, "sigma_range_m", minimum=0)
        self.sigma_range_frac = finite_scalar(sigma_range_frac, "sigma_range_frac", minimum=0)
        self.sigma_angle      = finite_scalar(sigma_angle_rad, "sigma_angle_rad", minimum=0)
        self.sigma_doppler    = finite_scalar(sigma_doppler_ms, "sigma_doppler_ms", minimum=0)
        self.fov_half         = np.radians(finite_scalar(fov_half_deg, "fov_half_deg", minimum=0))
        self.min_range        = finite_scalar(min_range_m, "min_range_m", minimum=0)
        self.max_range        = finite_scalar(max_range_m, "max_range_m", strictly_positive=True)
        if self.fov_half > np.pi or self.max_range <= self.min_range:
            raise ValueError("Require FOV half angle <= 180 deg and max_range > min_range")
        self._prev_pos_est    = None   # for lateral velocity diff

    def measure(self, dr_lvlh, sensor_pointing_lvlh=None):
        """
        Range + bearing measurement.

        Returns z=[range, az, el] or None, and 3x3 noise covariance.
        """
        if sensor_pointing_lvlh is None:
            sensor_pointing_lvlh = np.array([0., 1., 0.])
        dr_lvlh = vector(dr_lvlh, "dr_lvlh", size=3)
        sensor_pointing_lvlh = normalize(vector(sensor_pointing_lvlh, "sensor_pointing_lvlh", size=3))

        true_range = float(np.linalg.norm(dr_lvlh))
        if true_range == 0 or true_range < self.min_range or true_range > self.max_range:
            return None, self._noise_cov(true_range)

        dr_hat  = dr_lvlh / true_range
        if np.dot(sensor_pointing_lvlh, dr_hat) < np.cos(self.fov_half):
            return None, self._noise_cov(true_range)

        az_true = np.arctan2(dr_lvlh[1], dr_lvlh[0])
        el_true = np.arctan2(dr_lvlh[2], np.sqrt(dr_lvlh[0]**2 + dr_lvlh[1]**2))

        sigma_r = max(self.sigma_range_abs, self.sigma_range_frac * true_range)
        z = np.array([true_range + self.rng.normal(0, sigma_r),
                      az_true    + self.rng.normal(0, self.sigma_angle),
                      el_true    + self.rng.normal(0, self.sigma_angle)])
        return z, self._noise_cov(true_range)

    def measure_range_rate(self, dr_lvlh, dv_lvlh, sensor_pointing_lvlh=None):
        """Scalar radial Doppler with the SAME geometry gate as range/bearing."""
        position = vector(dr_lvlh, "dr_lvlh", size=3)
        velocity = vector(dv_lvlh, "dv_lvlh", size=3)
        distance = float(np.linalg.norm(position))
        pointing = normalize(vector([0, 1, 0] if sensor_pointing_lvlh is None
                                    else sensor_pointing_lvlh, "pointing", size=3))
        noise = np.array([[self.sigma_doppler**2]])
        if (distance == 0 or distance < self.min_range or distance > self.max_range
                or np.dot(pointing, position / distance) < np.cos(self.fov_half)):
            return None, noise
        value = np.dot(velocity, position / distance) + self.rng.normal(0, self.sigma_doppler)
        return np.array([value]), noise

    def measure_doppler(self, dr_lvlh, dv_lvlh, pos_est_ekf, dt):
        """
        3-D velocity estimate: Doppler radial only.

        Lateral velocity is set to ZERO and left to the TH-EKF STM
        to propagate via the CW dynamics model.

        Why radial-only?
        ----------------
        The position-diff lateral estimator (prev version) caused runaway:
            v_lateral = (pos_ekf_now - pos_ekf_prev) / dt
        If the EKF position changes by 1 m between steps (due to predict()
        integrating a bad velocity estimate), the lateral estimate blows up:
            1 m / 0.1 s = 10 m/s error
        This created a feedback loop: bad v → EKF diverges → worse v.

        The TH-EKF STM couples radial and along-track via CW equations,
        so a good radial measurement (5 mm/s sigma) propagates lateral
        information within the filter. Explicit lateral injection is not
        needed and causes instability.

        Parameters
        ----------
        dr_lvlh     : true relative position [m]
        dv_lvlh     : true relative velocity [m/s] (scalar radial only used)
        pos_est_ekf : EKF position estimate [m]
        dt          : unused (kept for API compatibility)

        Returns
        -------
        v_meas  : 3-D velocity estimate [m/s]  (radial only, lateral=0)
        sigma_v : per-axis 1-sigma noise [m/s]
        """
        true_range = float(np.linalg.norm(dr_lvlh))
        if true_range < 1e-3:
            return np.zeros(3), np.ones(3) * 1.0

        # ── Radial Doppler: only the scalar range-rate is used from truth ──
        # dot(dv_lvlh, r_hat) is a scalar — no truth vector in guidance path
        r_hat_true      = dr_lvlh / true_range
        true_range_rate = float(np.dot(dv_lvlh, r_hat_true))
        doppler_meas    = true_range_rate + self.rng.normal(0, self.sigma_doppler)

        # Project onto EKF range direction (not truth direction)
        r_hat_est = pos_est_ekf / max(np.linalg.norm(pos_est_ekf), 1e-3)
        v_radial  = doppler_meas * r_hat_est

        # ── Lateral = 0: let TH-EKF CW-STM propagate lateral from radial ──
        v_meas = v_radial   # lateral components set to zero

        # Noise: radial sigma = sigma_doppler; lateral sigma large (unknown)
        sigma_lateral = 10.0   # m/s — very uncertain, tells EKF to keep own estimate
        sigma_v = np.array([
            self.sigma_doppler + abs(r_hat_est[0]) * sigma_lateral,
            self.sigma_doppler + abs(r_hat_est[1]) * sigma_lateral,
            self.sigma_doppler + abs(r_hat_est[2]) * sigma_lateral,
        ])
        return v_meas, sigma_v

    def _noise_cov(self, range_m):
        sigma_r = max(self.sigma_range_abs,
                      self.sigma_range_frac * max(range_m, self.min_range))
        return np.diag([sigma_r**2, self.sigma_angle**2, self.sigma_angle**2])

    @staticmethod
    def invert(z):
        """[range, az, el] -> Cartesian LVLH [m]."""
        r, az, el = z
        return np.array([r * np.cos(el) * np.cos(az),
                         r * np.cos(el) * np.sin(az),
                         r * np.sin(el)]) 
