import numpy as np


class Gyro:
    """
    MEMS Gyroscope model — ADIS16488 class fidelity.

    Noise model based on Allan variance decomposition:
        - Angle Random Walk (ARW)  : white noise on rate
        - Bias Instability (BI)    : flicker noise, Gauss-Markov model
        - Rate Random Walk (RRW)   : bias drift (Brownian motion)

    Reference:
        IEEE Std 952-1997
        Woodman, "An Introduction to Inertial Navigation", §4.2

    Default parameters match ADIS16488A datasheet:
        ARW  = 0.3  deg/√hr
        BI   = 6.0  deg/hr
        RRW  = 0.02 deg/hr/√hr
    """

    ARW_DEG_SQRTHR     = 0.3
    BI_DEG_HR          = 6.0
    RRW_DEG_HR_SQRTHR  = 0.02

    def __init__(self, dt=0.01, bias_init_max_deg_s=0.5):
        self.dt = dt

        # ARW: deg/√hr → rad/√s → noise std per sample
        arw_rad_sqrts  = np.radians(self.ARW_DEG_SQRTHR) / 60.0
        self.sigma_arw = arw_rad_sqrts / np.sqrt(dt)

        # Bias instability: first-order Gauss-Markov, τ = 1 hr
        bi_rad_s       = np.radians(self.BI_DEG_HR) / 3600.0
        self.tau_bi    = 3600.0
        self.sigma_bi  = bi_rad_s * np.sqrt(2 * dt / self.tau_bi)

        # Rate random walk: deg/hr/√hr → rad/s/√s
        rrw_rad        = np.radians(self.RRW_DEG_HR_SQRTHR) / 3600.0
        self.sigma_rrw = rrw_rad * np.sqrt(dt)

        # Initial bias — uniform within spec
        bias_max   = np.radians(bias_init_max_deg_s)
        self.bias  = np.random.uniform(-bias_max, bias_max, 3)

        # Gauss-Markov state for bias instability
        self.bias_gm = np.zeros(3)

    def measure(self, omega_true):
        """
        omega_meas = omega_true + bias + bias_gm + arw_noise
        Bias evolves each call via RRW and Gauss-Markov.
        """
        arw_noise     = np.random.normal(0, self.sigma_arw, 3)
        self.bias    += np.random.normal(0, self.sigma_rrw, 3)
        self.bias_gm  = ((1.0 - self.dt / self.tau_bi) * self.bias_gm
                         + np.random.normal(0, self.sigma_bi, 3))
        return omega_true + self.bias + self.bias_gm + arw_noise
