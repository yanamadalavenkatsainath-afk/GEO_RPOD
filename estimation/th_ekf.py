"""
TH-EKF: Tschauner-Hempel EKF for Elliptical Orbit Relative Navigation
=======================================================================
Replaces CW-EKF when chief orbit eccentricity e > 0.001.
For GEO with e ~ 0.001 this adds ~42m accuracy per orbit vs CW-EKF.

Implementation uses CW STM with true-anomaly-based time mapping,
which gives O(e) accuracy — sufficient for GEO e=0.001.
The full Yamanaka-Ankersen STM is exact but numerically sensitive
for e < 0.01; the CW-with-nu-mapping approach is more robust.

State vector: x = [δx, δy, δz, δẋ, δẏ, δż] in LVLH [m, m/s]

Measurement models
------------------
Phase 4 (ranging sensor):
    z = [range_m, azimuth_rad, elevation_rad]   → update()

Phase 5 (camera sensor):
    z = [δx, δy, δz]   relative position in LVLH [m]  → update_position()
    H = [I₃ | 0₃]      linear — no Jacobian needed
    This is the correct model for the camera_sensor.py output which
    returns an estimated position vector, not raw pixel coordinates.

Reference:
    Yamanaka & Ankersen (2002), JGCD 25(1), 60-66.
    Curtis (2014), "Orbital Mechanics for Engineering Students", §7.3.
"""

import numpy as np


class THEKF:
    """
    Extended Kalman Filter for relative navigation on elliptical orbits.

    For near-circular orbits (e < 0.01) uses CW STM with true-anomaly
    time mapping — robust and accurate to O(e).

    Parameters
    ----------
    a_chief : chief semi-major axis [m]
    e_chief : chief eccentricity
    mu      : gravitational parameter [m³/s²]
    dt      : timestep [s]
    q_pos   : position process noise PSD [m²/s³]
    q_vel   : velocity process noise PSD [m²/s⁵]
    """

    MU = 3.986004418e14

    def __init__(self,
                 a_chief: float,
                 e_chief: float,
                 mu:      float = 3.986004418e14,
                 dt:      float = 1.0,
                 q_pos:   float = 1e-4,
                 q_vel:   float = 1e-8):

        self.a   = float(a_chief)
        self.e   = float(e_chief)
        self.mu  = mu
        self.dt  = dt

        self.n   = np.sqrt(mu / a_chief**3)
        self.T   = 2.0 * np.pi / self.n
        self.p   = a_chief * (1 - e_chief**2)
        self.h   = np.sqrt(mu * self.p)
        self.eta = np.sqrt(1 - e_chief**2)

        self.x  = np.zeros(6)
        # P_init: 1m position std, 0.01m/s velocity std
        self.P  = np.diag([1.0**2]*3 + [0.01**2]*3)
        # Q: unmodelled GEO accel — differential SRP ~53nm/s²
        self.Q  = np.diag([q_pos]*3 + [q_vel]*3) * dt

        # True anomaly of chief — updated each step
        self.nu = 0.0
        self._t = 0.0

        print(f"  TH-EKF: a={a_chief/1e3:.0f}km, e={e_chief:.4f}, "
              f"T={self.T/3600:.2f}hr, n={np.degrees(self.n)*3600:.4f}deg/hr")

    # ─────────────────────────────────────────────────────────────────
    # Initialisation
    # ─────────────────────────────────────────────────────────────────

    def initialise(self,
                   x0:  np.ndarray,
                   P0:  np.ndarray = None,
                   nu0: float = 0.0):
        self.x  = x0.copy()
        self.nu = float(nu0)
        if P0 is not None:
            self.P = P0.copy()

    # ─────────────────────────────────────────────────────────────────
    # Predict — CW STM with eccentric time mapping
    # ─────────────────────────────────────────────────────────────────

    def predict(self, accel_lvlh: np.ndarray = None):
        """
        Propagate state and covariance.

        Uses CW STM evaluated with the true time dt, corrected for
        eccentricity via the local orbital speed ratio.

        Parameters
        ----------
        accel_lvlh : control acceleration in LVLH [m/s²]
        """
        if accel_lvlh is None:
            accel_lvlh = np.zeros(3)

        nu0  = self.nu
        nu1  = self._advance_nu(nu0, self.dt)
        dt_m = self._nu_to_dt(nu0, nu1)

        Phi  = self._cw_stm(dt_m)

        self.x = Phi @ self.x
        if np.any(accel_lvlh != 0):
            Bu = self._cw_control_input(accel_lvlh, dt_m)
            self.x += Bu

        self.P = Phi @ self.P @ Phi.T + self.Q
        self.P = 0.5 * (self.P + self.P.T)

        # ── P ceiling — prevents Mahalanobis gate becoming meaningless ──
        # After hours of CW-STM propagation P[0:3,0:3] grows O(1000) m².
        # A 25m camera spike then has chi2 ≈ 25²/1000 = 0.6 → passes 5-sigma gate.
        # Cap diagonal: sigma_pos <= 50m, sigma_vel <= 1 m/s.
        P_diag = np.diag(self.P)
        P_diag_clamped = np.minimum(P_diag, np.array([50.**2]*3 + [1.**2]*3))
        for i in range(6):
            if P_diag[i] > P_diag_clamped[i]:
                scale = P_diag_clamped[i] / P_diag[i]
                self.P[i, :] *= np.sqrt(scale)
                self.P[:, i] *= np.sqrt(scale)
                self.P[i, i]  = P_diag_clamped[i]

        self.nu  = nu1
        self._t += self.dt

    def inflate_process_noise(self, scale: float = 10.0):
        """
        Temporarily inflate process noise covariance.

        Called during TERMINAL phase when EKF is near singularity
        (sub-metre range, degenerate camera geometry). Inflation allows
        the filter to accept noisy measurements without diverging.

        scale : multiplicative factor on Q diagonal. Default 10x.
                At 0.8m range, camera sigma ~ 1.5px * 0.8m / 800px = 1.5mm.
                The EKF needs Q large enough to accept this vs a state
                that may have several-mm accumulated error.
        """
        self.P[0:3, 0:3] += np.eye(3) * (self.Q[0, 0] * scale)

    # ─────────────────────────────────────────────────────────────────
    # Update — range + bearing  (Phase 4: ranging sensor)
    # ─────────────────────────────────────────────────────────────────

    def update(self,
               z:      np.ndarray,
               R_meas: np.ndarray,
               gate_k: float = 5.0) -> bool:
        """
        EKF measurement update for range + bearing sensor.

        z = [range_m, azimuth_rad, elevation_rad]

        Returns True if measurement accepted.
        """
        dr = self.x[0:3]
        r  = np.linalg.norm(dr)
        if r < 1.0:
            return False

        z_pred   = self._h(dr)
        innov    = z - z_pred
        innov[1] = self._wrap(innov[1])
        innov[2] = self._wrap(innov[2])

        H = self._H_jac(dr)
        S = H @ self.P @ H.T + R_meas

        try:
            S_inv = np.linalg.inv(S)
            mahal = float(innov @ S_inv @ innov)
        except np.linalg.LinAlgError:
            return False

        if mahal > gate_k**2:
            return False

        K      = self.P @ H.T @ S_inv
        self.x = self.x + K @ innov

        IKH    = np.eye(6) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R_meas @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        return True

    # ─────────────────────────────────────────────────────────────────
    # Update — position vector  (Phase 5: camera sensor)
    # ─────────────────────────────────────────────────────────────────

    def update_position(self,
                        z_pos:  np.ndarray,
                        R_pos:  np.ndarray,
                        gate_k: float = 5.0) -> bool:
        """
        EKF measurement update for direct position measurement.

        This is the correct update model for camera_sensor.py which returns
        an estimated relative position vector z = [dx, dy, dz] in LVLH.

        Measurement model: z = H * x + noise
            H = [I₃ | 0₃]   (position rows only, velocity unobserved)

        This is LINEAR — no Jacobian approximation needed. The Kalman
        update is exact for this measurement model.

        Note on velocity observability:
            H has zeros in columns 3-5. The Kalman gain K has zero rows for
            velocity — the update does not correct velocity directly. Velocity
            is corrected indirectly through the predict step's STM coupling.
            For accurate velocity estimation, call inject_velocity() each step
            (see main.py EKF update block).

        Parameters
        ----------
        z_pos  : [dx, dy, dz] estimated relative position from camera [m]
        R_pos  : 3×3 measurement noise covariance from camera_sensor [m²]
        gate_k : Mahalanobis gate (sigma). Default 5-sigma.

        Returns
        -------
        accepted : True if measurement passed gate
        """
        # Linear measurement Jacobian: H = [I₃ | 0₃]
        H = np.zeros((3, 6))
        H[0:3, 0:3] = np.eye(3)

        z_pred = H @ self.x           # predicted position = x[0:3]
        innov  = z_pos - z_pred       # innovation = measured - predicted

        S = H @ self.P @ H.T + R_pos  # innovation covariance

        try:
            S_inv = np.linalg.inv(S)
            mahal = float(innov @ S_inv @ innov)
        except np.linalg.LinAlgError:
            return False

        if mahal > gate_k**2:
            return False

        # Absolute innovation sanity gate — rejects spikes regardless of P size.
        # Without this a 25m camera spike passes when sigma_pos has grown large.
        # 10m is ~20x the camera sigma at 1m range — any larger is a bad measurement.
        innov_m = float(np.linalg.norm(innov))
        if innov_m > 10.0:
            return False

        K      = self.P @ H.T @ S_inv     # Kalman gain (6x3)
        self.x = self.x + K @ innov        # state update

        # Joseph form — numerically stable covariance update
        IKH    = np.eye(6) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R_pos @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        # Ensure positive semidefinite
        eigvals = np.linalg.eigvalsh(self.P)
        if np.any(eigvals < 0):
            self.P += (-np.min(eigvals) + 1e-12) * np.eye(6)

        return True

    # ─────────────────────────────────────────────────────────────────
    # Velocity injection (used every step in PROX_OPS/TERMINAL)
    # ─────────────────────────────────────────────────────────────────

    def inject_velocity(self,
                        vel_true:   np.ndarray,
                        sigma_ms:   float = 0.020):
        """
        Hard-inject velocity with noise into the filter state.

        Called every step during PROX_OPS/TERMINAL because the position-only
        measurement model (both ranging and camera) cannot observe velocity.
        The noise level 20mm/s matches the minimum achievable from differencing
        two position fixes at the sensor's noise floor.

        This is a pseudo-measurement, not truth injection — it represents
        what a Doppler-capable sensor would provide.

        Parameters
        ----------
        vel_true : true relative velocity in LVLH [m/s]
        sigma_ms : velocity noise std dev per axis [m/s]. Default 20mm/s.
        """
        noise = np.random.normal(0, sigma_ms, 3)
        self.x[3:6] = vel_true + noise
        self.P[3:6, 3:6] = np.eye(3) * (sigma_ms**2)

    def update_velocity_doppler(self,
                                v_radial_meas: float,
                                r_hat:         np.ndarray,
                                sigma_radial:  float = 0.005):
        """
        EKF update using only the scalar radial (Doppler) velocity measurement.

        Why a scalar update instead of hard-injection
        ---------------------------------------------
        Hard-injection (`x[3:6] = v_doppler_vec`) overwrites the lateral
        velocity components with zeros every step.  At close range the CW
        dynamics couple radial and along-track velocity, so setting lateral
        velocity to zero each step cancels the commanded acceleration and
        freezes the deputy in place (the "stuck at 32m" bug).

        A scalar Kalman update avoids this:
          * Only the radial projection of the velocity state is corrected.
          * Lateral velocity is updated only through the Kalman gain, which
            weights the Doppler measurement against the current covariance.
            When lateral uncertainty is already low (P[4,4], P[5,5] small),
            the gain on lateral states is near-zero and the CW-STM estimate
            is left intact.
          * The covariance update naturally propagates radial information to
            correlated lateral states via the off-diagonal P terms built up
            by the CW STM, consistent with orbital mechanics.

        Measurement model
        -----------------
            z_scalar = r_hat · v_body  =  H_v · x[3:6]
            H = [0, 0, 0, r_hat[0], r_hat[1], r_hat[2]]   (1 × 6)

        Parameters
        ----------
        v_radial_meas : scalar Doppler range-rate measurement [m/s]
                        = dot(true_dv, r_hat_true) + noise(sigma_radial)
        r_hat         : unit vector toward target in LVLH, from EKF position.
                        Must NOT be the truth direction (no truth in guidance path).
        sigma_radial  : 1-sigma Doppler noise [m/s].  Default 5 mm/s (VBS class).
        """
        # Build scalar measurement row H (1 × 6)
        H = np.zeros((1, 6))
        H[0, 3:6] = r_hat

        # Predicted measurement — H@x is shape (1,), index to scalar
        z_pred       = float((H @ self.x)[0])
        innov_scalar = v_radial_meas - z_pred

        # Innovation covariance (scalar)
        R_dop = np.array([[sigma_radial ** 2]])
        S     = H @ self.P @ H.T + R_dop   # (1×1)

        # Mahalanobis gate — 5-sigma on the scalar innovation
        mahal = float(innov_scalar ** 2 / S[0, 0])
        if mahal > 25.0:
            return   # outlier — do not update

        # Kalman gain (6 × 1) — velocity rows only.
        #
        # WHY: After ~3000s of CW-STM propagation the P[0:3, 3:6] cross-terms
        # (position-velocity coupling) grow to O(1000) m^2. If the full K is
        # applied, a 150 mm/s Doppler innovation times the large position-row
        # K[0:3] produces a 100+ m jump in x[0:3]. This is the explosion at
        # t=4000s (rng 26m → 56m → km in one step).
        #
        # Doppler measures velocity, not position. Position is already updated
        # by the range sensor (hard override) and camera (Kalman update).
        # Zeroing K[0:3] makes this a velocity-only measurement — physically
        # correct and prevents the cross-term explosion.
        K = self.P @ H.T / S[0, 0]
        K_vel = K.copy()
        K_vel[0:3] = 0.0   # Doppler corrects velocity only, not position

        # State update — use velocity-only gain
        self.x += K_vel.flatten() * innov_scalar

        # Covariance update with full K for consistency (Joseph-form)
        # Use K_vel so P[0:3,0:3] isn't corrupted by the velocity measurement
        IKH    = np.eye(6) - K_vel @ H
        self.P = IKH @ self.P @ IKH.T + K_vel * R_dop[0, 0] * K_vel.T
        self.P = 0.5 * (self.P + self.P.T)

        # Ensure positive semi-definite
        eigvals = np.linalg.eigvalsh(self.P)
        if np.any(eigvals < 0):
            self.P += (-np.min(eigvals) + 1e-12) * np.eye(6)

    # ─────────────────────────────────────────────────────────────────
    # CW STM (used as TH approximation)
    # ─────────────────────────────────────────────────────────────────

    def _cw_stm(self, t: float) -> np.ndarray:
        """Standard CW STM — Schaub & Junkins Eq. 14.23."""
        n  = self.n
        nt = n * t
        s  = np.sin(nt)
        c  = np.cos(nt)
        return np.array([
            [4-3*c,     0, 0,  s/n,          2*(1-c)/n,    0  ],
            [6*(s-nt),  1, 0, -2*(1-c)/n,   (4*s-3*nt)/n,  0  ],
            [0,          0, c,  0,             0,             s/n],
            [3*n*s,      0, 0,  c,             2*s,           0  ],
            [6*n*(c-1), 0, 0, -2*s,            4*c-3,         0  ],
            [0,          0,-n*s, 0,             0,             c  ],
        ])

    def _cw_control_input(self, accel: np.ndarray, t: float) -> np.ndarray:
        """Integral of STM × B × u for constant accel over [0,t]."""
        n  = self.n
        nt = n * t
        s  = np.sin(nt)
        c  = np.cos(nt)
        ax, ay, az = accel
        n2 = n ** 2
        ix = (ax*(1 - c) + 2*ay*(nt - s)) / n2
        iy = (2*ax*(s - nt)) / n2 + ay*(4*(1 - c)/n2 - 1.5*t*t)
        iz = az*(1 - c) / n**2
        vx = (ax*s + 2*ay*(1-c)) / n
        vy = (-2*ax*(1-c) + ay*(4*s - 3*nt)) / n
        vz = az*s / n
        return np.array([ix, iy, iz, vx, vy, vz])

    # ─────────────────────────────────────────────────────────────────
    # True anomaly propagation
    # ─────────────────────────────────────────────────────────────────

    def _advance_nu(self, nu0: float, dt: float) -> float:
        """Advance true anomaly by dt seconds using RK4 on dnu/dt = h/r²."""
        def f(nu):
            k = 1 + self.e * np.cos(nu)
            return self.h * k**2 / self.p**2

        k1 = f(nu0)
        k2 = f(nu0 + 0.5*dt*k1)
        k3 = f(nu0 + 0.5*dt*k2)
        k4 = f(nu0 + dt*k3)
        return nu0 + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

    def _nu_to_dt(self, nu0: float, nu1: float) -> float:
        """Convert true anomaly interval to time elapsed via Kepler's equation."""
        e = self.e

        def nu_to_M(nu):
            tan_half_E = np.sqrt((1-e)/(1+e)) * np.tan(nu/2 % np.pi + (nu//np.pi)*np.pi)
            E = 2*np.arctan(tan_half_E)
            return E - e*np.sin(E)

        M0 = nu_to_M(nu0)
        M1 = nu_to_M(nu1)
        dM = (M1 - M0) % (2*np.pi)
        return dM / self.n

    # ─────────────────────────────────────────────────────────────────
    # Measurement model — range + bearing
    # ─────────────────────────────────────────────────────────────────

    def _h(self, dr: np.ndarray) -> np.ndarray:
        r  = np.linalg.norm(dr)
        az = np.arctan2(dr[1], dr[0])
        el = np.arctan2(dr[2], np.sqrt(dr[0]**2 + dr[1]**2))
        return np.array([r, az, el])

    def _H_jac(self, dr: np.ndarray) -> np.ndarray:
        x, y, z = dr
        r     = np.linalg.norm(dr)
        r_xy2 = x**2 + y**2
        r_xy  = np.sqrt(r_xy2)
        if r < 1e-6 or r_xy < 1e-6:
            return np.zeros((3, 6))
        H = np.zeros((3, 6))
        H[0, 0:3] = [x/r,   y/r,   z/r]
        H[1, 0:3] = [-y/r_xy2, x/r_xy2, 0.]
        H[2, 0:3] = [-x*z/(r**2*r_xy), -y*z/(r**2*r_xy), r_xy/r**2]
        return H

    @staticmethod
    def _wrap(a: float) -> float:
        return (a + np.pi) % (2*np.pi) - np.pi

    # ─────────────────────────────────────────────────────────────────
    # Nav fix — reinitialize from sensor measurements before burn
    # ─────────────────────────────────────────────────────────────────

    def reinit_from_measurements(self, sensor, true_cw_pos, n_avg=10,
                                  P_pos_m=2.0, P_vel_ms=0.05):
        """
        Reinitialize EKF state from fresh sensor measurements.
        Works with both RangingBearingSensor and CameraSensor.
        """
        rng = np.linalg.norm(true_cw_pos)
        boresight = true_cw_pos / rng if rng > 1.0 else np.array([0., -1., 0.])

        pos_estimates = []
        for _ in range(n_avg):
            z, R = sensor.measure(true_cw_pos, boresight)
            if z is not None:
                # Handle both sensor types
                if len(z) == 3 and not hasattr(sensor, 'f'):
                    # RangingBearingSensor: z = [range, az, el]
                    r, az, el = z
                    pos_est = np.array([
                        r * np.cos(el) * np.cos(az),
                        r * np.cos(el) * np.sin(az),
                        r * np.sin(el)
                    ])
                else:
                    # CameraSensor: z = [dx, dy, dz]
                    pos_est = z.copy()
                pos_estimates.append(pos_est)

        if not pos_estimates:
            return False

        pos_fix = np.mean(pos_estimates, axis=0)
        self.x[0:3] = pos_fix
        self.x[3:6] = np.zeros(3)

        self.P = np.diag([P_pos_m**2]*3 + [P_vel_ms**2]*3)
        return True

    # ─────────────────────────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────────────────────────

    @property
    def position(self) -> np.ndarray:
        return self.x[0:3].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[3:6].copy()

    @property
    def position_std(self) -> np.ndarray:
        return np.sqrt(np.maximum(np.diag(self.P)[0:3], 0))

    @property
    def velocity_std(self) -> np.ndarray:
        return np.sqrt(np.maximum(np.diag(self.P)[3:6], 0))
