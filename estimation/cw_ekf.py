"""
CW-EKF: Extended Kalman Filter for Relative Navigation
=======================================================
Uses Clohessy-Wiltshire equations as the process model and
range + bearing (azimuth/elevation) as the measurement model.

State vector:
    x = [δx, δy, δz, δẋ, δẏ, δż]  [m, m/s]   in LVLH frame

Measurement vector:
    z = [range, azimuth, elevation]

The measurement model h(x) is nonlinear (range is a norm, angles
are arctangents), so we linearise via analytical Jacobian H = ∂h/∂x.

Process noise Q accounts for unmodelled accelerations:
    - CW is a linearisation → J2, drag, SRP not included
    - Q diagonal tuned to ~1e-6 m²/s⁴ process noise (0.1 mm/s² unmodelled)

Reference:
    Gaias & D'Amico, "Impulsive Maneuvers for Formation Reconfiguration
    Using Relative Orbital Elements", JGCD 38(6), 2015.
    
    Montenbruck et al., "Navigation and Control of the PRISMA Formation",
    JGCD 35(5), 2012.
"""

import numpy as np
from environment.cw_dynamics import CWDynamics


class CWEKF:
    """
    Extended Kalman Filter for relative navigation.

    Parameters
    ----------
    n        : chief mean motion [rad/s]
    dt       : propagation timestep [s]
    q_pos    : process noise PSD, position [m²/s³]. Default: 1e-6.
    q_vel    : process noise PSD, velocity [m²/s⁵]. Default: 1e-8.
    """

    def __init__(self,
                 n:     float,
                 dt:    float,
                 q_pos: float = 1e-6,
                 q_vel: float = 1e-8):

        self.n  = n
        self.dt = dt

        # State estimate: [δx, δy, δz, δẋ, δẏ, δż]
        self.x  = np.zeros(6)

        # Covariance: initial uncertainty — 50 m position, 0.5 m/s velocity
        self.P  = np.diag([50.0**2, 50.0**2, 50.0**2,
                           0.5**2,  0.5**2,  0.5**2])

        # Process noise covariance
        self.Q  = np.diag([q_pos, q_pos, q_pos,
                           q_vel, q_vel, q_vel]) * dt

    # ─────────────────────────────────────────────────────────────────
    # Initialisation
    # ─────────────────────────────────────────────────────────────────

    def initialise(self,
                   x0:      np.ndarray,
                   P0:      np.ndarray = None):
        """
        Seed the filter with an initial estimate.

        Parameters
        ----------
        x0 : initial state [δx,δy,δz,δẋ,δẏ,δż]
        P0 : initial covariance (6×6). Default: 50m/0.5m/s diagonal.
        """
        self.x = x0.copy()
        if P0 is not None:
            self.P = P0.copy()

    # ─────────────────────────────────────────────────────────────────
    # Predict step — CW linearised dynamics
    # ─────────────────────────────────────────────────────────────────

    def predict(self, accel_lvlh: np.ndarray = None):
        """
        Propagate state and covariance via CW state transition matrix.

        Uses the analytical CW state transition matrix Φ(dt) rather than
        numerical integration — exact for linear CW (no thrust case).
        For thrust, the control input term B·u is added.

        Parameters
        ----------
        accel_lvlh : control acceleration in LVLH [m/s²]. Default: zeros.
        """
        if accel_lvlh is None:
            accel_lvlh = np.zeros(3)

        Phi = self._cw_stm(self.dt)
        Bu  = self._cw_control_input(accel_lvlh, self.dt)

        self.x = Phi @ self.x + Bu
        self.P = Phi @ self.P @ Phi.T + self.Q

        # Enforce symmetry
        self.P = 0.5 * (self.P + self.P.T)

    # ─────────────────────────────────────────────────────────────────
    # Update step — range + bearing measurement
    # ─────────────────────────────────────────────────────────────────

    def update(self,
               z:       np.ndarray,
               R:       np.ndarray,
               gate_k:  float = 5.0) -> bool:
        """
        EKF measurement update with range + bearing.

        Parameters
        ----------
        z      : measurement [range, azimuth, elevation]
        R      : 3×3 measurement noise covariance
        gate_k : Mahalanobis distance gate (sigma). Default: 5σ.

        Returns
        -------
        accepted : True if measurement passed the gate
        """
        dr = self.x[0:3]
        r  = np.linalg.norm(dr)

        if r < 1e-3:   # too close to linearise safely
            return False

        # Predicted measurement
        z_pred = self._h(dr)

        # Innovation (handle azimuth wrap-around)
        innov      = z - z_pred
        innov[1]   = self._wrap_angle(innov[1])
        innov[2]   = self._wrap_angle(innov[2])

        # Jacobian H = ∂h/∂x  (3×6, only position columns nonzero)
        H = self._H_jacobian(dr)

        # Innovation covariance
        S = H @ self.P @ H.T + R

        # Mahalanobis gate
        try:
            S_inv  = np.linalg.inv(S)
            mahal  = float(innov @ S_inv @ innov)
        except np.linalg.LinAlgError:
            return False

        if mahal > gate_k**2:
            return False   # outlier rejected

        # Kalman gain
        K = self.P @ H.T @ S_inv

        # State update
        self.x = self.x + K @ innov

        # Joseph-form covariance update (numerically stable)
        IKH    = np.eye(6) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        return True

    # ─────────────────────────────────────────────────────────────────
    # CW State Transition Matrix  Φ(t)
    # ─────────────────────────────────────────────────────────────────

    def _cw_stm(self, t: float) -> np.ndarray:
        """
        Analytical CW state transition matrix Φ(t).
        Exact for zero-thrust relative motion on a circular orbit.

        Schaub & Junkins, Eq. 14.23.
        """
        n  = self.n
        nt = n * t
        s  = np.sin(nt)
        c  = np.cos(nt)

        Phi = np.array([
            # δx          δy   δz    δẋ            δẏ              δż
            [4-3*c,       0,   0,    s/n,          2*(1-c)/n,      0     ],
            [6*(s-nt),    1,   0,   -2*(1-c)/n,    (4*s-3*nt)/n,   0     ],
            [0,           0,   c,    0,             0,              s/n   ],
            [3*n*s,       0,   0,    c,             2*s,            0     ],
            [6*n*(c-1),   0,   0,   -2*s,           4*c-3,          0     ],
            [0,           0,  -n*s,  0,             0,              c     ],
        ])
        return Phi

    def _cw_control_input(self,
                          accel: np.ndarray,
                          t:     float) -> np.ndarray:
        """
        Integral of Φ(t-τ)·B·u dτ  for constant acceleration u over [0,t].

        B = [0_{3×3}; I_{3×3}]  (acceleration enters velocity states directly).

        Derived by evaluating ∫₀ᵀ Φ(T−τ)·[0;0;0;ax;ay;az] dτ row by row:

        Position (rows 0–2 of the integral):
            ix = ax*(1−c)/n²  +  ay*2*(t − s/n)/n
            iy = −ax*2*(t − s/n)/n  +  ay*(4*(1−c)/n² − 1.5*t²)
            iz = az*(1−c)/n²

        Velocity (rows 3–5 of the integral):
            vx = (ax*s + 2*ay*(1−c)) / n
            vy = (−2*ax*(1−c) + ay*(4*s − 3*nt)) / n
            vz = az*s / n

        Verified numerically against RK4 integration of CW EOM.
        Previous position formulas were incorrect by ~57% at half-orbit
        timescales; velocity formulas were already correct.
        """
        n  = self.n
        nt = n * t
        s  = np.sin(nt)
        c  = np.cos(nt)
        ax, ay, az = accel

        # Position integrals — exact closed form
        ix = ax*(1.0 - c)/n**2  +  2.0*ay*(t - s/n)/n
        iy = -2.0*ax*(t - s/n)/n  +  ay*(4.0*(1.0 - c)/n**2 - 1.5*t**2)
        iz = az*(1.0 - c)/n**2

        # Velocity integrals — exact closed form
        vx = (ax*s        + 2.0*ay*(1.0 - c)) / n
        vy = (-2.0*ax*(1.0 - c) + ay*(4.0*s - 3.0*nt)) / n
        vz = az*s / n

        return np.array([ix, iy, iz, vx, vy, vz])

    # ─────────────────────────────────────────────────────────────────
    # Measurement model h(x) and Jacobian H(x)
    # ─────────────────────────────────────────────────────────────────

    def _h(self, dr: np.ndarray) -> np.ndarray:
        """
        Nonlinear measurement function h(x) = [range, az, el].
        """
        r  = np.linalg.norm(dr)
        az = np.arctan2(dr[1], dr[0])
        el = np.arctan2(dr[2], np.sqrt(dr[0]**2 + dr[1]**2))
        return np.array([r, az, el])

    def _H_jacobian(self, dr: np.ndarray) -> np.ndarray:
        """
        Analytical Jacobian H = ∂h/∂x  (3×6).
        Only the first 3 columns (position) are nonzero.
        """
        x, y, z = dr
        r       = np.linalg.norm(dr)
        r_xy2   = x**2 + y**2
        r_xy    = np.sqrt(r_xy2)

        if r < 1e-6 or r_xy < 1e-6:
            return np.zeros((3, 6))

        # ∂range/∂[x,y,z]
        dR = np.array([x/r, y/r, z/r])

        # ∂azimuth/∂[x,y,z]
        dAz = np.array([-y/r_xy2, x/r_xy2, 0.0])

        # ∂elevation/∂[x,y,z]
        dEl = np.array([-x*z/(r**2 * r_xy),
                        -y*z/(r**2 * r_xy),
                         r_xy / r**2])

        H = np.zeros((3, 6))
        H[0, 0:3] = dR
        H[1, 0:3] = dAz
        H[2, 0:3] = dEl

        return H

    @staticmethod
    def _wrap_angle(a: float) -> float:
        """Wrap angle to [-π, π]."""
        return (a + np.pi) % (2 * np.pi) - np.pi

    # ─────────────────────────────────────────────────────────────────
    # Convenience
    # ─────────────────────────────────────────────────────────────────

    @property
    def position(self) -> np.ndarray:
        return self.x[0:3].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[3:6].copy()

    @property
    def position_std(self) -> np.ndarray:
        return np.sqrt(np.diag(self.P)[0:3])

    @property
    def velocity_std(self) -> np.ndarray:
        return np.sqrt(np.diag(self.P)[3:6])