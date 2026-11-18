"""
Clohessy-Wiltshire (Hill's Equations) Relative Orbital Dynamics
================================================================
Models the relative motion of a DEPUTY spacecraft with respect to a
CHIEF spacecraft in a circular reference orbit.

State vector (LVLH / Hill's frame):
    x = [δx, δy, δz, δẋ, δẏ, δż]  [m, m/s]

LVLH frame definition:
    +x : radial (outward from Earth)
    +y : along-track (direction of motion)
    +z : cross-track (completes right-hand system, ~ orbit normal)

Clohessy-Wiltshire equations (linearised, circular chief orbit):
    δẍ - 2n·δẏ - 3n²·δx = ax
    δÿ + 2n·δẋ          = ay
    δz̈ + n²·δz          = az

where n = sqrt(μ/a³) is the chief mean motion and ax,ay,az are
control accelerations in LVLH frame [m/s²].

Reference:
    Clohessy & Wiltshire (1960), J. Aerospace Sciences, 27(9).
    Schaub & Junkins, "Analytical Mechanics of Space Systems", §14.3
    Vallado, "Fundamentals of Astrodynamics and Applications", §6.3
"""

import numpy as np


class CWDynamics:
    """
    Clohessy-Wiltshire relative motion propagator.

    Parameters
    ----------
    chief_orbit_radius_km : float
        Semi-major axis of chief circular orbit [km].
        Default: 6871 km (~500 km altitude).
    """

    MU = 3.986004418e14    # m³/s²   Earth gravitational parameter

    def __init__(self, chief_orbit_radius_km: float = 6871.0):
        self.a   = chief_orbit_radius_km * 1e3          # m
        self.n   = np.sqrt(self.MU / self.a**3)         # rad/s  mean motion
        self.T   = 2.0 * np.pi / self.n                 # s      orbital period

        # State: [δx, δy, δz, δẋ, δẏ, δż]
        self.state = np.zeros(6)

        # Accumulated delta-V log [m/s]
        self.dv_total = np.zeros(3)

        print(f"  CW: chief orbit a={chief_orbit_radius_km:.1f} km, "
              f"n={np.degrees(self.n)*3600:.4f} deg/hr, "
              f"T={self.T/60:.2f} min")

    # ─────────────────────────────────────────────────────────────────
    # State initialisation helpers
    # ─────────────────────────────────────────────────────────────────

    def set_initial_offset(self,
                           dr_lvlh_m: np.ndarray,
                           dv_lvlh_ms: np.ndarray = None):
        """
        Set deputy initial conditions in LVLH frame.

        Parameters
        ----------
        dr_lvlh_m  : relative position [δx, δy, δz]  [m]
        dv_lvlh_ms : relative velocity [δẋ, δẏ, δż]  [m/s]
                     Default: natural drift-free (passive safety ellipse).
        """
        if dv_lvlh_ms is None:
            # Drift-free condition: δẏ₀ = -2n·δx₀  (no secular drift in y)
            dv_lvlh_ms = np.array([0.0,
                                   -2.0 * self.n * dr_lvlh_m[0],
                                   0.0])
        self.state = np.concatenate([dr_lvlh_m, dv_lvlh_ms])

    def set_passive_safety_ellipse(self,
                                   rho_m: float,
                                   phase_deg: float = 0.0):
        """
        Initialise a Passive Safety Ellipse (PSE) — a natural CW orbit
        that forms a 2:1 ellipse in the radial/along-track plane.

        The PSE is collision-safe because the deputy never drifts toward
        the chief without a maneuver.

        Parameters
        ----------
        rho_m     : semi-minor axis (radial amplitude) [m]
        phase_deg : initial phase in the ellipse [deg]
        """
        phi = np.radians(phase_deg)
        n   = self.n
        # Parametric: δx = ρ cos(φ),  δy = -2ρ sin(φ),  δz = 0
        # Velocities: δẋ = -nρ sin(φ), δẏ = -2nρ cos(φ)
        dr = np.array([rho_m * np.cos(phi),
                       -2.0 * rho_m * np.sin(phi),
                       0.0])
        dv = np.array([-self.n * rho_m * np.sin(phi),
                       -2.0 * self.n * rho_m * np.cos(phi),
                       0.0])
        self.state = np.concatenate([dr, dv])

    # ─────────────────────────────────────────────────────────────────
    # Equations of motion
    # ─────────────────────────────────────────────────────────────────

    def _eom(self, state: np.ndarray, accel_lvlh: np.ndarray) -> np.ndarray:
        """
        CW equations of motion — continuous form.

        ẋ = A·x + B·u

        State matrix A:
            [  0   0   0   1   0   0 ]
            [  0   0   0   0   1   0 ]
            [  0   0   0   0   0   1 ]
            [3n²  0   0   0   2n  0 ]
            [  0   0   0  -2n  0   0 ]
            [  0   0  -n²  0   0   0 ]
        """
        n  = self.n
        dx, dy, dz, ddx, ddy, ddz = state
        ax, ay, az = accel_lvlh

        d2x =  3.0*n**2 * dx + 2.0*n * ddy + ax
        d2y = -2.0*n * ddx               + ay
        d2z = -n**2 * dz                 + az

        return np.array([ddx, ddy, ddz, d2x, d2y, d2z])

    def step(self,
             dt: float,
             accel_lvlh: np.ndarray = None) -> np.ndarray:
        """
        Propagate relative state one timestep using RK4.

        Parameters
        ----------
        dt          : timestep [s]
        accel_lvlh  : control acceleration in LVLH [m/s²]. Default: zeros.

        Returns
        -------
        state : updated [δx, δy, δz, δẋ, δẏ, δż]
        """
        if accel_lvlh is None:
            accel_lvlh = np.zeros(3)

        s = self.state
        k1 = self._eom(s,                 accel_lvlh)
        k2 = self._eom(s + 0.5*dt*k1,    accel_lvlh)
        k3 = self._eom(s + 0.5*dt*k2,    accel_lvlh)
        k4 = self._eom(s + dt*k3,         accel_lvlh)

        self.state = s + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return self.state.copy()

    def apply_impulse(self, dv_lvlh: np.ndarray):
        """
        Apply an instantaneous delta-V to the deputy in LVLH frame.

        Parameters
        ----------
        dv_lvlh : [δvx, δvy, δvz] impulse [m/s]
        """
        self.state[3:6] += dv_lvlh
        self.dv_total   += np.abs(dv_lvlh)

    # ─────────────────────────────────────────────────────────────────
    # Analytical CW solution (zero thrust)
    # ─────────────────────────────────────────────────────────────────

    def analytical_state(self, t: float) -> np.ndarray:
        """
        Closed-form CW solution from initial state (no thrust).

        Useful for maneuver planning and targeting.
        Schaub & Junkins Eq. 14.22.
        """
        n  = self.n
        nt = n * t
        x0, y0, z0, vx0, vy0, vz0 = self.state

        s, c = np.sin(nt), np.cos(nt)

        x  = (4 - 3*c)*x0 + s*vx0/n + 2*(1 - c)*vy0/n
        y  = 6*(s - nt)*x0 + y0 - 2*(1 - c)*vx0/n + (4*s - 3*nt)*vy0/n
        z  =  c*z0 + s*vz0/n

        vx =  3*n*s*x0 + c*vx0 + 2*s*vy0
        vy =  (6*n*c - 6*n)*x0 - 2*s*vx0 + (4*c - 3)*vy0
        vz = -n*s*z0 + c*vz0

        return np.array([x, y, z, vx, vy, vz])

    # ─────────────────────────────────────────────────────────────────
    # LVLH ↔ ECI conversions
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def lvlh_to_eci(dr_lvlh: np.ndarray,
                    r_chief_eci: np.ndarray,
                    v_chief_eci: np.ndarray) -> np.ndarray:
        """
        Convert LVLH relative position to ECI absolute position.

        LVLH basis:
            x̂ = r̂_chief             (radial)
            ẑ = r̂ × v̂  (orbit normal, but reversed: ẑ_LVLH = h_hat)
            ŷ = ẑ × x̂               (along-track)

        Parameters
        ----------
        dr_lvlh     : relative position in LVLH [m]
        r_chief_eci : chief ECI position [m]
        v_chief_eci : chief ECI velocity [m/s]

        Returns
        -------
        r_deputy_eci : ECI position of deputy [m]
        """
        x_hat = r_chief_eci / np.linalg.norm(r_chief_eci)
        h_vec = np.cross(r_chief_eci, v_chief_eci)
        z_hat = h_vec / np.linalg.norm(h_vec)
        y_hat = np.cross(z_hat, x_hat)

        R_lvlh_to_eci = np.column_stack([x_hat, y_hat, z_hat])  # 3×3
        return r_chief_eci + R_lvlh_to_eci @ dr_lvlh

    @staticmethod
    def rotation_matrix_lvlh(r_chief: np.ndarray,
                              v_chief: np.ndarray) -> np.ndarray:
        """
        Returns 3×3 rotation matrix R such that v_LVLH = R @ v_ECI.
        (ECI → LVLH)
        """
        x_hat = r_chief / np.linalg.norm(r_chief)
        h_vec = np.cross(r_chief, v_chief)
        z_hat = h_vec / np.linalg.norm(h_vec)
        y_hat = np.cross(z_hat, x_hat)
        return np.row_stack([x_hat, y_hat, z_hat])   # rows = LVLH axes in ECI

    # ─────────────────────────────────────────────────────────────────
    # Convenience
    # ─────────────────────────────────────────────────────────────────

    @property
    def position(self) -> np.ndarray:
        return self.state[0:3].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.state[3:6].copy()

    @property
    def range_m(self) -> float:
        return float(np.linalg.norm(self.state[0:3]))

    @property
    def total_dv_ms(self) -> float:
        return float(np.sum(self.dv_total))
