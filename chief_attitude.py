"""
Chief Spacecraft Attitude Model — Free-Tumbling 6-DOF
======================================================
Models the rotational state of a non-cooperative GEO client satellite
(e.g. a fuel-depleted IS-1002 class comsat) that has lost attitude control.

At end-of-life, GEO satellites typically enter a slow tumble driven by
residual angular momentum and environmental torques (gravity gradient,
SRP). This model captures that behaviour with:

  - Euler rigid-body dynamics (RK4)
  - Gravity gradient torque (dominant at GEO for large inertia ratios)
  - Optional constant angular rate (simple tumble) or full dynamics

The docking port is defined in the chief body frame. Its ECI position
is exposed each timestep so the TERMINAL guidance can track it.

Typical GEO derelict tumble rates: 0.1–2.0 deg/s
IS-1002 class inertia (approx): 3000 kg × 30m span
    Ixx ~ 2.5e6 kg·m², Iyy ~ 2.5e6 kg·m², Izz ~ 4.0e5 kg·m²
    (large asymmetry drives gravity gradient coupling)

Reference:
    Shan et al., "Review of dynamics modelling of space debris objects",
    Acta Astronautica, 2016.
    Muñoz-Paez et al., "Attitude dynamics of uncontrolled GEO objects",
    AIAA SciTech 2019.
"""

import numpy as np


def _normalize(q):
    return q / np.linalg.norm(q)


def _quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _rot_matrix(q):
    """Rotation matrix from quaternion [w,x,y,z]: maps body → inertial."""
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z),  2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),    1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),    2*(y*z+w*x),   1-2*(x*x+y*y)],
    ])


class ChiefAttitude:
    """
    Free-tumbling rigid-body attitude for non-cooperative GEO chief.

    Parameters
    ----------
    inertia : array (3,) or (3,3)
        Principal moments of inertia [kg·m²].
        Pass a 3-vector for diagonal inertia tensor.
    omega0_deg_s : array (3,)
        Initial angular rate [deg/s] in body frame.
    q0 : array (4,) optional
        Initial quaternion [w,x,y,z]. Default: random orientation.
    dock_port_body : array (3,)
        Docking port offset from CoM in body frame [m].
        Default: [0, 0, 0.5] — port on +z face, 0.5m from CoM.
    dock_axis_body : array (3,)
        Docking port approach axis in body frame (unit vector).
        Deputy must align with this axis for capture.
        Default: [0, 0, 1] — approach from +z.
    mu : float
        Gravitational parameter [m³/s²].
    """

    MU = 3.986004418e14

    def __init__(self,
                 inertia=None,
                 omega0_deg_s=None,
                 q0=None,
                 dock_port_body=None,
                 dock_axis_body=None,
                 mu=3.986004418e14,
                 enable_gg_torque=True):

        # Inertia tensor — IS-1002 class defaults
        if inertia is None:
            inertia = np.array([2.5e6, 2.5e6, 4.0e5])   # kg·m²
        inertia = np.asarray(inertia, dtype=float)
        if inertia.ndim == 1:
            self.I     = np.diag(inertia)
            self.I_inv = np.diag(1.0 / inertia)
        else:
            self.I     = inertia
            self.I_inv = np.linalg.inv(inertia)

        # Initial angular rate
        if omega0_deg_s is None:
            omega0_deg_s = np.array([0.05, 0.10, 0.03])   # deg/s — typical derelict
        self.omega = np.radians(omega0_deg_s)   # rad/s

        # Initial quaternion
        if q0 is None:
            # Random orientation — non-cooperative sat has unknown attitude
            q0 = _normalize(np.random.randn(4))
        self.q = _normalize(np.asarray(q0, dtype=float))

        # Docking port geometry in body frame
        if dock_port_body is None:
            dock_port_body = np.array([0.0, 0.0, 0.5])   # +z face, 0.5m from CoM
        self.dock_port_body = np.asarray(dock_port_body, dtype=float)

        if dock_axis_body is None:
            dock_axis_body = np.array([0.0, 0.0, 1.0])   # approach along +z
        self.dock_axis_body = np.asarray(dock_axis_body, dtype=float)
        self.dock_axis_body /= np.linalg.norm(self.dock_axis_body)

        self.mu             = mu
        self.enable_gg      = enable_gg_torque

        # Log initial state
        rate_mag = np.degrees(np.linalg.norm(self.omega))
        print(f"  ChiefAttitude: |ω|={rate_mag:.3f} deg/s  "
              f"dock_port_body={self.dock_port_body}  "
              f"gg_torque={'ON' if enable_gg_torque else 'OFF'}")

    # ─────────────────────────────────────────────────────────────────
    # Step — integrate attitude dynamics one timestep
    # ─────────────────────────────────────────────────────────────────

    def step(self, dt: float, chi_pos_eci: np.ndarray):
        """
        Propagate chief attitude by dt seconds using RK4.

        Parameters
        ----------
        dt          : timestep [s]
        chi_pos_eci : chief ECI position [m] (for gravity gradient torque)
        """
        def f(q, omega):
            tau = self._gravity_gradient(q, chi_pos_eci) if self.enable_gg else np.zeros(3)
            omega_dot = self.I_inv @ (tau - np.cross(omega, self.I @ omega))
            qw, qx, qy, qz = q
            ox, oy, oz = omega
            q_dot = 0.5 * np.array([
                -qx*ox - qy*oy - qz*oz,
                 qw*ox + qy*oz - qz*oy,
                 qw*oy - qx*oz + qz*ox,
                 qw*oz + qx*oy - qy*ox,
            ])
            return q_dot, omega_dot

        k1q, k1w = f(self.q, self.omega)
        k2q, k2w = f(_normalize(self.q + 0.5*dt*k1q), self.omega + 0.5*dt*k1w)
        k3q, k3w = f(_normalize(self.q + 0.5*dt*k2q), self.omega + 0.5*dt*k2w)
        k4q, k4w = f(_normalize(self.q + dt*k3q),     self.omega + dt*k3w)

        self.q     = _normalize(self.q + (dt/6)*(k1q + 2*k2q + 2*k3q + k4q))
        self.omega = self.omega + (dt/6)*(k1w + 2*k2w + 2*k3w + k4w)

    # ─────────────────────────────────────────────────────────────────
    # Docking port position and approach axis in ECI
    # ─────────────────────────────────────────────────────────────────

    def dock_port_eci(self, chi_pos_eci: np.ndarray) -> np.ndarray:
        """
        ECI position of the docking port [m].
        = chief CoM ECI + R_body2eci @ dock_port_body
        """
        R = _rot_matrix(self.q)
        return chi_pos_eci + R @ self.dock_port_body

    def dock_axis_eci(self) -> np.ndarray:
        """
        Docking approach axis in ECI (unit vector).
        Deputy must approach along the negative of this vector.
        """
        R = _rot_matrix(self.q)
        return R @ self.dock_axis_body

    # ─────────────────────────────────────────────────────────────────
    # Gravity gradient torque
    # ─────────────────────────────────────────────────────────────────

    def _gravity_gradient(self, q: np.ndarray,
                          pos_eci: np.ndarray) -> np.ndarray:
        """
        Gravity gradient torque in body frame [N·m].
        τ_gg = 3μ/r³ × (r̂_body × I·r̂_body)

        This is the dominant environmental torque for large GEO satellites.
        For IS-1002 with Ixx=Iyy=2.5e6, Izz=4e5:
            max τ_gg ≈ 3μ/r³ × |Izz-Ixx| ≈ 3×3.986e14/(4.2e7)³ × 2.1e6 ≈ 10 N·m
        This drives tumble on a timescale of hours.
        """
        r_I   = pos_eci
        r_mag = np.linalg.norm(r_I)
        r_hat_I = r_I / r_mag

        R       = _rot_matrix(q)
        r_hat_b = R.T @ r_hat_I   # nadir unit vector in body frame

        coeff = 3.0 * self.mu / r_mag**3
        return coeff * np.cross(r_hat_b, self.I @ r_hat_b)

    # ─────────────────────────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────────────────────────

    @property
    def rate_deg_s(self) -> float:
        return float(np.degrees(np.linalg.norm(self.omega)))

    @property
    def quaternion(self) -> np.ndarray:
        return self.q.copy()

    @property
    def omega_body(self) -> np.ndarray:
        return self.omega.copy()