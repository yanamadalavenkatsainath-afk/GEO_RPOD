"""Near-circular, small-separation ROE dynamics in the chief RTN frame.

State: [da, dlambda, dex, dey, dix, diy], dimensionless, deputy minus chief.
RTN axes are radial, along-track, orbit normal; velocity is d(position)/dt
in that rotating frame. ``mean_anomaly`` is the historical API name for
the chief *mean argument of latitude* u (not M on an eccentric orbit).

The circular two-body mapping and impulse map follow Delurgio & D'Amico,
IEEE Aerospace 2024, Eqs. (3)-(5), Appendix Eq. (64). The optional J2
model is the circular-chief limit of the first-order secular mean-element
rates in Koenig, Guffanti & D'Amico, JGCD 2017, Eqs. (13)-(15),(24).
It omits chief-eccentricity terms, short-period corrections, drag and SRP.
J2 Cartesian velocity is the derivative of the adopted averaged position
map; it is NOT a reconstructed osculating inertial state.
"""

import numpy as np


class ROEDynamics:
    MU = 3.986004418e14
    J2 = 1.08263e-3
    RE = 6.3781e6

    def __init__(self, a_chief_m, e_chief=0.0,
                 i_chief=np.radians(51.6), j2=True):
        self.a, self.e, self.i = map(float, (a_chief_m, e_chief, i_chief))
        if self.a <= self.RE or not 0 <= self.e < 0.01:
            raise ValueError("Requires an Earth orbit with 0 <= chief e < 0.01")
        self.n = np.sqrt(self.MU / self.a**3)
        self.T = 2 * np.pi / self.n
        self.eta = np.sqrt(1 - self.e**2)
        self.j2_enabled = bool(j2)
        # Deliberately use e=0 coefficients: finite-chief-e terms are omitted.
        k = 1.5 * self.J2 * (self.RE / self.a)**2 * self.n if j2 else 0.0
        ci, si = np.cos(self.i), np.sin(self.i)
        self.omega_dot = 0.5 * k * (5 * ci**2 - 1)
        self.RAAN_dot = -k * ci
        self.phase_rate = self.n + k * (4 * ci**2 - 1)
        self._kappa_ei = self.omega_dot  # compatibility with older callers
        self.A = np.zeros((6, 6))
        self.A[1, 0] = -1.5 * self.n - 3.5 * k * (3 * ci**2 - 1)
        self.A[1, 4] = -7 * k * si * ci
        self.A[2, 3] = -self.omega_dot
        self.A[3, 2] = self.omega_dot
        self.A[5, 0] = 3.5 * k * si * ci
        self.A[5, 4] = k * si**2
        self.roe = np.zeros(6)
        self.mean_anomaly = 0.0
        self.t = 0.0
        self.dv_total = np.zeros(3)  # legacy componentwise accounting
        self._dv_used = 0.0         # physical sum of burn magnitudes

    def position_matrix(self, mean_anomaly=None):
        u = self.mean_anomaly if mean_anomaly is None else mean_anomaly
        s, c = np.sin(u), np.cos(u)
        return self.a * np.array([[1, 0, -c, -s, 0, 0],
                                  [0, 1, 2*s, -2*c, 0, 0],
                                  [0, 0, 0, 0, s, -c]])

    def mapping_matrix(self, mean_anomaly=None):
        """ROE -> [RTN position, rotating-frame velocity], at a given u."""
        u = self.mean_anomaly if mean_anomaly is None else mean_anomaly
        s, c = np.sin(u), np.cos(u)
        dD_du = self.a * np.array([[0, 0, s, -c, 0, 0],
                                    [0, 0, 2*c, 2*s, 0, 0],
                                    [0, 0, 0, 0, c, s]])
        D = self.position_matrix(u)
        return np.vstack((D, self.phase_rate * dD_du + D @ self.A))

    def from_lvlh(self, state, mean_anomaly=None):
        """Inverse of the same map; does not mutate truth."""
        return np.linalg.solve(self.mapping_matrix(mean_anomaly),
                               np.asarray(state, dtype=float))

    def set_from_lvlh(self, dr_lvlh, dv_lvlh):
        self.roe = self.from_lvlh(np.r_[dr_lvlh, dv_lvlh])

    def set_roe(self, roe):
        self.roe = np.asarray(roe, dtype=float).copy()

    def to_lvlh(self, mean_anomaly=None, roe=None):
        return self.mapping_matrix(mean_anomaly) @ (self.roe if roe is None else roe)

    def state_transition(self, dt):
        """Exact exponential of this reduced constant secular model."""
        F = np.eye(6)
        F[1, 0], F[1, 4] = self.A[1, 0] * dt, self.A[1, 4] * dt
        F[5, 0], F[5, 4] = self.A[5, 0] * dt, self.A[5, 4] * dt
        c, s = np.cos(self.omega_dot * dt), np.sin(self.omega_dot * dt)
        F[2:4, 2:4] = [[c, -s], [s, c]]
        return F

    def impulse_matrix(self, mean_anomaly=None):
        """Command-only map that preserves position and adds requested RTN dv.

        At J2=0 this is the standard circular GVE impulse matrix exactly.
        With averaged J2 it preserves the velocity convention above.
        """
        return np.linalg.solve(self.mapping_matrix(mean_anomaly),
                               np.vstack((np.zeros((3, 3)), np.eye(3))))

    def control_matrix(self, dt, mean_anomaly=None):
        """Midpoint equivalent impulse for constant RTN acceleration, O(dt^3)."""
        u = self.mean_anomaly if mean_anomaly is None else mean_anomaly
        return (self.state_transition(dt / 2)
                @ self.impulse_matrix(u + self.phase_rate * dt / 2) * dt)

    def cartesian_transition(self, dt, mean_anomaly=None):
        """Cartesian STM of the selected ROE model; CW equivalent at J2=0."""
        u = self.mean_anomaly if mean_anomaly is None else mean_anomaly
        return (self.mapping_matrix(u + self.phase_rate * dt)
                @ self.state_transition(dt)
                @ np.linalg.inv(self.mapping_matrix(u)))

    def step(self, dt, accel_lvlh=None):
        if dt <= 0:
            raise ValueError("dt must be positive")
        accel = np.zeros(3) if accel_lvlh is None else np.asarray(accel_lvlh)
        self.roe = (self.state_transition(dt) @ self.roe
                    + self.control_matrix(dt) @ accel)
        self.mean_anomaly = (self.mean_anomaly + self.phase_rate * dt) % (2*np.pi)
        self.t += dt
        self._dv_used += float(np.linalg.norm(accel)) * dt
        self.dv_total += np.abs(accel) * dt
        return self.to_lvlh()

    def apply_impulse_lvlh(self, dv_lvlh, mean_anomaly=None):
        dv = np.asarray(dv_lvlh, dtype=float)
        self.roe += self.impulse_matrix(mean_anomaly) @ dv
        self.dv_total += np.abs(dv)
        self._dv_used += float(np.linalg.norm(dv))

    @property
    def range_m(self):
        return float(np.linalg.norm(self.to_lvlh()[:3]))

    @property
    def total_dv_ms(self):
        return self._dv_used
