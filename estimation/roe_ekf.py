"""ROE EKF using one shared near-circular mean-element model.

Truth is never used during prediction, commands or measurement updates.
The model object supplies only chief orbit constants and mapping methods.
Chief argument of latitude is synchronized explicitly at initialization.
"""
import numpy as np
from environment.roe_dynamics import ROEDynamics


class ROEEKF:
    def __init__(self, roe_dyn: ROEDynamics, dt: float, sigma_accel=1e-6):
        self.dyn, self.dt = roe_dyn, float(dt)
        self.a, self.n = roe_dyn.a, roe_dyn.n
        self.mean_anomaly = roe_dyn.mean_anomaly
        self.x = np.zeros(6)
        self.sigma_accel = float(sigma_accel)
        self.Phi = roe_dyn.state_transition(dt)
        self.Q = np.zeros((6, 6))
        self.initialise_cartesian(np.zeros(6), self.mean_anomaly)

    def initialise(self, roe0, P0=None, mean_anomaly=None):
        """Explicit prior initialization only; never called after a burn."""
        self.x = np.asarray(roe0, dtype=float).copy()
        if mean_anomaly is not None:
            self.mean_anomaly = float(mean_anomaly) % (2*np.pi)
        if P0 is not None:
            self.P = np.asarray(P0, dtype=float).copy()

    def initialise_cartesian(self, state, mean_anomaly, P_cart=None):
        self.mean_anomaly = float(mean_anomaly) % (2*np.pi)
        Jinv = np.linalg.inv(self.dyn.mapping_matrix(self.mean_anomaly))
        self.x = Jinv @ state
        if P_cart is None:
            P_cart = np.diag([50.**2]*3 + [0.5**2]*3)
        self.P = Jinv @ P_cart @ Jinv.T

    def apply_impulse_lvlh(self, dv_lvlh, sigma_dv=0.0):
        """Apply a known command to the estimate, retaining navigation error."""
        G = self.dyn.impulse_matrix(self.mean_anomaly)
        self.x += G @ np.asarray(dv_lvlh)
        self.P += sigma_dv**2 * (G @ G.T)

    def predict(self, accel_lvlh=None):
        accel = np.zeros(3) if accel_lvlh is None else np.asarray(accel_lvlh)
        L = self.dyn.control_matrix(self.dt, self.mean_anomaly)
        self.x = self.Phi @ self.x + L @ accel
        self.Q = self.sigma_accel**2 * (L @ L.T)
        self.P = self.Phi @ self.P @ self.Phi.T + self.Q
        self.P = 0.5 * (self.P + self.P.T)
        self.mean_anomaly = (self.mean_anomaly + self.dyn.phase_rate*self.dt) % (2*np.pi)

    def update(self, z, R_meas, gate_k=5.0):
        dr = self.position
        if np.linalg.norm(dr) < 1e-3 or np.linalg.norm(dr[:2]) < 1e-6:
            return False
        innov = np.asarray(z) - self._h(dr)
        innov[1:] = (innov[1:] + np.pi) % (2*np.pi) - np.pi
        H = self._H_bearing(dr) @ self._jacobian_roe2pos()
        S = H @ self.P @ H.T + R_meas
        try:
            solved = np.linalg.solve(S, innov)
            if float(innov @ solved) > gate_k**2:
                return False
            K = np.linalg.solve(S, H @ self.P).T
        except np.linalg.LinAlgError:
            return False
        self.x += K @ innov
        IKH = np.eye(6) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R_meas @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        return True

    def _roe_to_lvlh(self, roe):
        return self.dyn.to_lvlh(self.mean_anomaly, roe=roe)

    def _jacobian_roe2pos(self):
        return self.dyn.position_matrix(self.mean_anomaly)

    @staticmethod
    def _h(dr):
        x, y, z = dr
        return np.array([np.linalg.norm(dr), np.arctan2(y, x),
                         np.arctan2(z, np.hypot(x, y))])

    @staticmethod
    def _H_bearing(dr):
        x, y, z = dr
        r, xy = np.linalg.norm(dr), np.hypot(x, y)
        return np.array([[x/r, y/r, z/r],
                         [-y/xy**2, x/xy**2, 0.],
                         [-x*z/(r*r*xy), -y*z/(r*r*xy), xy/(r*r)]])

    @staticmethod
    def _wrap(angle):
        return (angle + np.pi) % (2*np.pi) - np.pi

    @property
    def position(self):
        return self._roe_to_lvlh(self.x)[:3]

    @property
    def velocity(self):
        return self._roe_to_lvlh(self.x)[3:]

    @property
    def position_std(self):
        J = self._jacobian_roe2pos()
        return np.sqrt(np.maximum(np.diag(J @ self.P @ J.T), 0.0))
