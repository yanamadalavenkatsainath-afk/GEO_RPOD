"""Impulsive ROE formation hold and model-consistent two-burn rendezvous.

At J2=0 the boundary-value solver is exactly CW. With secular J2 enabled
it uses the same averaged model as navigation, rather than an unrelated
Cartesian trajectory. This is a proximity-rendezvous demonstration;
no docking, keep-out-zone or passive-safety qualification is implied.
"""
from enum import Enum, auto
import numpy as np


class ROEMode(Enum):
    FORMATION_HOLD = auto()
    RENDEZVOUS = auto()
    COASTING = auto()


class ROEController:
    def __init__(self, roe_dyn, mode=ROEMode.FORMATION_HOLD, target_roe=None):
        self.dyn, self.mode = roe_dyn, mode
        self.a, self.n, self.T = roe_dyn.a, roe_dyn.n, roe_dyn.T
        self.target_roe = np.zeros(6) if target_roe is None else np.array(target_roe)
        self.dv_max, self.pos_deadband_m = 0.5, 20.0
        self._last_burn_t, self._burn_interval = -999., self.T/2
        self._rdv_burn1_applied = self._rdv_burn2_applied = False
        self._rdv_t_start = self._rdv_T = None
        self._rdv_dv1 = self._rdv_dv2 = None
        self.mode_history = []

    def compute(self, roe_est, mean_anomaly, t=0.):
        if self.mode == ROEMode.FORMATION_HOLD:
            dv = self._formation_hold(roe_est, mean_anomaly, t)
        elif self.mode == ROEMode.RENDEZVOUS:
            dv = self._rendezvous(roe_est, mean_anomaly, t)
        else:
            dv = None
        return np.zeros(3), dv

    def set_mode(self, mode, roe_est=None, mean_anomaly=0., t=0.,
                 t_sim_max=None, lvlh_est=None):
        self.mode = mode
        self.mode_history.append((t, mode))
        if mode == ROEMode.RENDEZVOUS:
            self._rdv_burn1_applied = self._rdv_burn2_applied = False
            self._rdv_t_start, self._rdv_T = t, None
            self._rdv_dv1 = self._rdv_dv2 = None
            if roe_est is not None:
                self._plan_rendezvous(roe_est, t, t_sim_max,
                                      lvlh_est, mean_anomaly)

    def _formation_hold(self, roe_est, mean_anomaly, t):
        if t - self._last_burn_t < self._burn_interval:
            return None
        err = roe_est - self.target_roe
        if np.linalg.norm(self._roe_to_pos(err, mean_anomaly)) < self.pos_deadband_m:
            return None
        G = self.dyn.impulse_matrix(mean_anomaly)
        dv = -np.linalg.lstsq(G, err, rcond=None)[0]
        mag = np.linalg.norm(dv)
        if mag < 1e-7:
            return None
        if mag > self.dv_max:
            dv *= self.dv_max / mag
        self._last_burn_t = t
        return dv

    def _plan_rendezvous(self, roe_est, t, t_sim_max=None,
                         lvlh_est=None, mean_anomaly=None):
        """Select lowest-dv candidate within a bounded transfer-time grid.

        This grid search is not a global optimality proof. ``lvlh_est`` is
        a caller-selected estimator state, never a truth fallback.
        """
        phase = self.dyn.mean_anomaly if mean_anomaly is None else mean_anomaly
        state = (self.dyn.to_lvlh(phase, roe_est) if lvlh_est is None
                 else np.asarray(lvlh_est).copy())
        max_transfer = 0.55*self.T
        if t_sim_max is not None:
            max_transfer = min(max_transfer, t_sim_max - t - 600.)
        if max_transfer < 0.25*self.T:
            self.mode = ROEMode.FORMATION_HOLD
            return
        best = None
        for transfer in np.linspace(0.25*self.T, max_transfer, 80):
            Phi = self.dyn.cartesian_transition(transfer, phase)
            dv1, dv2 = self._solve_two_impulse(state, Phi)
            if dv1 is None or max(np.linalg.norm(dv1), np.linalg.norm(dv2)) > 0.10:
                continue
            cost = np.linalg.norm(dv1) + np.linalg.norm(dv2)
            if best is None or cost < best[0]:
                best = cost, transfer, dv1, dv2
        if best is None:
            self.mode = ROEMode.FORMATION_HOLD
            return
        _, self._rdv_T, self._rdv_dv1, self._rdv_dv2 = best

    @staticmethod
    def _solve_two_impulse(state0, Phi):
        rr, rv = Phi[:3, :3], Phi[:3, 3:]
        if np.linalg.cond(rv) > 1e8:
            return None, None
        try:
            dv1 = -np.linalg.solve(rv, rr @ state0[:3] + rv @ state0[3:])
        except np.linalg.LinAlgError:
            return None, None
        dv2 = -(Phi[3:, :3] @ state0[:3] + Phi[3:, 3:] @ (state0[3:] + dv1))
        return dv1, dv2

    def _cw_two_impulse(self, state0, transfer_time):
        """Retained explicit unperturbed CW reference solver."""
        n, nt = self.n, self.n*transfer_time
        s, c = np.sin(nt), np.cos(nt)
        Phi = np.array([[4-3*c, 0, 0, s/n, 2*(1-c)/n, 0],
                        [6*(s-nt), 1, 0, -2*(1-c)/n, (4*s-3*nt)/n, 0],
                        [0, 0, c, 0, 0, s/n],
                        [3*n*s, 0, 0, c, 2*s, 0],
                        [6*n*(c-1), 0, 0, -2*s, 4*c-3, 0],
                        [0, 0, -n*s, 0, 0, c]])
        return self._solve_two_impulse(np.asarray(state0), Phi)

    def _rendezvous(self, roe_est, mean_anomaly, t):
        if self._rdv_T is None:
            return None
        if not self._rdv_burn1_applied:
            self._rdv_burn1_applied = True
            return self._rdv_dv1.copy()
        if not self._rdv_burn2_applied and t - self._rdv_t_start >= self._rdv_T:
            self._rdv_burn2_applied = True
            self.mode = ROEMode.COASTING
            # Arrival braking uses estimated velocity, not scheduled truth.
            dv = -self.dyn.to_lvlh(mean_anomaly, roe=roe_est)[3:]
            mag = np.linalg.norm(dv)
            return dv if mag <= self.dv_max else dv*self.dv_max/mag
        return None

    def _roe_to_pos(self, roe, mean_anomaly=None):
        return self.dyn.position_matrix(mean_anomaly) @ roe

    def _roe_to_lvlh_est(self, roe, mean_anomaly=None):
        return self.dyn.to_lvlh(mean_anomaly, roe=roe)
