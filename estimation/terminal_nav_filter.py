import numpy as np


class TerminalNavFilter:
    """
    Alpha-beta terminal relative-navigation filter.

    The terminal camera provides a close-range position measurement. This
    filter estimates both position and velocity without raw finite
    differencing, which is too sensitive to centimeter-scale frame jitter.
    """

    def __init__(self,
                 alpha=0.35,
                 beta=0.06,
                 v_max_ms=0.10,
                 innovation_gate_m=0.25):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.v_max_ms = float(v_max_ms)
        self.innovation_gate_m = float(innovation_gate_m)
        self.initialized = False
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.rejected_updates = 0
        self.accepted_updates = 0

    def reset(self, pos=None, vel=None):
        self.initialized = False
        self.pos = np.zeros(3) if pos is None else np.asarray(pos, dtype=float).copy()
        self.vel = np.zeros(3) if vel is None else np.asarray(vel, dtype=float).copy()
        self.rejected_updates = 0
        self.accepted_updates = 0

    def update(self, pos_meas, dt, measurement_valid=True, vel_seed=None):
        pos_meas = np.asarray(pos_meas, dtype=float)
        dt = max(float(dt), 1e-6)

        if not self.initialized:
            self.pos = pos_meas.copy()
            self.vel = (np.zeros(3) if vel_seed is None
                        else np.asarray(vel_seed, dtype=float).copy())
            self._limit_velocity()
            self.initialized = True
            return self.pos.copy(), self.vel.copy()

        pos_pred = self.pos + self.vel * dt

        if not measurement_valid:
            self.pos = pos_pred
            self._limit_velocity()
            return self.pos.copy(), self.vel.copy()

        residual = pos_meas - pos_pred
        residual_norm = float(np.linalg.norm(residual))
        if residual_norm > self.innovation_gate_m:
            residual *= self.innovation_gate_m / residual_norm
            self.rejected_updates += 1
        else:
            self.accepted_updates += 1

        self.pos = pos_pred + self.alpha * residual
        self.vel = self.vel + self.beta * residual / dt
        self._limit_velocity()
        return self.pos.copy(), self.vel.copy()

    def _limit_velocity(self):
        speed = float(np.linalg.norm(self.vel))
        if speed > self.v_max_ms:
            self.vel *= self.v_max_ms / speed
