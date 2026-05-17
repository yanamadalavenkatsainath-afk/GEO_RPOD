import numpy as np


class PortTracker:
    """
    Lightweight gated tracker for direct dock-port measurements.

    It replaces one-off exponential smoothing with explicit state, validity,
    and innovation limiting. The port velocity used by guidance still comes
    from chief tumble geometry; this tracker owns the measured port position.
    """

    def __init__(self, alpha=0.25, innovation_gate_m=0.15, max_coast_s=5.0):
        self.alpha = float(alpha)
        self.innovation_gate_m = float(innovation_gate_m)
        self.max_coast_s = float(max_coast_s)
        self.initialized = False
        self.pos = np.zeros(3)
        self.coast_s = 0.0
        self.accepted_updates = 0
        self.rejected_updates = 0

    def reset(self):
        self.initialized = False
        self.pos = np.zeros(3)
        self.coast_s = 0.0
        self.accepted_updates = 0
        self.rejected_updates = 0

    def update(self, pos_meas, dt, measurement_valid=True):
        dt = max(float(dt), 1e-6)
        if not measurement_valid:
            self.coast_s += dt
            if self.coast_s > self.max_coast_s:
                self.reset()
            return self.pos.copy(), self.initialized

        pos_meas = np.asarray(pos_meas, dtype=float)
        self.coast_s = 0.0

        if not self.initialized:
            self.pos = pos_meas.copy()
            self.initialized = True
            self.accepted_updates = 1
            return self.pos.copy(), True

        residual = pos_meas - self.pos
        residual_norm = float(np.linalg.norm(residual))
        if residual_norm > self.innovation_gate_m:
            residual *= self.innovation_gate_m / residual_norm
            self.rejected_updates += 1
        else:
            self.accepted_updates += 1

        self.pos = self.pos + self.alpha * residual
        return self.pos.copy(), True
