import numpy as np


class ThrusterAllocation:
    def __init__(self, forces_n, force_body_n, torque_body_nm, residual):
        self.forces_n = np.asarray(forces_n, dtype=float)
        self.force_body_n = np.asarray(force_body_n, dtype=float)
        self.torque_body_nm = np.asarray(torque_body_nm, dtype=float)
        self.residual = np.asarray(residual, dtype=float)


class ThrusterLayout:
    """
    Physical deputy thruster layout and bounded wrench allocation.

    The allocator maps a requested body-frame force/torque wrench into
    non-negative single-thruster force commands. This is the first bridge from
    ideal acceleration guidance toward full 6DOF force/torque actuation.
    """

    def __init__(self, positions_body_m, directions_body, max_force_n):
        self.positions_body_m = np.asarray(positions_body_m, dtype=float)
        self.directions_body = np.asarray(directions_body, dtype=float)
        norms = np.linalg.norm(self.directions_body, axis=1)
        self.directions_body = self.directions_body / np.maximum(norms[:, None], 1e-12)
        self.max_force_n = np.asarray(max_force_n, dtype=float)
        if self.max_force_n.ndim == 0:
            self.max_force_n = np.full(len(self.directions_body), float(self.max_force_n))

        if self.positions_body_m.shape != self.directions_body.shape:
            raise ValueError("thruster positions and directions must have the same shape")
        if len(self.max_force_n) != len(self.directions_body):
            raise ValueError("max_force_n must be scalar or one value per thruster")

        torque_dirs = np.cross(self.positions_body_m, self.directions_body)
        self._A = np.vstack((self.directions_body.T, torque_dirs.T))

    @classmethod
    def box_16(cls, half_extents_m=(0.30, 0.30, 0.40), max_force_n=0.25):
        """
        Sixteen-thruster box layout for a 60 x 60 x 80 cm deputy.

        Thrusters are placed near the body corners in opposing force pairs so
        translation can be produced with small net torque when the vehicle is
        well aligned, while still retaining torque authority for later coupling.
        """
        hx, hy, hz = map(float, half_extents_m)
        entries = []
        # +/-X force thrusters, offset in y/z
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                entries.append(([0.0, sy * hy, sz * hz], [1.0, 0.0, 0.0]))
                entries.append(([0.0, sy * hy, sz * hz], [-1.0, 0.0, 0.0]))
        # +/-Y force thrusters, offset in x/z
        for sx in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                entries.append(([sx * hx, 0.0, sz * hz], [0.0, 1.0, 0.0]))
                entries.append(([sx * hx, 0.0, sz * hz], [0.0, -1.0, 0.0]))

        pos = np.array([e[0] for e in entries], dtype=float)
        dirs = np.array([e[1] for e in entries], dtype=float)
        return cls(pos, dirs, max_force_n)

    def allocate(self, force_body_n, torque_body_nm=None):
        force_body_n = np.asarray(force_body_n, dtype=float)
        if torque_body_nm is None:
            torque_body_nm = np.zeros(3)
        torque_body_nm = np.asarray(torque_body_nm, dtype=float)
        desired = np.concatenate((force_body_n, torque_body_nm))

        free = np.ones(self._A.shape[1], dtype=bool)
        forces = np.zeros(self._A.shape[1])
        remaining = desired.copy()

        for _ in range(self._A.shape[1] + 1):
            if not np.any(free):
                break
            A_free = self._A[:, free]
            sol, *_ = np.linalg.lstsq(A_free, remaining, rcond=None)
            trial = np.zeros_like(forces)
            trial[~free] = forces[~free]
            trial[free] = sol

            low = trial < 0.0
            high = trial > self.max_force_n
            violated = (low | high) & free
            if not np.any(violated):
                forces = np.clip(trial, 0.0, self.max_force_n)
                break

            clamp_idx = np.where(violated)[0]
            forces[clamp_idx] = np.where(low[clamp_idx], 0.0, self.max_force_n[clamp_idx])
            free[clamp_idx] = False
            remaining = desired - self._A[:, ~free] @ forces[~free]

        achieved = self._A @ forces
        return ThrusterAllocation(
            forces_n=forces,
            force_body_n=achieved[0:3],
            torque_body_nm=achieved[3:6],
            residual=desired - achieved)
