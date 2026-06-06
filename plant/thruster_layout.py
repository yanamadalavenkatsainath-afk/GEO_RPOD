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

    @classmethod
    def box_24(cls, half_extents_m=(0.30, 0.30, 0.40), max_force_n=0.25):
        """
        Twenty-four-thruster box layout: ±X, ±Y, ±Z opposing pairs at face edges.

        Extends box_16 with four ±Z thruster pairs mounted at the x/y corners,
        giving full 6-axis force authority and better torque decoupling.

            ±X:  8 thrusters at (0, ±hy, ±hz)
            ±Y:  8 thrusters at (±hx, 0, ±hz)
            ±Z:  8 thrusters at (±hx, ±hy, 0)   ← new
        """
        hx, hy, hz = map(float, half_extents_m)
        entries = []
        # ±X thrusters at y/z face edges
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                entries.append(([0.0, sy * hy, sz * hz], [ 1.0, 0.0, 0.0]))
                entries.append(([0.0, sy * hy, sz * hz], [-1.0, 0.0, 0.0]))
        # ±Y thrusters at x/z face edges
        for sx in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                entries.append(([sx * hx, 0.0, sz * hz], [0.0,  1.0, 0.0]))
                entries.append(([sx * hx, 0.0, sz * hz], [0.0, -1.0, 0.0]))
        # ±Z thrusters at x/y face edges
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                entries.append(([sx * hx, sy * hy, 0.0], [0.0, 0.0,  1.0]))
                entries.append(([sx * hx, sy * hy, 0.0], [0.0, 0.0, -1.0]))

        pos  = np.array([e[0] for e in entries], dtype=float)
        dirs = np.array([e[1] for e in entries], dtype=float)
        return cls(pos, dirs, max_force_n)

    @classmethod
    def quad_16(cls, half_extents_m=(0.30, 0.30, 0.40), cant_deg=15.0, max_force_n=0.25):
        """
        Standard 16-thruster quad-cluster RCS (Apollo/Dragon style).

        Four clusters of four nozzles placed 90° apart at the body midplane
        (±X and ±Y faces, z=0). Each nozzle fires tangentially (±Y for ±X-face
        clusters, ±X for ±Y-face clusters) and is canted cant_deg toward ±Z so
        the docking axis gets force authority without dedicated axial thrusters.

        Z-axis authority: 8 × sin(cant_deg) × max_force_n per direction.
        Plume toward chief (+Z): only aft-canted nozzles (dz<0); use chief_mask
        to block these during final approach.
        """
        hx, hy, _ = map(float, half_extents_m)
        c = float(np.cos(np.radians(cant_deg)))
        s = float(np.sin(np.radians(cant_deg)))
        entries = []
        for px in (+hx, -hx):
            for dy, dz in [(+c, +s), (+c, -s), (-c, +s), (-c, -s)]:
                entries.append(([px, 0.0, 0.0], [0.0, dy, dz]))
        for py in (+hy, -hy):
            for dx, dz in [(+c, +s), (+c, -s), (-c, +s), (-c, -s)]:
                entries.append(([0.0, py, 0.0], [dx, 0.0, dz]))
        pos  = np.array([e[0] for e in entries], dtype=float)
        dirs = np.array([e[1] for e in entries], dtype=float)
        return cls(pos, dirs, max_force_n)

    @classmethod
    def corner_pod_16(cls, half_extents_m=(0.30, 0.30, 0.40), cant_deg=35.26, max_force_n=0.25):
        """
        Isometric 16-thruster corner-pod layout (Apollo/Dragon-derived).

        Four pods at body midplane corners (±hx, ±hy, 0). Each pod has four
        thrusters: ±X and ±Y lateral directions each canted cant_deg toward
        ±Z. At the isometric angle (35.26°): Z authority = 8 sin(θ) F_max,
        lateral = 4 cos(θ) F_max — balanced 6-DoF with no dedicated axial
        thrusters.
        """
        hx, hy, _ = map(float, half_extents_m)
        c = float(np.cos(np.radians(cant_deg)))
        s = float(np.sin(np.radians(cant_deg)))
        entries = []
        for sx in (+1, -1):
            for sy in (+1, -1):
                pos = [sx * hx, sy * hy, 0.0]
                for dx, dz in [(+c, +s), (+c, -s), (-c, +s), (-c, -s)]:
                    entries.append((pos, [dx * abs(sx), 0.0, dz]))
                for dy, dz in [(+c, +s), (+c, -s), (-c, +s), (-c, -s)]:
                    entries.append((pos, [0.0, dy * abs(sy), dz]))
        # 4 pods × 8 thrusters = 32 total; keep only the canonical 4 per pod
        # (±X cant ±Z, ±Y cant ±Z) → 4 × 4 = 16
        entries = []
        for sx in (+1, -1):
            for sy in (+1, -1):
                pos = [sx * hx, sy * hy, 0.0]
                entries.append((pos, [sx * c, 0.0,  s]))
                entries.append((pos, [sx * c, 0.0, -s]))
                entries.append((pos, [0.0, sy * c,  s]))
                entries.append((pos, [0.0, sy * c, -s]))
        pos  = np.array([e[0] for e in entries], dtype=float)
        dirs = np.array([e[1] for e in entries], dtype=float)
        return cls(pos, dirs, max_force_n)

    def chief_mask(self, chief_dir_body, cone_half_deg=45.0):
        """
        Boolean mask of thrusters whose exhaust plume points toward the chief.

        A thruster fires with force in direction d. Its exhaust plume goes in
        -d. We block thrusters where the plume direction (-d) falls within
        cone_half_deg of the chief direction c_hat:

            dot(-d, c_hat) >= cos(cone_half_deg)
            ⟺  dot(d, c_hat) <= -cos(cone_half_deg)

        Parameters
        ----------
        chief_dir_body : (3,) unit vector from deputy toward chief, in body frame
        cone_half_deg  : plume exclusion cone half-angle [deg]. Default 45°.

        Returns
        -------
        mask : (n_thrusters,) bool — True = thruster is blocked
        """
        c = np.asarray(chief_dir_body, dtype=float)
        norm = float(np.linalg.norm(c))
        if norm < 1e-9:
            return np.zeros(len(self.directions_body), dtype=bool)
        c_hat = c / norm
        cos_thresh = float(np.cos(np.radians(cone_half_deg)))
        # dot(d, c_hat) <= -cos_thresh  ↔  plume within cone of chief
        return (self.directions_body @ c_hat) <= -cos_thresh

    def allocate(self, force_body_n, torque_body_nm=None, excluded=None):
        """
        Map a requested wrench to non-negative thruster forces.

        Parameters
        ----------
        force_body_n   : (3,) desired force in body frame [N]
        torque_body_nm : (3,) desired torque in body frame [N·m], default zeros
        excluded       : (n_thrusters,) bool mask — True = thruster locked at 0
                         (used for chief plume-avoidance)
        """
        force_body_n = np.asarray(force_body_n, dtype=float)
        if torque_body_nm is None:
            torque_body_nm = np.zeros(3)
        torque_body_nm = np.asarray(torque_body_nm, dtype=float)
        desired = np.concatenate((force_body_n, torque_body_nm))

        free = np.ones(self._A.shape[1], dtype=bool)
        forces = np.zeros(self._A.shape[1])

        # Lock excluded thrusters at zero before solving
        if excluded is not None:
            exc = np.asarray(excluded, dtype=bool)
            free[exc] = False

        remaining = desired - self._A[:, ~free] @ forces[~free]

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
