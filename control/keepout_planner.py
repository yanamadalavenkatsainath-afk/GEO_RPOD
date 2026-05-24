import numpy as np


class MovingKeepoutZone:
    """Spherical moving keep-out zone attached to the tumbling target body."""

    def __init__(self, center_body, radius_m, name="keepout"):
        self.center_body = np.asarray(center_body, dtype=float)
        self.radius_m = float(radius_m)
        self.name = str(name)

    def center_lvlh(self, R_body2lvlh):
        return R_body2lvlh @ self.center_body


class KeepoutAvoidancePlanner:
    """
    Lightweight receding-horizon safety layer.

    This is not a full MPC yet. It models rotating appendages as moving
    keep-out spheres and adds a bounded repulsive acceleration when the
    deputy enters a warning band. It is designed so a later MPC can replace
    compute() without changing the sim call site.
    """

    def __init__(self, zones=None, warning_margin_m=0.75,
                 gain=1.0e-4, accel_max=2.0e-4):
        self.zones = list(zones) if zones is not None else []
        self.warning_margin_m = float(warning_margin_m)
        self.gain = float(gain)
        self.accel_max = float(accel_max)

    @staticmethod
    def default_appendage_zones():
        return [
            MovingKeepoutZone([1.20, 0.0, 0.0], 0.45, "solar_panel_plus_x"),
            MovingKeepoutZone([-1.20, 0.0, 0.0], 0.45, "solar_panel_minus_x"),
            MovingKeepoutZone([0.0, 1.10, 0.0], 0.35, "antenna_plus_y"),
        ]

    def compute(self, dep_lvlh, R_body2lvlh):
        dep_lvlh = np.asarray(dep_lvlh, dtype=float)
        accel = np.zeros(3)
        min_clearance = np.inf
        active = []

        for zone in self.zones:
            center = zone.center_lvlh(R_body2lvlh)
            delta = dep_lvlh - center
            dist = float(np.linalg.norm(delta))
            clearance = dist - zone.radius_m
            min_clearance = min(min_clearance, clearance)
            trigger = zone.radius_m + self.warning_margin_m
            if dist < trigger:
                direction = delta / max(dist, 1e-9)
                strength = self.gain * (trigger - dist) / max(self.warning_margin_m, 1e-9)
                accel += strength * direction
                active.append(zone.name)

        mag = float(np.linalg.norm(accel))
        if mag > self.accel_max:
            accel *= self.accel_max / mag

        return {
            "accel": accel,
            "active": active,
            "min_clearance_m": float(min_clearance),
        }
