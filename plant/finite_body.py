import numpy as np


class BoxBody:
    """
    Oriented box geometry helper for coarse spacecraft collision checks.

    The body frame is centered at the spacecraft COM. `R_body_to_world` maps
    body-frame vectors into the caller's world frame, usually LVLH.
    """

    def __init__(self, half_extents_m, name="body"):
        self.half_extents_m = np.asarray(half_extents_m, dtype=float)
        self.name = str(name)

    @property
    def bounding_radius_m(self):
        return float(np.linalg.norm(self.half_extents_m))

    def corners_body(self):
        hx, hy, hz = self.half_extents_m
        return np.array([
            [sx * hx, sy * hy, sz * hz]
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ], dtype=float)

    def points_world(self, com_world, R_body_to_world):
        com_world = np.asarray(com_world, dtype=float)
        R_body_to_world = np.asarray(R_body_to_world, dtype=float)
        return com_world + (R_body_to_world @ self.corners_body().T).T

    def contains_world_point(self, point_world, com_world, R_body_to_world,
                             margin_m=0.0):
        point_world = np.asarray(point_world, dtype=float)
        com_world = np.asarray(com_world, dtype=float)
        R_body_to_world = np.asarray(R_body_to_world, dtype=float)
        point_body = R_body_to_world.T @ (point_world - com_world)
        limits = self.half_extents_m + float(margin_m)
        return bool(np.all(np.abs(point_body) <= limits))


class FiniteBodyPair:
    """
    Coarse pairwise finite-body geometry between chief and deputy.

    This is intentionally conservative: it checks deputy box corners against
    the chief box and uses a bounding-radius precheck to keep MC inexpensive.
    """

    def __init__(self, chief_body, deputy_body):
        self.chief = chief_body
        self.deputy = deputy_body

    def clearance(self, chief_com_world, R_chief_to_world,
                  deputy_com_world, R_deputy_to_world):
        chief_com_world = np.asarray(chief_com_world, dtype=float)
        deputy_com_world = np.asarray(deputy_com_world, dtype=float)
        center_dist = float(np.linalg.norm(deputy_com_world - chief_com_world))
        sphere_clearance = (center_dist - self.chief.bounding_radius_m
                            - self.deputy.bounding_radius_m)

        if sphere_clearance > 0.5:
            return {
                "collision": False,
                "sphere_clearance_m": sphere_clearance,
                "corner_inside_count": 0,
            }

        dep_corners = self.deputy.points_world(deputy_com_world, R_deputy_to_world)
        inside = [
            self.chief.contains_world_point(p, chief_com_world, R_chief_to_world)
            for p in dep_corners
        ]
        return {
            "collision": bool(any(inside)),
            "sphere_clearance_m": sphere_clearance,
            "corner_inside_count": int(np.sum(inside)),
        }
