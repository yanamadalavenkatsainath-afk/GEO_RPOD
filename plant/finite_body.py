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
    Exact oriented-box overlap for the declared bus geometries.

    The sphere precheck avoids work at range. Near the buses, all 15
    separating axes are tested; corner containment alone misses crossed
    edges and complete containment. Appendages are not represented here.
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
        chief_axes = np.asarray(R_chief_to_world, dtype=float)
        deputy_axes = np.asarray(R_deputy_to_world, dtype=float)
        delta = deputy_com_world-chief_com_world
        axes = [*chief_axes.T, *deputy_axes.T]
        axes.extend(np.cross(a, b) for a in chief_axes.T for b in deputy_axes.T)
        collision = True
        for axis in axes:
            norm = float(np.linalg.norm(axis))
            if norm < 1e-10:
                continue  # Parallel edges do not define a separating axis.
            axis = axis/norm
            chief_radius = float(np.dot(self.chief.half_extents_m,
                                         np.abs(chief_axes.T@axis)))
            deputy_radius = float(np.dot(self.deputy.half_extents_m,
                                          np.abs(deputy_axes.T@axis)))
            # Mere surface contact is not positive-volume bus penetration.
            if abs(float(np.dot(delta, axis))) >= chief_radius+deputy_radius-1e-9:
                collision = False
                break
        return {
            "collision": collision,
            "sphere_clearance_m": sphere_clearance,
            "corner_inside_count": int(np.sum(inside)),
        }
