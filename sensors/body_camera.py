import numpy as np


class BodyMountedCamera:
    """
    Simple body-mounted camera visibility model.

    The existing camera sensor produces the measurement. This class answers
    whether a body-mounted camera could see the target from the current deputy
    attitude, using boresight, field of view, and optional line-of-sight range.
    """

    def __init__(self, boresight_body=(0.0, 0.0, 1.0),
                 fov_half_angle_deg=35.0, max_range_m=5000.0):
        self.boresight_body = np.asarray(boresight_body, dtype=float)
        self.boresight_body /= max(np.linalg.norm(self.boresight_body), 1e-12)
        self.fov_half_angle_deg = float(fov_half_angle_deg)
        self.max_range_m = float(max_range_m)

    def visibility(self, target_from_deputy_world, R_body_to_world):
        target_from_deputy_world = np.asarray(target_from_deputy_world, dtype=float)
        rng = float(np.linalg.norm(target_from_deputy_world))
        if rng <= 1e-9 or rng > self.max_range_m:
            return {
                "visible": False,
                "angle_deg": np.inf,
                "range_m": rng,
            }

        los = target_from_deputy_world / rng
        boresight_world = np.asarray(R_body_to_world, dtype=float) @ self.boresight_body
        boresight_world /= max(np.linalg.norm(boresight_world), 1e-12)
        angle_deg = float(np.degrees(np.arccos(
            np.clip(np.dot(los, boresight_world), -1.0, 1.0))))
        return {
            "visible": bool(angle_deg <= self.fov_half_angle_deg),
            "angle_deg": angle_deg,
            "range_m": rng,
        }
