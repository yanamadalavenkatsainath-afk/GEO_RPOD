import numpy as np


class GimbaledTrackingCamera:
    """
    2-axis gimbaled camera that actively tracks a target.

    The gimbal is centred on `center_body` (a body-frame axis, normally the
    docking face +z).  Within `gimbal_range_deg` of that centre the camera
    always slews to point directly at the target — no pointing loss inside
    the envelope.  Outside the envelope the target is physically occluded by
    the spacecraft body.

    This is a drop-in replacement for BodyMountedCamera: same visibility()
    signature, same returned keys (angle_deg renamed to gimbal_angle_deg but
    angle_deg is also kept for compatibility).
    """

    def __init__(self, center_body=(0., 0., 1.),
                 gimbal_range_deg=110.0,
                 max_range_m=5000.0):
        self.center_body = np.asarray(center_body, dtype=float)
        self.center_body /= max(np.linalg.norm(self.center_body), 1e-12)
        self.gimbal_range_deg = float(gimbal_range_deg)
        self.max_range_m = float(max_range_m)

    def visibility(self, target_from_deputy_world, R_body_to_world):
        target_from_deputy_world = np.asarray(target_from_deputy_world, dtype=float)
        rng = float(np.linalg.norm(target_from_deputy_world))
        if rng <= 1e-9 or rng > self.max_range_m:
            return {"visible": False, "angle_deg": np.inf,
                    "gimbal_angle_deg": np.inf, "range_m": rng}

        los = target_from_deputy_world / rng
        center_world = np.asarray(R_body_to_world, dtype=float) @ self.center_body
        center_world /= max(np.linalg.norm(center_world), 1e-12)
        gimbal_deg = float(np.degrees(np.arccos(
            np.clip(np.dot(los, center_world), -1.0, 1.0))))
        visible = bool(gimbal_deg <= self.gimbal_range_deg)
        return {"visible": visible, "angle_deg": gimbal_deg,
                "gimbal_angle_deg": gimbal_deg, "range_m": rng}


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
