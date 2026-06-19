import numpy as np


class UncooperativePoseMeasurement:
    def __init__(self, r_body2lvlh, t_lvlh, R_pose, quality, valid, reason="OK"):
        self.R_body2lvlh = r_body2lvlh
        self.t_lvlh = t_lvlh
        self.R_pose = R_pose
        self.quality = float(quality)
        self.valid = bool(valid)
        self.reason = reason


class UncooperativePoseSensor:
    """
    Surrogate uncooperative-target pose sensor.

    This is a replaceable interface for a future FoundPose/SpeedPlus-style
    neural pose estimator. It does not assume fiducials or a cooperative
    docking target. It outputs a noisy 6DOF pose estimate plus a quality score
    derived from range, apparent size, lighting, and model-feature visibility.
    """

    def __init__(self,
                 sigma_pos_m=0.20,
                 sigma_att_deg=30.0,
                 min_quality=0.25,
                 max_range_m=250.0,
                 rng=None):
        self.sigma_pos_m = float(sigma_pos_m)
        self.sigma_att_rad = np.radians(float(sigma_att_deg))
        self.min_quality = float(min_quality)
        self.max_range_m = float(max_range_m)
        self.rng = rng

    def measure(self, dr_lvlh, R_body2lvlh_true, sun_lvlh=None,
                body_points=None):
        rng = self.rng if self.rng is not None else np.random
        dr_lvlh = np.asarray(dr_lvlh, dtype=float)
        R_body2lvlh_true = np.asarray(R_body2lvlh_true, dtype=float)
        r = float(np.linalg.norm(dr_lvlh))
        if r < 0.05 or r > self.max_range_m:
            return UncooperativePoseMeasurement(
                None, None, None, 0.0, False, reason="RANGE")

        los = dr_lvlh / max(r, 1e-12)
        feature_score = self._feature_score(los, R_body2lvlh_true, body_points)
        lighting_score = self._lighting_score(los, R_body2lvlh_true, sun_lvlh)
        range_score = np.clip((self.max_range_m - r) / self.max_range_m, 0.0, 1.0)
        quality = float(np.clip(0.25 + 0.35 * feature_score
                                + 0.25 * lighting_score
                                + 0.15 * range_score, 0.0, 1.0))

        if quality < self.min_quality:
            return UncooperativePoseMeasurement(
                None, None, None, quality, False, reason="LOW_QUALITY")

        sigma_pos = self.sigma_pos_m / max(quality, 0.1)
        sigma_att = self.sigma_att_rad / max(quality, 0.1)

        t_meas = dr_lvlh + rng.normal(0.0, sigma_pos, 3)
        R_meas = self._small_angle_perturb(R_body2lvlh_true,
                                           rng.normal(0.0, sigma_att, 3))
        R_pose = np.diag([sigma_pos ** 2] * 3 + [sigma_att ** 2] * 3)
        return UncooperativePoseMeasurement(
            R_meas, t_meas, R_pose, quality, True, reason="OK")

    @staticmethod
    def _small_angle_perturb(R, dtheta):
        x, y, z = dtheta
        K = np.array([[0.0, -z, y],
                      [z, 0.0, -x],
                      [-y, x, 0.0]])
        return (np.eye(3) + K) @ R

    @staticmethod
    def _feature_score(los_lvlh, R_body2lvlh, body_points):
        if body_points is None or len(body_points) < 4:
            return 0.5
        pts = np.asarray(body_points, dtype=float)
        pts_lvlh = (R_body2lvlh @ pts.T).T
        depths = pts_lvlh @ los_lvlh
        visible = depths > np.percentile(depths, 35)
        spread = np.linalg.norm(np.std(pts_lvlh[visible], axis=0)) if np.any(visible) else 0.0
        return float(np.clip(spread / 0.5, 0.0, 1.0))

    @staticmethod
    def _lighting_score(los_lvlh, R_body2lvlh, sun_lvlh):
        if sun_lvlh is None or np.linalg.norm(sun_lvlh) < 1e-9:
            return 0.7
        sun = sun_lvlh / np.linalg.norm(sun_lvlh)
        phase = float(np.dot(-los_lvlh, sun))
        return float(np.clip(0.5 + 0.5 * phase, 0.0, 1.0))
