"""Rate/acceleration/angle-limited azimuth/elevation pointing mechanism."""
from dataclasses import dataclass

import numpy as np

from utils.validation import finite_scalar, rotation_matrix, vector


@dataclass(frozen=True)
class GimbalState:
    angles_rad: np.ndarray
    rates_rad_s: np.ndarray
    boresight_body: np.ndarray
    limited: bool


class RateLimitedGimbal:
    """Base +X boresight, azimuth about +Z and elevation toward +Z.

    Position servo dynamics only. Motor bus reaction, backlash, flexible modes,
    power/thermal and joint-axis inertia require a higher-fidelity model.
    """
    def __init__(self, *, minimum_angles_rad, maximum_angles_rad, maximum_rates_rad_s,
                 maximum_accelerations_rad_s2, base_to_body=None):
        self.lower = vector(minimum_angles_rad, "minimum_angles_rad", size=2)
        self.upper = vector(maximum_angles_rad, "maximum_angles_rad", size=2)
        self.rate_limit = vector(maximum_rates_rad_s, "maximum_rates_rad_s", size=2)
        self.acceleration_limit = vector(maximum_accelerations_rad_s2, "maximum_accelerations_rad_s2", size=2)
        if np.any(self.lower >= self.upper) or np.any(self.rate_limit <= 0) or np.any(self.acceleration_limit <= 0):
            raise ValueError("Ordered angle bounds and positive rate/acceleration limits required")
        self.mount = rotation_matrix(np.eye(3) if base_to_body is None else base_to_body)
        self.reset()

    def reset(self, angles_rad=(0., 0.)):
        angles = vector(angles_rad, "angles_rad", size=2)
        if np.any(angles < self.lower) or np.any(angles > self.upper):
            raise ValueError("Initial gimbal angles exceed bounds")
        self.angles, self.rates = angles, np.zeros(2)

    def step(self, requested_angles_rad, duration_s):
        requested = vector(requested_angles_rad, "requested_angles_rad", size=2)
        dt = finite_scalar(duration_s, "duration_s", strictly_positive=True)
        target = np.clip(requested, self.lower, self.upper)
        steps = max(1, int(np.ceil(dt/.01)))
        h = dt/steps
        for _ in range(steps):
            error = target-self.angles
            desired = np.sign(error)*np.minimum(self.rate_limit, np.sqrt(2*self.acceleration_limit*np.abs(error)))
            rate = self.rates+np.clip(desired-self.rates, -self.acceleration_limit*h, self.acceleration_limit*h)
            angles = self.angles+(rate+self.rates)*h/2
            stopped = (angles < self.lower) | (angles > self.upper)
            self.angles = np.clip(angles, self.lower, self.upper)
            self.rates = np.where(stopped, 0., rate)
        azimuth, elevation = self.angles
        boresight = self.mount @ [np.cos(elevation)*np.cos(azimuth), np.cos(elevation)*np.sin(azimuth), np.sin(elevation)]
        return GimbalState(self.angles.copy(), self.rates.copy(), boresight,
                           bool(np.any(requested != target)))
