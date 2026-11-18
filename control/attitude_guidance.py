"""Body-to-reference pointing, smooth slew references and physical bus PD."""
from dataclasses import dataclass

import numpy as np

from spec import Timestamp, WrenchCommand
from utils.quaternion import (normalize, quat_conjugate, quat_from_matrix,
                              quat_from_rotvec, quat_multiply, rot_matrix,
                              rotvec_from_quat)
from utils.validation import finite_scalar, text_label, vector


@dataclass(frozen=True)
class AttitudeReference:
    time: Timestamp
    frame: str
    body_frame: str
    quaternion: np.ndarray
    omega_body_rad_s: np.ndarray
    alpha_body_rad_s2: np.ndarray

    def __post_init__(self):
        if not isinstance(self.time, Timestamp):
            raise ValueError("time must be a Timestamp")
        text_label(self.frame, "frame")
        text_label(self.body_frame, "body_frame")
        q = normalize(vector(self.quaternion, "quaternion", size=4))
        q.setflags(write=False)
        object.__setattr__(self, "quaternion", q)
        for name in ("omega_body_rad_s", "alpha_body_rad_s2"):
            object.__setattr__(self, name, vector(getattr(self, name), name, size=3, readonly=True))


def pointing_quaternion(direction_reference, up_reference, *, boresight_body=(0., 0., 1.), up_body=(0., 1., 0.)):
    """Align boresight and projected roll reference; reject singular roll geometry.

    For nadir use estimated -position and orbit normal as roll reference.
    For target tracking use estimated target-observer position. No occultation,
    Sun exclusion, slew feasibility or target visibility is implied.
    """
    def basis(primary, secondary):
        first = normalize(vector(primary, "primary", size=3))
        up = normalize(vector(secondary, "up", size=3))
        second = up-first*(up @ first)
        if np.linalg.norm(second) < 1e-8:
            raise ValueError("Boresight and roll reference are collinear")
        second = normalize(second)
        return np.column_stack((first, second, np.cross(first, second)))
    return quat_from_matrix(basis(direction_reference, up_reference) @ basis(boresight_body, up_body).T)


class SlewReference:
    """Shortest fixed-axis quintic slew, bounded reference rate/acceleration.

    Reference bounds do not guarantee achieved pointing, wheel momentum,
    actuator authority, plume safety or exclusion-cone avoidance. These require
    trajectory/plant checks by the scenario wrapper.
    """
    def __init__(self, *, start_time, start_quaternion, end_quaternion,
                 max_rate_rad_s, max_accel_rad_s2, frame="earth/gcrs", body_frame="body/chaser"):
        if not isinstance(start_time, Timestamp):
            raise ValueError("start_time must be a Timestamp")
        self.time = start_time
        self.start = normalize(vector(start_quaternion, "start_quaternion", size=4))
        end = normalize(vector(end_quaternion, "end_quaternion", size=4))
        self.rotation = rotvec_from_quat(quat_multiply(quat_conjugate(self.start), end))
        angle = np.linalg.norm(self.rotation)
        rate = finite_scalar(max_rate_rad_s, "max_rate_rad_s", strictly_positive=True)
        accel = finite_scalar(max_accel_rad_s2, "max_accel_rad_s2", strictly_positive=True)
        self.duration_s = max(1.875*angle/rate, np.sqrt((10*np.sqrt(3)/3)*angle/accel))
        self.frame, self.body_frame = text_label(frame, "frame"), text_label(body_frame, "body_frame")

    def at(self, time):
        elapsed = time.seconds_since(self.time)
        if self.duration_s == 0:
            u, rate, accel = 0., 0., 0.
        else:
            u = np.clip(elapsed/self.duration_s, 0, 1)
            rate = (30*u**2-60*u**3+30*u**4)/self.duration_s
            accel = (60*u-180*u**2+120*u**3)/self.duration_s**2
        fraction = 10*u**3-15*u**4+6*u**5
        q = quat_multiply(self.start, quat_from_rotvec(self.rotation*fraction))
        return AttitudeReference(time, self.frame, self.body_frame, q,
                                 self.rotation*rate, self.rotation*accel)


class QuaternionPD:
    """Requested physical torque ON the bus; wheel motor torque is opposite.

    Gains are diagonal, nonnegative SI torque-per-radian and torque-per-rate.
    No implicit truth feedforward, momentum dumping or actuator allocation.
    """
    def __init__(self, *, kp_nm_rad, kd_nm_s_rad, torque_limit_nm,
                 frame="earth/gcrs", body_frame="body/chaser"):
        self.kp = vector(kp_nm_rad, "kp_nm_rad", size=3)
        self.kd = vector(kd_nm_s_rad, "kd_nm_s_rad", size=3)
        self.limit = vector(torque_limit_nm, "torque_limit_nm", size=3)
        if np.any(self.kp < 0) or np.any(self.kd < 0) or np.any(self.limit <= 0):
            raise ValueError("Nonnegative gains and positive torque limits are required")
        self.frame, self.body_frame = text_label(frame, "frame"), text_label(body_frame, "body_frame")

    def command(self, *, time, quaternion_estimate, omega_estimate_body_rad_s, reference):
        if not isinstance(reference, AttitudeReference) or reference.time != time or reference.frame != self.frame or reference.body_frame != self.body_frame:
            raise ValueError("Current attitude reference with matching frames is required")
        q = normalize(vector(quaternion_estimate, "quaternion_estimate", size=4))
        omega = vector(omega_estimate_body_rad_s, "omega_estimate_body_rad_s", size=3)
        error = rotvec_from_quat(quat_multiply(quat_conjugate(q), reference.quaternion))
        target_rate = rot_matrix(q).T @ rot_matrix(reference.quaternion) @ reference.omega_body_rad_s
        torque = self.kp*error-self.kd*(omega-target_rate)
        return WrenchCommand("quaternion_pd", time, self.body_frame,
                             torque_nm=np.clip(torque, -self.limit, self.limit))
