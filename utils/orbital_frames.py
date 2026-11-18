"""Explicit velocity conventions at inertial/rotating LVLH boundaries.

Let rho = R (r_dep-r_chief), where R maps ECI to radial/along-track/normal.
Differentiation gives rho_dot = R (v_dep-v_chief) - Omega_L x rho.
The CW/TH estimator and terminal position filter use [rho, rho_dot].
Plant contact velocities use the physical vector R (v_dep-v_chief).
An instantaneous commanded delta-v increments either convention equally.
"""

import numpy as np
from utils.validation import covariance, finite_array, rotation_matrix, vector


def _chief_vectors(chief_pos, chief_vel):
    position = vector(chief_pos, "chief_pos", size=3)
    velocity = vector(chief_vel, "chief_vel", size=3)
    radius = float(np.linalg.norm(position))
    angular_momentum = np.cross(position, velocity)
    h = float(np.linalg.norm(angular_momentum))
    if not np.isfinite(radius) or radius <= 0 or not np.isfinite(h) or h <= 0:
        raise ValueError("LVLH requires nonzero position and nonradial finite motion")
    return position, velocity, radius, angular_momentum, h


def skew(value):
    """Matrix S(a) for which S(a) @ b equals cross(a, b)."""
    x, y, z = vector(value, "skew vector", size=3)
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def eci_to_lvlh_rotation(chief_pos, chief_vel):
    position, _, radius, momentum, h = _chief_vectors(chief_pos, chief_vel)
    radial = position / radius
    normal = momentum / h
    return np.vstack([radial, np.cross(normal, radial), normal])


def lvlh_frame_rate(chief_pos, chief_vel, chief_accel=None):
    """Frame angular velocity expressed in LVLH, radians/second.

    h/r² is the osculating plane's in-plane rotation. Supplying acceleration
    also includes the plane precession caused by a normal acceleration.
    The default matches the planar CW/TH reference used by the GNC model.
    """
    _, _, radius, momentum, h = _chief_vectors(chief_pos, chief_vel)
    if chief_accel is not None:
        chief_accel = vector(chief_accel, "chief_accel", size=3)
    radial_rate = (0.0 if chief_accel is None else
                   radius * float(np.dot(chief_accel, momentum / h)) / h)
    return np.array([radial_rate, 0.0, h / radius**2])


def projected_to_rotating_velocity(projected_velocity, rho, frame_rate):
    return (vector(projected_velocity, "projected_velocity", size=3)
            - np.cross(vector(frame_rate, "frame_rate", size=3), vector(rho, "rho", size=3)))


def rotating_to_projected_velocity(rho_dot, rho, frame_rate):
    return (vector(rho_dot, "rho_dot", size=3)
            + np.cross(vector(frame_rate, "frame_rate", size=3), vector(rho, "rho", size=3)))


def relative_state_from_eci(chief_pos, chief_vel, deputy_pos, deputy_vel,
                             R_eci_to_lvlh=None, chief_accel=None):
    R = (eci_to_lvlh_rotation(chief_pos, chief_vel) if R_eci_to_lvlh is None
         else rotation_matrix(R_eci_to_lvlh))
    chief_pos = vector(chief_pos, "chief_pos", size=3)
    chief_vel = vector(chief_vel, "chief_vel", size=3)
    deputy_pos = vector(deputy_pos, "deputy_pos", size=3)
    deputy_vel = vector(deputy_vel, "deputy_vel", size=3)
    rho = R @ (deputy_pos - chief_pos)
    rho_dot = projected_to_rotating_velocity(
        R @ (deputy_vel - chief_vel), rho,
        lvlh_frame_rate(chief_pos, chief_vel, chief_accel))
    return np.concatenate([rho, rho_dot])


def relative_state_to_eci(chief_pos, chief_vel, relative_state,
                           R_eci_to_lvlh=None, chief_accel=None):
    R = (eci_to_lvlh_rotation(chief_pos, chief_vel) if R_eci_to_lvlh is None
         else rotation_matrix(R_eci_to_lvlh))
    chief_pos = vector(chief_pos, "chief_pos", size=3)
    chief_vel = vector(chief_vel, "chief_vel", size=3)
    relative_state = vector(relative_state, "relative_state", size=6)
    rho, rho_dot = relative_state[:3], relative_state[3:6]
    velocity = rotating_to_projected_velocity(
        rho_dot, rho, lvlh_frame_rate(chief_pos, chief_vel, chief_accel))
    return chief_pos + R.T @ rho, chief_vel + R.T @ velocity


def relative_state_jacobian(chief_pos, chief_vel, chief_accel=None):
    """Deputy ECI [r,v] -> rotating LVLH [rho,rho_dot], chief held fixed.

    This does not include uncertainty in the chief state. Joint chief/deputy
    covariance requires the full joint-state linearization instead.
    """
    rotation = eci_to_lvlh_rotation(chief_pos, chief_vel)
    rate = lvlh_frame_rate(chief_pos, chief_vel, chief_accel)
    return np.block([[rotation, np.zeros((3, 3))],
                     [-skew(rate) @ rotation, rotation]])


def transform_covariance(matrix, jacobian):
    """P_out = J P_in J.T for explicitly supplied coordinate Jacobian J."""
    jacobian = finite_array(jacobian, "jacobian")
    if jacobian.ndim != 2 or not all(jacobian.shape):
        raise ValueError("jacobian must be a nonempty matrix")
    matrix = covariance(matrix, jacobian.shape[1])
    return covariance(jacobian @ matrix @ jacobian.T, jacobian.shape[0])
