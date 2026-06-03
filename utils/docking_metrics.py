"""
Pure geometry helpers for docking alignment and approach-cone checks.

Both main.py and monte_carlo.py use these; extracted here so they can be
unit-tested without importing the simulation loop.
"""

import numpy as np
from sim_config import (
    DEP_DOCK_AXIS_BODY, DOCK_ALIGN_MAX_DEG,
    DOCK_AXIS_BODY, DOCK_PORT_BODY,
    CHIEF_BODY_HALF_EXTENTS_M, DOCK_FACE_TOL_M,
    DOCK_PORT_APERTURE_M, DOCK_CONE_MIN_RANGE_M,
    DOCK_CONE_HALF_ANGLE_DEG,
)


def docking_alignment_metrics(R_dep_body_to_lvlh: np.ndarray,
                              port_axis_lvlh: np.ndarray) -> dict:
    dep_axis = R_dep_body_to_lvlh @ DEP_DOCK_AXIS_BODY
    dep_axis /= max(np.linalg.norm(dep_axis), 1e-12)
    desired_axis = -port_axis_lvlh
    desired_axis /= max(np.linalg.norm(desired_axis), 1e-12)
    align_deg = float(np.degrees(np.arccos(
        np.clip(np.dot(dep_axis, desired_axis), -1.0, 1.0))))
    return {
        "ok": bool(align_deg <= DOCK_ALIGN_MAX_DEG),
        "align_deg": align_deg,
    }


def docking_geometry_metrics(dep_lvlh: np.ndarray,
                             R_body_to_lvlh: np.ndarray) -> dict:
    """Finite chief body and port approach-cone checks in the chief body frame."""
    axis_body = DOCK_AXIS_BODY / max(np.linalg.norm(DOCK_AXIS_BODY), 1e-12)
    dep_body = R_body_to_lvlh.T @ dep_lvlh
    port_to_dep_body = dep_body - DOCK_PORT_BODY
    port_range = float(np.linalg.norm(port_to_dep_body))
    axial = float(np.dot(port_to_dep_body, axis_body))
    lateral_vec = port_to_dep_body - axial * axis_body
    lateral = float(np.linalg.norm(lateral_vec))

    inside_body = bool(np.all(np.abs(dep_body) < CHIEF_BODY_HALF_EXTENTS_M))
    on_dock_face = dep_body[2] > CHIEF_BODY_HALF_EXTENTS_M[2] - DOCK_FACE_TOL_M
    in_aperture = lateral <= DOCK_PORT_APERTURE_M
    body_clear = (not inside_body) or (on_dock_face and in_aperture)
    capture_core = port_range <= DOCK_CONE_MIN_RANGE_M and in_aperture

    if port_range <= 1e-9:
        cone_angle_deg = 0.0
    else:
        cos_ang = np.clip(axial / port_range, -1.0, 1.0)
        cone_angle_deg = float(np.degrees(np.arccos(cos_ang)))

    cone_ok = capture_core or (axial > 0.0
                               and cone_angle_deg <= DOCK_CONE_HALF_ANGLE_DEG)
    cone_error_deg = max(0.0, cone_angle_deg - DOCK_CONE_HALF_ANGLE_DEG)
    if capture_core:
        cone_error_deg = 0.0

    return {
        "ok": bool(body_clear and cone_ok and in_aperture),
        "body_clear": bool(body_clear),
        "cone_ok": bool(cone_ok),
        "in_aperture": bool(in_aperture),
        "capture_core": bool(capture_core),
        "inside_body": inside_body,
        "lateral_m": lateral,
        "axial_m": axial,
        "cone_angle_deg": cone_angle_deg,
        "cone_error_deg": cone_error_deg,
    }
