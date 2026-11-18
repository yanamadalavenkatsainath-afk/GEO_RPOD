"""Rigid impulse/contact and attachment bookkeeping, not capture certification."""
from dataclasses import dataclass

import numpy as np

from plant.coupled import MassElement, combine_mass_properties, inertia_tensor
from utils.orbital_frames import skew
from utils.quaternion import normalize, quat_from_matrix, rot_matrix
from utils.validation import finite_scalar, vector
from utils.validation import text_label
from spec import Timestamp


@dataclass(frozen=True)
class RigidBodyState:
    mass_kg: float
    inertia_body_kg_m2: np.ndarray
    position_m: np.ndarray
    velocity_m_s: np.ndarray
    quaternion: np.ndarray
    omega_body_rad_s: np.ndarray
    time: Timestamp
    reference_frame: str

    def __post_init__(self):
        if not isinstance(self.time, Timestamp):
            raise ValueError("Rigid body state requires explicit Timestamp")
        text_label(self.reference_frame, "reference_frame")
        object.__setattr__(self, "mass_kg", finite_scalar(self.mass_kg, "mass_kg", strictly_positive=True))
        inertia = inertia_tensor(self.inertia_body_kg_m2)
        inertia.setflags(write=False)
        object.__setattr__(self, "inertia_body_kg_m2", inertia)
        for name in ("position_m", "velocity_m_s", "omega_body_rad_s"):
            object.__setattr__(self, name, vector(getattr(self, name), name, size=3, readonly=True))
        q = normalize(vector(self.quaternion, "quaternion", size=4))
        q.setflags(write=False)
        object.__setattr__(self, "quaternion", q)


def contact_impulse(first, second, *, contact_position_m, normal_first_to_second, restitution=0., friction=0.):
    """Single contact-point rigid impact; friction uses a projected impulse cap.

    Caller detects actual geometry/contact and supplies a normal. This does not
    resolve penetration, multiple simultaneous contacts or compliant deformation.
    Friction/normal coupling uses a low-order sequential impulse approximation.
    """
    if not isinstance(first, RigidBodyState) or not isinstance(second, RigidBodyState):
        raise ValueError("Two RigidBodyStates required")
    if first.time != second.time or first.reference_frame != second.reference_frame:
        raise ValueError("Contact states must share time and reference frame")
    point = vector(contact_position_m, "contact_position_m", size=3)
    normal = normalize(vector(normal_first_to_second, "normal_first_to_second", size=3))
    restitution = finite_scalar(restitution, "restitution", minimum=0)
    friction = finite_scalar(friction, "friction", minimum=0)
    if restitution > 1:
        raise ValueError("Restitution cannot exceed one")
    ra, rb = point-first.position_m, point-second.position_m
    a, b = rot_matrix(first.quaternion), rot_matrix(second.quaternion)
    ia, ib = a @ first.inertia_body_kg_m2 @ a.T, b @ second.inertia_body_kg_m2 @ b.T
    wa, wb = a @ first.omega_body_rad_s, b @ second.omega_body_rad_s
    relative = second.velocity_m_s+np.cross(wb, rb)-first.velocity_m_s-np.cross(wa, ra)
    closing = relative @ normal
    if closing >= 0:
        return first, second, np.zeros(3)
    effective = ((1/first.mass_kg+1/second.mass_kg)*np.eye(3)
        -skew(ra) @ np.linalg.solve(ia, skew(ra))-skew(rb) @ np.linalg.solve(ib, skew(rb)))
    magnitude = -(1+restitution)*closing/(normal @ effective @ normal)
    impulse = normal*magnitude
    if friction:
        tangent = relative-normal*closing
        if np.linalg.norm(tangent) > 1e-12:
            tangent /= np.linalg.norm(tangent)
            amount = -(tangent @ (relative+effective @ impulse))/(tangent @ effective @ tangent)
            impulse += tangent*np.clip(amount, -friction*magnitude, friction*magnitude)
    va = first.velocity_m_s-impulse/first.mass_kg
    vb = second.velocity_m_s+impulse/second.mass_kg
    wa -= np.linalg.solve(ia, np.cross(ra, impulse))
    wb += np.linalg.solve(ib, np.cross(rb, impulse))
    return (RigidBodyState(first.mass_kg, first.inertia_body_kg_m2, first.position_m, va, first.quaternion, a.T @ wa, first.time, first.reference_frame),
            RigidBodyState(second.mass_kg, second.inertia_body_kg_m2, second.position_m, vb, second.quaternion, b.T @ wb, second.time, second.reference_frame), impulse)


@dataclass(frozen=True)
class RigidAttachment:
    stack: RigidBodyState
    components: tuple[MassElement, ...]
    component_rotations: tuple[np.ndarray, ...]


def attach_rigidly(bodies):
    """Perfectly inelastic rigid attachment preserving total linear/angular momentum.

    An external capture/load policy must authorize attachment first. Dissipated
    kinetic energy is not a calculation of peak contact force or structural safety.
    """
    bodies = tuple(bodies)
    if not bodies or any(not isinstance(b, RigidBodyState) for b in bodies):
        raise ValueError("Rigid body states required")
    if any(b.time != bodies[0].time or b.reference_frame != bodies[0].reference_frame for b in bodies):
        raise ValueError("Attachment states must share time and reference frame")
    axis = rot_matrix(bodies[0].quaternion)
    total_mass = sum(b.mass_kg for b in bodies)
    position = sum(b.mass_kg*b.position_m for b in bodies)/total_mass
    velocity = sum(b.mass_kg*b.velocity_m_s for b in bodies)/total_mass
    elements, rotations, momentum = [], [], np.zeros(3)
    for body in bodies:
        rotation = rot_matrix(body.quaternion)
        relative = axis.T @ rotation
        offset = body.position_m-position
        elements.append(MassElement(body.mass_kg, axis.T @ offset, relative @ body.inertia_body_kg_m2 @ relative.T))
        rotations.append(relative)
        momentum += rotation @ body.inertia_body_kg_m2 @ body.omega_body_rad_s+np.cross(offset, body.mass_kg*(body.velocity_m_s-velocity))
    aggregate = combine_mass_properties(elements)
    omega = np.linalg.solve(aggregate.inertia_at_com_kg_m2, axis.T @ momentum)
    stack = RigidBodyState(total_mass, aggregate.inertia_at_com_kg_m2, position, velocity, bodies[0].quaternion, omega, bodies[0].time, bodies[0].reference_frame)
    return RigidAttachment(stack, tuple(elements), tuple(rotations))


def release_rigidly(attachment, *, stack_state=None):
    """Passive release from current rigid-stack state, without separation impulse."""
    if not isinstance(attachment, RigidAttachment):
        raise ValueError("Expected RigidAttachment")
    stack = attachment.stack if stack_state is None else stack_state
    if not isinstance(stack, RigidBodyState) or not np.isclose(stack.mass_kg, attachment.stack.mass_kg) or stack.reference_frame != attachment.stack.reference_frame:
        raise ValueError("Compatible current stack state required")
    rotation = rot_matrix(stack.quaternion)
    omega = rotation @ stack.omega_body_rad_s
    result = []
    for element, local_rotation in zip(attachment.components, attachment.component_rotations):
        offset = rotation @ element.com_m
        body_rotation = rotation @ local_rotation
        inertia = local_rotation.T @ element.inertia_at_com_kg_m2 @ local_rotation
        result.append(RigidBodyState(element.mass_kg, inertia, stack.position_m+offset,
            stack.velocity_m_s+np.cross(omega, offset), quat_from_matrix(body_rotation), body_rotation.T @ omega, stack.time, stack.reference_frame))
    return tuple(result)


def compliant_normal_force(penetration_m, closing_speed_m_s, *, stiffness_n_m, damping_n_s_m):
    """Unilateral Kelvin-Voigt normal force; integration stability is caller-owned."""
    depth = finite_scalar(penetration_m, "penetration_m")
    closing = finite_scalar(closing_speed_m_s, "closing_speed_m_s")
    stiffness = finite_scalar(stiffness_n_m, "stiffness_n_m", minimum=0)
    damping = finite_scalar(damping_n_s_m, "damping_n_s_m", minimum=0)
    return max(0., stiffness*depth+damping*closing) if depth > 0 else 0.
