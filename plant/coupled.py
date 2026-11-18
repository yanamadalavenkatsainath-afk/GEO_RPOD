"""Coupled Cartesian and rigid-body plant using achieved physical wrench.

Baseline: concentric propellant depletion; no slosh, flex, moving COM, or contact
solver. Torque and body origin are at the current fixed centre of mass.
"""
from dataclasses import dataclass

import numpy as np

from spec import ActuationResult, Timestamp
from utils.quaternion import normalize, quat_multiply, rot_matrix
from utils.validation import covariance, finite_scalar, rotation_matrix, vector
from utils.sampling import SimulationClock


def inertia_tensor(value):
    inertia = covariance(value, 3, "inertia_kg_m2")
    moments = np.linalg.eigvalsh(inertia)
    if moments[0] <= 0 or moments[2] > moments[0]+moments[1]+1e-12*max(moments):
        raise ValueError("Inertia must be positive definite with physical principal moments")
    return inertia


@dataclass(frozen=True)
class MassElement:
    """Rigid mass, local COM position, and inertia expressed in stack axes."""
    mass_kg: float
    com_m: np.ndarray
    inertia_at_com_kg_m2: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "mass_kg", finite_scalar(self.mass_kg, "mass_kg", strictly_positive=True))
        object.__setattr__(self, "com_m", vector(self.com_m, "com_m", size=3, readonly=True))
        inertia = inertia_tensor(self.inertia_at_com_kg_m2)
        inertia.setflags(write=False)
        object.__setattr__(self, "inertia_at_com_kg_m2", inertia)

    def transformed(self, rotation, translation_m):
        rotation = rotation_matrix(rotation)
        offset = vector(translation_m, "translation_m", size=3)
        return MassElement(self.mass_kg, rotation @ self.com_m+offset,
                           rotation @ self.inertia_at_com_kg_m2 @ rotation.T)


def combine_mass_properties(elements):
    """Parallel-axis aggregate of already attached, rigidly aligned components.

    Does not model a contact, determine latch success, or reconcile pre-contact
    velocities. Those decisions must precede selecting this aggregate plant.
    """
    elements = tuple(elements)
    if not elements or any(not isinstance(e, MassElement) for e in elements):
        raise ValueError("At least one MassElement is required")
    mass = sum(e.mass_kg for e in elements)
    centre = sum(e.mass_kg*e.com_m for e in elements)/mass
    inertia = np.zeros((3, 3))
    for element in elements:
        delta = element.com_m-centre
        inertia += element.inertia_at_com_kg_m2+element.mass_kg*(np.dot(delta, delta)*np.eye(3)-np.outer(delta, delta))
    return MassElement(mass, centre, inertia)


class CoupledSpacecraft:
    """SI RK4 plant with environment acceleration injected explicitly.

    The environment callback receives (Timestamp, r_reference, v_reference)
    and returns reference acceleration. Each ActuationResult is the interval's
    mean body force/torque on the bus, about the fixed COM. Mean torque alone
    omits gyroscopic coupling from moving internal wheel momentum; optionally
    supply midpoint internal momentum and evolve it in the actuator layer.
    """
    def __init__(self, *, time, position_m, velocity_m_s, quaternion,
                 omega_body_rad_s, dry_mass_kg, propellant_kg, dry_inertia_kg_m2,
                 fuel_inertia_per_kg=None, body_frame="body/chaser", max_step_s=.1):
        from utils.validation import text_label
        if not isinstance(time, Timestamp):
            raise ValueError("time must be a Timestamp")
        self.time = time
        self._clock = SimulationClock(time)
        self.r = vector(position_m, "position_m", size=3)
        self.v = vector(velocity_m_s, "velocity_m_s", size=3)
        self.q = normalize(vector(quaternion, "quaternion", size=4))
        self.omega = vector(omega_body_rad_s, "omega_body_rad_s", size=3)
        self.dry_mass = finite_scalar(dry_mass_kg, "dry_mass_kg", strictly_positive=True)
        self.fuel = finite_scalar(propellant_kg, "propellant_kg", minimum=0)
        self.dry_inertia = inertia_tensor(dry_inertia_kg_m2)
        self.fuel_inertia = (np.zeros((3, 3)) if fuel_inertia_per_kg is None
                             else covariance(fuel_inertia_per_kg, 3, "fuel_inertia_per_kg"))
        inertia_tensor(self.dry_inertia+self.fuel*self.fuel_inertia)
        self.body_frame = text_label(body_frame, "body_frame")
        self.max_step = finite_scalar(max_step_s, "max_step_s", strictly_positive=True)

    @property
    def mass_kg(self):
        return self.dry_mass+self.fuel

    @property
    def inertia_kg_m2(self):
        return self.dry_inertia+self.fuel*self.fuel_inertia

    def step(self, achieved, *, environment_acceleration=None, external_torque_body_nm=None,
             internal_momentum_body_nms=None, environment_wrench=None):
        """Advance achieved actuation and optional state-dependent environment.

        environment_wrench(time,r,v,q,omega,mass) returns physical body
        (force, torque) at each RK stage, enabling attitude-dependent plate
        forces without holding an earlier attitude through the orbit step.
        """
        if not isinstance(achieved, ActuationResult):
            raise ValueError("Plant requires achieved ActuationResult, not a WrenchCommand")
        if achieved.command.time != self.time or achieved.command.body_frame != self.body_frame:
            raise ValueError("Achieved wrench epoch/time/body frame mismatch")
        if achieved.propellant_used_kg > self.fuel+1e-12:
            raise ValueError("Actuator consumed more propellant than the plant carries")
        torque = achieved.torque_nm+(np.zeros(3) if external_torque_body_nm is None else
                                    vector(external_torque_body_nm, "external_torque_body_nm", size=3))
        momentum = np.zeros(3) if internal_momentum_body_nms is None else vector(internal_momentum_body_nms, "internal_momentum_body_nms", size=3)
        dt = achieved.duration_s
        mass = self.mass_kg-achieved.propellant_used_kg/2
        inertia = self.dry_inertia+(self.fuel-achieved.propellant_used_kg/2)*self.fuel_inertia
        state = np.r_[self.r, self.v, self.q, self.omega]
        steps = max(1, int(np.ceil(dt/self.max_step)))
        h = dt/steps
        def rhs(value, offset):
            r, v, q, omega = value[:3], value[3:6], value[6:10], value[10:]
            env = np.zeros(3) if environment_acceleration is None else vector(
                environment_acceleration(self.time.shifted(offset), r.copy(), v.copy()), "environment_acceleration", size=3)
            force=achieved.force_n
            stage_torque=torque
            if environment_wrench is not None:
                extra_force,extra_torque=environment_wrench(self.time.shifted(offset),r.copy(),v.copy(),
                    normalize(q),omega.copy(),mass)
                force=force+vector(extra_force,'environment force',size=3)
                stage_torque=stage_torque+vector(extra_torque,'environment torque',size=3)
            acceleration = env+rot_matrix(q) @ force/mass
            alpha = np.linalg.solve(inertia, stage_torque-np.cross(omega, inertia @ omega+momentum))
            qdot = .5*quat_multiply(q, np.r_[0., omega])
            return np.r_[v, acceleration, qdot, alpha]
        for i in range(steps):
            a = rhs(state, i*h)
            b = rhs(state+h*a/2, (i+.5)*h)
            c = rhs(state+h*b/2, (i+.5)*h)
            d = rhs(state+h*c, (i+1)*h)
            state += h*(a+2*b+2*c+d)/6
            state[6:10] = normalize(state[6:10])
        self.r, self.v, self.q, self.omega = state[:3], state[3:6], state[6:10], state[10:]
        self.fuel = max(0., self.fuel-achieved.propellant_used_kg)
        self.time = self._clock.advance(dt)
