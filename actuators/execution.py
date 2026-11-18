"""Low-order device execution: achieved physical wrench ON the spacecraft.

Models cover saturation, failures, finite intervals and fuel. Electrical,
thermal, valve lag, pressure blowdown, slosh and microvibration are not modeled.
"""
from copy import deepcopy
import numpy as np

from actuators.thruster_layout import ThrusterLayout
from spec import ActuationResult, WrenchCommand
from utils.validation import finite_array, finite_scalar, vector


def _limits(value, count, name, *, zero=False):
    result = finite_array(value, name)
    if result.ndim == 0:
        result = np.full(count, float(result))
    if result.shape != (count,) or np.any(result < 0 if zero else result <= 0):
        raise ValueError(f"{name} must contain {count} {'nonnegative' if zero else 'positive'} values")
    return result


def _mask(value, count):
    if value is None:
        return np.zeros(count, dtype=bool)
    result = np.asarray(value)
    if result.shape != (count,) or result.dtype != np.dtype(bool):
        raise ValueError("disabled must be one boolean per channel")
    return result.copy()


class _Timeline:
    def __init__(self, body_frame):
        if not isinstance(body_frame, str) or not body_frame.strip():
            raise ValueError("body_frame must be a nonempty string")
        self.body_frame, self._end = body_frame, None

    def _interval(self, command, duration_s):
        if not isinstance(command, WrenchCommand) or command.body_frame != self.body_frame:
            raise ValueError("Actuator requires a WrenchCommand in its configured body frame")
        duration = finite_scalar(duration_s, "duration_s", strictly_positive=True)
        if self._end is not None:
            gap = command.time.seconds_since(self._end)
            tolerance = 64*np.finfo(float).eps*max(1., abs(command.time.seconds))
            if abs(gap) > tolerance:
                raise ValueError("Intervals must be contiguous; advance idle periods with zero commands")
        return duration

    def _finish(self, command, duration):
        self._end = command.time.shifted(duration)


class FuelTank:
    def __init__(self, dry_mass_kg, propellant_kg):
        self._dry_mass = finite_scalar(dry_mass_kg, "dry_mass_kg", strictly_positive=True)
        self._initial = finite_scalar(propellant_kg, "propellant_kg", minimum=0)
        self.reset()

    @property
    def dry_mass_kg(self):
        return self._dry_mass

    @property
    def propellant_kg(self):
        return self._fuel

    @property
    def mass_kg(self):
        return self._dry_mass+self._fuel

    def consume(self, requested_kg):
        requested = finite_scalar(requested_kg, "requested_kg", minimum=0)
        used = min(requested, self._fuel)
        self._fuel -= used
        return used

    def reset(self):
        self._fuel = self._initial


class ThrusterArray(_Timeline):
    """Allocate body wrench and execute continuous or minimum-impulse jets.

    Pulses can span steps and complete after demand becomes zero. Disabling
    a channel cancels its queued/pending pulse. Depletion truncates available
    impulse and returns the mean achieved wrench over the requested interval.
    Gain/direction errors are fixed parameters for a future MC runner to sample.
    """
    def __init__(self, layout, tank, isp_s=220., body_frame="body/chaser",
                 minimum_impulse_ns=0., thrust_gain=1., actual_directions_body=None,
                 allocation_method='bounded_least_squares',valve_delay_s=0.,
                 valve_time_constant_s=0.,blowdown_exponent=0.):
        super().__init__(body_frame)
        if not isinstance(layout, ThrusterLayout) or not isinstance(tank, FuelTank):
            raise ValueError("Provide a ThrusterLayout and FuelTank")
        self.layout, self.tank = deepcopy(layout), tank
        if allocation_method not in ('bounded_least_squares','minimum_force'):
            raise ValueError('Unknown thruster allocation method')
        self.allocation_method=allocation_method
        self.isp_s = finite_scalar(isp_s, "isp_s", strictly_positive=True)
        count = len(layout.max_force_n)
        self.minimum_impulse = _limits(minimum_impulse_ns, count, "minimum_impulse_ns", zero=True)
        self.gain = _limits(thrust_gain, count, "thrust_gain", zero=True)
        from actuators.valve_dynamics import ValveBank
        self.valves=ValveBank(count,delay_s=valve_delay_s,time_constant_s=valve_time_constant_s)
        self.blowdown_exponent=finite_scalar(blowdown_exponent,'blowdown_exponent',minimum=0)
        self.reference_fuel_kg=tank.propellant_kg
        directions = layout.directions_body if actual_directions_body is None else actual_directions_body
        directions = finite_array(directions, "actual_directions_body", shape=(count, 3))
        norms = np.linalg.norm(directions, axis=1)
        if np.any(norms == 0):
            raise ValueError("Actual thrust directions must be nonzero")
        self.directions = directions / norms[:, None]
        self._pending = np.zeros(count)
        self._pulse_remaining = np.zeros(count)
        self.last_forces_n = np.zeros(count)

    def reset(self, *, reset_tank=True):
        self._end = None
        self._pending[:] = 0
        self._pulse_remaining[:] = 0
        self.last_forces_n[:] = 0
        self.valves.reset()
        if reset_tank:
            self.tank.reset()

    def execute(self, command, duration_s, *, disabled=None,stuck_on=None):
        dt = self._interval(command, duration_s)
        blocked = _mask(disabled, len(self.gain)) | (self.gain == 0)
        stuck=_mask(stuck_on,len(self.gain)) & ~blocked
        allocator=(self.layout.allocate_minimum_force if self.allocation_method=='minimum_force' else self.layout.allocate)
        allocation = allocator(command.force_n, command.torque_nm, excluded=blocked)
        flags = []
        if np.linalg.norm(allocation.residual) > 1e-9:
            flags.append("ALLOCATION_RESIDUAL")
        if np.any(blocked):
            flags.append("DISABLED_CHANNELS")
        self._pending[blocked] = 0
        self._pulse_remaining[blocked] = 0
        delivered = np.zeros_like(self.gain)
        for k, requested_force in enumerate(allocation.forces_n):
            if blocked[k]:
                continue
            capacity, mib = self.layout.max_force_n[k]*dt, self.minimum_impulse[k]
            if mib == 0:
                delivered[k] = requested_force*dt
                continue
            self._pending[k] += requested_force*dt
            carried = min(self._pulse_remaining[k], capacity)
            delivered[k] = carried
            self._pulse_remaining[k] -= carried
            capacity -= carried
            if capacity > 0 and self._pulse_remaining[k] == 0:
                complete = np.floor((self._pending[k]+1e-12*mib)/mib)
                additional = min(complete*mib, capacity)
                if additional > 0:
                    started = np.ceil(additional/mib-1e-12)
                    delivered[k] += additional
                    self._pending[k] = max(0., self._pending[k]-started*mib)
                    self._pulse_remaining[k] = max(0., started*mib-additional)
        if np.any(self.minimum_impulse > 0):
            flags.append("PULSED")
        if np.any(stuck):
            delivered[stuck]=self.layout.max_force_n[stuck]*dt
            flags.append('STUCK_ON_CHANNELS')
        delivered=self.valves.integrate(command.time.seconds,dt,delivered/dt,blocked)
        if np.any(self.valves.delay>0) or np.any(self.valves.tau>0):flags.append('VALVE_DYNAMICS')
        if self.blowdown_exponent:
            fraction=self.tank.propellant_kg/self.reference_fuel_kg if self.reference_fuel_kg else 0.
            delivered*=fraction**self.blowdown_exponent
            flags.append('PRESCRIBED_BLOWDOWN')
        delivered *= self.gain
        requested_fuel = float(np.sum(delivered)/(self.isp_s*9.80665))
        used = self.tank.consume(requested_fuel)
        if requested_fuel > used:
            delivered *= used/requested_fuel
            self._pending[:] = 0
            self._pulse_remaining[:] = 0
            self.valves.reset()
            flags.append("FUEL_EXHAUSTED")
        elif self.tank.propellant_kg == 0:
            flags.append("FUEL_EXHAUSTED")
        self.last_forces_n = delivered/dt
        jet_forces = self.directions*self.last_forces_n[:, None]
        force = np.sum(jet_forces, axis=0)
        torque = np.sum(np.cross(self.layout.positions_body_m, jet_forces), axis=0)
        self._finish(command, dt)
        return ActuationResult(command, dt, force, torque, used, tuple(flags))


class ReactionWheelArray(_Timeline):
    """Axis allocation with motor-torque, momentum/speed and failed-wheel limits."""
    def __init__(self, axes_body=None, max_torque_nm=.01, max_momentum_nms=.05,
                 rotor_inertia_kg_m2=1e-5, body_frame="body/chaser"):
        super().__init__(body_frame)
        axes = finite_array(np.eye(3) if axes_body is None else axes_body, "axes_body")
        if axes.ndim != 2 or axes.shape[1] != 3 or len(axes) == 0:
            raise ValueError("axes_body must contain one axis per wheel")
        norms = np.linalg.norm(axes, axis=1)
        if np.any(norms == 0):
            raise ValueError("Wheel axes must be nonzero")
        self.axes = axes/norms[:, None]
        self.max_torque = _limits(max_torque_nm, len(axes), "max_torque_nm")
        self.max_momentum = _limits(max_momentum_nms, len(axes), "max_momentum_nms")
        self.rotor_inertia = _limits(rotor_inertia_kg_m2, len(axes), "rotor_inertia_kg_m2")
        self._h = np.zeros(len(axes))
        self.last_power_w=0.

    @property
    def momentum(self):
        return self._h.copy()

    @property
    def angular_momentum_body(self):
        return self.axes.T @ self._h

    @property
    def speed_rad_s(self):
        return self._h/self.rotor_inertia

    def reset(self):
        self._h[:] = 0
        self._end = None
        self.last_power_w=0.

    def execute(self, command, duration_s, *, disabled=None,power_limit_w=None,motor_efficiency=.8):
        dt = self._interval(command, duration_s)
        blocked = _mask(disabled, len(self._h))
        device = np.zeros(len(self._h))
        if np.any(~blocked):
            device[~blocked] = np.linalg.lstsq(self.axes[~blocked].T, -command.torque_nm, rcond=None)[0]
        limited = np.clip(device, -self.max_torque, self.max_torque)
        old = self._h.copy()
        efficiency=finite_scalar(motor_efficiency,'motor_efficiency',strictly_positive=True)
        if efficiency>1:raise ValueError('Motor efficiency cannot exceed one')
        def electrical_power(scale):
            momentum=np.clip(old+limited*dt*scale,-self.max_momentum,self.max_momentum)
            energy=(momentum**2-old**2)/(2*self.rotor_inertia)
            return float(np.sum(np.maximum(energy,0))/(dt*efficiency))
        power_limited=False
        if power_limit_w is not None:
            budget=finite_scalar(power_limit_w,'power_limit_w',minimum=0)
            if electrical_power(1.)>budget:
                low,high=0.,1.
                for _ in range(40):
                    middle=(low+high)/2
                    if electrical_power(middle)<=budget:low=middle
                    else:high=middle
                limited*=low;power_limited=True
        self._h = np.clip(old+limited*dt, -self.max_momentum, self.max_momentum)
        self.last_power_w=float(np.sum(np.maximum((self._h**2-old**2)/(2*self.rotor_inertia),0))/(dt*efficiency))
        torque = -self.axes.T @ ((self._h-old)/dt)
        flags = []
        if power_limited:flags.append('POWER_LIMIT')
        if np.linalg.norm(torque-command.torque_nm) > 1e-10:
            flags.append("TORQUE_RESIDUAL")
        if np.any(blocked):
            flags.append("DISABLED_CHANNELS")
        if np.any(command.force_n != 0):
            flags.append("UNSUPPORTED_FORCE")
        self._finish(command, dt)
        return ActuationResult(command, dt, np.zeros(3), torque, flags=tuple(flags))


class MagneticDipoleActuator(_Timeline):
    """Bounded dipoles; body torque parallel to the magnetic field is unavailable."""
    def __init__(self, max_dipole_am2=.4, body_frame="body/chaser"):
        super().__init__(body_frame)
        self.max_dipole = _limits(max_dipole_am2, 3, "max_dipole_am2")
        self.dipole_am2 = np.zeros(3)

    def reset(self):
        self.dipole_am2[:] = 0
        self._end = None

    def execute(self, command, duration_s, *, magnetic_field_body_t, disabled=None,dipole_command_am2=None):
        dt = self._interval(command, duration_s)
        field = vector(magnetic_field_body_t, "magnetic_field_body_t", size=3)
        blocked = _mask(disabled, 3)
        norm2 = float(np.dot(field, field))
        dipole = (np.zeros(3) if norm2 == 0 else np.cross(field, command.torque_nm)/norm2) if dipole_command_am2 is None else vector(dipole_command_am2,'dipole_command_am2',size=3)
        self.dipole_am2 = np.clip(dipole, -self.max_dipole, self.max_dipole)
        self.dipole_am2[blocked] = 0
        torque = np.cross(self.dipole_am2, field)
        flags = []
        if norm2 == 0:
            flags.append("NO_MAGNETIC_FIELD")
        if np.linalg.norm(torque-command.torque_nm) > 1e-12:
            flags.append("TORQUE_RESIDUAL")
        if np.any(blocked):
            flags.append("DISABLED_CHANNELS")
        if np.any(command.force_n != 0):
            flags.append("UNSUPPORTED_FORCE")
        self._finish(command, dt)
        return ActuationResult(command, dt, np.zeros(3), torque, flags=tuple(flags))


def combine_actuation(results, source_id="actuator_bank"):
    """Sum partitioned device requests/results on the same interval, not duplicates."""
    results = tuple(results)
    if not results or any(not isinstance(r, ActuationResult) for r in results):
        raise ValueError("Provide at least one ActuationResult")
    first = results[0]
    for result in results:
        if (result.command.time != first.command.time or result.duration_s != first.duration_s
                or result.command.body_frame != first.command.body_frame):
            raise ValueError("Results must share epoch, time, duration and body frame")
    command = WrenchCommand(source_id, first.command.time, first.command.body_frame,
        sum((r.command.force_n for r in results), np.zeros(3)),
        sum((r.command.torque_nm for r in results), np.zeros(3)))
    return ActuationResult(command, first.duration_s,
        sum((r.force_n for r in results), np.zeros(3)),
        sum((r.torque_nm for r in results), np.zeros(3)),
        sum(r.propellant_used_kg for r in results),
        tuple(sorted({flag for r in results for flag in r.flags})))
