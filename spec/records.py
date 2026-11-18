"""Mission-independent state, measurement and physical actuation records.

    Frames are explicit names such as earth/gcrs, body/chaser or lvlh/chief.
    A name is an identity, not a request to perform a frame transformation.
"""
from dataclasses import dataclass, field

import numpy as np

from spec.time import Timestamp
from utils.validation import covariance, finite_array, finite_scalar, text_label, vector


def _units(values, name):
    if isinstance(values, str):
        raise ValueError(f"{name} must contain one unit label per coordinate")
    result = tuple(values)
    if not result:
        raise ValueError(f"{name} must not be empty")
    for item in result:
        text_label(item, name)
    return result


def _identity(record):
    for name in ("source_id", "kind", "frame"):
        text_label(getattr(record, name), name)
    if not isinstance(record.time, Timestamp):
        raise ValueError("time must be a Timestamp")
    object.__setattr__(record, "units", _units(record.units, "units"))


@dataclass(frozen=True)
class Measurement:
    """One observation, including explicitly unavailable observations.

    ``error_units`` defines covariance coordinates when different from value
    coordinates (e.g. a 4-value quaternion with a 3-angle error covariance).
    Availability is separate from acquisition time; neither implies freshness.
    A valid sample requires finite values and a declared PSD covariance.
    """
    source_id: str
    kind: str
    time: Timestamp
    frame: str
    units: tuple[str, ...]
    value: np.ndarray | None = None
    covariance: np.ndarray | None = None
    valid: bool = True
    reason: str = "OK"
    available_at: Timestamp | None = None
    error_units: tuple[str, ...] | None = None
    error_frame: str | None = None

    def __post_init__(self):
        _identity(self)
        if not isinstance(self.valid, (bool, np.bool_)):
            raise ValueError("valid must be boolean")
        text_label(self.reason, "reason")
        if not self.valid and self.reason == "OK":
            raise ValueError("Unavailable/rejected measurements require a reason")
        if self.value is not None:
            object.__setattr__(self, "value", self._checked_value())
        error_units = self.units if self.error_units is None else _units(self.error_units, "error_units")
        object.__setattr__(self, "error_units", error_units)
        error_frame = self.frame if self.error_frame is None else self.error_frame
        text_label(error_frame, "error_frame")
        object.__setattr__(self, "error_frame", error_frame)
        if self.covariance is not None:
            object.__setattr__(self, "covariance", covariance(
                self.covariance, len(error_units), readonly=True))
        if self.valid and (self.value is None or self.covariance is None):
            raise ValueError("Valid measurements require both value and covariance")
        available = self.time if self.available_at is None else self.available_at
        if not isinstance(available, Timestamp) or available.seconds_since(self.time) < 0:
            raise ValueError("available_at must be at or after acquisition in the same epoch")
        object.__setattr__(self, "available_at", available)

    def _checked_value(self):
        return vector(self.value, "value", size=len(self.units), readonly=True)

    def usable_at(self, now: Timestamp, *, max_age_s: float | None = None) -> bool:
        age = now.seconds_since(self.time)
        delivered = now.seconds_since(self.available_at) >= 0
        limit = None if max_age_s is None else finite_scalar(max_age_s, "max_age_s", minimum=0)
        return bool(self.valid and delivered and age >= 0 and (limit is None or age <= limit))


@dataclass(frozen=True)
class StateEstimate(Measurement):
    """Estimated state with covariance, distinct in type from simulator truth."""
    def __post_init__(self):
        super().__post_init__()
        from telemetry.component_events import emit_component_event
        emit_component_event('estimate',self)


@dataclass(frozen=True)
class PointCloudMeasurement(Measurement):
    """N independent XYZ observations; covariance is shared per point (3x3).

    This avoids a dense (3N)x(3N) covariance. Correlated clouds need a different
    noise representation and must not claim this independent-point contract.
    """
    def __post_init__(self):
        super().__post_init__()
        if len(self.units) != 3 or len(self.error_units) != 3:
            raise ValueError("Point clouds require three value/error units")

    def _checked_value(self):
        points = finite_array(self.value, "points", readonly=True)
        if (points.ndim != 2 or points.shape[1] != 3 or len(self.units) != 3
                or (self.valid and len(points) == 0)):
            raise ValueError("A valid point cloud requires shape (N,3), N>0, and three units")
        return points


@dataclass(frozen=True)
class _VectorRecord:
    source_id: str
    kind: str
    time: Timestamp
    frame: str
    units: tuple[str, ...]
    value: np.ndarray

    def __post_init__(self):
        _identity(self)
        object.__setattr__(self, "value", vector(
            self.value, "value", size=len(self.units), readonly=True))


@dataclass(frozen=True)
class TruthState(_VectorRecord):
    """Simulator-only vector. Estimator APIs must consume observations instead."""


@dataclass(frozen=True)
class Reference(_VectorRecord):
    """Deterministic reference/setpoint vector, not an observation of truth."""


@dataclass(frozen=True)
class WrenchCommand:
    """Requested physical force and torque ON the spacecraft, in a body frame.

    A wheel's motor torque is a device command, with opposite bus reaction;
    convert it explicitly before constructing a physical body wrench.
    """
    source_id: str
    time: Timestamp
    body_frame: str
    force_n: np.ndarray = field(default_factory=lambda: np.zeros(3))
    torque_nm: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        text_label(self.source_id, "source_id")
        text_label(self.body_frame, "body_frame")
        if not isinstance(self.time, Timestamp):
            raise ValueError("time must be a Timestamp")
        for name in ("force_n", "torque_nm"):
            object.__setattr__(self, name, vector(getattr(self, name), name, size=3, readonly=True))


@dataclass(frozen=True)
class ActuationResult:
    """Mean achieved body wrench over an interval beginning at command.time.

    ``flags`` records saturation, disabled/failed channels, etc. Fuel consumed
    covers this interval only. Models must not report requested thrust as
    achieved when allocation, device limits or depletion reduce it.
    """
    command: WrenchCommand
    duration_s: float
    force_n: np.ndarray
    torque_nm: np.ndarray
    propellant_used_kg: float = 0.0
    flags: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.command, WrenchCommand):
            raise ValueError("command must be a WrenchCommand")
        object.__setattr__(self, "duration_s", finite_scalar(
            self.duration_s, "duration_s", strictly_positive=True))
        object.__setattr__(self, "propellant_used_kg", finite_scalar(
            self.propellant_used_kg, "propellant_used_kg", minimum=0))
        for name in ("force_n", "torque_nm"):
            object.__setattr__(self, name, vector(getattr(self, name), name, size=3, readonly=True))
        if isinstance(self.flags, str):
            raise ValueError("flags must be a sequence of reason strings")
        flags = tuple(self.flags)
        for flag in flags:
            text_label(flag, "flag")
        object.__setattr__(self, "flags", flags)
        from telemetry.component_events import emit_component_event
        emit_component_event('actuation_result',self)

    @property
    def residual(self):
        """Requested minus achieved [force N, torque Nm], in command.body_frame."""
        return np.concatenate((self.command.force_n - self.force_n,
                               self.command.torque_nm - self.torque_nm))
