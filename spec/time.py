"""Explicit time references; this module does not convert physical time scales."""
from dataclasses import dataclass, field
from enum import Enum

from utils.validation import finite_scalar, text_label


class TimeScale(str, Enum):
    SIMULATION = "simulation"
    TAI = "TAI"
    TT = "TT"
    TDB = "TDB"
    GPS = "GPS"


@dataclass(frozen=True)
class Epoch:
    """Identity of a time origin on a uniform seconds scale.

    Physical scales require an explicit origin description, e.g. an ISO epoch
    labelled in that scale. No leap-second, relativistic or frame conversion
    is implied. An ephemeris/time provider must perform those conversions.
    """
    name: str = "simulation"
    scale: TimeScale = TimeScale.SIMULATION
    origin: str | None = None

    def __post_init__(self):
        text_label(self.name, "epoch name")
        object.__setattr__(self, "scale", TimeScale(self.scale))
        if self.origin is not None:
            text_label(self.origin, "epoch origin")
        if self.scale != TimeScale.SIMULATION and self.origin is None:
            raise ValueError("Physical time scales require an explicit epoch origin")


@dataclass(frozen=True)
class Timestamp:
    """Finite seconds relative to an explicitly identified epoch."""
    seconds: float
    epoch: Epoch = field(default_factory=Epoch)

    def __post_init__(self):
        object.__setattr__(self, "seconds", finite_scalar(self.seconds, "seconds"))
        if not isinstance(self.epoch, Epoch):
            raise ValueError("epoch must be an Epoch")

    def seconds_since(self, other: "Timestamp") -> float:
        if not isinstance(other, Timestamp) or self.epoch != other.epoch:
            raise ValueError("Cannot combine timestamps from different epochs/time scales")
        return self.seconds - other.seconds

    def shifted(self, seconds: float) -> "Timestamp":
        return Timestamp(self.seconds + finite_scalar(seconds, "offset"), self.epoch)
