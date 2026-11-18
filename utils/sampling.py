"""Time-grid sampling without accumulated floating-point schedule drift."""
from dataclasses import dataclass
import math

import numpy as np

from spec.time import Timestamp
from utils.validation import finite_scalar


@dataclass(frozen=True)
class SensorTiming:
    period_s: float
    latency_s: float = 0.0
    dropout_probability: float = 0.0
    start_s: float = 0.0

    def __post_init__(self):
        for name in ("period_s", "latency_s", "dropout_probability", "start_s"):
            object.__setattr__(self, name, finite_scalar(
                getattr(self, name), name,
                minimum=0 if name in ("latency_s", "dropout_probability") else None,
                strictly_positive=name == "period_s"))
        if self.dropout_probability > 1:
            raise ValueError("dropout_probability must lie in [0, 1]")


class SampleClock:
    """At most one acquisition per time-grid slot.

    A late poll samples current truth once; missed slots are not fabricated.
    Backward time or an epoch change requires an explicit reset.
    """
    def __init__(self, timing: SensorTiming):
        if not isinstance(timing, SensorTiming):
            raise ValueError("timing must be SensorTiming")
        self.timing = timing
        self.reset()

    def reset(self):
        self._last_time = None
        self._last_slot = None

    def due(self, time: Timestamp):
        if not isinstance(time, Timestamp):
            raise ValueError("time must be Timestamp")
        if self._last_time is not None and time.seconds_since(self._last_time) < 0:
            raise ValueError("Sampling time moved backward; reset before starting a new run")
        self._last_time = time
        if time.seconds < self.timing.start_s:
            return False
        phase = (time.seconds - self.timing.start_s) / self.timing.period_s
        slot = math.floor(phase + 64 * np.finfo(float).eps * max(1.0, abs(phase)))
        if self._last_slot == slot:
            return False
        self._last_slot = slot
        return True


class SimulationClock:
    """Compensated accumulated step durations relative to an explicit origin.

    Avoids long repeated-dt additions falling on the wrong sampling-grid slot.
    Physical time-scale conversion is still the epoch/provider's responsibility.
    """
    def __init__(self, time):
        if not isinstance(time, Timestamp):
            raise ValueError("time must be Timestamp")
        self.origin = self.time = time
        self._elapsed = self._correction = 0.

    def advance(self, duration_s):
        duration = finite_scalar(duration_s, "duration_s", strictly_positive=True)
        adjusted = duration-self._correction
        elapsed = self._elapsed+adjusted
        self._correction = (elapsed-self._elapsed)-adjusted
        self._elapsed = elapsed
        self.time = self.origin.shifted(self._elapsed)
        return self.time
