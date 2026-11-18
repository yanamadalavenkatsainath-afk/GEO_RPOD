"""Configurable commanded latch state model, independent of mission phases."""
from dataclasses import dataclass

import numpy as np

from spec import Timestamp
from utils.validation import finite_scalar, vector


@dataclass(frozen=True)
class LatchStatus:
    latched: bool
    broken: bool
    state: str
    reason: str


class CaptureLatch:
    """Contact/position/speed/alignment/load/dwell acceptance model.

    Thresholds are explicit design assumptions, not a certified mechanism.
    Loads must come from the selected contact/plant model or hardware sensing.
    It does not project translation or silently cancel relative velocity.
    """
    def __init__(self, *, time, maximum_position_error_m, maximum_speed_m_s,
                 maximum_alignment_error_rad, maximum_load_n, dwell_s):
        if not isinstance(time, Timestamp):
            raise ValueError("Timestamp required")
        self.start_time = time
        self.position_limit = finite_scalar(maximum_position_error_m, "maximum_position_error_m", minimum=0)
        self.speed_limit = finite_scalar(maximum_speed_m_s, "maximum_speed_m_s", minimum=0)
        self.alignment_limit = finite_scalar(maximum_alignment_error_rad, "maximum_alignment_error_rad", minimum=0)
        self.load_limit = finite_scalar(maximum_load_n, "maximum_load_n", strictly_positive=True)
        self.dwell = finite_scalar(dwell_s, "dwell_s", minimum=0)
        self.reset()

    def reset(self):
        self.time, self.good_since = self.start_time, None
        self.latched = self.broken = False

    def update(self, *, time, contact, request_latch, request_release,
               position_error_m, relative_speed_m_s, alignment_error_rad, contact_force_n):
        if any(not isinstance(v, bool) for v in (contact, request_latch, request_release)):
            raise ValueError("Contact/commands must be boolean")
        if time.seconds_since(self.time) < 0:
            raise ValueError("Latch timeline cannot go backwards")
        if request_latch and request_release:
            raise ValueError("Conflicting latch/release commands")
        position = finite_scalar(position_error_m, "position_error_m", minimum=0)
        speed = finite_scalar(relative_speed_m_s, "relative_speed_m_s", minimum=0)
        alignment = finite_scalar(alignment_error_rad, "alignment_error_rad", minimum=0)
        load = np.linalg.norm(vector(contact_force_n, "contact_force_n", size=3))
        self.time = time
        if self.broken:
            return LatchStatus(False, True, "BROKEN", "LOAD_FAILURE")
        if self.latched and load > self.load_limit:
            self.latched, self.broken = False, True
            return LatchStatus(False, True, "BROKEN", "LOAD_FAILURE")
        if request_release:
            self.latched, self.good_since = False, None
            return LatchStatus(False, False, "OPEN", "RELEASE_COMMANDED")
        if self.latched:
            return LatchStatus(True, False, "LATCHED", "HOLDING")
        conditions = (contact and position <= self.position_limit and speed <= self.speed_limit
                      and alignment <= self.alignment_limit and load <= self.load_limit)
        if request_latch and conditions:
            if self.good_since is None:
                self.good_since = time
            if time.seconds_since(self.good_since) >= self.dwell:
                self.latched = True
                return LatchStatus(True, False, "LATCHED", "ACCEPTED")
            return LatchStatus(False, False, "PENDING", "DWELL")
        self.good_since = None
        return LatchStatus(False, False, "CONTACT" if contact else "OPEN",
                           "CONDITIONS_REJECTED" if request_latch else "NO_LATCH_COMMAND")
