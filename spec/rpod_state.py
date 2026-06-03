"""
spec/rpod_state.py — Explicit RPOD guidance state structs.

These dataclasses are the direct Python equivalent of the C structs that will
be defined in rpod_state.h. Fields map 1-to-1: no Python objects, no hidden
state, no Optional that can't be represented as a sentinel value in C.

C mapping
---------
  TerminalState  → typedef struct TerminalState { ... } TerminalState;
  RpodGuidanceState → typedef struct RpodGuidanceState { ... } RpodGuidanceState;

Do NOT add fields that are [TRUTH]-only (EKF refs, truth resets, etc.).
Those stay in the sim harness, not in this spec.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class TerminalState:
    """
    Per-entry state for TERMINAL guidance.

    Reset when mode_entry_key changes (i.e. each time TERMINAL is entered).
    In C: compare mode_entry_key to int(mode_entry_t) each tick.
    """
    braking: bool  = False   # True while entry-brake phase is active
    entry_v: float = 0.0     # |vel| at TERMINAL entry [m/s]
    entry_key: int = -1      # int(mode_entry_t) at last entry; -1 = uninitialised


@dataclass
class RpodGuidanceState:
    """
    Complete RPOD guidance state — all fields that the C FSW must persist
    between ticks.

    This is the single struct passed into every rpod_guidance function and
    mutated in place (mirrors C pass-by-pointer convention).
    """
    # ── Mode ──────────────────────────────────────────────────────────
    # Use integer here so it maps to the C enum directly.
    # 0=FORMATION_HOLD 1=LAMBERT 2=PROX_OPS 3=TERMINAL
    # 4=SOFT_CAPTURE   5=DOCKING 6=LOST_TARGET
    mode: int        = 0       # RPODMode value
    mode_entry_t: float = 0.0  # time of last mode transition [s]

    # ── TERMINAL sub-state ────────────────────────────────────────────
    term: TerminalState = field(default_factory=TerminalState)

    # Closest approach range seen while in TERMINAL (for abort guard).
    term_min_range: float = 1.0e9   # m; reset each TERMINAL entry

    # Time camera was first lost in TERMINAL (-1 = not currently lost).
    term_cam_lost_since: float = -1.0   # s; -1 sentinel = "not set"

    # ── LOST_TARGET sub-state ─────────────────────────────────────────
    # (no extra fields — transition timing is in mode_entry_t)

    # ── Lambert sub-state (not ported to C FSW — Lambert runs on ground) ──
    # Kept here for completeness so Python controller can use same struct.
    lam_active: bool      = False
    lam_burn2_t: float    = -1.0        # -1 sentinel = "not scheduled"
    lam_dv2_lvlh: np.ndarray = field(
        default_factory=lambda: np.zeros(3))
    lam_last_plan_t: float = -9999.0
    lam_replan_count: int  = 0

    def time_in_mode(self, t: float) -> float:
        return t - self.mode_entry_t
