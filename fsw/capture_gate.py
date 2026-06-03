"""
fsw/capture_gate.py — Pure capture classification logic.

Faithfully matches main.py's inline capture logic. Both main.py and
monte_carlo.py call evaluate_capture() so the classification is identical
and divergence-free.

C signature:
  void capture_gate(const CaptureGateIn *in, CaptureGateOut *out);

Caller responsibilities (not done inside this function):
  - Apply ENABLE_FINITE_BODY_COLLISION flag before setting body_clear.
  - Handle None/unavailable align_deg by passing NaN.
    NaN align_deg → no alignment constraint on soft_capture_ready (matches
    main.py's `attitude_align_deg_cmd is None` branch).
"""

from __future__ import annotations
from dataclasses import dataclass
import math


@dataclass
class CaptureGateIn:
    """All instantaneous inputs.  Maps 1-to-1 to the C struct."""
    port_range_m:    float   # deputy-to-port distance [m]
    port_vrel_ms:    float   # port-relative closing speed [m/s]
    align_deg:       float   # attitude alignment error [deg]; NaN = not available
    body_clear:      bool    # dock_geom["body_clear"] (finite-body flag applied by caller)
    capture_core:    bool    # dock_geom["capture_core"]
    geometry_ok:     bool    # dock_geom["ok"]  (body_clear & cone_ok & in_aperture)
    align_ok:        bool    # align_geom["ok"] (align_deg <= DOCK_ALIGN_MAX_DEG)

    # Config — passed explicitly, no globals (maps to C config struct)
    soft_capture_range_m:            float = 0.30
    soft_capture_vrel_ms:            float = 0.05
    soft_capture_entry_align_max_deg: float = 60.0
    soft_capture_latch_vrel_ms:      float = 0.030
    soft_capture_core_align_max_deg:  float = 20.0
    hard_capture_range_m:            float = 0.08
    hard_capture_vrel_ms:            float = 0.010
    dock_align_max_deg:              float = 10.0


@dataclass
class CaptureGateOut:
    """Instantaneous classification flags.  Maps 1-to-1 to the C struct."""
    soft_capture_ready:     bool   # translational + alignment entry gate
    hard_capture_ready:     bool   # tight envelope; geometry + alignment certified
    soft_core_ready:        bool   # inside capture core with attitude ok
    soft_capture_stable:    bool   # port-relative motion below latch threshold
    soft_capture_certified: bool   # stable + core + aligned → safe to latch


def evaluate_capture(i: CaptureGateIn) -> CaptureGateOut:
    """
    Classify capture state from instantaneous measurements.

    All logic mirrors main.py exactly.  No state, no globals.
    """
    align_known = not math.isnan(i.align_deg)

    # Soft capture entry: translational only + coarse alignment.
    # None/NaN align_deg means estimator not ready → no alignment constraint.
    align_ok_entry = (not align_known) or (i.align_deg < i.soft_capture_entry_align_max_deg)
    soft_capture_ready = (
        i.port_range_m < i.soft_capture_range_m
        and i.port_vrel_ms < i.soft_capture_vrel_ms
        and i.body_clear
        and align_ok_entry
    )

    # Hard capture: tight envelope, full geometry + attitude certified.
    hard_capture_ready = (
        i.port_range_m < i.hard_capture_range_m
        and i.port_vrel_ms < i.hard_capture_vrel_ms
        and i.geometry_ok
        and i.body_clear
        and i.align_ok
    )

    # Soft core: inside capture sphere with attitude good enough for latch.
    soft_core_ready = (
        i.capture_core
        and align_known
        and i.align_deg <= i.soft_capture_core_align_max_deg
    )

    # Stable: port-relative motion below latch certification speed.
    soft_capture_stable = (
        i.port_range_m < i.soft_capture_range_m
        and i.port_vrel_ms < i.soft_capture_latch_vrel_ms
        and i.body_clear
    )

    # Certified: all three conditions for autonomous hard latch.
    soft_capture_certified = (
        soft_capture_stable
        and soft_core_ready
        and align_known
        and i.align_deg <= i.dock_align_max_deg
    )

    return CaptureGateOut(
        soft_capture_ready    = soft_capture_ready,
        hard_capture_ready    = hard_capture_ready,
        soft_core_ready       = soft_core_ready,
        soft_capture_stable   = soft_capture_stable,
        soft_capture_certified = soft_capture_certified,
    )
