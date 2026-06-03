import math
import pytest
from fsw.capture_gate import evaluate_capture, CaptureGateIn


def _inp(**kwargs):
    """Build a CaptureGateIn with all-passing defaults, then apply overrides."""
    defaults = dict(
        port_range_m  = 0.10,
        port_vrel_ms  = 0.02,
        align_deg     = 5.0,
        body_clear    = True,
        capture_core  = True,
        geometry_ok   = True,
        align_ok      = True,
    )
    defaults.update(kwargs)
    return CaptureGateIn(**defaults)


class TestSoftCaptureReady:
    def test_all_conditions_met(self):
        assert evaluate_capture(_inp()).soft_capture_ready

    def test_blocked_when_range_too_large(self):
        assert not evaluate_capture(_inp(port_range_m=0.31)).soft_capture_ready

    def test_blocked_when_too_fast(self):
        assert not evaluate_capture(_inp(port_vrel_ms=0.06)).soft_capture_ready

    def test_blocked_when_body_not_clear(self):
        assert not evaluate_capture(_inp(body_clear=False)).soft_capture_ready

    def test_blocked_when_misaligned(self):
        assert not evaluate_capture(_inp(align_deg=61.0)).soft_capture_ready

    def test_nan_align_deg_no_constraint(self):
        # NaN means estimator not ready → no alignment constraint
        assert evaluate_capture(_inp(align_deg=float('nan'))).soft_capture_ready

    def test_exactly_at_range_threshold_ok(self):
        # strictly less than, so 0.30 is blocked
        assert not evaluate_capture(_inp(port_range_m=0.30)).soft_capture_ready
        assert evaluate_capture(_inp(port_range_m=0.299)).soft_capture_ready


class TestHardCaptureReady:
    def test_all_conditions_met(self):
        assert evaluate_capture(_inp(port_range_m=0.05, port_vrel_ms=0.005)).hard_capture_ready

    def test_blocked_when_range_outside_hard_envelope(self):
        assert not evaluate_capture(_inp(port_range_m=0.09)).hard_capture_ready

    def test_blocked_when_geometry_not_ok(self):
        assert not evaluate_capture(
            _inp(port_range_m=0.05, port_vrel_ms=0.005, geometry_ok=False)).hard_capture_ready

    def test_blocked_when_align_not_ok(self):
        assert not evaluate_capture(
            _inp(port_range_m=0.05, port_vrel_ms=0.005, align_ok=False)).hard_capture_ready

    def test_stricter_than_soft(self):
        # Something in soft range but not hard range
        out = evaluate_capture(_inp(port_range_m=0.20, port_vrel_ms=0.02))
        assert out.soft_capture_ready
        assert not out.hard_capture_ready


class TestSoftCore:
    def test_core_ready_when_in_core_and_aligned(self):
        assert evaluate_capture(_inp()).soft_core_ready

    def test_blocked_when_not_in_capture_core(self):
        assert not evaluate_capture(_inp(capture_core=False)).soft_core_ready

    def test_blocked_when_align_deg_nan(self):
        # NaN align_deg → can't certify alignment → soft_core not ready
        assert not evaluate_capture(_inp(align_deg=float('nan'))).soft_core_ready

    def test_blocked_when_core_align_exceeded(self):
        assert not evaluate_capture(_inp(align_deg=21.0)).soft_core_ready


class TestSoftCaptureStable:
    def test_stable_when_slow_and_close_and_clear(self):
        assert evaluate_capture(_inp(port_vrel_ms=0.01)).soft_capture_stable

    def test_not_stable_when_too_fast(self):
        assert not evaluate_capture(_inp(port_vrel_ms=0.031)).soft_capture_stable

    def test_not_stable_when_body_not_clear(self):
        assert not evaluate_capture(_inp(port_vrel_ms=0.01, body_clear=False)).soft_capture_stable


class TestSoftCaptureCertified:
    def test_certified_when_stable_and_core_and_aligned(self):
        out = evaluate_capture(_inp(port_vrel_ms=0.01))
        assert out.soft_capture_certified

    def test_not_certified_when_not_stable(self):
        out = evaluate_capture(_inp(port_vrel_ms=0.05))
        assert not out.soft_capture_certified

    def test_not_certified_when_not_in_core(self):
        out = evaluate_capture(_inp(port_vrel_ms=0.01, capture_core=False))
        assert not out.soft_capture_certified

    def test_not_certified_when_align_exceeds_dock_max(self):
        out = evaluate_capture(_inp(port_vrel_ms=0.01, align_deg=11.0, align_ok=False))
        assert not out.soft_capture_certified
