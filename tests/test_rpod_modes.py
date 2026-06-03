import numpy as np
import pytest
from control.lambert_controller import (
    RPODMode, GEORPODController,
    FAR_FIELD_M, TERMINAL_M, PROX_V_PROFILE,
)


def _ctrl():
    return GEORPODController(dep_mass_kg=50.0, dep_thrust_N=1.0)


# ── Enum integrity ────────────────────────────────────────────────────

class TestRPODModeEnum:
    EXPECTED = {"FORMATION_HOLD", "LAMBERT", "PROX_OPS", "TERMINAL",
                "SOFT_CAPTURE", "DOCKING", "LOST_TARGET"}

    def test_all_modes_present(self):
        assert {m.name for m in RPODMode} == self.EXPECTED

    def test_distinct_values(self):
        vals = [m.value for m in RPODMode]
        assert len(vals) == len(set(vals))


# ── Velocity profile sanity ───────────────────────────────────────────

class TestProxVProfile:
    def test_ranges_strictly_decreasing(self):
        ranges = [r for r, _ in PROX_V_PROFILE]
        assert all(r1 > r2 for r1, r2 in zip(ranges, ranges[1:]))

    def test_speeds_strictly_decreasing(self):
        speeds = [v for _, v in PROX_V_PROFILE]
        assert all(v1 > v2 for v1, v2 in zip(speeds, speeds[1:]))

    def test_min_profile_range_above_terminal_threshold(self):
        min_range = min(r for r, _ in PROX_V_PROFILE)
        assert min_range > TERMINAL_M, (
            "PROX_V_PROFILE must cover all the way down to near TERMINAL_M "
            "so there is always a target speed in PROX_OPS"
        )

    def test_max_profile_range_at_or_below_far_field(self):
        max_range = max(r for r, _ in PROX_V_PROFILE)
        assert max_range <= FAR_FIELD_M


# ── Mode transitions ──────────────────────────────────────────────────

# Minimal orbit state: 100 m along LVLH-x, at rest
_STATE_100M = np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
# GEO chief position / velocity (approximate circular)
_CHI_POS = np.array([42164e3, 0.0, 0.0])
_CHI_VEL = np.array([0.0, 3074.0, 0.0])


class TestLostTargetTransitions:
    def test_prox_ops_to_lost_target_on_cam_loss(self):
        ctrl = _ctrl()
        ctrl._set_mode(RPODMode.PROX_OPS, 0.0)
        ctrl.compute(_STATE_100M, _CHI_POS, _CHI_VEL, t=0.0, cam_lost=True)
        assert ctrl.mode == RPODMode.LOST_TARGET

    def test_no_lost_target_below_15m(self):
        ctrl = _ctrl()
        ctrl._set_mode(RPODMode.PROX_OPS, 0.0)
        state_10m = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        ctrl.compute(state_10m, _CHI_POS, _CHI_VEL, t=0.0, cam_lost=True)
        # Below 15 m the centroid fallback keeps tracking — no lost-target
        assert ctrl.mode == RPODMode.PROX_OPS

    def test_lost_target_stays_until_10s(self):
        ctrl = _ctrl()
        ctrl._set_mode(RPODMode.LOST_TARGET, 0.0)
        ctrl.compute(_STATE_100M, _CHI_POS, _CHI_VEL, t=9.0, cam_lost=False)
        assert ctrl.mode == RPODMode.LOST_TARGET

    def test_lost_target_recovers_after_10s(self):
        ctrl = _ctrl()
        ctrl._set_mode(RPODMode.LOST_TARGET, 0.0)
        ctrl.compute(_STATE_100M, _CHI_POS, _CHI_VEL, t=10.0, cam_lost=False)
        assert ctrl.mode == RPODMode.PROX_OPS

    def test_set_mode_records_history(self):
        ctrl = _ctrl()
        ctrl._set_mode(RPODMode.PROX_OPS, 0.0)
        ctrl._set_mode(RPODMode.LOST_TARGET, 10.0)
        modes = [m for _, m in ctrl.mode_history]
        assert RPODMode.PROX_OPS in modes
        assert RPODMode.LOST_TARGET in modes
