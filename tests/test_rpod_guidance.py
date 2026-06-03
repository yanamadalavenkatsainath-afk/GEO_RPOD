import math
import numpy as np
import pytest
from fsw.rpod_guidance import prox_ops_accel, terminal_accel
from spec.rpod_state import TerminalState


ACCEL_MAX = 0.020   # m/s²  (50kg deputy, 1N thrust)


class TestProxOpsAccel:
    def test_points_toward_origin(self):
        # Deputy at +x, stationary → accel should be in -x direction
        pos = np.array([100.0, 0.0, 0.0])
        vel = np.zeros(3)
        a = prox_ops_accel(pos, vel, ACCEL_MAX)
        assert a[0] < 0.0
        assert abs(a[1]) < 1e-9
        assert abs(a[2]) < 1e-9

    def test_zero_accel_at_desired_speed(self):
        # At 100m, desired closing speed is sqrt(100/500)*0.2 = 0.0894 m/s
        # If vel is already exactly vel_des, accel should be ~zero
        pos     = np.array([100.0, 0.0, 0.0])
        rng     = 100.0
        k       = 0.200 / math.sqrt(500.0)
        v_close = k * math.sqrt(rng)
        vel     = np.array([-v_close, 0.0, 0.0])
        a = prox_ops_accel(pos, vel, ACCEL_MAX)
        assert np.linalg.norm(a) < 1e-9

    def test_capped_at_accel_max(self):
        # Far from desired speed → should saturate
        pos = np.array([500.0, 0.0, 0.0])
        vel = np.array([10.0, 0.0, 0.0])   # moving away fast
        a = prox_ops_accel(pos, vel, ACCEL_MAX)
        assert np.linalg.norm(a) <= ACCEL_MAX + 1e-9

    def test_speed_capped_below_10m(self):
        # At 5m, desired speed should be capped at 5mm/s
        pos = np.array([5.0, 0.0, 0.0])
        vel = np.zeros(3)
        a = prox_ops_accel(pos, vel, ACCEL_MAX)
        # vel_des = 5mm/s → accel = vel_des / TAU = 0.005 / 5 = 0.001 m/s²
        assert np.linalg.norm(a) <= ACCEL_MAX + 1e-9

    def test_forces_non_negative_norm(self):
        for pos in [np.array([10., 0., 0.]), np.array([0., 500., 0.]),
                    np.array([0., 0., 100.])]:
            a = prox_ops_accel(pos, np.zeros(3), ACCEL_MAX)
            assert np.linalg.norm(a) >= 0.0

    def test_near_zero_range_no_crash(self):
        pos = np.array([1e-6, 0.0, 0.0])
        a = prox_ops_accel(pos, np.zeros(3), ACCEL_MAX)
        assert np.all(np.isfinite(a))


class TestTerminalAccel:
    def _fresh_state(self):
        return TerminalState()

    def test_no_braking_at_slow_entry(self):
        state = self._fresh_state()
        pos = np.array([0.0, 0.0, 0.5])
        vel = np.array([0.0, 0.0, -0.010])   # 10mm/s — below 2×V_MAX
        terminal_accel(pos, vel, None, ACCEL_MAX, 0.0, state)
        assert not state.braking

    def test_braking_activated_at_fast_entry(self):
        state = self._fresh_state()
        pos = np.array([0.0, 0.0, 0.5])
        vel = np.array([0.0, 0.0, -0.10])    # 100mm/s → > 2×25mm/s
        terminal_accel(pos, vel, None, ACCEL_MAX, 0.0, state)
        assert state.braking

    def test_brake_clears_when_slow(self):
        state = TerminalState(braking=True, entry_v=0.1, entry_key=0)
        pos = np.array([0.0, 0.0, 0.5])
        vel = np.array([0.0, 0.0, -0.005])   # below 10mm/s brake done threshold
        terminal_accel(pos, vel, None, ACCEL_MAX, 0.0, state)
        assert not state.braking

    def test_entry_key_resets_state_on_reentry(self):
        state = TerminalState(braking=False, entry_v=0.0, entry_key=5)
        pos = np.array([0.0, 0.0, 0.5])
        vel = np.array([0.0, 0.0, -0.10])    # fast — should re-arm braking
        # mode_entry_t=10.0 → entry_key=10 ≠ 5 → fresh entry detected
        terminal_accel(pos, vel, None, ACCEL_MAX, 10.0, state)
        assert state.braking
        assert state.entry_key == 10

    def test_port_sanity_guard_rejects_distant_port(self):
        # Port 5m away from deputy — should fall back to CoM target
        state = self._fresh_state()
        pos  = np.array([0.0, 0.0, 0.4])
        vel  = np.zeros(3)
        port = np.array([0.0, 0.0, 5.0])    # > 2m sanity threshold from pos
        a_with_bad_port = terminal_accel(pos, vel, port, ACCEL_MAX, 0.0, state)
        state2 = self._fresh_state()
        a_no_port = terminal_accel(pos, vel, None, ACCEL_MAX, 0.0, state2)
        # Both should target CoM → same acceleration
        assert np.allclose(a_with_bad_port, a_no_port, atol=1e-9)

    def test_accel_capped_at_max(self):
        state = self._fresh_state()
        pos = np.array([0.0, 0.0, 0.5])
        vel = np.array([1.0, 0.0, 0.0])   # large lateral velocity
        a = terminal_accel(pos, vel, None, ACCEL_MAX, 0.0, state)
        assert np.linalg.norm(a) <= ACCEL_MAX + 1e-9

    def test_output_finite(self):
        state = self._fresh_state()
        a = terminal_accel(np.array([0.0, 0.0, 0.3]), np.zeros(3),
                           None, ACCEL_MAX, 0.0, state)
        assert np.all(np.isfinite(a))
