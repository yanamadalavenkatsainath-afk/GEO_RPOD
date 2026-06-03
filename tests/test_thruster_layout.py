import numpy as np
import pytest
from plant.thruster_layout import ThrusterLayout


class TestBox24:
    def test_thruster_count(self):
        assert len(ThrusterLayout.box_24().directions_body) == 24

    def test_allocate_plus_x(self):
        layout = ThrusterLayout.box_24(max_force_n=1.0)
        result = layout.allocate(np.array([1.0, 0.0, 0.0]))
        assert result.force_body_n[0] > 0.5
        assert abs(result.force_body_n[1]) < 0.15
        assert abs(result.force_body_n[2]) < 0.15

    def test_allocate_minus_z_dock_axis(self):
        # Docking-axis deceleration — box_16 couldn't do this efficiently
        layout = ThrusterLayout.box_24(max_force_n=1.0)
        result = layout.allocate(np.array([0.0, 0.0, -1.0]))
        assert result.force_body_n[2] < -0.5

    def test_forces_non_negative(self):
        layout = ThrusterLayout.box_24()
        result = layout.allocate(np.array([0.5, -0.3, 0.1]))
        assert np.all(result.forces_n >= -1e-9)

    def test_forces_within_max(self):
        layout = ThrusterLayout.box_24(max_force_n=0.25)
        result = layout.allocate(np.array([1.0, 1.0, 1.0]))
        assert np.all(result.forces_n <= 0.25 + 1e-9)

    def test_excluded_thruster_is_zero(self):
        layout = ThrusterLayout.box_24()
        excluded = np.zeros(24, dtype=bool)
        excluded[0] = True
        result = layout.allocate(np.array([1.0, 0.0, 0.0]), excluded=excluded)
        assert result.forces_n[0] == pytest.approx(0.0)

    def test_zero_request_gives_zero_force(self):
        layout = ThrusterLayout.box_24()
        result = layout.allocate(np.zeros(3))
        assert np.allclose(result.force_body_n, np.zeros(3), atol=1e-9)


class TestBox16:
    def test_thruster_count(self):
        assert len(ThrusterLayout.box_16().directions_body) == 16

    def test_poor_z_authority(self):
        # box_16 has no ±Z thrusters; Z residual should be large
        layout = ThrusterLayout.box_16(max_force_n=1.0)
        result = layout.allocate(np.array([0.0, 0.0, -1.0]))
        assert abs(result.residual[2]) > 0.5

    def test_box24_beats_box16_on_z(self):
        b16 = ThrusterLayout.box_16(max_force_n=1.0)
        b24 = ThrusterLayout.box_24(max_force_n=1.0)
        req = np.array([0.0, 0.0, -1.0])
        assert abs(b24.allocate(req).residual[2]) < abs(b16.allocate(req).residual[2])


class TestChiefMask:
    def test_blocks_thrusters_toward_chief(self):
        layout = ThrusterLayout.box_24()
        mask = layout.chief_mask(np.array([1.0, 0.0, 0.0]))
        assert np.any(mask)

    def test_all_unmasked_when_no_chief_dir(self):
        layout = ThrusterLayout.box_24()
        mask = layout.chief_mask(np.array([0.0, 0.0, 0.0]))
        assert not np.any(mask)

    def test_masked_thrusters_excluded_reduces_allocation(self):
        layout = ThrusterLayout.box_24(max_force_n=1.0)
        req = np.array([1.0, 0.0, 0.0])
        free = layout.allocate(req)
        mask = layout.chief_mask(np.array([1.0, 0.0, 0.0]))
        blocked = layout.allocate(req, excluded=mask)
        # Masking thrusters pointing toward chief increases residual
        assert np.linalg.norm(blocked.residual) >= np.linalg.norm(free.residual) - 1e-9
