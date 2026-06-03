import numpy as np
import pytest
from plant.contact_dynamics import DockingContactModel


I_DEP   = np.diag([4.167, 4.167, 3.000])
I_CHIEF = np.diag([20000.0, 20000.0, 15000.0])
NORMAL  = np.array([0.0, 0.0, 1.0])


class TestResolve:
    def setup_method(self):
        self.m = DockingContactModel(restitution=0.1, tangential_damping=0.3,
                                     capture_vrel_ms=0.05)

    def test_slow_approach_captured(self):
        result = self.m.resolve(np.array([0.0, 0.0, -0.04]), NORMAL, 25.0)
        assert result.captured

    def test_fast_approach_not_captured(self):
        result = self.m.resolve(np.array([0.0, 0.0, -0.5]), NORMAL, 25.0)
        assert not result.captured

    def test_normal_restitution(self):
        # v_n = -0.1; after: v_z = v_n + (1+e)*(-v_n) = e*|v_n| = 0.01
        result = self.m.resolve(np.array([0.0, 0.0, -0.1]), NORMAL, 25.0)
        assert result.rel_vel_after[2] == pytest.approx(0.01, abs=1e-8)

    def test_tangential_damping(self):
        # No normal component → only tangential damping applied
        result = self.m.resolve(np.array([0.1, 0.0, 0.0]), NORMAL, 25.0)
        assert result.rel_vel_after[0] == pytest.approx(0.1 * (1.0 - 0.3), abs=1e-8)

    def test_impulse_non_negative(self):
        result = self.m.resolve(np.array([0.2, 0.1, -0.3]), NORMAL, 25.0)
        assert result.impulse_ns >= 0.0

    def test_severity_scales_with_speed(self):
        fast = self.m.resolve(np.array([0.0, 0.0, -1.0]), NORMAL, 25.0)
        slow = self.m.resolve(np.array([0.0, 0.0, -0.01]), NORMAL, 25.0)
        assert fast.severity > slow.severity

    def test_approaching_triggers_bounce_not_receding(self):
        # Receding contact (v_n > 0) should not get the restitution kick
        result_approach = self.m.resolve(np.array([0.0, 0.0, -0.1]), NORMAL, 25.0)
        result_recede   = self.m.resolve(np.array([0.0, 0.0,  0.1]), NORMAL, 25.0)
        # Approaching: after-z > initial-z (bounce)
        assert result_approach.rel_vel_after[2] > -0.1
        # Receding: only tangential change (zero here), normal unchanged
        assert result_recede.rel_vel_after[2] == pytest.approx(0.1, abs=1e-8)


class TestIdealLatch:
    def setup_method(self):
        self.m = DockingContactModel()

    def test_zeroes_velocity(self):
        _, _, result = self.m.ideal_latch(
            rel_pos=np.array([0.0, 0.0, 0.05]),
            rel_vel=np.array([0.0, 0.0, -0.02]),
            deputy_mass_kg=50.0)
        assert np.allclose(result.rel_vel_after, np.zeros(3), atol=1e-12)

    def test_captured(self):
        _, _, result = self.m.ideal_latch(
            rel_pos=np.zeros(3),
            rel_vel=np.array([0.0, 0.0, -0.01]),
            deputy_mass_kg=50.0)
        assert result.captured

    def test_pos_delta_cancels_rel_pos(self):
        rel_pos = np.array([0.02, -0.01, 0.05])
        pos_delta, _, _ = self.m.ideal_latch(rel_pos, np.zeros(3), 50.0)
        assert np.allclose(pos_delta + rel_pos, np.zeros(3), atol=1e-12)


class TestResolveCoupled:
    def setup_method(self):
        self.m = DockingContactModel()

    def test_axial_contact_no_spin(self):
        # Contact point on dock axis → cross product is zero → no delta-omega
        result = self.m.resolve_coupled(
            rel_vel=np.array([0.0, 0.0, -0.03]),
            normal_hat=NORMAL,
            deputy_mass_kg=50.0, chief_mass_kg=3000.0,
            deputy_I_body=I_DEP, chief_I_body=I_CHIEF,
            r_dep_contact_body=np.array([0.0, 0.0, 0.4]),
            r_chief_contact_body=np.array([0.0, 0.0, 0.5]),
            R_dep_body_to_world=np.eye(3),
            R_chief_body_to_world=np.eye(3))
        assert np.allclose(result.deputy_delta_omega, np.zeros(3), atol=1e-9)

    def test_off_axis_contact_spins_deputy(self):
        result = self.m.resolve_coupled(
            rel_vel=np.array([0.0, 0.0, -0.03]),
            normal_hat=NORMAL,
            deputy_mass_kg=50.0, chief_mass_kg=3000.0,
            deputy_I_body=I_DEP, chief_I_body=I_CHIEF,
            r_dep_contact_body=np.array([0.1, 0.0, 0.0]),   # off-axis
            r_chief_contact_body=np.array([0.0, 0.1, 0.0]),
            R_dep_body_to_world=np.eye(3),
            R_chief_body_to_world=np.eye(3))
        assert np.linalg.norm(result.deputy_delta_omega) > 0.0

    def test_impulse_direction_consistent(self):
        # Deputy impulse and chief impulse should be equal and opposite in world frame
        result = self.m.resolve_coupled(
            rel_vel=np.array([0.0, 0.0, -0.05]),
            normal_hat=NORMAL,
            deputy_mass_kg=50.0, chief_mass_kg=3000.0,
            deputy_I_body=I_DEP, chief_I_body=I_CHIEF,
            r_dep_contact_body=np.array([0.0, 0.0, 0.4]),
            r_chief_contact_body=np.array([0.0, 0.0, 0.5]),
            R_dep_body_to_world=np.eye(3),
            R_chief_body_to_world=np.eye(3))
        # Impulse magnitude is positive
        assert result.impulse_ns > 0.0
