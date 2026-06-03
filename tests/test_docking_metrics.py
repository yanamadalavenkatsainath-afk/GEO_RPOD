import numpy as np
import pytest
import sim_config as cfg
from utils.docking_metrics import docking_alignment_metrics, docking_geometry_metrics


class TestAlignmentMetrics:
    def test_perfect_alignment(self):
        # DEP_DOCK_AXIS_BODY = [0,0,1]; desired = -port_axis = [0,0,1] → 0°
        result = docking_alignment_metrics(np.eye(3), np.array([0.0, 0.0, -1.0]))
        assert result["align_deg"] == pytest.approx(0.0, abs=1e-6)
        assert result["ok"]

    def test_180_deg_misalignment(self):
        R = np.diag([1.0, 1.0, -1.0])   # flips Z axis
        result = docking_alignment_metrics(R, np.array([0.0, 0.0, -1.0]))
        assert result["align_deg"] == pytest.approx(180.0, abs=1e-4)
        assert not result["ok"]

    def test_exactly_at_threshold_is_ok(self):
        a = np.radians(cfg.DOCK_ALIGN_MAX_DEG)
        R = np.array([[1, 0, 0],
                      [0, np.cos(a), -np.sin(a)],
                      [0, np.sin(a),  np.cos(a)]])
        result = docking_alignment_metrics(R, np.array([0.0, 0.0, -1.0]))
        assert result["align_deg"] == pytest.approx(cfg.DOCK_ALIGN_MAX_DEG, abs=1e-3)
        assert result["ok"]

    def test_over_threshold_not_ok(self):
        a = np.radians(cfg.DOCK_ALIGN_MAX_DEG + 1.0)
        R = np.array([[1, 0, 0],
                      [0, np.cos(a), -np.sin(a)],
                      [0, np.sin(a),  np.cos(a)]])
        result = docking_alignment_metrics(R, np.array([0.0, 0.0, -1.0]))
        assert not result["ok"]

    def test_90_deg_misalignment(self):
        # Rotate 90° around X: Z body → Y_lvlh
        R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        result = docking_alignment_metrics(R, np.array([0.0, 0.0, -1.0]))
        assert result["align_deg"] == pytest.approx(90.0, abs=1e-4)


class TestGeometryMetrics:
    def test_on_axis_approaching_is_ok(self):
        # Deputy at [0, 0, 2]; port at [0, 0, 0.5] → port_to_dep = [0,0,1.5], in cone
        result = docking_geometry_metrics(np.array([0.0, 0.0, 2.0]), np.eye(3))
        assert result["in_aperture"]
        assert result["cone_ok"]
        assert result["body_clear"]
        assert result["ok"]

    def test_large_lateral_not_in_aperture(self):
        result = docking_geometry_metrics(np.array([1.0, 0.0, 2.0]), np.eye(3))
        assert not result["in_aperture"]
        assert not result["ok"]

    def test_capture_core_when_very_close(self):
        # Deputy at port + tiny epsilon: [0, 0, 0.51]
        result = docking_geometry_metrics(np.array([0.0, 0.0, 0.51]), np.eye(3))
        assert result["capture_core"]
        assert result["cone_error_deg"] == pytest.approx(0.0, abs=1e-9)

    def test_inside_chief_body_not_clear(self):
        # At chief center, not on dock face → body_clear = False
        result = docking_geometry_metrics(np.array([0.0, 0.0, 0.0]), np.eye(3))
        assert not result["body_clear"]
        assert not result["ok"]

    def test_axial_output_matches_geometry(self):
        dep = np.array([0.0, 0.0, 2.0])
        result = docking_geometry_metrics(dep, np.eye(3))
        # axial = dep_z - port_z = 2.0 - 0.5 = 1.5
        assert result["axial_m"] == pytest.approx(1.5, abs=1e-9)
        assert result["lateral_m"] == pytest.approx(0.0, abs=1e-9)

    def test_cone_error_zero_on_axis(self):
        result = docking_geometry_metrics(np.array([0.0, 0.0, 1.0]), np.eye(3))
        assert result["cone_error_deg"] == pytest.approx(0.0, abs=1e-9)
