import unittest
import numpy as np
from fsw.rpod import RPODMission
from fsw.leo_rpod import NonCooperativeRPOD
from fsw.rpod_supervisor import HandoffSupervisor
from mission_specifics.rpod.wrapper import configuration
from estimation.rpod_relative_navigation import RelativeNavigationScenario


class UnifiedRPODTests(unittest.TestCase):
    def test_one_engine_initializes_both_orbits(self):
        for orbit,radius in [('leo',6781e3),('geo',42164e3)]:
            scenario=RPODMission(configuration(orbit,0,1.))
            self.assertAlmostEqual(np.linalg.norm(scenario.orbit_truth.chief_position),radius)
            self.assertEqual(type(scenario),RPODMission)

    def test_chief_inclination_and_far_model_are_configured(self):
        for orbit in ['leo','geo']:
            cfg=configuration(orbit,0,1.)
            scenario=RelativeNavigationScenario(dt=.5,chief_radius_km=cfg.chief_radius_m/1000,
                chief_inclination_deg=cfg.chief_inclination_deg,truth_model='cartesian_j2',physical_config=cfg)
            self.assertAlmostEqual(scenario.truth.a,cfg.chief_radius_m)
            self.assertAlmostEqual(scenario.truth.i,np.radians(cfg.chief_inclination_deg))

    def test_stand_off_hold_resets_when_navigation_is_stale(self):
        gate=HandoffSupervisor(hold_s=2)
        args=dict(navigation_fresh=True,actuation_ready=True,burn_active=False)
        state=[0,12,0,0,0,0]
        self.assertFalse(gate.update(state,1,**args))
        self.assertFalse(gate.update(state,1,**dict(args,navigation_fresh=False)))
        self.assertFalse(gate.update(state,1,**args))
        self.assertTrue(gate.update(state,1,**args))

    def test_handoff_rejects_active_burn_or_unhealthy_actuation(self):
        gate=HandoffSupervisor(hold_s=1)
        self.assertFalse(gate.update([0,12,0,0,0,0],1,navigation_fresh=True,actuation_ready=True,burn_active=True))
        self.assertFalse(gate.update([0,12,0,0,0,0],1,navigation_fresh=True,actuation_ready=False,burn_active=False))

    def test_legacy_alias_and_orbit_validation(self):
        self.assertIs(NonCooperativeRPOD,RPODMission)
        with self.assertRaises(ValueError):configuration('unknown',0,1.)
        with self.assertRaises(ValueError):configuration('leo',0,1.,chief_radius_m=1.)
