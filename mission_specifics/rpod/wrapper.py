"""One experimental physical RPOD workflow with LEO/GEO orbit presets."""
from dataclasses import asdict
from pathlib import Path
import numpy as np
from fsw.rpod import RPODConfig,RPODMission
from fsw.rpod_supervisor import HandoffSupervisor
from estimation.rpod_relative_navigation import RelativeNavigationScenario
from mission_specifics.benchmarks import report
from telemetry.geo_run import atomic_json
from telemetry.timeseries import save_timeseries
from utils.validation import finite_scalar

ORBIT_PRESETS={'leo':dict(chief_radius_m=6781e3,chief_inclination_deg=51.6),
               'geo':dict(chief_radius_m=42164e3,chief_inclination_deg=0.)}


def describe():
    return dict(mission='rpod',status='experimental',orbits=ORBIT_PRESETS,
        engine='Single Cartesian/finite-Lambert/achieved-RCS/sensor-attitude/mapped-pixel/compliant-contact path',
        limitation='GEO is a new configuration of the shared engine, not parity with the heritage GEO acquisition pipeline')


def configuration(orbit,seed,duration_s,chief_radius_m=None,chief_inclination_deg=None):
    if orbit not in ORBIT_PRESETS:raise ValueError('orbit must be leo or geo')
    values=dict(ORBIT_PRESETS[orbit])
    if chief_radius_m is not None:values['chief_radius_m']=chief_radius_m
    if chief_inclination_deg is not None:values['chief_inclination_deg']=chief_inclination_deg
    return RPODConfig(seed=seed,duration=duration_s,truth_model='cartesian_j2',actuation_model='allocated_rcs',
        attitude_model='sensor_mekf',vision_model='mapped_pixels',contact_model='compliant',environment_model='earth_plates',**values)


def run_shared(*,seed=0,duration_s=2400.,orbit='leo',far_duration_s=None,initial_separation_m=100.,
               chief_radius_m=None,chief_inclination_deg=None,artifacts_output=None,mission_inspired=False):
    cfg=configuration(orbit,seed,duration_s,chief_radius_m,chief_inclination_deg)
    finite_scalar(initial_separation_m,'initial_separation_m',strictly_positive=True)
    period=2*np.pi*np.sqrt(cfg.chief_radius_m**3/3.986004418e14)
    transfer=2700.*(cfg.chief_radius_m/6781e3)**1.5
    far_duration=float(far_duration_s) if far_duration_s is not None else 4500.*period/(2*np.pi*np.sqrt(6781e3**3/3.986004418e14))
    finite_scalar(far_duration,'far_duration_s',minimum=.5)
    if abs(far_duration/.5-round(far_duration/.5))>1e-9:
        far_duration=.5*np.ceil(far_duration/.5)
    folder=Path(artifacts_output) if artifacts_output is not None else None
    if folder is not None:folder.mkdir(parents=True,exist_ok=False)
    scenario=RelativeNavigationScenario(dt=.5,chief_radius_km=cfg.chief_radius_m/1000,
        chief_inclination_deg=cfg.chief_inclination_deg,formation_offset=[0,initial_separation_m,0],
        seed=seed,truth_model='cartesian_j2',physical_config=cfg,transfer_time_s=transfer)
    supervisor=HandoffSupervisor() if mission_inspired else None
    trace={k:[] for k in ('t','truth_state','estimated_state','phase')}
    events=[];last_fix=0.;updates=0;handoff=False;elapsed=0.;terminal=None
    try:
        for _ in range(60):scenario.coast()
        scenario.activate()
        for step in range(int(round(far_duration/.5))):
            t=step*.5
            state=scenario.step(t,allow_rendezvous=t>=120.,stop_time=far_duration)
            estimate=scenario.roe_ekf._roe_to_lvlh(scenario.roe_ekf.x)
            elapsed=t+.5
            if scenario.roe_updates>updates:last_fix=elapsed;updates=scenario.roe_updates
            no_burn=scenario.finite_transfer.active is None
            inside=np.linalg.norm(estimate[:3])<20. and np.linalg.norm(estimate[3:])<.15
            if supervisor is None:handoff=bool(inside and no_burn)
            else:
                handoff=supervisor.update(estimate,.5,navigation_fresh=elapsed-last_fix<=2.,
                    actuation_ready=scenario.physical_attitude_chain.healthy and scenario.physical_actuation_health.ready,
                    burn_active=not no_burn)
                if supervisor.phase=='HANDOFF_HOLD':
                    # Hold at the declared stand-off using estimated state and
                    # the same bounded physical actuation, not a state reset.
                    scenario.cleanup_active=True;scenario.cleanup_ctrl.target=np.array([0.,12.,0.])
                    if scenario.lambert_plan is not None:scenario.lambert_plan['arrival_issued']=True
            phase=supervisor.phase if supervisor is not None else 'FAR_RENDEZVOUS'
            if step%20==0 or handoff:
                for key,value in zip(trace,(elapsed,state.copy(),estimate.copy(),phase)):trace[key].append(value)
            if step%120==0 or handoff:
                event=dict(t_s=elapsed,phase=phase,estimated_range_m=float(np.linalg.norm(estimate[:3])),
                    handoff_hold_s=0. if supervisor is None else supervisor.held)
                events.append(event);print(event,flush=True)
                if folder is not None:
                    atomic_json(folder/'checkpoint.json',dict(status='running',orbit=orbit,event=event,far=scenario.summary()))
                    save_timeseries(folder/'far_partial.npz',trace)
            if handoff:break
        if handoff:
            terminal=RPODMission(cfg,scenario)
            def checkpoint(current):
                if folder is not None:atomic_json(folder/'checkpoint.json',dict(status='running',orbit=orbit,terminal=current.summary()))
            terminal_result=terminal.run(checkpoint)
        else:terminal_result=None
    finally:
        if folder is not None:
            save_timeseries(folder/'far_partial.npz',trace)
            if terminal is not None:atomic_json(folder/'terminal_records.json',terminal.records)
    metrics=dict(orbit=orbit,orbital_period_s=float(period),far=scenario.summary(),terminal=terminal_result,
        supervisor_hold_s=None if supervisor is None else supervisor.held)
    criteria=dict(far_handoff=handoff,terminal_capture=terminal_result is not None and terminal_result['success'])
    return report('master_rpod' if mission_inspired else 'rpod',seed,elapsed+(0. if terminal is None else terminal.t),
        metrics,criteria,trace,
        ['One shared Earth-orbit RPOD engine; LEO/GEO change declared orbit and scaled transfer timing',
         'Same illustrative vehicle, target and sensors for both presets; not heritage GEO pipeline parity',
         'Master mode initially adds fresh-navigation/actuation qualification and a 5 s estimated stand-off hold',
         'Complete mission-inspired retreat/abort trajectory enforcement remains to be integrated; component checks are separate'],
        {},dict(orbit=orbit,far_duration_s=float(far_duration),transfer_time_s=float(transfer),mission_inspired=mission_inspired,vehicle=asdict(cfg)))


def run_single(*,seed=0,duration_s=2400.,artifacts_output=None,**parameters):
    if 'mission_inspired' in parameters:raise ValueError('Select the master_rpod entry point for supervisor mode')
    return run_shared(seed=seed,duration_s=duration_s,artifacts_output=artifacts_output,**parameters)
