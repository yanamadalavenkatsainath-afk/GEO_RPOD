"""Mission-inspired qualification around the same shared RPOD mission engine."""
from mission_specifics.rpod.wrapper import run_shared


def describe():
    return dict(mission='master_rpod',status='experimental',engine='mission_specifics.rpod.wrapper.run_shared',
        implemented='Fresh-navigation/actuation handoff qualification and 5-second stand-off hold',
        pending='Integrated hold/retreat/abort fault trajectories and complete operational verification')


def run_single(*,seed=0,duration_s=2400.,artifacts_output=None,**parameters):
    if 'mission_inspired' in parameters:raise ValueError('Master supervisor mode cannot be overridden')
    return run_shared(seed=seed,duration_s=duration_s,artifacts_output=artifacts_output,mission_inspired=True,**parameters)


def run_terminal_window(*,seed=1,duration_s=180.,orbit='leo',artifacts_output=None):
    """Fresh terminal-only reproduction; not a restart from historical telemetry."""
    from fsw.rpod import RPODMission
    from mission_specifics.rpod.wrapper import configuration
    from mission_specifics.benchmarks import report
    from telemetry.geo_run import atomic_json
    from pathlib import Path
    mission=RPODMission(configuration(orbit,seed,duration_s))
    folder=None if artifacts_output is None else Path(artifacts_output)
    if folder is not None:folder.mkdir(parents=True,exist_ok=False)
    def checkpoint(current):
        if folder is not None:atomic_json(folder/'checkpoint.json',current.summary())
    result=mission.run(checkpoint)
    if folder is not None:atomic_json(folder/'terminal_records.json',mission.records)
    return report('master_rpod_terminal_diagnostic',seed,mission.t,result,result['criteria'],{},
        ['Fresh nominal terminal acquisition near 14 m; not historical full-state restart',
         'Same orbit, Cartesian/J2 plant, allocated RCS, measured attitude, mapped camera and contact model',
         'No inherited far-leg hardware state, filter history or independent range instrument; diagnostic only'],
        {},dict(orbit=orbit,scope='bounded terminal reproduction'))


def execution_profiles():return {'full':run_single,'terminal-window':run_terminal_window}
DEFAULT_SINGLE_PROFILE='full'
DEFAULT_MC_PROFILE='full'
