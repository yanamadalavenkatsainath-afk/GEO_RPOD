"""Ground-only entry/logging helpers for migrated legacy scenario wrappers."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr,ExitStack
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import traceback

from telemetry.geo_run import atomic_json, clean


def validate(seed, duration_s):
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if not math.isfinite(duration_s) or duration_s < .1:
        raise ValueError("duration must be finite and >= 0.1 s")


def source_hashes():
    """Fingerprint the active source subset, including render/native assets."""
    from telemetry.provenance import collect_sources
    return collect_sources()


def _trial(function, seed, duration_s, log_path, save_original_plots=False, component_events=False, configuration=None):
    # Each worker owns a separate line-buffered log, durable during execution.
    with Path(log_path).open('w',encoding='utf-8',buffering=1) as stream:
        try:
            with redirect_stdout(stream), redirect_stderr(stream),ExitStack() as event_stack:
                if component_events:
                    from telemetry.component_events import ComponentEventLog
                    event_stack.enter_context(ComponentEventLog(Path(log_path).with_suffix('')/'component_events',
                        metadata=dict(seed=seed,runner=function.__module__+'.'+getattr(function,'__name__',type(function).__name__))))
                options=dict(configuration or {})
                if save_original_plots:options['plots_output']=Path(log_path).parent/(Path(log_path).stem+'_plots')
                import inspect
                if 'artifacts_output' in inspect.signature(function).parameters:
                    options['artifacts_output']=Path(log_path).parent/(Path(log_path).stem+'_artifacts')
                result = function(seed=seed, duration_s=duration_s, **options)
            # Keep full histories on disk rather than accumulating every
            # trajectory in the parent process and campaign summary JSON.
            telemetry = result.pop('telemetry', None)
            if telemetry is not None:
                import gzip
                import numpy as np
                arrays, irregular = {}, {}
                for key, values in telemetry.items():
                    try:
                        array = np.asarray(values)
                        if array.dtype.hasobject:
                            raise ValueError('ragged telemetry')
                        arrays[key] = array
                    except ValueError:
                        irregular[key] = clean(values)
                trace = Path(log_path).with_suffix('.npz')
                temporary = trace.with_suffix('.npz.tmp')
                with temporary.open('wb') as handle:
                    np.savez_compressed(handle, **arrays)
                temporary.replace(trace)
                result['telemetry_file'] = trace.name
                if irregular:
                    extra = trace.with_suffix('.json.gz')
                    with gzip.open(extra, 'wt', encoding='utf-8') as handle:
                        json.dump(irregular, handle, allow_nan=False)
                    result['irregular_telemetry_file'] = extra.name
            return dict(seed=seed, result=clean(result), error=None)
        except (Exception,KeyboardInterrupt):
            error=traceback.format_exc()
            stream.write(error)
            return dict(seed=seed, result=None, error=error)


def campaign(function, *, runs=300, seed=0, workers=1, duration_s=1800., output=None,
             save_original_plots=False, component_events=False,resume=False,retry_failed=False,
             configuration=None):
    """Durable trial-boundary resume; interrupted trials restart from initial state.

    This is deliberately not mid-simulation state restoration. Every attempt
    has its own log/artifacts. Source/configuration changes reject resume.
    """
    validate(seed,duration_s)
    import inspect
    configuration={} if configuration is None else dict(configuration)
    reserved={'seed','duration_s','artifacts_output','plots_output'}
    if reserved.intersection(configuration):raise ValueError('Runner-owned parameters cannot be overridden in configuration')
    # Validate before creating result folders or launching any workers.
    inspect.signature(function).bind(seed=seed,duration_s=duration_s,**configuration)
    if any(isinstance(v,bool) or not isinstance(v,int) or v<1 for v in (runs,workers)) or seed+runs>2**32:
        raise ValueError('Require positive integer runs/workers and seeds below 2**32')
    if retry_failed and not resume:raise ValueError('Retrying failed trials requires resume')
    if resume and output is None:raise ValueError('Resume requires an explicit existing output folder')
    folder=Path(output) if output else Path(__file__).resolve().parent/function.__module__.split('.')[-2]/'results'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    from telemetry.provenance import archive_sources,execution_archive
    hashes=source_hashes();frozen=execution_archive()
    from dataclasses import is_dataclass,asdict
    identity=dict(schema_version=4,runner=function.__module__+'.'+getattr(function,'__qualname__',type(function).__qualname__),
        runner_parameters=clean(asdict(function)) if is_dataclass(function) else {},runs=runs,
        seed=seed,duration_s=duration_s,save_original_plots=save_original_plots,
        component_events=component_events,source_hashes=hashes,configuration=clean(configuration))
    records={}
    if resume:
        manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        if any(manifest.get(key)!=value for key,value in identity.items()):
            raise ValueError('Resume configuration, runner or source differs from the saved campaign')
        for trial_seed in range(seed,seed+runs):
            path=folder/f'trial_{trial_seed}.json'
            if path.exists():
                row=json.loads(path.read_text(encoding='utf-8'))
                if row.get('seed')!=trial_seed or ('error' not in row):raise ValueError('Invalid saved trial record: '+str(path))
                records[trial_seed]=row
        manifest.setdefault('resume_invocations',[]).append(dict(time=datetime.now().isoformat(),workers=workers,retry_failed=retry_failed))
        atomic_json(folder/'manifest.json',manifest)
    else:
        if folder.exists() and any(folder.iterdir()):raise ValueError('Output folder must be empty: '+str(folder))
        folder.mkdir(parents=True,exist_ok=True)
        if frozen is None:archive_sources(folder/'source',expected=hashes)
        atomic_json(folder/'manifest.json',dict(**identity,workers=workers,
            dispersion='Wrapper-specific declared dispersions; run count is not qualification',
            source_archive=frozen if frozen else 'source',execution_from_snapshot=frozen is not None,
            resume_scope='Completed trial records retained; interrupted trials restart in a new attempt folder'))
    print('Outputs:',folder.resolve(),flush=True)
    def save(record,log_path):
        previous=records.get(record['seed'])
        if previous is not None:
            record['previous_attempts']=previous.get('previous_attempts',[])+[{k:v for k,v in previous.items() if k!='previous_attempts'}]
        record['log_file']=str(log_path.relative_to(folder))
        result=record.get('result') or {}
        for key in ('telemetry_file','irregular_telemetry_file'):
            if key in result:result[key]=str(log_path.parent.relative_to(folder)/result[key])
        atomic_json(folder/f"trial_{record['seed']}.json",record)
        records[record['seed']]=record
        ordered=sorted(records.values(),key=lambda row:row['seed'])
        outcomes=[(row.get('result') or {}).get('success') for row in ordered]
        atomic_json(folder/'summary.json',dict(completed=len(records),requested=runs,
            errors=sum(row['error'] is not None for row in ordered),
            mission_passes=sum(value is True for value in outcomes),
            mission_failures=sum(value is False for value in outcomes),
            mission_unevaluated=sum(value is None for value in outcomes),trials=ordered))
        with (folder/'run.log').open('a',encoding='utf-8') as stream:
            stream.write(log_path.read_text(encoding='utf-8'))
            stream.write(f"Trial seed={record['seed']} completed; error={record['error'] is not None}\n")
        print(f"[{len(records)}/{runs}] seed={record['seed']} error={record['error'] is not None} "
              f"mission_success={result.get('success','UNEVALUATED')}",flush=True)
        if result.get('plot_metrics'):
            try:
                from telemetry.benchmark_plots import write_benchmark_plots
                write_benchmark_plots(folder,ordered)
                atomic_json(folder/'plot_status.json',dict(ok=True))
            except Exception:atomic_json(folder/'plot_status.json',dict(ok=False,error=traceback.format_exc()))
    pending=[]
    for trial_seed in range(seed,seed+runs):
        if trial_seed in records and (records[trial_seed]['error'] is None or not retry_failed):continue
        attempt=0
        while (folder/f'trial_{trial_seed}_attempt_{attempt:03d}').exists():attempt+=1
        attempt_folder=folder/f'trial_{trial_seed}_attempt_{attempt:03d}'
        attempt_folder.mkdir()
        pending.append((trial_seed,attempt_folder/f'trial_{trial_seed}.log'))
    if workers==1:
        for trial_seed,log_path in pending:
            save(_trial(function,trial_seed,duration_s,log_path,save_original_plots,component_events,configuration),log_path)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures={pool.submit(_trial,function,s,duration_s,path,save_original_plots,component_events,configuration):path for s,path in pending}
            for future in as_completed(futures):save(future.result(),futures[future])
    return sorted(records.values(),key=lambda row:row['seed'])


def cli(wrapper, *, monte_carlo=False, default_duration=1800.):
    parser=argparse.ArgumentParser(description=wrapper.__doc__)
    parser.add_argument('--describe',action='store_true')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--duration',type=float,default=default_duration)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--parameters',type=Path,help='JSON object of explicit selected-wrapper parameters; saved in the manifest')
    parser.add_argument('--resume',action='store_true',help='Resume completed trials; interrupted trials restart from initial state')
    parser.add_argument('--retry-failed',action='store_true',help='Preserve failed attempt evidence and run a new attempt with --resume')
    parser.add_argument('--component-events',action='store_true',help='Save chunked raw measurements, covariance, pixel data and actuator records')
    parser.add_argument('--original-plots',action='store_true',help='Save retained original plots where supported')
    if monte_carlo:
        parser.add_argument('--runs',type=int,default=300)
        parser.add_argument('--workers',type=int,default=1)
    profiles=wrapper.execution_profiles() if hasattr(wrapper,'execution_profiles') else None
    if profiles:
        default=getattr(wrapper,'DEFAULT_MC_PROFILE' if monte_carlo else 'DEFAULT_SINGLE_PROFILE')
        parser.add_argument('--profile',choices=tuple(profiles),default=default)
    args=parser.parse_args()
    if args.describe:
        print(json.dumps(wrapper.describe(),indent=2))
        return
    function=profiles[args.profile] if profiles else wrapper.run_single
    if args.original_plots:
        import inspect
        if 'plots_output' not in inspect.signature(function).parameters:
            parser.error('The selected profile does not provide original plots')
    try:
        configuration=json.loads(args.parameters.read_text(encoding='utf-8-sig')) if args.parameters else {}
        if not isinstance(configuration,dict):raise ValueError('Parameters must be a JSON object')
        rows=campaign(profiles[args.profile] if profiles else wrapper.run_single,seed=args.seed,duration_s=args.duration,output=args.output,
            runs=args.runs if monte_carlo else 1,workers=args.workers if monte_carlo else 1,save_original_plots=args.original_plots,component_events=args.component_events,resume=args.resume,retry_failed=args.retry_failed,configuration=configuration)
    except (ValueError,TypeError,OSError) as error:
        parser.error(str(error))
    if any(row['error'] for row in rows):
        raise SystemExit("One or more trials failed; see saved logs")
