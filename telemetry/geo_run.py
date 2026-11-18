"""Ground-only artifacts and progress; never imported by flight algorithms."""
from dataclasses import asdict
from pathlib import Path
import hashlib,json,os
import numpy as np
from utils.atomic_io import atomic_replace


def clean(value):
    if isinstance(value,dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [clean(v) for v in value]
    if isinstance(value,np.ndarray): return clean(value.tolist())
    if isinstance(value,np.generic): return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value): return None
    return value


def atomic_json(path,value):
    path=Path(path);temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(clean(value),indent=2,allow_nan=False),encoding='utf-8')
    atomic_replace(temporary,path)


def source_manifest(config):
    from telemetry.provenance import collect_sources
    return dict(config=asdict(config),schema_version=2,
                source_hashes=collect_sources(include_vision_weights=config.cnn_pose_enabled),
                frames='Chief RTN position and coordinate-derivative velocity; SI units')


class Tee:
    def __init__(self,terminal,log): self.terminal,self.log=terminal,log
    def write(self,text):
        self.terminal.write(text);self.log.write(text);self.log.flush()
        return len(text)
    def flush(self): self.terminal.flush();self.log.flush()


class RunSink:
    def __init__(self,path,config,checkpoint_s=60.):
        self.path=Path(path)
        if self.path.exists() and any(self.path.iterdir()):
            raise ValueError('Output folder is not empty: '+str(self.path))
        self.path.mkdir(parents=True,exist_ok=True)
        self.rows=[];self.last_mode=None;self.next_checkpoint=0.
        self.checkpoint_s=checkpoint_s;self.last_cloud=np.empty((0,3))
        from telemetry.provenance import archive_sources, execution_archive
        manifest=source_manifest(config)
        frozen=execution_archive()
        if frozen is None:
            archive_sources(self.path/'source',expected=manifest['source_hashes'])
        manifest.update(source_archive=frozen if frozen else 'source',execution_from_snapshot=frozen is not None)
        atomic_json(self.path/'manifest.json',manifest)

    def __call__(self,sample,cloud):
        row=asdict(sample)
        if self.rows and self.rows[-1]['time_s']==row['time_s']:
            self.rows[-1]=row
        else:
            self.rows.append(row)
        self.last_cloud=cloud
        if sample.mode!=self.last_mode or sample.time_s>=self.next_checkpoint:
            print(f'[{sample.time_s:.1f}s] {sample.adcs_mode}/{sample.mode} '
                  f'range={np.linalg.norm(sample.truth_position_m):.3f}m '
                  f'dv={sample.net_delta_v_m_s:.5f}m/s fuel={sample.modeled_propellant_g:.3f}g '
                  f'bell_conf={sample.nozzle_confidence:.2f} '
                  f'approach={sample.approach_state}/{sample.approach_reason} '
                  f'survey={sample.survey_state}/{sample.survey_reason} '
                  f'pose_age={sample.pose_age_s:.1f}s close_cam={sample.close_camera_active} '
                  f'timing_ok={sample.final_timing_ok} coverage_ok={sample.final_coverage_ok} '
                  f'align={sample.estimated_alignment_deg:.2f}deg '
                  f'rate_err={np.degrees(sample.estimated_rate_error_rad_s):.4f}deg/s',flush=True)
            self.checkpoint()
            self.next_checkpoint=sample.time_s+self.checkpoint_s
            self.last_mode=sample.mode

    def checkpoint(self):
        arrays={key:np.asarray([row[key] for row in self.rows]) for key in self.rows[0]} if self.rows else {}
        temporary=self.path/'telemetry.tmp'
        with temporary.open('wb') as handle: np.savez(handle,**arrays)
        atomic_replace(temporary,self.path/'telemetry.npz')
        temporary=self.path/'nozzle_cloud.tmp'
        with temporary.open('wb') as handle: np.savez(handle,points=self.last_cloud)
        atomic_replace(temporary,self.path/'nozzle_cloud.npz')
        atomic_json(self.path/'checkpoint.json',dict(outcome='running',last=self.rows[-1] if self.rows else None))

    def finish(self,result=None,error=None):
        self.checkpoint()
        payload=result.to_dict() if result is not None else dict(outcome='exception',success=False,error=str(error))
        atomic_json(self.path/'summary.json',payload)
        atomic_json(self.path/'checkpoint.json',dict(outcome=payload['outcome'],last=self.rows[-1] if self.rows else None))
