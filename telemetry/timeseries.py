"""Atomic ground telemetry storage shared by checkpoints and completed runs."""
import gzip
import json
from pathlib import Path
import numpy as np
from utils.atomic_io import atomic_replace
from telemetry.geo_run import clean


def save_timeseries(path,telemetry):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    arrays,irregular={},{}
    for key,values in telemetry.items():
        try:
            array=np.asarray(values)
            if array.dtype.hasobject:raise ValueError('irregular telemetry')
            arrays[key]=array
        except ValueError:irregular[key]=clean(values)
    temporary=path.with_suffix('.npz.tmp')
    with temporary.open('wb') as handle:np.savez_compressed(handle,**arrays)
    atomic_replace(temporary,path)
    result=dict(telemetry_file=path.name)
    if irregular:
        extra=path.with_suffix('.json.gz');temporary=extra.with_suffix('.gz.tmp')
        with gzip.open(temporary,'wt',encoding='utf-8') as handle:json.dump(irregular,handle,allow_nan=False)
        atomic_replace(temporary,extra)
        result['irregular_telemetry_file']=extra.name
    return result
