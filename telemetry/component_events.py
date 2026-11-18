"""Opt-in, bounded-memory component records with durable closed chunks.

This is evidence logging, not a simulation checkpoint or restart mechanism.
Actuation records include device and combined results; do not sum all records
as independent fuel use. Mission resource totals remain authoritative.
"""
from contextvars import ContextVar
from dataclasses import asdict,is_dataclass
from enum import Enum
from pathlib import Path
import gzip,hashlib,json,os
import numpy as np
from telemetry.geo_run import atomic_json

_active=ContextVar('component_event_log',default=None)


def _encode(value):
    if is_dataclass(value):return _encode(asdict(value))
    if isinstance(value,Enum):return value.value
    if isinstance(value,np.ndarray):return _encode(value.tolist())
    if isinstance(value,np.generic):return _encode(value.item())
    if isinstance(value,dict):return {str(k):_encode(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_encode(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value):return None
    if isinstance(value,Path):return str(value)
    return value


def emit_component_event(kind,record,**extra):
    sink=_active.get()
    if sink is not None:sink.emit(kind,record,**extra)


def component_logging_enabled():
    return _active.get() is not None


class ComponentEventLog:
    def __init__(self,path,*,metadata=None,chunk_records=256):
        if isinstance(chunk_records,bool) or not isinstance(chunk_records,int) or chunk_records<1:
            raise ValueError('Positive integer chunk size required')
        self.path=Path(path);self.metadata={} if metadata is None else metadata
        self.limit=chunk_records;self.files=[];self.total=0;self.count=0;self.handle=None;self.token=None

    def __enter__(self):
        self.path.mkdir(parents=True,exist_ok=False)
        self.token=_active.set(self)
        self._index('RUNNING')
        return self

    def _index(self,status,error=None):
        atomic_json(self.path/'index.json',dict(schema_version=1,status=status,error=error,
            records_in_closed_chunks=sum(item['records'] for item in self.files),chunks=self.files,metadata=_encode(self.metadata),
            scope='Component evidence only; no state resume; device and combined actuation records overlap'))

    def emit(self,kind,record,**extra):
        if self.handle is None:
            self.temporary=self.path/f'chunk_{len(self.files):06d}.jsonl.gz.tmp'
            self.handle=gzip.open(self.temporary,'wt',encoding='utf-8');self.count=0
        self.handle.write(json.dumps(_encode(dict(sequence=self.total,kind=kind,record=record,**extra)),allow_nan=False,separators=(',',':'))+'\n')
        self.total+=1;self.count+=1
        if self.count>=self.limit:self._close_chunk()

    def _close_chunk(self):
        if self.handle is None:return
        self.handle.close();self.handle=None
        with self.temporary.open('r+b') as stream:
            os.fsync(stream.fileno())
            digest=hashlib.file_digest(stream,'sha256').hexdigest()
        final=self.temporary.with_suffix('');self.temporary.replace(final)
        self.files.append(dict(file=final.name,records=self.count,sha256=digest))
        self._index('RUNNING')

    def __exit__(self,kind,error,traceback):
        try:
            self._close_chunk()
            self._index('COMPLETE' if error is None else 'STOPPED',None if error is None else str(error))
        finally:
            if self.token is not None:_active.reset(self.token)
