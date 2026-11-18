"""Versioned JSONL component-event log with explicit provenance and reader.

Replays data records, not Python objects or mission execution. Source files
are hashed only when explicitly supplied; no directory scan or secret capture.
"""
from dataclasses import fields, is_dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path

import numpy as np

from spec import Timestamp
from utils.validation import text_label


def _json_value(value):
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {"record_type": type(value).__name__, **{f.name: _json_value(getattr(value, f.name)) for f in fields(value)}}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("Log dictionaries require string keys")
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ValueError(f"Unsupported log type: {type(value).__name__}")


class ComponentLog:
    """Create a new file exclusively; flush every record for visible run logs."""
    def __init__(self, path, *, configuration, source_files=()):
        self.path = Path(path)
        sources = []
        for filename in source_files:
            source = Path(filename).resolve(strict=True)
            sources.append({"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
        header = {"schema": "gnc.component-log", "version": 1, "type": "manifest",
                  "configuration": _json_value(configuration), "sources": sources}
        line = json.dumps(header, allow_nan=False, sort_keys=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("x", encoding="utf-8")
        self.file.write(line+"\n")
        self.file.flush()
        self.sequence = 0
        self._last_time = None

    def record(self, *, time, source_id, event, data):
        if not isinstance(time, Timestamp):
            raise ValueError("time must be a Timestamp")
        text_label(source_id, "source_id")
        text_label(event, "event")
        if self._last_time is not None and time.seconds_since(self._last_time) < 0:
            raise ValueError("Log delivery/event times must be nondecreasing in one epoch")
        # Payload keeps acquisition and availability times; event time is when
        # the wrapper processes/records it. Dropouts are logged like any event.
        record = {"type": "event", "sequence": self.sequence, "time": _json_value(time),
                  "source_id": source_id, "event": event, "data": _json_value(data)}
        line = json.dumps(record, allow_nan=False, sort_keys=True)
        self.file.write(line+"\n")
        self.file.flush()
        self.sequence += 1
        self._last_time = time

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def read_component_log(path):
    """Yield decoded dictionaries; reject wrong schema, sequence, or truncation."""
    with Path(path).open(encoding="utf-8") as stream:
        header = json.loads(next(stream))
        if header.get("schema") != "gnc.component-log" or header.get("version") != 1 or header.get("type") != "manifest":
            raise ValueError("Unsupported component log schema")
        yield header
        for sequence, line in enumerate(stream):
            record = json.loads(line)
            if record.get("type") != "event" or record.get("sequence") != sequence:
                raise ValueError("Component log event sequence is invalid")
            yield record
