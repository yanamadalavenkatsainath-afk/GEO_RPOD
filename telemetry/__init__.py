"""Common component logging; no optional analysis libraries on import."""
from telemetry.component_log import ComponentLog, read_component_log

__all__ = ["ComponentLog", "read_component_log"]
