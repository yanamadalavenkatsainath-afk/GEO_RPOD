"""Common scheduled interface for vector and independent-point sensors."""
from dataclasses import dataclass

from spec import Measurement, Timestamp
from utils.random_streams import RandomStreams
from utils.sampling import SampleClock, SensorTiming
from utils.validation import text_label


@dataclass
class Readout:
    value: object = None
    covariance: object = None
    valid: bool = True
    reason: str = "OK"


class Sensor:
    """Adapter: sample(time, **truth_inputs) -> a documented measurement.

    The observer is simulator-side. An estimator receives only the resulting
    record, never this object or the truth inputs. Model and dropout RNG streams
    are independent. Reset reconstructs the model and replays its random stream.
    """
    def __init__(self, *, source_id, kind, frame, units, timing, model_factory,
                 observer, seed=0, error_units=None, error_frame=None, record_type=Measurement):
        for name, value in (("source_id", source_id), ("kind", kind), ("frame", frame)):
            text_label(value, name)
        if not isinstance(timing, SensorTiming):
            raise ValueError("timing must be SensorTiming")
        if not isinstance(record_type, type) or not issubclass(record_type, Measurement):
            raise ValueError("record_type must be a Measurement type")
        self.source_id, self.kind, self.frame = source_id, kind, frame
        self.units, self.error_units = tuple(units), error_units
        self.error_frame = error_frame
        self.timing, self.record_type = timing, record_type
        self._model_factory, self._observer, self._seed = model_factory, observer, seed
        self.reset()

    def reset(self, *, seed=None):
        if seed is not None:
            self._seed = seed
        self._streams = RandomStreams(self._seed)
        self._clock = SampleClock(self.timing)
        self.model = self._model_factory(self._streams.generator(self.source_id + "/model"))
        self._dropout_rng = self._streams.generator(self.source_id + "/dropout")

    def sample(self, time: Timestamp, *, enabled=True, **inputs):
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be boolean")
        due = self._clock.due(time)
        if not enabled:
            readout = Readout(valid=False, reason="DISABLED")
        elif not due:
            readout = Readout(valid=False, reason="NOT_DUE")
        elif self.timing.dropout_probability > 0 and self._dropout_rng.random() < self.timing.dropout_probability:
            readout = Readout(valid=False, reason="DROPOUT")
        else:
            readout = self._observer(self.model, time, **inputs)
            if not isinstance(readout, Readout):
                raise TypeError("Sensor observer must return a Readout")
        record=self.record_type(source_id=self.source_id, kind=self.kind, time=time,
            frame=self.frame, units=self.units, error_units=self.error_units, error_frame=self.error_frame,
            value=readout.value, covariance=readout.covariance, valid=readout.valid,
            reason=readout.reason, available_at=time.shifted(self.timing.latency_s))
        if due:
            from telemetry.component_events import emit_component_event
            extras={name:getattr(self.model,name) for name in ('last_pixels','last_ids','mount') if hasattr(self.model,name)}
            emit_component_event('sensor',record,model=type(self.model).__name__,**extras)
        return record
