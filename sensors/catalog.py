"""Explicit, lazy sensor construction for future scenario wrappers."""
import numpy as np
from copy import deepcopy
from sensors.base import Readout, Sensor
from spec import PointCloudMeasurement
from utils.quaternion import quat_from_matrix
from utils.sampling import SensorTiming

def sensor_names():
    return ('landmark_pose', 'performance_gyro', 'star_tracker')

def _pair(result):
    value, noise = result
    return Readout(value, noise, value is not None, 'OK' if value is not None else 'UNAVAILABLE')

def _gyro(model, time, **inputs):
    previous = getattr(model, '_last_sample_s', None)
    dt = model.dt if previous is None else time.seconds - previous
    value = model.measure(dt=dt, **inputs)
    model._last_sample_s = time.seconds
    return Readout(value, np.eye(3) * model.sigma_arw ** 2)

def _performance_gyro(model, time, **inputs):
    readout = _gyro(model, time, **inputs)
    if model.saturated:
        return Readout(valid=False, reason='RATE_SATURATION')
    quantum = np.deg2rad(model.performance.quantization_deg_s)
    readout.covariance += np.eye(3) * quantum ** 2 / 12
    return readout

def _magnetic(model, time, **inputs):
    return Readout(model.measure(**inputs), np.eye(3) * model.sigma ** 2)

def _sun(model, time, illuminated=True, **inputs):
    if not isinstance(illuminated, bool):
        raise ValueError('illuminated must be boolean')
    if not illuminated:
        return Readout(valid=False, reason='ECLIPSE')
    value = model.measure(**inputs)
    return Readout(value, model.sigma ** 2 * (np.eye(3) - np.outer(value, value)))

def _star(model, time, **inputs):
    value, _, valid = model.measure(t=time.seconds, scheduled=True, **inputs)
    return Readout(value, model.error_covariance_body if valid else None, valid, 'OK' if valid else 'ACQUIRING_OR_EXCLUDED')

def _pose(model, time, **inputs):
    result = model.measure(**inputs)
    if not result.valid:
        return Readout(valid=False, reason=result.reason)
    return Readout(np.concatenate((result.t_lvlh, quat_from_matrix(result.R_body2lvlh))), result.R_pose)

def _lidar(model, time, pointing_lvlh, **inputs):
    points = model.measure(pointing_lvlh=pointing_lvlh, **inputs)
    if len(points) == 0:
        return Readout(valid=False, reason='NO_RETURN')
    return Readout(points, np.eye(3) * model._noise_sigma ** 2)

def create_sensor(name, *, source_id=None, seed=0, timing=None, frame=None, error_frame=None, **parameters):
    """Construct only the selected model; optional ML/astronomy libs stay lazy.

    ``frame`` names output coordinates, not a transform. Inputs must satisfy
    each model's documented coordinate convention. Examples/default rates are
    simulation choices, not hardware specifications.
    """
    source_id = name if source_id is None else source_id
    parameters = deepcopy(parameters)
    error_units, record_type = (None, None)
    if name == 'performance_gyro':
        from sensors.gyro_performance import PerformanceGyro
        cls, observer, kind, output, units, period = (PerformanceGyro, _performance_gyro, 'angular_rate', 'body/chaser', ('rad/s',) * 3, 0.01)
    elif name == 'landmark_pose':
        from sensors.landmark_camera import LandmarkPoseCamera
        cls, kind, output, units, period = (LandmarkPoseCamera, 'target_pose', 'sensor/camera', ('m',) * 3 + ('1',) * 4, 0.5)
        error_units = ('m',) * 3 + ('rad',) * 3
        observer = lambda model, time, **inputs: model.observe(**inputs)
    elif name == 'star_tracker':
        from sensors.star_tracker import StarTracker
        cls, observer, kind, output, units, period = (StarTracker, _star, 'quaternion_body_to_reference', 'earth/gcrs', ('1',) * 4, 0.25)
        error_units = ('rad',) * 3
        error_frame = 'body/chaser' if error_frame is None else error_frame
    else:
        raise ValueError(f'Unknown sensor {name!r}; select one of {sensor_names()}')
    timing = SensorTiming(period) if timing is None else timing
    if not isinstance(timing, SensorTiming):
        raise ValueError('timing must be SensorTiming')
    if name in ('gyro', 'performance_gyro', 'accelerometer'):
        if 'dt' in parameters and parameters['dt'] != timing.period_s:
            raise ValueError('gyro dt must match the sampling period')
        parameters = dict(parameters, dt=timing.period_s)
    if name == 'star_tracker':
        if 'update_rate_hz' in parameters and (not np.isclose(parameters['update_rate_hz'], 1 / timing.period_s)):
            raise ValueError('star tracker rate must match the sampling period')
        parameters = dict(parameters, update_rate_hz=1 / timing.period_s)
    fields = dict(source_id=source_id, kind=kind, frame=output if frame is None else frame, units=units, error_units=error_units, error_frame=error_frame, timing=timing, model_factory=lambda rng: cls(**parameters, rng=rng), observer=observer, seed=seed)
    if record_type is not None:
        fields['record_type'] = record_type
    return Sensor(**fields)
