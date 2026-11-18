"""Common readiness predicates; mission sequencing remains in future wrappers."""
from dataclasses import dataclass

import numpy as np

from spec import ActuationResult, Measurement, Timestamp
from utils.validation import finite_scalar, vector


@dataclass(frozen=True)
class Readiness:
    ready: bool
    reasons: tuple[str, ...]


def measurement_readiness(measurement, *, now, max_age_s, maximum_variances=None):
    """Check validity, delivery, age and optional covariance diagonal limits.

    Variance limits are supplied in measurement.error_units squared. Passing
    these checks alone does not establish estimator consistency or integrity.
    """
    if not isinstance(measurement, Measurement) or not isinstance(now, Timestamp):
        raise ValueError("Expected a measurement/estimate and current Timestamp")
    age_limit = finite_scalar(max_age_s, "max_age_s", minimum=0)
    age = now.seconds_since(measurement.time)
    reasons = []
    if not measurement.valid:
        reasons.append("INVALID:"+measurement.reason)
    if now.seconds_since(measurement.available_at) < 0:
        reasons.append("NOT_DELIVERED")
    if age < 0:
        reasons.append("FUTURE_SAMPLE")
    elif age > age_limit:
        reasons.append("STALE")
    if maximum_variances is not None:
        limits = vector(maximum_variances, "maximum_variances", size=len(measurement.error_units))
        if np.any(limits < 0):
            raise ValueError("Variance limits must be nonnegative")
        if measurement.covariance is None or np.any(np.diag(measurement.covariance) > limits):
            reasons.append("COVARIANCE_LIMIT")
    return Readiness(not reasons, tuple(reasons))


def actuation_readiness(result, *, force_residual_limit_n, torque_residual_limit_nm):
    """Separate force and torque thresholds; never compare mixed-unit norms."""
    if not isinstance(result, ActuationResult):
        raise ValueError("Expected an ActuationResult")
    force_limit = finite_scalar(force_residual_limit_n, "force_residual_limit_n", minimum=0)
    torque_limit = finite_scalar(torque_residual_limit_nm, "torque_residual_limit_nm", minimum=0)
    reasons = []
    if np.linalg.norm(result.residual[:3]) > force_limit:
        reasons.append("FORCE_SHORTFALL")
    if np.linalg.norm(result.residual[3:]) > torque_limit:
        reasons.append("TORQUE_SHORTFALL")
    return Readiness(not reasons, tuple(reasons))


def all_ready(*reports):
    if any(not isinstance(r, Readiness) for r in reports):
        raise ValueError("Expected Readiness reports")
    reasons = tuple(reason for report in reports for reason in report.reasons)
    return Readiness(all(r.ready for r in reports), reasons)


class ActuationHealthMonitor:
    """Causal dwell qualification of achieved-wrench shortfall.

    Independent force/torque dwell timers avoid accumulating unrelated faults.
    Resource loss latches immediately. A latched fault requires an explicit
    reset after mission-level recovery authorization; a later zero command
    does not erase it. Thresholds are vehicle/mission configuration, not
    hardware-qualified defaults. This monitor does not diagnose fault origin.
    """
    def __init__(self,*,force_residual_limit_n,torque_residual_limit_nm,dwell_s,
                 immediate_flags=('FUEL_EXHAUSTED',)):
        self.force_limit=finite_scalar(force_residual_limit_n,'force_residual_limit_n',minimum=0)
        self.torque_limit=finite_scalar(torque_residual_limit_nm,'torque_residual_limit_nm',minimum=0)
        self.dwell=finite_scalar(dwell_s,'dwell_s',minimum=0)
        self.immediate_flags=tuple(immediate_flags)
        self.reset()

    def reset(self):
        self.latched_reasons=()
        self.pending={}
        self.events=[]
        self.end=None

    @property
    def ready(self):
        return not self.latched_reasons

    def rearm(self,*,time,authorization_reason):
        """Mission-authorized retry; preserve fault history and clock continuity.

        This method does not establish recovery. The caller must independently
        verify resource and device status before authorizing a bounded retry.
        """
        if not isinstance(authorization_reason,str) or not authorization_reason.strip():
            raise ValueError('Explicit recovery authorization reason required')
        if self.end is None or time!=self.end:
            raise ValueError('Rearm must occur at the last observed interval end')
        self.events.append(dict(time_s=time.seconds,rearmed=True,previous_reasons=self.latched_reasons,
            authorization_reason=authorization_reason))
        self.latched_reasons=();self.pending={}

    def update(self,result):
        if not isinstance(result,ActuationResult):raise ValueError('Expected achieved actuation')
        if self.end is not None:
            gap=result.command.time.seconds_since(self.end)
            tolerance=64*np.finfo(float).eps*max(1.,abs(self.end.seconds))
            if abs(gap)>tolerance:raise ValueError('Health observations must cover contiguous actuator intervals')
        report=actuation_readiness(result,force_residual_limit_n=self.force_limit,
            torque_residual_limit_nm=self.torque_limit)
        self.end=result.command.time.shifted(result.duration_s)
        for reason in ('FORCE_SHORTFALL','TORQUE_SHORTFALL'):
            self.pending[reason]=self.pending.get(reason,0.)+result.duration_s if reason in report.reasons else 0.
        qualified=[reason for reason in report.reasons if self.pending[reason]>=self.dwell]
        qualified.extend(flag for flag in result.flags if flag in self.immediate_flags)
        new=tuple(reason for reason in dict.fromkeys(qualified) if reason not in self.latched_reasons)
        if new:
            self.latched_reasons+=new
            self.events.append(dict(time_s=self.end.seconds,reasons=new))
        return Readiness(self.ready,self.latched_reasons)
