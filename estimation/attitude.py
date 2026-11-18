"""Common right-error attitude filter; Hamilton q maps body to reference.

No truth input, implicit frame conversion, replay, or automatic reacquisition.
Legacy mission-specific filters remain in their original modules.
"""
from dataclasses import dataclass

import numpy as np

from spec import Measurement, Reference, StateEstimate, Timestamp
from utils.quaternion import (normalize, quat_conjugate, quat_from_matrix,
                              quat_from_rotvec, quat_multiply, rot_matrix,
                              rotvec_from_quat)
from utils.validation import covariance, finite_scalar, text_label, vector


def _skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def _transition(omega, dt):
    """Exact constant-rate transition for [body-angle error, gyro-bias error]."""
    a = omega * dt
    angle = np.linalg.norm(a)
    cross = _skew(a)
    if angle < 1e-5:
        jacobian = np.eye(3) - cross / 2 + cross @ cross / 6
    else:
        jacobian = (np.eye(3) - (1-np.cos(angle))/angle**2 * cross
                    + (angle-np.sin(angle))/angle**3 * cross @ cross)
    phi = np.eye(6)
    phi[:3, :3] = rot_matrix(quat_from_rotvec(-a))
    phi[:3, 3:] = -dt * jacobian
    return phi


@dataclass(frozen=True)
class UpdateResult:
    accepted: bool
    reason: str
    nis: float | None = None


def attitude_from_directions(body_vectors, reference_vectors, weights=None):
    """Weighted Wahba SVD solution, returning body-to-reference quaternion.

Requires at least two non-collinear directions in each frame. Weights are
relative positive scalars, not a substitute for a filter covariance.
"""
    body = np.asarray(body_vectors, dtype=float)
    reference = np.asarray(reference_vectors, dtype=float)
    if (body.ndim != 2 or body.shape[1] != 3 or len(body) < 2
            or reference.shape != body.shape or not np.all(np.isfinite(body))
            or not np.all(np.isfinite(reference))):
        raise ValueError("Expected matching finite (N,3) direction arrays, N>=2")
    body = np.array([normalize(v) for v in body])
    reference = np.array([normalize(v) for v in reference])
    w = np.ones(len(body)) if weights is None else vector(weights, "weights", size=len(body))
    if np.any(w <= 0):
        raise ValueError("weights must be positive")
    w = w / np.max(w)
    for directions in (body, reference):
        singular = np.linalg.svd(directions * np.sqrt(w[:, None]), compute_uv=False)
        if singular[1] <= 1e-8 * singular[0]:
            raise ValueError("Direction geometry is collinear or ill-conditioned")
    u, _, vt = np.linalg.svd((reference * w[:, None]).T @ body)
    correction = np.diag([1., 1., np.linalg.det(u @ vt)])
    return quat_from_matrix(u @ correction @ vt)


class AttitudeMEKF:
    """Six-state local MEKF with explicit initial state and SI noise densities.

    q_true = q_hat * Exp(dtheta_body); dbias = bias_true - bias_hat.
    gyro_noise_density is rad/sqrt(s); bias_walk_density is rad/s/sqrt(s).
    Delayed attitude/direction observations are rejected, not fused at a wrong
    time. The wrapper must propagate to the sample time before updating.
    """
    def __init__(self, *, time, quaternion, bias_rad_s, initial_covariance,
                 reference_frame="earth/gcrs", body_frame="body/chaser",
                 gyro_noise_density=1e-5, bias_walk_density=1e-8,
                 gate_nis=25., max_gyro_age_s=0.1):
        if not isinstance(time, Timestamp):
            raise ValueError("time must be a Timestamp")
        self.reference_frame = text_label(reference_frame, "reference_frame")
        self.body_frame = text_label(body_frame, "body_frame")
        self.gyro_noise_density = finite_scalar(gyro_noise_density, "gyro_noise_density", minimum=0)
        self.bias_walk_density = finite_scalar(bias_walk_density, "bias_walk_density", minimum=0)
        self.gate_nis = finite_scalar(gate_nis, "gate_nis", strictly_positive=True)
        self.max_gyro_age_s = finite_scalar(max_gyro_age_s, "max_gyro_age_s", minimum=0)
        self.reset(time=time, quaternion=quaternion, bias_rad_s=bias_rad_s,
                   initial_covariance=initial_covariance)

    def reset(self, *, time, quaternion, bias_rad_s, initial_covariance):
        if not isinstance(time, Timestamp):
            raise ValueError("time must be a Timestamp")
        q = normalize(vector(quaternion, "quaternion", size=4))
        bias = vector(bias_rad_s, "bias_rad_s", size=3)
        p = covariance(initial_covariance, 6)
        self.time, self.q, self.bias, self.P = time, q, bias, p
        self._consumed = set()

    def _check(self, sample, kind, frame, units, *, current=True):
        # State estimates and truth must not masquerade as sensor observations.
        if type(sample) is not Measurement:
            raise ValueError("Expected a sensor Measurement")
        if sample.kind != kind or sample.frame != frame or sample.units != units:
            raise ValueError("Measurement kind, frame or SI units do not match")
        sample.time.seconds_since(self.time)  # also checks epoch
        if not sample.usable_at(self.time):
            return UpdateResult(False, "UNAVAILABLE_OR_FUTURE")
        if current and sample.time != self.time:
            return UpdateResult(False, "OUT_OF_SEQUENCE")
        key = (sample.source_id, sample.kind, sample.time)
        if current and key in self._consumed:
            return UpdateResult(False, "DUPLICATE")
        return None

    def predict(self, gyro, to_time):
        """Hold the supplied gyro from the current state time to to_time.

        The reading must already be available and no older than max_gyro_age_s
        at the end of the interval; gyro covariance is not added twice to the
        configured continuous white-noise density.
        """
        dt = to_time.seconds_since(self.time)
        if dt <= 0:
            raise ValueError("Propagation must advance time")
        rejected = self._check(gyro, "angular_rate", self.body_frame, ("rad/s",)*3, current=False)
        if rejected:
            return rejected
        if not gyro.usable_at(to_time, max_age_s=self.max_gyro_age_s):
            return UpdateResult(False, "STALE_GYRO")
        omega = gyro.value - self.bias
        # Gauss-Legendre quadrature of Phi(t) Qc Phi(t)^T. Limit rotation per
        # interval to .1 rad; exact transition preserves zero-rate bias coupling.
        steps = max(1, int(np.ceil(np.linalg.norm(omega)*dt / .1)))
        h = dt / steps
        qc = np.diag([self.gyro_noise_density**2]*3 + [self.bias_walk_density**2]*3)
        nodes = (.5-np.sqrt(15)/10, .5, .5+np.sqrt(15)/10)
        weights = (5/18, 4/9, 5/18)
        phi = _transition(omega, h)
        qd = sum(w * (_transition(omega, h*x) @ qc @ _transition(omega, h*x).T)
                 for x, w in zip(nodes, weights)) * h
        p = self.P.copy()
        for _ in range(steps):
            p = phi @ p @ phi.T + qd
        self.q = normalize(quat_multiply(self.q, quat_from_rotvec(omega*dt)))
        self.P = (p+p.T)/2
        self.time = to_time
        self._consumed.clear()
        return UpdateResult(True, "PROPAGATED")

    def _correct(self, sample, residual, h, noise):
        noise = covariance(noise, len(residual))
        s = h @ self.P @ h.T + noise
        try:
            nis = float(residual @ np.linalg.solve(s, residual))
            k = np.linalg.solve(s, h @ self.P).T
        except np.linalg.LinAlgError:
            return UpdateResult(False, "SINGULAR_INNOVATION")
        self._consumed.add((sample.source_id, sample.kind, sample.time))
        if nis > self.gate_nis:
            return UpdateResult(False, "INNOVATION_GATE", nis)
        dx = k @ residual
        ikh = np.eye(6)-k @ h
        p = ikh @ self.P @ ikh.T + k @ noise @ k.T
        reset = np.eye(6)
        reset[:3, :3] -= .5 * _skew(dx[:3])
        p = reset @ p @ reset.T
        self.q = normalize(quat_multiply(self.q, quat_from_rotvec(dx[:3])))
        self.bias = self.bias + dx[3:]
        self.P = (p+p.T)/2
        return UpdateResult(True, "UPDATED", nis)

    def update_star_tracker(self, sample):
        rejected = self._check(sample, "quaternion_body_to_reference", self.reference_frame, ("1",)*4)
        if rejected:
            return rejected
        if sample.error_units != ("rad",)*3 or sample.error_frame != self.body_frame:
            raise ValueError("Star-tracker covariance must be in right/body angle coordinates")
        q = normalize(sample.value)
        residual = rotvec_from_quat(quat_multiply(quat_conjugate(self.q), q))
        return self._correct(sample, residual, np.column_stack((np.eye(3), np.zeros((3, 3)))), sample.covariance)

    def update_direction(self, sample, reference):
        """Fuse sun direction or magnetic vector against a deterministic model.

        Reference has the same kind/units and sample time, in reference_frame.
        Model uncertainty is not inferred; callers must include it in the
        measurement covariance. Only the two tangent coordinates are fused.
        """
        if not isinstance(reference, Reference):
            raise ValueError("Expected a deterministic Reference, not simulator truth")
        if (reference.kind not in ("sun_direction", "magnetic_field")
                or reference.frame != self.reference_frame or reference.time != self.time):
            raise ValueError("Reference kind, frame or time does not match")
        expected_units = ("T",)*3 if reference.kind == "magnetic_field" else ("1",)*3
        if reference.units != expected_units:
            raise ValueError("Reference must use SI units")
        rejected = self._check(sample, reference.kind, self.body_frame, expected_units)
        if rejected:
            return rejected
        if sample.error_frame != self.body_frame or sample.error_units != sample.units:
            raise ValueError("Direction covariance must use body-frame vector coordinates")
        z = normalize(sample.value)
        predicted = rot_matrix(self.q).T @ normalize(reference.value)
        if np.dot(z, predicted) <= 0:
            return UpdateResult(False, "OUTSIDE_LOCAL_HEMISPHERE")
        # A radial component is unobservable after normalization. Project into
        # a tangent basis; this also handles physically singular sun covariance.
        axis = np.eye(3)[np.argmin(np.abs(predicted))]
        first = normalize(np.cross(predicted, axis))
        tangent = np.vstack((first, np.cross(predicted, first)))
        jac = (np.eye(3)-np.outer(z, z)) / np.linalg.norm(sample.value)
        noise = tangent @ jac @ sample.covariance @ jac.T @ tangent.T
        h = np.column_stack((tangent @ _skew(predicted), np.zeros((2, 3))))
        return self._correct(sample, tangent @ (z-predicted), h, noise)

    def estimate(self):
        return StateEstimate(
            source_id="attitude_mekf", kind="attitude_and_gyro_bias", time=self.time,
            frame=self.reference_frame, units=("1",)*4+("rad/s",)*3,
            value=np.concatenate((self.q, self.bias)), covariance=self.P,
            error_units=("rad",)*3+("rad/s",)*3, error_frame=self.body_frame)
