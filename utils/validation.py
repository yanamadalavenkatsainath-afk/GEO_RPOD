"""Validation and defensive copies at component boundaries."""
from numbers import Real

import numpy as np


def finite_scalar(value, name, *, minimum=None, strictly_positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if strictly_positive and result <= 0:
        raise ValueError(f"{name} must be positive")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def text_label(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def finite_array(value, name, *, shape=None, readonly=False):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    try:
        result = np.array(value, dtype=float, copy=True)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must contain real numbers") from exc
    if shape is not None and result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    if readonly:
        result.setflags(write=False)
    return result


def vector(value, name, *, size=None, readonly=False):
    result = finite_array(value, name)
    if result.ndim != 1 or not result.size or (size is not None and result.size != size):
        expected = "nonempty vector" if size is None else f"vector of length {size}"
        raise ValueError(f"{name} must be a {expected}")
    if readonly:
        result.setflags(write=False)
    return result


def covariance(value, size, name="covariance", *, readonly=False):
    """Accept symmetric PSD matrices, including zero and singular covariance.

    Tolerances scale with the matrix, not with an arbitrary unit variance.
    Tiny covariances therefore do not accidentally admit large negative modes.
    """
    result = finite_array(value, name, shape=(size, size))
    scale = float(np.max(np.abs(result))) if result.size else 0.0
    tolerance = 64 * np.finfo(float).eps * scale * max(size, 1)
    if float(np.max(np.abs(result - result.T))) > tolerance:
        raise ValueError(f"{name} must be symmetric")
    result = result * 0.5 + result.T * 0.5
    if np.linalg.eigvalsh(result).min() < -tolerance:
        raise ValueError(f"{name} must be positive semidefinite")
    if readonly:
        result.setflags(write=False)
    return result


def rotation_matrix(value, name="rotation"):
    result = finite_array(value, name, shape=(3, 3))
    if not np.allclose(result @ result.T, np.eye(3), rtol=0, atol=1e-10):
        raise ValueError(f"{name} must be orthonormal")
    if not np.isclose(np.linalg.det(result), 1.0, rtol=0, atol=1e-10):
        raise ValueError(f"{name} must be a proper rotation (determinant +1)")
    return result
