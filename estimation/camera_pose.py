"""Known-correspondence camera pose, without measurement generation or truth.

DLT initialization plus local reprojection refinement for >=6 non-coplanar
landmarks. This baseline rejects planar geometry and does not identify landmarks.
"""
from dataclasses import dataclass

import numpy as np

from estimation.attitude import _skew
from spec import Measurement
from utils.quaternion import quat_from_matrix, quat_from_rotvec, rot_matrix
from utils.validation import covariance, finite_array, finite_scalar, rotation_matrix, vector


@dataclass(frozen=True)
class CameraPose:
    body_to_camera: np.ndarray
    translation_camera_m: np.ndarray
    covariance: np.ndarray
    reprojection_rms_px: float

    def __post_init__(self):
        rotation = rotation_matrix(self.body_to_camera)
        rotation.setflags(write=False)
        object.__setattr__(self, "body_to_camera", rotation)
        object.__setattr__(self, "translation_camera_m", vector(self.translation_camera_m, "translation_camera_m", size=3, readonly=True))
        object.__setattr__(self, "covariance", covariance(self.covariance, 6, readonly=True))
        object.__setattr__(self, "reprojection_rms_px", finite_scalar(self.reprojection_rms_px, "reprojection_rms_px", minimum=0))

    def measurement(self, *, time, source_id, reference_frame, camera_to_reference, camera_position_reference_m):
        rotation = rotation_matrix(camera_to_reference)
        p = vector(camera_position_reference_m, "camera_position_reference_m", size=3)
        transform = np.zeros((6, 6))
        transform[:3, :3] = transform[3:, 3:] = rotation
        noise = transform @ self.covariance @ transform.T
        return Measurement(source_id, "target_pose", time, reference_frame,
            ("m",)*3+("1",)*4, np.r_[rotation @ self.translation_camera_m+p,
                quat_from_matrix(rotation @ self.body_to_camera)], (noise+noise.T)/2,
            error_units=("m",)*3+("rad",)*3)


def solve_landmark_pose(points_body_m, pixels, camera_matrix, *, sigma_px=1., maximum_rms_px=5., iterations=20):
    body, image = finite_array(points_body_m, "points_body_m"), finite_array(pixels, "pixels")
    k = finite_array(camera_matrix, "camera_matrix", shape=(3, 3))
    noise = finite_scalar(sigma_px, "sigma_px", strictly_positive=True)
    gate = finite_scalar(maximum_rms_px, "maximum_rms_px", strictly_positive=True)
    if body.ndim != 2 or body.shape[1] != 3 or len(body) < 6 or image.shape != (len(body), 2):
        raise ValueError("At least six matched 3D landmarks and pixels are required")
    if np.linalg.matrix_rank(body-body.mean(axis=0), tol=1e-8) < 3:
        raise ValueError("This DLT initializer requires non-coplanar landmarks")
    if not np.allclose(k[2], [0, 0, 1]) or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError("Expected conventional calibrated pinhole intrinsics")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be positive")
    # Normalize object coordinates for DLT conditioning.
    centre = body.mean(axis=0)
    scale = np.sqrt(np.mean(np.sum((body-centre)**2, axis=1)))
    points = np.column_stack(((body-centre)/scale, np.ones(len(body))))
    rays = (np.linalg.solve(k, np.column_stack((image, np.ones(len(image)))).T)).T
    rows = []
    for point, ray in zip(points, rays):
        x, y = ray[:2]/ray[2]
        rows.extend((np.r_[point, np.zeros(4), -x*point], np.r_[np.zeros(4), point, -y*point]))
    _, singular, vt = np.linalg.svd(np.asarray(rows))
    if singular[-2] <= singular[0]*1e-10:
        raise ValueError("Landmark pose geometry is rank deficient")
    projection = vt[-1].reshape(3, 4)
    left = projection[:, :3]/scale
    translation = projection[:, 3]-left @ centre
    if np.linalg.det(left) < 0:
        left, translation = -left, -translation
    factor = np.linalg.svd(left, compute_uv=False).mean()
    u, _, vt = np.linalg.svd(left)
    rotation = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
    translation /= factor
    def linearize(rotation, translation):
        relative = body @ rotation.T
        camera = relative+translation
        if np.any(camera[:, 2] <= 0):
            raise ValueError("Pose places a landmark behind the camera")
        homogeneous = camera @ k.T
        predicted = homogeneous[:, :2]/homogeneous[:, 2, None]
        jacobians = []
        for point, ray in zip(relative, homogeneous):
            pixel_jac = (k[:2]-np.outer(ray[:2]/ray[2], k[2]))/ray[2]
            jacobians.append(pixel_jac @ np.column_stack((np.eye(3), -_skew(point))))
        return predicted, np.vstack(jacobians)
    for _ in range(iterations):
        predicted, h = linearize(rotation, translation)
        dx, _, rank, _ = np.linalg.lstsq(h, (image-predicted).ravel(), rcond=1e-10)
        if rank < 6:
            raise ValueError("Local pose is unobservable")
        translation += dx[:3]
        rotation = rot_matrix(quat_from_rotvec(dx[3:])) @ rotation
        if np.linalg.norm(dx) < 1e-9:
            break
    predicted, h = linearize(rotation, translation)
    rms = float(np.sqrt(np.mean((image-predicted)**2)))
    if rms > gate:
        raise ValueError("Pose reprojection gate failed")
    _, singular, vt = np.linalg.svd(h/noise, full_matrices=False)
    if singular[-1] <= singular[0]*1e-10:
        raise ValueError("Pose covariance is ill-conditioned")
    p = (vt.T/singular**2) @ vt
    return CameraPose(rotation, translation, (p+p.T)/2, rms)


def robust_landmark_pose(points_body_m, pixels, camera_matrix, *, rng,
                         sigma_px=1., gate_px=5., attempts=12):
    """Six-point RANSAC followed by refinement of matched nonplanar landmarks.

    Requires explicit feature correspondences and calibrated intrinsics. No
    true range, attitude or translation is supplied to this estimator.
    """
    from itertools import combinations
    body = finite_array(points_body_m, 'points_body_m')
    image = finite_array(pixels, 'pixels')
    k = finite_array(camera_matrix, 'camera_matrix', shape=(3, 3))
    gate = finite_scalar(gate_px, 'gate_px', strictly_positive=True)
    if body.ndim != 2 or body.shape[1] != 3 or len(body) < 6 or image.shape != (len(body), 2):
        raise ValueError('At least six matched landmarks are required')
    if not isinstance(rng, np.random.Generator):
        raise ValueError('An explicit random generator is required')
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
        raise ValueError('attempts must be a positive integer')
    # Small target sets: enumerate combinations without replacement. Larger
    # sets: bound memory and work rather than constructing N-choose-six tuples.
    if len(body) <= 12:
        subsets = list(combinations(range(len(body)), 6))
        order = rng.permutation(len(subsets))[:attempts]
        subsets = [subsets[i] for i in order]
    else:
        subsets = [rng.choice(len(body), 6, replace=False) for _ in range(attempts)]
    best = None
    for subset in subsets:
        index = np.asarray(subset)
        try:
            pose = solve_landmark_pose(body[index], image[index], k, sigma_px=sigma_px,
                                       maximum_rms_px=gate, iterations=10)
        except (ValueError, np.linalg.LinAlgError):
            continue
        camera = body @ pose.body_to_camera.T + pose.translation_camera_m
        projected = camera @ k.T
        safe_depth = np.where(camera[:, 2] > 0, camera[:, 2], 1.)
        residual = np.linalg.norm(projected[:, :2]/safe_depth[:, None]-image, axis=1)
        inliers = (camera[:, 2] > 0) & (residual <= gate)
        score = (int(inliers.sum()), -float(np.sum(residual[inliers]**2)))
        if best is None or score > best[0]:
            best = score, inliers
        if inliers.all():
            break
    if best is None or best[0][0] < 6:
        raise ValueError('No six-landmark pose consensus')
    inliers = best[1]
    return solve_landmark_pose(body[inliers], image[inliers], k, sigma_px=sigma_px,
                               maximum_rms_px=gate)
