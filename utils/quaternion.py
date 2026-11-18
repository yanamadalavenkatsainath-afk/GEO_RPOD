"""Hamilton scalar-first quaternions: [w, x, y, z].

For attitude q, rot_matrix(q) maps body coordinates into the reference frame.
Use its transpose to express a reference-frame vector in body coordinates.
"""
import numpy as np

from utils.validation import finite_scalar, rotation_matrix, vector

def normalize(q):
    """Normalize a nonzero finite vector (also used by legacy QUEST/TRIAD)."""
    q = vector(q, "vector")
    scale = float(np.max(np.abs(q)))
    if scale == 0:
        raise ValueError("Cannot normalize a zero vector")
    scaled = q / scale
    return scaled / np.linalg.norm(scaled)

def quat_multiply(q1, q2):
    """Hamilton product; no implicit normalization of algebraic inputs."""
    q1 = vector(q1, "q1", size=4)
    q2 = vector(q2, "q2", size=4)
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quat_conjugate(q):
    q = vector(q, "q", size=4)
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quat_error(q_true, q_est):
    """Right error conj(q_est) * q_true; retained legacy API/order."""
    return quat_multiply(quat_conjugate(q_est), q_true)

def rot_matrix(q):
    q = normalize(vector(q, "q", size=4))
    w, x, y, z = q
    return np.array([
        [1-2*(y**2+z**2), 2*(x*y-z*w),     2*(x*z+y*w)],
        [2*(x*y+z*w),     1-2*(x**2+z**2), 2*(y*z-x*w)],
        [2*(x*z-y*w),     2*(y*z+x*w),     1-2*(x**2+y**2)]
    ])


def body_to_reference(q, body_vector):
    return rot_matrix(q) @ vector(body_vector, "body_vector", size=3)


def reference_to_body(q, reference_vector):
    return rot_matrix(q).T @ vector(reference_vector, "reference_vector", size=3)


def quat_from_rotvec(rotation_vector):
    """Rotation vector (axis times angle in radians) to unit quaternion."""
    value = vector(rotation_vector, "rotation_vector", size=3)
    angle = finite_scalar(np.linalg.norm(value), "rotation angle")
    # sinc avoids division by zero and preserves small-angle precision.
    return normalize(np.concatenate(([np.cos(angle / 2)],
                                     value * (0.5 * np.sinc(angle / (2 * np.pi))))))


def rotvec_from_quat(q):
    """Shortest rotation vector in radians; q and -q describe the same rotation."""
    q = normalize(vector(q, "q", size=4))
    if q[0] < 0:
        q = -q
    sine = float(np.linalg.norm(q[1:]))
    if sine < 1e-12:
        return 2 * q[1:]
    return q[1:] * (2 * np.arctan2(sine, q[0]) / sine)


def quat_from_matrix(matrix):
    """Proper active rotation matrix to Hamilton scalar-first quaternion."""
    matrix = rotation_matrix(matrix)
    trace = float(np.trace(matrix))
    if trace > 0:
        s = 2 * np.sqrt(trace + 1)
        q = np.array([s / 4, (matrix[2, 1] - matrix[1, 2]) / s,
                      (matrix[0, 2] - matrix[2, 0]) / s,
                      (matrix[1, 0] - matrix[0, 1]) / s])
    else:
        index = int(np.argmax(np.diag(matrix)))
        j, k = (index + 1) % 3, (index + 2) % 3
        s = 2 * np.sqrt(1 + matrix[index, index] - matrix[j, j] - matrix[k, k])
        q = np.zeros(4)
        q[0] = (matrix[k, j] - matrix[j, k]) / s
        q[index + 1] = s / 4
        q[j + 1] = (matrix[j, index] + matrix[index, j]) / s
        q[k + 1] = (matrix[k, index] + matrix[index, k]) / s
    q = normalize(q)
    return -q if q[0] < 0 else q
