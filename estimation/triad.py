import numpy as np
from utils.quaternion import normalize


class TRIAD:
    """
    TRIAD (Three-Axis Attitude Determination) algorithm.

    Computes attitude quaternion from two vector observations.
    Used to initialise the MEKF at the start of Phase 2,
    replacing the idealised warm-start from detumbling exit attitude.

    Reference:
        Shuster & Oh (1981), "Three-Axis Attitude Determination
        from Vector Observations", JGCD 4(1).

    Convention:
        q = [w, x, y, z]  (scalar-first)
        Vectors ordered by accuracy: v1 = primary (more trusted),
        v2 = secondary.  Typically: v1 = magnetometer, v2 = sun sensor.
        Swap if sun is available and mag is noisy.

    Accuracy:
        Limited by sensor noise on the two input vectors.
        Typically ~1-3° for MEMS gyros + magnetometer + coarse sun sensor.
        Good enough to warm-start MEKF, which will converge in <20 s.
    """

    def __init__(self):
        pass

    def compute(self,
                v1_body: np.ndarray, v1_inertial: np.ndarray,
                v2_body: np.ndarray, v2_inertial: np.ndarray
                ) -> tuple[np.ndarray, bool]:
        """
        Compute attitude quaternion from two vector pair observations.

        Parameters
        ----------
        v1_body      : unit vector in body frame  (primary)
        v1_inertial  : unit vector in inertial frame (primary)
        v2_body      : unit vector in body frame  (secondary)
        v2_inertial  : unit vector in inertial frame (secondary)

        Returns
        -------
        q   : attitude quaternion [w, x, y, z]
        ok  : True if geometry was non-degenerate (|cross| > threshold)
        """
        # ── Normalise inputs ───────────────────────────────────────────
        b1 = self._safe_normalise(v1_body)
        r1 = self._safe_normalise(v1_inertial)
        b2 = self._safe_normalise(v2_body)
        r2 = self._safe_normalise(v2_inertial)

        # ── Build triad triads in body and inertial frames ─────────────
        # t1 = primary vector (most trusted)
        # t2 = t1 x t2 normalised (encodes angular relationship)
        # t3 = t1 x t2 (completes right-hand set)

        # Inertial triad
        r12   = np.cross(r1, r2)
        norm_r = np.linalg.norm(r12)
        if norm_r < 1e-6:
            # Vectors are nearly parallel — geometry is degenerate
            return np.array([1., 0., 0., 0.]), False

        r12_hat = r12 / norm_r
        r3_hat  = np.cross(r1, r12_hat)

        # Body triad
        b12   = np.cross(b1, b2)
        norm_b = np.linalg.norm(b12)
        if norm_b < 1e-6:
            return np.array([1., 0., 0., 0.]), False

        b12_hat = b12 / norm_b
        b3_hat  = np.cross(b1, b12_hat)

        # ── Attitude matrix A (body ← inertial) ────────────────────────
        # A = [b1 | b12_hat | b3_hat] @ [r1 | r12_hat | r3_hat]^T
        M_body    = np.column_stack([b1, b12_hat, b3_hat])     # (3,3)
        M_inertial = np.column_stack([r1, r12_hat, r3_hat])    # (3,3)

        A = M_body @ M_inertial.T   # rotation matrix: body = A @ inertial

        # ── Convert rotation matrix to quaternion ───────────────────────
        q = self._dcm_to_quat(A)
        return q, True

    # ─────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _safe_normalise(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        if n < 1e-12:
            return np.array([1., 0., 0.])
        return v / n

    @staticmethod
    def _dcm_to_quat(A: np.ndarray) -> np.ndarray:
        """
        Shepperd's method — numerically stable DCM → quaternion.
        Returns [w, x, y, z].
        """
        trace = A[0, 0] + A[1, 1] + A[2, 2]

        K = np.array([
            [trace,          A[1,2]-A[2,1],  A[2,0]-A[0,2],  A[0,1]-A[1,0]],
            [A[1,2]-A[2,1],  A[0,0]-A[1,1]-A[2,2], A[0,1]+A[1,0],  A[2,0]+A[0,2]],
            [A[2,0]-A[0,2],  A[0,1]+A[1,0], -A[0,0]+A[1,1]-A[2,2], A[1,2]+A[2,1]],
            [A[0,1]-A[1,0],  A[2,0]+A[0,2],  A[1,2]+A[2,1], -A[0,0]-A[1,1]+A[2,2]],
        ]) / 3.0   # Symmetric K matrix (Shepperd)

        # Largest eigenvalue → quaternion components
        eigvals, eigvecs = np.linalg.eigh(K)
        idx = np.argmax(eigvals)
        q_wxyz = eigvecs[:, idx]          # [w, x, y, z]

        # Enforce positive scalar part for uniqueness
        if q_wxyz[0] < 0:
            q_wxyz = -q_wxyz

        return normalize(q_wxyz)


# ═══════════════════════════════════════════════════════════════════
# Quick unit test — run this file directly
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from utils.quaternion import rot_matrix

    np.random.seed(42)
    print("TRIAD unit test")
    print("=" * 50)

    # True attitude — some arbitrary rotation
    q_true = normalize(np.array([0.9, 0.2, -0.3, 0.1]))
    R_true = rot_matrix(q_true)

    # Two reference vectors in inertial frame
    mag_I = normalize(np.array([0.3, -0.8, 0.5]))
    sun_I = normalize(np.array([0.6,  0.5, 0.6]))

    # Rotate to body — add small noise
    mag_b = R_true @ mag_I + 1e-3 * np.random.randn(3)
    sun_b = R_true @ sun_I + 5e-3 * np.random.randn(3)

    triad = TRIAD()
    q_est, ok = triad.compute(mag_b, mag_I, sun_b, sun_I)

    print(f"  Geometry valid : {ok}")
    print(f"  q_true : {q_true}")
    print(f"  q_est  : {q_est}")

    # Quaternion error angle
    from utils.quaternion import quat_error
    qe  = quat_error(q_true, q_est)
    err = np.degrees(2 * np.linalg.norm(qe[1:]))
    print(f"  Error  : {err:.3f} deg")
    print("  Expected: <1 deg with this noise level")
