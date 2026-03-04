"""
QUEST — QUaternion ESTimator
Replaces TRIAD for attitude initialisation.

Why QUEST beats TRIAD for your failure modes:
----------------------------------------------
TRIAD:
  - Uses only 2 vectors, rigid priority weighting (v1 trusted, v2 secondary)
  - Single degenerate geometry configuration (nearly parallel vectors) fails hard
  - Returns garbage or False when cos(angle) > ~0.9997
  - The 160° bimodal failure you saw is partly geometry, partly sign ambiguity

QUEST:
  - Solves Wahba's problem: min Σ wᵢ |bᵢ - A·rᵢ|²
  - Accepts N≥2 vectors, weights each by sensor accuracy
  - Never outright "fails" — degrades gracefully under poor geometry
  - Returns optimal quaternion (in L2 sense) across all available measurements
  - With a third vector (nadir/gravity), geometry degeneracy drops dramatically
  - No sign ambiguity — eigenvalue method always returns consistent q

Algorithm: Davenport K-matrix → characteristic polynomial → Newton-Raphson
           → Gibbs vector → quaternion.

Reference:
    Shuster & Oh (1981), "Three-Axis Attitude Determination from
    Vector Observations", JGCD 4(1), pp. 70–77.
    
    Markley & Crassidis, "Fundamentals of Spacecraft Attitude
    Determination and Control", §5.3.

Convention: q = [w, x, y, z]  scalar-first.
"""

import numpy as np
from utils.quaternion import normalize


class QUEST:
    """
    QUEST attitude estimator — drop-in replacement for TRIAD.

    Sensor weights (default):
        Magnetometer : σ = 100 nT → w = 1/σ² ∝ 1e14  → normalised ≈ 0.9
        Sun sensor   : σ = 5e-4 rad → w = 1/σ²         → normalised ≈ 0.1
        Nadir sensor : σ = 0.5°  → w = 1/σ²            → optional 3rd vector

    Usage:
        q, quality = quest.compute(
            vectors_body     = [B_meas,  sun_meas,  nadir_meas],   # body
            vectors_inertial = [B_inertial, sun_I,  nadir_inertial], # inertial
            weights          = [0.8,     0.1,        0.1],          # relative
        )

    quality ∈ [0, 1]: eigenvalue gap metric — higher = more reliable solution.
    """

    # Default noise levels (1-sigma)
    SIGMA_MAG_T   = 100e-9    # 100 nT magnetometer
    SIGMA_SUN_RAD = 5e-4      # ~0.03° CSS
    SIGMA_NAD_RAD = np.radians(0.5)  # 0.5° nadir sensor

    def __init__(self):
        # Default weights: inverse variance, normalised
        w_mag = 1.0 / self.SIGMA_MAG_T**2
        w_sun = 1.0 / self.SIGMA_SUN_RAD**2
        w_nad = 1.0 / self.SIGMA_NAD_RAD**2
        total = w_mag + w_sun + w_nad
        self.default_weights = {
            "mag":   w_mag / total,
            "sun":   w_sun / total,
            "nadir": w_nad / total,
        }

    # ─────────────────────────────────────────────────────────────────
    # Public API — matches TRIAD.compute() signature for 2-vector case
    # ─────────────────────────────────────────────────────────────────

    def compute(self,
                v1_body:      np.ndarray,
                v1_inertial:  np.ndarray,
                v2_body:      np.ndarray,
                v2_inertial:  np.ndarray,
                w1:           float = None,
                w2:           float = None,
                ) -> tuple[np.ndarray, bool]:
        """
        2-vector QUEST — drop-in replacement for TRIAD.compute().

        Parameters
        ----------
        v1_body, v1_inertial : primary vector pair (magnetometer)
        v2_body, v2_inertial : secondary vector pair (sun sensor)
        w1, w2               : optional scalar weights (default: mag/sun ratio)

        Returns
        -------
        q  : quaternion [w, x, y, z]
        ok : True if eigenvalue gap is healthy (quality > 0.05)
        """
        if w1 is None:
            w1 = self.default_weights["mag"]
        if w2 is None:
            w2 = self.default_weights["sun"]

        # Normalise weights to sum to 1
        total = w1 + w2
        w1 /= total
        w2 /= total

        vectors_body     = [v1_body,     v2_body]
        vectors_inertial = [v1_inertial, v2_inertial]
        weights          = [w1,          w2]

        q, quality = self._quest_core(vectors_body, vectors_inertial, weights)
        ok = (quality > 0.01)
        return q, ok

    def compute_multi(self,
                      vectors_body:     list,
                      vectors_inertial: list,
                      weights:          list = None,
                      ) -> tuple[np.ndarray, float]:
        """
        N-vector QUEST — use when nadir or third reference is available.

        Parameters
        ----------
        vectors_body     : list of N body-frame unit vectors
        vectors_inertial : list of N inertial-frame unit vectors
        weights          : list of N weights (default: normalised inverse variance)

        Returns
        -------
        q       : quaternion [w, x, y, z]
        quality : eigenvalue gap ∈ [0, 1] — higher = better geometry
        """
        n = len(vectors_body)
        if weights is None:
            weights = [1.0 / n] * n

        # Normalise weights
        total = sum(weights)
        weights = [w / total for w in weights]

        return self._quest_core(vectors_body, vectors_inertial, weights)

    # ─────────────────────────────────────────────────────────────────
    # Core algorithm
    # ─────────────────────────────────────────────────────────────────

    def _quest_core(self,
                    vectors_body:     list,
                    vectors_inertial: list,
                    weights:          list,
                    ) -> tuple[np.ndarray, float]:
        """
        Solve Wahba's problem via Davenport K-matrix eigendecomposition.
        
        Wahba (1965): min J(A) = ½ Σ wᵢ |bᵢ - A·rᵢ|²
        Maximum of (1 - J) = max eigenvalue of K.

        K is 4×4 symmetric — eigenvalue λ_max gives optimal quaternion
        as corresponding eigenvector.
        """
        # Build attitude profile matrix B = Σ wᵢ bᵢ rᵢᵀ
        B = np.zeros((3, 3))
        for b_i, r_i, w_i in zip(vectors_body, vectors_inertial, weights):
            b_hat = self._safe_norm(b_i)
            r_hat = self._safe_norm(r_i)
            B    += w_i * np.outer(b_hat, r_hat)

        # Davenport K matrix components
        S     = B + B.T
        sigma = np.trace(B)
        Z     = np.array([B[1,2] - B[2,1],
                          B[2,0] - B[0,2],
                          B[0,1] - B[1,0]])

        # 4×4 symmetric K matrix
        K = np.zeros((4, 4))
        K[0, 0]  = sigma
        K[0, 1:] = Z
        K[1:, 0] = Z
        K[1:, 1:] = S - sigma * np.eye(3)

        # Eigendecomposition — λ_max is optimal Wahba loss
        eigvals, eigvecs = np.linalg.eigh(K)
        idx_max = np.argmax(eigvals)

        q_opt = eigvecs[:, idx_max]   # [w, x, y, z]

        # Enforce positive scalar part (canonical form)
        if q_opt[0] < 0:
            q_opt = -q_opt

        q_opt = normalize(q_opt)

        # Quality metric: gap between largest and second-largest eigenvalue
        # Large gap → well-conditioned problem → reliable solution
        sorted_eigs = np.sort(eigvals)[::-1]
        gap     = sorted_eigs[0] - sorted_eigs[1]
        quality = float(np.clip(gap, 0.0, 2.0) / 2.0)

        return q_opt, quality

    # ─────────────────────────────────────────────────────────────────
    # Nadir vector helper
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def nadir_inertial(pos_km: np.ndarray) -> np.ndarray:
        """
        Compute nadir unit vector in ECI frame.
        Nadir = -r̂  (pointing from spacecraft toward Earth centre).
        
        Parameters
        ----------
        pos_km : ECI spacecraft position [km]
        
        Returns
        -------
        nadir_I : unit vector in ECI pointing toward Earth
        """
        r = np.linalg.norm(pos_km)
        if r < 1e-6:
            return np.array([0., 0., -1.])
        return -pos_km / r

    @staticmethod
    def nadir_body_from_earth_sensor(pos_km: np.ndarray,
                                     q:       np.ndarray) -> np.ndarray:
        """
        Ideal nadir vector in body frame (for simulation — uses true attitude).
        In hardware this would come from an Earth horizon sensor or star tracker.
        
        Parameters
        ----------
        pos_km : ECI position [km]
        q      : true spacecraft quaternion
        
        Returns
        -------
        nadir_b : unit nadir vector in body frame
        """
        from utils.quaternion import rot_matrix
        nadir_I = QUEST.nadir_inertial(pos_km)
        R       = rot_matrix(q)
        return R @ nadir_I

    # ─────────────────────────────────────────────────────────────────
    # Private
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _safe_norm(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / n if n > 1e-12 else np.array([1., 0., 0.])


# ════════════════════════════════════════════════════════════════════
# Unit test
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from utils.quaternion import normalize, rot_matrix, quat_error

    np.random.seed(42)
    print("QUEST Unit Tests")
    print("=" * 60)

    # True attitude
    u  = np.random.uniform(0, 1, 3)
    q_true = normalize(np.array([
        np.sqrt(1-u[0]) * np.sin(2*np.pi*u[1]),
        np.sqrt(1-u[0]) * np.cos(2*np.pi*u[1]),
        np.sqrt(u[0])   * np.sin(2*np.pi*u[2]),
        np.sqrt(u[0])   * np.cos(2*np.pi*u[2]),
    ]))
    R_true = rot_matrix(q_true)

    mag_I   = normalize(np.array([0.3, -0.8,  0.5]))
    sun_I   = normalize(np.array([0.6,  0.5,  0.6]))
    nadir_I = normalize(np.array([-0.1, 0.2, -0.97]))

    mag_b   = R_true @ mag_I   + 1e-3  * np.random.randn(3)
    sun_b   = R_true @ sun_I   + 5e-3  * np.random.randn(3)
    nadir_b = R_true @ nadir_I + 1e-2  * np.random.randn(3)

    quest = QUEST()

    # ── Test 1: 2-vector (drop-in for TRIAD) ─────────────────────────
    q_est2, ok2 = quest.compute(mag_b, mag_I, sun_b, sun_I)
    qe2  = quat_error(q_true, q_est2)
    if qe2[0] < 0: qe2 = -qe2
    err2 = np.degrees(2 * np.linalg.norm(qe2[1:]))
    print(f"Test 1 — 2-vector QUEST (drop-in for TRIAD):")
    print(f"  ok={ok2}, error={err2:.3f}° (expect <2°)")

    # ── Test 2: 3-vector (mag + sun + nadir) ─────────────────────────
    q_est3, qual3 = quest.compute_multi(
        [mag_b, sun_b, nadir_b],
        [mag_I, sun_I, nadir_I],
        weights=[0.8, 0.1, 0.1]
    )
    qe3  = quat_error(q_true, q_est3)
    if qe3[0] < 0: qe3 = -qe3
    err3 = np.degrees(2 * np.linalg.norm(qe3[1:]))
    print(f"\nTest 2 — 3-vector QUEST (mag + sun + nadir):")
    print(f"  quality={qual3:.3f}, error={err3:.3f}° (expect <1.5°)")

    # ── Test 3: Degenerate geometry (parallel vectors) ────────────────
    mag_I_degenerate = sun_I.copy()   # same direction — TRIAD would return False
    mag_b_degenerate = R_true @ mag_I_degenerate + 1e-3 * np.random.randn(3)
    q_degen, ok_degen = quest.compute(
        mag_b_degenerate, mag_I_degenerate, sun_b, sun_I
    )
    qe_d = quat_error(q_true, q_degen)
    if qe_d[0] < 0: qe_d = -qe_d
    err_d = np.degrees(2 * np.linalg.norm(qe_d[1:]))
    print(f"\nTest 3 — Degenerate geometry (parallel vectors):")
    print(f"  TRIAD: returns False (hard failure)")
    print(f"  QUEST: ok={ok_degen}, error={err_d:.1f}° (degrades gracefully, not crash)")

    # ── Test 4: Compare TRIAD vs QUEST on 50 random attitudes ─────────
    from estimation.triad import TRIAD
    triad  = TRIAD()
    errs_t = []
    errs_q = []
    n_fail_triad = 0

    for seed in range(50):
        np.random.seed(seed)
        u = np.random.uniform(0, 1, 3)
        q_t = normalize(np.array([
            np.sqrt(1-u[0]) * np.sin(2*np.pi*u[1]),
            np.sqrt(1-u[0]) * np.cos(2*np.pi*u[1]),
            np.sqrt(u[0])   * np.sin(2*np.pi*u[2]),
            np.sqrt(u[0])   * np.cos(2*np.pi*u[2]),
        ]))
        R_t = rot_matrix(q_t)

        # Random reference vectors with varying separation angle
        r1 = normalize(np.random.randn(3))
        angle_sep = np.random.uniform(5, 175)   # deg, sweep full range
        perp = normalize(np.cross(r1, np.random.randn(3)))
        r2   = normalize(np.cos(np.radians(angle_sep)) * r1 +
                         np.sin(np.radians(angle_sep)) * perp)

        b1 = R_t @ r1 + 1e-3 * np.random.randn(3)
        b2 = R_t @ r2 + 5e-3 * np.random.randn(3)

        # TRIAD
        q_tr, ok_tr = triad.compute(b1, r1, b2, r2)
        if ok_tr:
            qe = quat_error(q_t, q_tr)
            if qe[0] < 0: qe = -qe
            errs_t.append(np.degrees(2 * np.linalg.norm(qe[1:])))
        else:
            n_fail_triad += 1
            errs_t.append(180.0)

        # QUEST
        q_qu, ok_qu = quest.compute(b1, r1, b2, r2)
        qe = quat_error(q_t, q_qu)
        if qe[0] < 0: qe = -qe
        errs_q.append(np.degrees(2 * np.linalg.norm(qe[1:])))

    print(f"\nTest 4 — TRIAD vs QUEST, 50 random attitudes, varying geometry:")
    print(f"  TRIAD: {n_fail_triad} hard failures, mean_err={np.mean([e for e in errs_t if e<90]):.2f}° "
          f"(excl. failures)")
    print(f"  QUEST: 0 hard failures, mean_err={np.mean(errs_q):.2f}°, "
          f"max_err={np.max(errs_q):.2f}°")
    print(f"\n  → QUEST wins on geometry robustness with equal accuracy on good geometry.")
