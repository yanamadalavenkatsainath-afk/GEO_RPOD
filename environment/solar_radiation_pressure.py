import numpy as np
from utils.quaternion import rot_matrix


class SolarRadiationPressure:
    """
    SRP torque model for 3U CubeSat.

    Eclipse model : dual-cone (cylindrical shadow + penumbra)
    Optical model : Mignard-Farinella (specular + diffuse coefficients)

    Reference:
        Mignard & Farinella (1984), Celest. Mech. 33, 239.
        Montenbruck & Gill, "Satellite Orbits", §3.4.
    """

    # Physical constants
    AU_km     = 1.495978707e8    # km
    R_sun_km  = 696000.0         # solar radius (km)
    R_earth_km = 6371.0          # Earth radius (km)
    P_srp_1AU = 4.56e-6          # N/m² at 1 AU

    def __init__(self):
        # 3U CubeSat face areas (m²)
        # Faces: +x, -x, +y, -y, +z, -z
        self.A_face = np.array([0.01, 0.01,   # 10×10 cm side panels
                                 0.01, 0.01,
                                 0.03, 0.03])  # 10×30 cm top/bottom

        # Outward unit normals in body frame
        self.normals = np.array([
            [ 1,  0,  0],
            [-1,  0,  0],
            [ 0,  1,  0],
            [ 0, -1,  0],
            [ 0,  0,  1],
            [ 0,  0, -1],
        ], dtype=float)

        # Centre-of-pressure offset from CoM (m)
        # Small offset gives non-zero torque arm
        self.r_cop = np.array([0.001, 0.001, 0.005])

        # Mignard-Farinella optical coefficients per face
        # rho  = specular reflectivity
        # kappa = diffuse reflectivity
        # absorbed fraction = 1 - rho - kappa
        self.rho   = np.array([0.1, 0.1, 0.1, 0.1, 0.3, 0.3])   # specular
        self.kappa = np.array([0.6, 0.6, 0.6, 0.6, 0.4, 0.4])   # diffuse

    # ─────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────

    def compute(self, q: np.ndarray,
                sun_inertial: np.ndarray,
                pos_km: np.ndarray,
                sun_pos_km: np.ndarray
                ) -> tuple[np.ndarray, float]:
        """
        Compute SRP torque in body frame.

        Parameters
        ----------
        q            : quaternion [w, x, y, z] body→inertial
        sun_inertial : unit sun vector in ECI (from SunModel)
        pos_km       : spacecraft ECI position [km]
        sun_pos_km   : sun ECI position [km]  (from astropy, scaled)

        Returns
        -------
        T_srp : SRP torque vector in body frame [N·m]
        nu    : shadow function  1 = full sun, 0 = umbra, (0,1) = penumbra
        """
        nu = self._shadow_function(pos_km, sun_pos_km)

        if nu < 1e-9:
            return np.zeros(3), 0.0   # full eclipse — no SRP

        # SRP pressure at spacecraft (scales with distance²)
        r_sun_km = np.linalg.norm(sun_pos_km)
        P = self.P_srp_1AU * (self.AU_km / r_sun_km) ** 2

        # Sun direction in body frame
        R        = rot_matrix(q)
        sun_body = R @ sun_inertial
        sun_body = sun_body / (np.linalg.norm(sun_body) + 1e-12)

        T_srp = np.zeros(3)

        for i in range(6):
            cos_theta = np.dot(self.normals[i], sun_body)
            if cos_theta <= 0.0:
                continue   # face in shadow — no illumination

            rho_i   = self.rho[i]
            kappa_i = self.kappa[i]
            A_i     = self.A_face[i]

            # Mignard-Farinella force on face i (body frame)
            # F = -P·A·cos θ · [(1 − rho)·s_hat + 2·rho·cos θ·n_hat
            #                    + (2/3)·kappa·n_hat]
            F_i = -nu * P * A_i * cos_theta * (
                (1.0 - rho_i) * sun_body
                + 2.0 * rho_i * cos_theta * self.normals[i]
                + (2.0 / 3.0) * kappa_i * self.normals[i]
            )

            T_srp += np.cross(self.r_cop, F_i)

        return T_srp, nu

    # ─────────────────────────────────────────────────────────────────
    # Eclipse (shadow) model — dual-cone cylindrical
    # ─────────────────────────────────────────────────────────────────

    def _shadow_function(self,
                         pos_km: np.ndarray,
                         sun_pos_km: np.ndarray) -> float:
        """
        Dual-cone shadow model.

        Returns nu ∈ [0, 1]:
            1.0  — full sunlight
            0.0  — full umbra (eclipse)
            (0,1) — penumbra (partial eclipse)

        Reference: Montenbruck & Gill, Eq. 3.85–3.87.
        """
        r_sat = pos_km                          # km
        r_sun = sun_pos_km                      # km (ECI)

        # Vector from Sun to spacecraft
        d = r_sat - r_sun                       # km
        d_mag = np.linalg.norm(d)

        # Apparent angular radius of Sun and Earth as seen from spacecraft
        # alpha_sun : half-angle of Sun disc
        # alpha_earth: half-angle of Earth disc
        r_sun_mag   = np.linalg.norm(r_sun)    # Sun distance from Earth
        r_sat_mag   = np.linalg.norm(r_sat)    # sat distance from Earth

        # Satellite-to-Earth vector magnitude
        # (same as r_sat_mag since Earth is at origin)

        alpha_sun   = np.arcsin(self.R_sun_km   / r_sun_mag)   # rad
        alpha_earth = np.arcsin(self.R_earth_km / r_sat_mag)   # rad

        # Angular separation between Sun and Earth as seen from satellite
        # cos(theta) = -r_sat · r_sun / (|r_sat|·|r_sun|)
        cos_sep = -np.dot(r_sat, r_sun) / (r_sat_mag * r_sun_mag + 1e-12)
        cos_sep = np.clip(cos_sep, -1.0, 1.0)
        theta   = np.arccos(cos_sep)           # rad, separation angle

        # Penumbra/umbra boundary check
        # Full sun  : theta > alpha_sun + alpha_earth
        # Penumbra  : |alpha_sun - alpha_earth| < theta < alpha_sun + alpha_earth
        # Umbra     : theta < |alpha_earth - alpha_sun|  (and alpha_earth > alpha_sun → LEO)

        if theta >= alpha_sun + alpha_earth:
            return 1.0   # full sunlight

        if alpha_earth > alpha_sun:
            # LEO case — Earth disc larger than Sun disc → true umbra possible
            if theta <= alpha_earth - alpha_sun:
                return 0.0   # full umbra

        # Penumbra — linear interpolation of shadow fraction
        nu = (theta - abs(alpha_earth - alpha_sun)) / (
            (alpha_sun + alpha_earth) - abs(alpha_earth - alpha_sun) + 1e-12
        )
        return float(np.clip(nu, 0.0, 1.0))


# ═══════════════════════════════════════════════════════════════════
# Quick validation
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    srp = SolarRadiationPressure()

    q = np.array([1., 0., 0., 0.])
    sun_I = np.array([1., 0., 0.])

    # Full sun, 500 km altitude
    pos_km     = np.array([6871., 0., 0.])
    sun_pos_km = np.array([1.496e8, 0., 0.])

    T, nu = srp.compute(q, sun_I, pos_km=pos_km, sun_pos_km=sun_pos_km)
    print(f"Full sun   : nu={nu:.3f}, |T|={np.linalg.norm(T)*1e9:.2f} nN·m")

    # Eclipsed: satellite directly behind Earth from Sun
    pos_eclipse = np.array([-6871., 0., 0.])
    T2, nu2 = srp.compute(q, sun_I, pos_km=pos_eclipse, sun_pos_km=sun_pos_km)
    print(f"Eclipse    : nu={nu2:.3f}, |T|={np.linalg.norm(T2)*1e9:.2f} nN·m")
    print("Expected: full sun ~10-100 nN·m, eclipse ~0 nN·m")
