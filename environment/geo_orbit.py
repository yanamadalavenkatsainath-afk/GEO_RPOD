"""
GEO Orbit Propagator — RK4 + J2 + SRP + Eclipse
=================================================
Numerical propagator for GEO and near-GEO elliptical orbits.
Replaces SGP4 which is only valid for LEO.

Key differences from LEO:
    - No atmospheric drag (rho < 1e-21 kg/m³ at GEO — effectively zero)
    - Solar radiation pressure is the DOMINANT perturbation (not drag)
    - Eclipse geometry is different: rare events near equinoxes only
    - GEO period = 23.93 hr (sidereal day), n = 7.292e-5 rad/s
    - Magnetic field is negligible — no magnetorquers at GEO

Eclipse model:
    Dual-cone (umbra + penumbra) using apparent angular radii
    of Sun and Earth as seen from the spacecraft.
    At GEO, eclipses occur ~44 days per year near equinoxes,
    lasting up to 72 minutes per event.
    nu = 1: full sun, nu = 0: full umbra, (0,1): penumbra

Reference:
    Vallado, "Fundamentals of Astrodynamics", §9.2-9.4
    Montenbruck & Gill, "Satellite Orbits", §3.2-3.4
"""

import numpy as np


class GEOOrbitPropagator:
    """
    RK4 orbit propagator for GEO spacecraft.

    Parameters
    ----------
    a_km      : semi-major axis [km]. Default: 42164 km (geostationary)
    e         : eccentricity. Default: 0.0001
    i_deg     : inclination [deg]. Default: 0.0
    raan_deg  : RAAN [deg]. Default: 0.0
    omega_deg : argument of perigee [deg]. Default: 0.0
    M0_deg    : initial mean anomaly [deg]. Default: 0.0
    Cr        : SRP coefficient (1=absorb, 2=reflect). Default: 1.5
    Am_ratio  : area/mass ratio [m²/kg]. Default: 0.003 (3U ~0.01m²/3kg)
    """

    MU        = 3.986004418e14   # m³/s²
    J2        = 1.08263e-3
    RE        = 6.3781e6         # m
    AU_m      = 1.495978707e11   # m
    P_SRP_1AU = 4.56e-6          # N/m²
    R_SUN_m   = 6.96e8           # m

    def __init__(self,
                 a_km:      float = 42164.0,
                 e:         float = 0.0001,
                 i_deg:     float = 0.0,
                 raan_deg:  float = 0.0,
                 omega_deg: float = 0.0,
                 M0_deg:    float = 0.0,
                 Cr:        float = 1.5,
                 Am_ratio:  float = 0.003):

        self.a      = a_km * 1e3
        self.e      = e
        self.i      = np.radians(i_deg)
        self.raan   = np.radians(raan_deg)
        self.omega  = np.radians(omega_deg)
        self.Cr     = Cr
        self.Am     = Am_ratio

        self.n      = np.sqrt(self.MU / self.a**3)
        self.T      = 2.0 * np.pi / self.n

        # Convert orbital elements → ECI (stored in km, km/s for compat)
        M0 = np.radians(M0_deg)
        self.pos, self.vel = self._elements_to_eci(
            self.a, e, self.i, self.raan, self.omega, M0)
        self.pos /= 1e3   # → km
        self.vel /= 1e3   # → km/s

        self.t_elapsed = 0.0

        print(f"  GEO Orbit: a={a_km:.0f} km, e={e:.4f}, i={i_deg:.1f}°, "
              f"T={self.T/3600:.2f} hr")

    # ─────────────────────────────────────────────────────────────────
    # Public API  (returns km, km/s — same convention as LEO SGP4)
    # ─────────────────────────────────────────────────────────────────

    def step(self, dt: float):
        """
        Propagate by dt seconds (RK4).

        Returns
        -------
        pos : ECI position [km]
        vel : ECI velocity [km/s]
        """
        p_m  = self.pos * 1e3
        v_ms = self.vel * 1e3

        k1p, k1v = self._deriv(p_m, v_ms)
        k2p, k2v = self._deriv(p_m + 0.5*dt*k1p, v_ms + 0.5*dt*k1v)
        k3p, k3v = self._deriv(p_m + 0.5*dt*k2p, v_ms + 0.5*dt*k2v)
        k4p, k4v = self._deriv(p_m + dt*k3p,      v_ms + dt*k3v)

        p_m  += (dt/6) * (k1p + 2*k2p + 2*k3p + k4p)
        v_ms += (dt/6) * (k1v + 2*k2v + 2*k3v + k4v)

        self.pos = p_m / 1e3
        self.vel = v_ms / 1e3
        self.t_elapsed += dt
        return self.pos.copy(), self.vel.copy()

    def get_pos_m(self)  -> np.ndarray: return self.pos * 1e3
    def get_vel_ms(self) -> np.ndarray: return self.vel * 1e3

    def get_eclipse_nu(self) -> float:
        """Shadow function nu ∈ [0,1]. 1=sun, 0=umbra."""
        return eclipse_nu(self.pos * 1e3, self.t_elapsed)

    def get_sun_vector_eci(self) -> np.ndarray:
        """Unit vector from spacecraft toward Sun, ECI."""
        p_m    = self.pos * 1e3
        sun_p  = _sun_pos_m(self.t_elapsed)
        r      = sun_p - p_m
        return r / np.linalg.norm(r)

    # ─────────────────────────────────────────────────────────────────
    # Equations of motion (SI)
    # ─────────────────────────────────────────────────────────────────

    def _deriv(self, pos: np.ndarray, vel: np.ndarray):
        a_2b  = -self.MU / np.linalg.norm(pos)**3 * pos
        a_j2  = self._j2(pos)
        a_srp = self._srp(pos)
        return vel, a_2b + a_j2 + a_srp

    def _j2(self, pos):
        x, y, z = pos
        r  = np.linalg.norm(pos)
        c  = -1.5 * self.J2 * self.MU * self.RE**2 / r**5
        f  = 5 * z**2 / r**2
        return np.array([c*x*(1-f), c*y*(1-f), c*z*(3-f)])

    def _srp(self, pos):
        sun_p = _sun_pos_m(self.t_elapsed)
        nu    = _shadow_dual_cone(pos, sun_p, self.RE, self.R_SUN_m)
        if nu < 1e-9:
            return np.zeros(3)
        r_sun = np.linalg.norm(sun_p)
        P     = self.P_SRP_1AU * (self.AU_m / r_sun)**2
        r_sc  = pos - sun_p
        return nu * self.Cr * self.Am * P * r_sc / np.linalg.norm(r_sc)

    # ─────────────────────────────────────────────────────────────────
    # Orbital elements → ECI
    # ─────────────────────────────────────────────────────────────────

    def _elements_to_eci(self, a, e, i, raan, omega, M):
        E = M
        for _ in range(50):
            dE = (M - E + e*np.sin(E)) / (1 - e*np.cos(E))
            E += dE
            if abs(dE) < 1e-12:
                break
        nu = 2*np.arctan2(np.sqrt(1+e)*np.sin(E/2), np.sqrt(1-e)*np.cos(E/2))
        r  = a*(1 - e*np.cos(E))
        p  = a*(1 - e**2)
        rp = np.array([r*np.cos(nu), r*np.sin(nu), 0.])
        vp = np.sqrt(self.MU/p) * np.array([-np.sin(nu), e+np.cos(nu), 0.])
        R  = self._peri_to_eci(i, raan, omega)
        return R @ rp, R @ vp

    @staticmethod
    def _peri_to_eci(i, raan, omega):
        co, so = np.cos(omega), np.sin(omega)
        ci, si = np.cos(i), np.sin(i)
        cr, sr = np.cos(raan), np.sin(raan)
        return np.array([
            [cr*co-sr*so*ci, -cr*so-sr*co*ci,  sr*si],
            [sr*co+cr*so*ci, -sr*so+cr*co*ci, -cr*si],
            [so*si,           co*si,            ci   ]
        ])


# ─────────────────────────────────────────────────────────────────────
# Module-level eclipse utilities (importable standalone)
# ─────────────────────────────────────────────────────────────────────

def _sun_pos_m(t_elapsed_s: float) -> np.ndarray:
    """
    Approximate Sun ECI position [m].
    Accurate to ~1° — sufficient for eclipse and SRP.
    """
    days = t_elapsed_s / 86400.0
    lam  = np.radians(280.46 + 360.985647 * days)
    eps  = np.radians(23.439)
    AU   = 1.495978707e11
    return AU * np.array([np.cos(lam),
                           np.cos(eps)*np.sin(lam),
                           np.sin(eps)*np.sin(lam)])


def _shadow_dual_cone(pos_m: np.ndarray,
                       sun_pos_m: np.ndarray,
                       R_earth: float = 6.3781e6,
                       R_sun:   float = 6.96e8) -> float:
    """
    Dual-cone shadow function nu ∈ [0,1].
    1.0 = full sun, 0.0 = full umbra, (0,1) = penumbra.

    Reference: Montenbruck & Gill, §3.4, Eq. 3.85–3.87.
    """
    r_sat = np.linalg.norm(pos_m)
    r_sun = np.linalg.norm(sun_pos_m)

    alpha_sun   = np.arcsin(np.clip(R_sun   / r_sun, -1, 1))
    alpha_earth = np.arcsin(np.clip(R_earth / r_sat, -1, 1))

    cos_sep = np.dot(-pos_m, sun_pos_m) / (r_sat * r_sun + 1e-30)
    theta   = np.arccos(np.clip(cos_sep, -1.0, 1.0))

    if theta >= alpha_sun + alpha_earth:
        return 1.0   # full sun

    if alpha_earth > alpha_sun and theta <= alpha_earth - alpha_sun:
        return 0.0   # full umbra (only possible in LEO, but handle for GEO inclined)

    denom = (alpha_sun + alpha_earth) - abs(alpha_earth - alpha_sun) + 1e-30
    return float(np.clip((theta - abs(alpha_earth - alpha_sun)) / denom, 0.0, 1.0))


def eclipse_nu(pos_km: np.ndarray, t_elapsed_s: float) -> float:
    """
    Shadow function for GEO spacecraft.

    Parameters
    ----------
    pos_km      : ECI position [km]
    t_elapsed_s : elapsed time since epoch [s]

    Returns
    -------
    nu : 1.0 = full sun, 0.0 = eclipse
    """
    return _shadow_dual_cone(pos_km * 1e3, _sun_pos_m(t_elapsed_s))


# ─────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== GEO Orbit Propagator Validation ===\n")

    orb = GEOOrbitPropagator(a_km=42164.0, e=0.0, i_deg=0.0)
    r0  = np.linalg.norm(orb.pos)
    print(f"Initial radius: {r0:.3f} km  (expected 42164 km)")

    # Propagate one sidereal day
    for _ in range(int(orb.T / 60)):
        orb.step(60.0)
    r1 = np.linalg.norm(orb.pos)
    print(f"After 1 GEO period ({orb.T/3600:.2f} hr): {r1:.3f} km")
    print(f"Radius drift: {abs(r1-r0)*1000:.1f} m  ({'✓ PASS' if abs(r1-r0)*1000 < 500 else '✗ FAIL'})")

    # Eclipse test at equinox
    print("\nEclipse test (vernal equinox, day 80):")
    nu_sun  = eclipse_nu(np.array([ 42164., 0., 0.]), 80*86400)
    nu_shad = eclipse_nu(np.array([-42164., 0., 0.]), 80*86400)
    print(f"  Sunlit side:  nu={nu_sun:.3f}  ({'✓' if nu_sun > 0.9 else '✗'})")
    print(f"  Shadow side:  nu={nu_shad:.3f}  ({'✓' if nu_shad < 0.1 else '✗'})")

    # Annual eclipse statistics
    print("\nAnnual eclipse statistics (60s resolution):")
    orb2 = GEOOrbitPropagator(a_km=42164.0, e=0.0, i_deg=0.0)
    total_s, events, in_ecl = 0.0, 0, False
    max_dur, cur_dur = 0.0, 0.0

    for _ in range(int(365*86400/60)):
        orb2.step(60.0)
        nu = orb2.get_eclipse_nu()
        if nu < 0.5:
            total_s  += 60
            cur_dur  += 60
            if not in_ecl:
                events += 1
                in_ecl  = True
        else:
            if in_ecl:
                max_dur = max(max_dur, cur_dur)
                cur_dur = 0.0
            in_ecl = False

    print(f"  Eclipse events/year : {events}")
    print(f"  Total eclipse time  : {total_s/3600:.1f} hr/year")
    print(f"  Max duration        : {max_dur/60:.0f} min")
    print(f"  Expected            : ~88 events, ~430 hr/year, ~72 min max")
