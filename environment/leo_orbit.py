"""
LEO Orbit Propagator — RK4 + J2 + Atmospheric Drag + SRP
=========================================================
Numerical propagator for LEO spacecraft.

Key differences from GEO:
    - Atmospheric drag is the DOMINANT perturbation (not SRP)
    - J2 is much stronger (closer to Earth)
    - Eclipse periods every ~90 min (frequent power cycling)
    - Period ~ 5600 s at 550 km
    - NRLMSISE-00 density model via simplified exponential fallback
      (avoids nrlmsise00 package dependency for the CA stack)

Public API is identical to GEOOrbitPropagator for drop-in reuse.
"""

import numpy as np


# Exponential atmosphere table (Vallado Table 8-4, two-layer fit)
# Columns: base_alt_km, scale_height_km, rho0_kg_m3
_ATMO_TABLE = np.array([
    [  0,   8.44, 1.225],
    [ 25,   6.49, 3.899e-2],
    [ 30,   6.75, 1.774e-2],
    [ 40,   7.07, 3.972e-3],
    [ 50,   7.47, 1.057e-3],
    [ 60,   7.83, 3.206e-4],
    [ 70,   7.95, 8.770e-5],
    [ 80,   9.47, 1.905e-5],
    [ 90,  10.83, 3.396e-6],
    [100,  16.84, 5.297e-7],
    [110,  23.84, 9.661e-8],
    [120,  37.56, 2.438e-8],
    [130,  41.94, 8.484e-9],
    [140,  45.62, 3.845e-9],
    [150,  52.72, 2.070e-9],
    [180, 111.84, 5.464e-10],
    [200, 167.43, 2.789e-10],
    [250, 225.55, 7.248e-11],
    [300, 281.49, 2.418e-11],
    [350, 313.22, 9.158e-12],
    [400, 347.51, 3.725e-12],
    [450, 378.14, 1.585e-12],
    [500, 408.57, 6.967e-13],
    [600, 470.81, 1.454e-13],
    [700, 553.42, 3.614e-14],
    [800, 639.20, 1.170e-14],
    [900, 701.12, 5.245e-15],
    [1000, 906.89, 3.019e-15],
], dtype=float)


def _atmo_density(alt_km: float) -> float:
    """Exponential atmosphere density [kg/m³]."""
    alt_km = float(np.clip(alt_km, 0.0, 1000.0))
    idx = np.searchsorted(_ATMO_TABLE[:, 0], alt_km, side='right') - 1
    idx = int(np.clip(idx, 0, len(_ATMO_TABLE) - 2))
    h0, H, rho0 = _ATMO_TABLE[idx, 0], _ATMO_TABLE[idx, 1], _ATMO_TABLE[idx, 2]
    return rho0 * np.exp(-(alt_km - h0) / H)


def _sun_pos_m(t_elapsed_s: float) -> np.ndarray:
    days = t_elapsed_s / 86400.0
    # Legacy mean annual Sun; precise common models use a physical ephemeris.
    lam  = np.radians(280.46 + 0.985647 * days)
    eps  = np.radians(23.439)
    AU   = 1.495978707e11
    return AU * np.array([np.cos(lam),
                           np.cos(eps) * np.sin(lam),
                           np.sin(eps) * np.sin(lam)])


class LEOOrbitPropagator:
    """
    RK4 orbit propagator for LEO spacecraft.

    Parameters
    ----------
    a_km      : semi-major axis [km]
    e         : eccentricity
    i_deg     : inclination [deg]
    raan_deg  : RAAN [deg]
    omega_deg : argument of perigee [deg]
    M0_deg    : initial mean anomaly [deg]
    Cd        : drag coefficient
    Am_ratio  : area/mass [m²/kg]
    Cr        : SRP reflectivity coefficient
    label     : name tag for print output
    """

    MU        = 3.986004418e14
    J2        = 1.08263e-3
    RE        = 6.3781e6
    AU_m      = 1.495978707e11
    P_SRP_1AU = 4.56e-6
    OMEGA_E   = np.array([0., 0., 7.2921150e-5])  # Earth rotation [rad/s]

    def __init__(self,
                 a_km:      float = 6928.0,
                 e:         float = 0.001,
                 i_deg:     float = 53.0,
                 raan_deg:  float = 0.0,
                 omega_deg: float = 0.0,
                 M0_deg:    float = 0.0,
                 Cd:        float = 2.2,
                 Am_ratio:  float = 0.005,
                 Cr:        float = 1.5,
                 label:     str   = "LEO"):

        self.a      = a_km * 1e3
        self.e      = e
        self.i      = np.radians(i_deg)
        self.raan   = np.radians(raan_deg)
        self.omega  = np.radians(omega_deg)
        self.Cd     = Cd
        self.Am     = Am_ratio
        self.Cr     = Cr
        self.label  = label

        self.n  = np.sqrt(self.MU / self.a**3)
        self.T  = 2.0 * np.pi / self.n

        M0 = np.radians(M0_deg)
        pos_m, vel_ms = self._elements_to_eci(self.a, e, self.i,
                                               self.raan, self.omega, M0)
        self.pos = pos_m  / 1e3   # km
        self.vel = vel_ms / 1e3   # km/s

        self.t_elapsed = 0.0

        alt_km = a_km - 6378.0
        print(f"  {label} Orbit: a={a_km:.1f} km  alt={alt_km:.0f} km  "
              f"i={i_deg:.1f}deg  T={self.T/60:.1f} min")

    # ── Public API (km / km/s — same as GEOOrbitPropagator) ───────────

    def step(self, dt: float):
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
        pos_m   = self.pos * 1e3
        sun_p   = _sun_pos_m(self.t_elapsed)
        RE, RS  = self.RE, 6.96e8
        r_sat   = np.linalg.norm(pos_m)
        r_sun   = np.linalg.norm(sun_p)
        a_sun   = np.arcsin(np.clip(RS / r_sun, -1, 1))
        a_earth = np.arcsin(np.clip(RE / r_sat, -1, 1))
        cos_sep = np.dot(-pos_m, sun_p) / (r_sat * r_sun + 1e-30)
        theta   = np.arccos(np.clip(cos_sep, -1.0, 1.0))
        if theta >= a_sun + a_earth:
            return 1.0
        if a_earth > a_sun and theta <= a_earth - a_sun:
            return 0.0
        denom = (a_sun + a_earth) - abs(a_earth - a_sun) + 1e-30
        return float(np.clip((theta - abs(a_earth - a_sun)) / denom, 0.0, 1.0))

    def get_sun_vector_eci(self) -> np.ndarray:
        p_m   = self.pos * 1e3
        sun_p = _sun_pos_m(self.t_elapsed)
        r     = sun_p - p_m
        return r / np.linalg.norm(r)

    def get_density(self) -> float:
        alt_km = np.linalg.norm(self.pos) - 6378.0
        return _atmo_density(alt_km)

    # ── Equations of motion ────────────────────────────────────────────

    def _deriv(self, pos: np.ndarray, vel: np.ndarray):
        return vel, self._a_2b(pos) + self._a_j2(pos) + self._a_drag(pos, vel) + self._a_srp(pos)

    def _a_2b(self, pos):
        return -self.MU / np.linalg.norm(pos)**3 * pos

    def _a_j2(self, pos):
        x, y, z = pos
        r = np.linalg.norm(pos)
        c = -1.5 * self.J2 * self.MU * self.RE**2 / r**5
        f = 5 * z**2 / r**2
        return np.array([c*x*(1-f), c*y*(1-f), c*z*(3-f)])

    def _a_drag(self, pos, vel):
        alt_km = np.linalg.norm(pos) / 1e3 - 6378.0
        rho = _atmo_density(alt_km)
        v_rel = vel - np.cross(self.OMEGA_E, pos)
        v_mag = np.linalg.norm(v_rel)
        if v_mag < 1.0 or rho < 1e-25:
            return np.zeros(3)
        return -0.5 * self.Cd * self.Am * rho * v_mag * v_rel

    def _a_srp(self, pos):
        sun_p = _sun_pos_m(self.t_elapsed)
        nu    = self.get_eclipse_nu()
        if nu < 1e-9:
            return np.zeros(3)
        r_sun  = np.linalg.norm(sun_p)
        P      = self.P_SRP_1AU * (self.AU_m / r_sun)**2
        r_sc   = pos - sun_p
        return nu * self.Cr * self.Am * P * r_sc / np.linalg.norm(r_sc)

    # ── Elements → ECI ────────────────────────────────────────────────

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
        ci, si = np.cos(i),     np.sin(i)
        cr, sr = np.cos(raan),  np.sin(raan)
        return np.array([
            [cr*co - sr*so*ci, -cr*so - sr*co*ci,  sr*si],
            [sr*co + cr*so*ci, -sr*so + cr*co*ci, -cr*si],
            [so*si,             co*si,              ci   ]
        ])
