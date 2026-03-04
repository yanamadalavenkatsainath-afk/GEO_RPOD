import numpy as np

# ================================================================
# IGRF-13 Spherical Harmonic Coefficients (2020.0 epoch)
# Source: International Association of Geomagnetism and Aeronomy
# https://www.ngdc.noaa.gov/IAGA/vmod/igrf.html
# Units: nanoTesla (nT)
# Truncated to degree/order 6 for computational efficiency
# Full model uses degree/order 13
# ================================================================

# Gauss coefficients g(n,m) and h(n,m) for IGRF-13 (2020.0)
# Organised as: g[n][m] and h[n][m]
IGRF13_G = {
    (1,0): -29404.5,
    (1,1): -1450.7,
    (2,0): -2500.0,
    (2,1):  2982.0,
    (2,2):  1676.7,
    (3,0):  1363.3,
    (3,1): -2381.2,
    (3,2):  1236.2,
    (3,3):   525.7,
    (4,0):   903.1,
    (4,1):   809.4,
    (4,2):   86.2,
    (4,3):  -309.4,
    (4,4):   47.9,
    (5,0):  -234.4,
    (5,1):   363.1,
    (5,2):   187.8,
    (5,3):  -140.7,
    (5,4):  -151.2,
    (5,5):    13.7,
    (6,0):    65.9,
    (6,1):    65.6,
    (6,2):    73.0,
    (6,3):  -121.5,
    (6,4):   -36.2,
    (6,5):    13.5,
    (6,6):   -64.7,
}

IGRF13_H = {
    (1,0):    0.0,
    (1,1):  4652.9,
    (2,0):    0.0,
    (2,1): -2991.6,
    (2,2):  -734.8,
    (3,0):    0.0,
    (3,1):   -82.2,
    (3,2):   241.8,
    (3,3):  -542.9,
    (4,0):    0.0,
    (4,1):   281.9,
    (4,2):   -158.4,
    (4,3):   199.7,
    (4,4):  -350.1,
    (5,0):    0.0,
    (5,1):    47.7,
    (5,2):   208.4,
    (5,3):  -121.3,
    (5,4):    32.2,
    (5,5):    99.1,
    (6,0):    0.0,
    (6,1):   -19.1,
    (6,2):    25.0,
    (6,3):    52.7,
    (6,4):   -64.4,
    (6,5):     9.0,
    (6,6):    68.1,
}

# Secular variation coefficients (nT/year) for 2020-2025
IGRF13_GD = {
    (1,0):  5.7,  (1,1): 7.4,
    (2,0): -11.5, (2,1): -7.0, (2,2):  2.8,
    (3,0):  2.4,  (3,1): -6.2, (3,2):  3.4, (3,3): -27.4,
    (4,0): -0.5,  (4,1):  2.0, (4,2): -3.8, (4,3):  -0.3, (4,4): 0.0,
}

IGRF13_HD = {
    (1,1): -25.9,
    (2,1): -30.2, (2,2): -23.9,
    (3,1):  5.7,  (3,2): -1.2, (3,3): 1.1,
    (4,1): -1.4,  (4,2):  5.0, (4,3): 3.0,  (4,4): 0.0,
}


class MagneticField:
    """
    IGRF-13 Magnetic Field Model
    Computes Earth's magnetic field at any position and time
    using spherical harmonic expansion up to degree/order 6.

    Reference: Alken et al. (2021), Earth, Planets and Space
    """

    def __init__(self, epoch_year=2025.0, n_max=6):
        self.a      = 6371.2          # Reference radius (km)
        self.n_max  = n_max
        self.epoch  = epoch_year

        # Build coefficient arrays with secular variation
        dt = epoch_year - 2020.0
        self.g = {}
        self.h = {}

        for key in IGRF13_G:
            self.g[key] = IGRF13_G[key] + IGRF13_GD.get(key, 0.0) * dt

        for key in IGRF13_H:
            self.h[key] = IGRF13_H[key] + IGRF13_HD.get(key, 0.0) * dt

        # Pre-compute Schmidt quasi-normal associated Legendre polynomials
        self._P_cache = {}

    def _associated_legendre(self, n_max, theta):
        """
        Compute Schmidt quasi-normalised associated Legendre polynomials
        P(n,m) and their derivatives dP(n,m) for given colatitude theta.
        """
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        P  = np.zeros((n_max + 2, n_max + 2))
        dP = np.zeros((n_max + 2, n_max + 2))

        P[0][0] = 1.0

        # Diagonal terms
        for n in range(1, n_max + 1):
            P[n][n] = sin_t * P[n-1][n-1] * np.sqrt(
                (2*n - 1) / (2*n)
            )

        # Off-diagonal terms
        for n in range(1, n_max + 1):
            P[n][n-1] = cos_t * P[n-1][n-1] * np.sqrt(2*n - 1)

        # Remaining terms
        for n in range(2, n_max + 1):
            for m in range(0, n - 1):
                K = ((n-1)**2 - m**2) / ((2*n-1) * (2*n-3))
                P[n][m] = cos_t * P[n-1][m] - np.sqrt(K) * P[n-2][m]

        # Derivatives
        for n in range(1, n_max + 1):
            for m in range(0, n + 1):
                if m == 0:
                    dP[n][0] = -P[n][1] / np.sqrt(2)
                else:
                    term1 = np.sqrt((n+m) * (n-m+1)) * P[n][m-1]
                    if m < n:
                        term2 = -np.sqrt((n-m) * (n+m+1)) * P[n][m+1]
                    else:
                        term2 = 0.0
                    dP[n][m] = 0.5 * (term1 + term2)

        return P, dP

    def _eci_to_spherical(self, pos_km):
        """Convert ECI Cartesian position to spherical coordinates."""
        x, y, z = pos_km
        r       = np.linalg.norm(pos_km)
        theta   = np.arccos(z / r)          # colatitude (0 at N pole)
        phi     = np.arctan2(y, x)          # longitude
        return r, theta, phi

    def get_field(self, pos_km=None):
        """
        Compute magnetic field vector in ECI frame.

        Parameters:
            pos_km: ECI position vector [x, y, z] in km

        Returns:
            B_eci: Magnetic field vector in ECI frame (Tesla)
        """
        # Fallback for backward compatibility
        if pos_km is None:
            return np.array([2e-5, -1e-5, 3e-5])

        r, theta, phi = self._eci_to_spherical(pos_km)

        # Compute Legendre polynomials
        P, dP = self._associated_legendre(self.n_max, theta)

        # Magnetic field in spherical coordinates (nT)
        Br     = 0.0    # radial component
        Btheta = 0.0    # colatitude component
        Bphi   = 0.0    # longitude component

        for n in range(1, self.n_max + 1):
            ratio = (self.a / r) ** (n + 2)

            for m in range(0, n + 1):
                g_nm = self.g.get((n, m), 0.0)
                h_nm = self.h.get((n, m), 0.0)

                cos_m_phi = np.cos(m * phi)
                sin_m_phi = np.sin(m * phi)

                gh_cos = g_nm * cos_m_phi + h_nm * sin_m_phi
                gh_sin = m * (-g_nm * sin_m_phi + h_nm * cos_m_phi)

                Br     -= (n + 1) * ratio * P[n][m]  * gh_cos
                Btheta -= ratio * dP[n][m] * gh_cos
                Bphi   += ratio * P[n][m]  * gh_sin / np.sin(theta + 1e-10)

        # Convert nT to Tesla
        Br     *= 1e-9
        Btheta *= 1e-9
        Bphi   *= 1e-9

        # Convert spherical to ECI Cartesian
        sin_t = np.sin(theta)
        cos_t = np.cos(theta)
        sin_p = np.sin(phi)
        cos_p = np.cos(phi)

        Bx = (Br * sin_t * cos_p
              + Btheta * cos_t * cos_p
              - Bphi * sin_p)

        By = (Br * sin_t * sin_p
              + Btheta * cos_t * sin_p
              + Bphi * cos_p)

        Bz = (Br * cos_t
              - Btheta * sin_t)

        return np.array([Bx, By, Bz])


# ================================================================
# Quick validation — run this file directly to check
# ================================================================
if __name__ == "__main__":
    mag = MagneticField(epoch_year=2025.0)

    # Test at equator, 500km altitude
    pos_test = np.array([6871.0, 0.0, 0.0])   # km
    B = mag.get_field(pos_test)
    B_nT = B * 1e9

    print("IGRF-13 Magnetic Field Test")
    print(f"Position: {pos_test} km")
    print(f"B field:  [{B_nT[0]:.1f}, {B_nT[1]:.1f}, {B_nT[2]:.1f}] nT")
    print(f"|B|:      {np.linalg.norm(B_nT):.1f} nT")
    print(f"Expected: ~25,000 - 65,000 nT for LEO")