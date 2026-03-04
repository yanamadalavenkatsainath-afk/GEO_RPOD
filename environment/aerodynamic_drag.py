import numpy as np
import datetime
from utils.quaternion import rot_matrix
from nrlmsise00 import msise_flat


def _nrlmsise_density(alt_km, lat_deg, lon_deg, t_seconds, f107a, f107, ap):
    """
    Total mass density [kg/m³] from NRLMSISE-00.

    Confirmed signature from inspect:
        msise_model(time, alt, lat, lon, f107a, f107, ap, ...)
        msise_flat is a vectorized wrapper with identical arg order.

    d[5] = total mass density [g/cm³] → kg/m³.
    """
    epoch = datetime.datetime(2025, 1, 1, 0, 0, 0)
    t_dt  = epoch + datetime.timedelta(seconds=float(t_seconds))

    result = msise_flat(t_dt, alt_km, lat_deg, lon_deg, f107a, f107, ap)
    return float(result[5] * 1000.0)   # g/cm³ → kg/m³


class AerodynamicDrag:
    """
    Aerodynamic drag torque for a 3U CubeSat in LEO.

    Atmosphere : NRLMSISE-00 via the nrlmsise00 package.
    Force model: Flat-plate per-face box model, consistent with
                 solar_radiation_pressure.py.

    Reference:
        Vallado & McClain, "Fundamentals of Astrodynamics", §8.6
        Sentman (1961) free-molecular drag coefficient derivation.
    """

    R_EARTH_KM  = 6371.0
    OMEGA_EARTH = np.array([0., 0., 7.2921150e-5])   # rad/s

    def __init__(self, Cd=2.2, f107=150.0, f107a=150.0, ap=4.0):
        self.Cd    = Cd
        self.f107  = f107
        self.f107a = f107a
        self.ap    = ap

        # 3U CubeSat face areas [m²]: +x,-x,+y,-y,+z,-z
        self.A_face = np.array([0.01, 0.01,
                                 0.01, 0.01,
                                 0.03, 0.03])

        # Outward unit normals in body frame
        self.normals = np.array([
            [ 1,  0,  0],
            [-1,  0,  0],
            [ 0,  1,  0],
            [ 0, -1,  0],
            [ 0,  0,  1],
            [ 0,  0, -1],
        ], dtype=float)

        # Centre-of-pressure offset from CoM [m]
        self.r_cop = np.array([0.001, 0.001, 0.005])

        # Validate at init — confirms correct signature before sim runs
        print("  AeroDrag: validating NRLMSISE-00 call... ", end="", flush=True)
        rho_test = _nrlmsise_density(
            alt_km=500.0, lat_deg=0.0, lon_deg=0.0,
            t_seconds=0.0,
            f107a=self.f107a, f107=self.f107, ap=self.ap
        )
        print(f"OK  (rho @ 500km = {rho_test:.3e} kg/m³)")

    def compute(self, q, pos_km, vel_km_s, t_seconds=0.0):
        """
        Compute aerodynamic drag torque in body frame.

        Parameters
        ----------
        q         : quaternion [w,x,y,z]
        pos_km    : ECI position [km]
        vel_km_s  : ECI velocity [km/s]
        t_seconds : simulation time [s]

        Returns
        -------
        T_drag : drag torque in body frame [N·m]
        rho    : atmospheric density [kg/m³]
        """
        alt_km = np.linalg.norm(pos_km) - self.R_EARTH_KM

        r       = np.linalg.norm(pos_km)
        lat_deg = np.degrees(np.arcsin(pos_km[2] / r))
        lon_deg = np.degrees(np.arctan2(pos_km[1], pos_km[0]))

        rho = _nrlmsise_density(alt_km, lat_deg, lon_deg,
                                t_seconds,
                                self.f107a, self.f107, self.ap)

        # Velocity relative to rotating atmosphere [m/s]
        v_rel_I = (vel_km_s * 1e3
                   - np.cross(self.OMEGA_EARTH, pos_km * 1e3))
        v_mag = np.linalg.norm(v_rel_I)
        if v_mag < 1.0:
            return np.zeros(3), rho

        R      = rot_matrix(q)
        v_body = R @ v_rel_I
        v_hat  = v_body / v_mag

        T_drag = np.zeros(3)
        for i in range(6):
            cos_theta = -np.dot(self.normals[i], v_hat)
            if cos_theta <= 0.0:
                continue
            F_i = (0.5 * rho * v_mag**2
                   * self.Cd * self.A_face[i] * cos_theta
                   * (-v_hat))
            T_drag += np.cross(self.r_cop, F_i)

        return T_drag, rho


if __name__ == "__main__":
    drag = AerodynamicDrag()
    print()

    q        = np.array([1., 0., 0., 0.])
    pos_km   = np.array([6781., 0., 0.])    # ~410 km altitude
    vel_km_s = np.array([0., 7.67, 0.])

    T, rho = drag.compute(q, pos_km, vel_km_s, t_seconds=0.0)
    print("Aerodynamic Drag — 3U CubeSat")
    print("=" * 40)
    print(f"  Altitude : {np.linalg.norm(pos_km) - 6371:.0f} km")
    print(f"  rho      : {rho:.3e} kg/m³")
    print(f"  T_drag   : [{T[0]*1e9:.3f}, {T[1]*1e9:.3f}, {T[2]*1e9:.3f}] nN·m")
    print(f"  |T_drag| : {np.linalg.norm(T)*1e9:.3f} nN·m")
    print()
    print("Expected: rho ~3-7e-13 kg/m³ at 410 km, |T| ~1-50 nN·m")