"""
Chief Tumble Model — Phase 6
=============================
Models a free-tumbling non-cooperative target (IS-1002 class GEO comsat).
Provides:
    1. Chief attitude quaternion (free precession, no control)
    2. Docking port position in LVLH — the moving target for terminal guidance
    3. Angular rate estimation from pose differences (what the servicer can observe)

Physical model
--------------
The chief is a fuel-depleted comsat. Without propulsion or attitude control
it will develop a slow tumble from:
    - Residual angular momentum at fuel depletion
    - Gravity gradient torque (aligns long axis with nadir, ~1e-7 Nm)
    - Solar radiation pressure torque (varies with attitude, ~1e-8 Nm)

Typical tumble rates for a dead GEO satellite: 0.5–5 deg/s.
IS-1002 class (3-axis stabilised box + 30m² solar panels):
    Mass: ~2000 kg EOL, Inertia approximately diag([5000, 8000, 3000]) kg·m²
    Characteristic period: ~60–600s depending on tumble axis

Docking port
------------
The servicer must dock at a known port on the chief body frame.
For IS-1002 class: docking port assumed at +Z body face (anti-Earth face
on a healthy satellite, now tumbling). Port defined in chief body frame.

In Phase 6, terminal guidance tracks the port position in LVLH by:
    1. Knowing the docking port location in chief body frame
    2. Knowing the chief attitude (from vision pose estimation — Phase 5+)
    3. Computing port LVLH position = chief_pos_lvlh + R_body2lvlh @ port_body

Phase 7 will replace truth attitude with estimated attitude from the EKF.

Reference:
    Hablani et al., "Guidance and Relative Navigation for Autonomous
    Rendezvous in a Circular Orbit", JGCD 25(3), 2002.
    Setterfield et al., "Inertial Estimation of a Spinning Object", 2018.
"""

import numpy as np
from utils.quaternion import normalize, quat_multiply


# ── Chief physical parameters (IS-1002 class) ─────────────────────────
CHIEF_MASS_KG = 2000.0
CHIEF_INERTIA  = np.diag([5000.0, 8000.0, 3000.0])   # kg·m²
CHIEF_INERTIA_INV = np.linalg.inv(CHIEF_INERTIA)

# Docking port in chief body frame [m]
# Positioned at the +Z face centre, 1.2m from CoM
DOCKING_PORT_BODY = np.array([0.0, 0.0, 1.2])

# Approach corridor: deputy must arrive within this half-angle
# of the docking port normal (+Z body axis) for capture
DOCK_CONE_HALF_ANGLE_DEG = 10.0


class ChiefTumble:
    """
    Free-tumbling chief spacecraft dynamics.

    Integrates Euler's equations for a rigid body with no active control.
    Gravity gradient torque is included as the dominant environmental torque
    at GEO (SRP torque is smaller and attitude-dependent — included as
    optional perturbation for fidelity).

    Parameters
    ----------
    omega0      : initial angular rate [rad/s] in body frame
    q0          : initial quaternion [w,x,y,z] body → LVLH
    inertia     : 3×3 inertia tensor [kg·m²]. Default: IS-1002 class
    tumble_rate_deg_s : if omega0 is None, sets a random tumble at this rate
    """

    def __init__(self,
                 omega0:           np.ndarray = None,
                 q0:               np.ndarray = None,
                 inertia:          np.ndarray = None,
                 tumble_rate_deg_s: float = 1.5):

        self.I     = inertia if inertia is not None else CHIEF_INERTIA.copy()
        self.I_inv = np.linalg.inv(self.I)

        # Initial attitude — random if not specified
        if q0 is not None:
            self.q = normalize(np.array(q0, dtype=float))
        else:
            # Random initial attitude (uniform on SO(3))
            axis  = np.random.randn(3)
            axis /= np.linalg.norm(axis)
            angle = np.random.uniform(0, 2*np.pi)
            self.q = np.array([
                np.cos(angle/2),
                *(np.sin(angle/2) * axis)
            ])

        # Initial angular rate
        if omega0 is not None:
            self.omega = np.array(omega0, dtype=float)
        else:
            # Random tumble axis, specified rate
            axis  = np.random.randn(3)
            axis /= np.linalg.norm(axis)
            self.omega = np.radians(tumble_rate_deg_s) * axis

        self.t = 0.0
        print(f"  ChiefTumble: |omega|={np.degrees(np.linalg.norm(self.omega)):.2f} deg/s  "
              f"period~{2*np.pi/max(np.linalg.norm(self.omega), 1e-6):.0f}s")

    # ─────────────────────────────────────────────────────────────────
    # Integration
    # ─────────────────────────────────────────────────────────────────

    def step(self, dt: float, tau_ext: np.ndarray = None) -> tuple:
        """
        Propagate chief tumble dynamics by dt seconds (RK4).

        Parameters
        ----------
        dt      : timestep [s]
        tau_ext : external torque in body frame [N·m]. Default: zeros.

        Returns
        -------
        q     : updated quaternion [w,x,y,z]
        omega : updated angular rate [rad/s]
        """
        if tau_ext is None:
            tau_ext = np.zeros(3)

        def deriv(q, omega):
            # Euler's equation: I·ω̇ = τ_ext − ω × (I·ω)
            omega_dot = self.I_inv @ (tau_ext - np.cross(omega, self.I @ omega))

            # Quaternion kinematics: q̇ = 0.5 * Ω(ω) * q
            wx, wy, wz = omega
            Omega = np.array([
                [ 0,  -wx, -wy, -wz],
                [ wx,  0,   wz, -wy],
                [ wy, -wz,  0,   wx],
                [ wz,  wy, -wx,  0 ]
            ])
            q_dot = 0.5 * Omega @ q
            return q_dot, omega_dot

        # RK4
        k1q, k1w = deriv(self.q,                    self.omega)
        k2q, k2w = deriv(self.q + 0.5*dt*k1q,      self.omega + 0.5*dt*k1w)
        k3q, k3w = deriv(self.q + 0.5*dt*k2q,      self.omega + 0.5*dt*k2w)
        k4q, k4w = deriv(self.q + dt*k3q,           self.omega + dt*k3w)

        self.q     = normalize(self.q + (dt/6)*(k1q + 2*k2q + 2*k3q + k4q))
        self.omega += (dt/6)*(k1w + 2*k2w + 2*k3w + k4w)
        self.t     += dt

        return self.q.copy(), self.omega.copy()

    # ─────────────────────────────────────────────────────────────────
    # Port tracking
    # ─────────────────────────────────────────────────────────────────

    def get_port_lvlh(self, chief_pos_lvlh: np.ndarray = None) -> np.ndarray:
        """
        Docking port position in LVLH frame.

        Parameters
        ----------
        chief_pos_lvlh : chief CoM position in LVLH [m].
                         At origin for the chief (LVLH is chief-centred),
                         so pass np.zeros(3) or None.

        Returns
        -------
        port_lvlh : docking port position in LVLH [m]
        """
        R_b2l = self._q_to_R(self.q)   # body → LVLH rotation matrix
        offset = R_b2l @ DOCKING_PORT_BODY
        if chief_pos_lvlh is None:
            return offset
        return chief_pos_lvlh + offset

    def get_port_normal_lvlh(self) -> np.ndarray:
        """
        Docking port approach normal in LVLH frame.
        Port is on +Z body face, so normal = R_body2lvlh @ [0,0,1].
        Deputy must approach along this vector.
        """
        R_b2l = self._q_to_R(self.q)
        return R_b2l @ np.array([0., 0., 1.])

    def get_port_velocity_lvlh(self, dt: float = 0.1) -> np.ndarray:
        """
        Velocity of the docking port in LVLH frame [m/s].
        Computed as ω × r_port (in LVLH).

        This is what the servicer must match at the moment of capture.
        """
        R_b2l    = self._q_to_R(self.q)
        omega_l  = R_b2l @ self.omega              # angular rate in LVLH
        port_l   = R_b2l @ DOCKING_PORT_BODY      # port offset in LVLH
        return np.cross(omega_l, port_l)

    def is_approach_window_open(self, dep_pos_lvlh: np.ndarray) -> bool:
        """
        Returns True if deputy is within the docking cone.

        The deputy approach direction must be within DOCK_CONE_HALF_ANGLE_DEG
        of the port normal (docking funnel constraint).
        """
        port_lvlh = self.get_port_lvlh()
        dep_to_port = port_lvlh - dep_pos_lvlh
        rng = np.linalg.norm(dep_to_port)
        if rng < 1e-3:
            return True
        approach_dir = -dep_to_port / rng         # deputy approach direction
        normal       = self.get_port_normal_lvlh()
        cos_angle    = np.dot(approach_dir, normal)
        angle_deg    = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        return angle_deg < DOCK_CONE_HALF_ANGLE_DEG

    # ─────────────────────────────────────────────────────────────────
    # Observable quantities (what the servicer can estimate)
    # ─────────────────────────────────────────────────────────────────

    def observe_omega(self, sigma_deg_s: float = 0.05) -> np.ndarray:
        """
        Noisy angular rate estimate from pose differentiation.

        In reality this comes from: omega_est ≈ (R_k - R_{k-1}) / dt
        mapped to a rotation vector. Here we add Gaussian noise to truth.

        Parameters
        ----------
        sigma_deg_s : angular rate estimation noise [deg/s per axis]

        Returns
        -------
        omega_est : estimated angular rate in body frame [rad/s]
        """
        noise = np.random.normal(0, np.radians(sigma_deg_s), 3)
        return self.omega + noise

    def observe_quaternion(self, sigma_deg: float = 0.5) -> np.ndarray:
        """
        Noisy attitude estimate from vision pose estimation (PnP output).

        Parameters
        ----------
        sigma_deg : attitude estimation noise [deg per axis]

        Returns
        -------
        q_est : estimated quaternion [w,x,y,z]
        """
        noise_axis = np.random.randn(3)
        noise_axis_norm = np.linalg.norm(noise_axis)
        if noise_axis_norm > 1e-9:
            noise_axis /= noise_axis_norm
        noise_angle = np.radians(sigma_deg) * np.random.randn()
        dq = np.array([
            np.cos(noise_angle/2),
            *(np.sin(noise_angle/2) * noise_axis)
        ])
        return normalize(quat_multiply(dq, self.q))

    # ─────────────────────────────────────────────────────────────────
    # Gravity gradient torque on chief
    # ─────────────────────────────────────────────────────────────────

    def gravity_gradient_torque(self,
                                 r_chief_eci: np.ndarray,
                                 mu: float = 3.986004418e14) -> np.ndarray:
        """
        Gravity gradient torque in chief body frame [N·m].

        T_gg = 3μ/r³ * r̂_body × (I · r̂_body)

        This is the dominant attitude perturbation for a dead GEO satellite
        and drives the long-term attitude evolution.
        """
        r_mag   = np.linalg.norm(r_chief_eci)
        r_hat_e = r_chief_eci / r_mag

        # Rotate nadir vector to body frame
        R_l2b   = self._q_to_R(self.q).T    # LVLH→body (q is body→LVLH)
        r_hat_b = R_l2b @ r_hat_e

        coeff = 3.0 * mu / r_mag**3
        return coeff * np.cross(r_hat_b, self.I @ r_hat_b)

    # ─────────────────────────────────────────────────────────────────
    # Diagnostics
    # ─────────────────────────────────────────────────────────────────

    def print_status(self, t: float):
        port = self.get_port_lvlh()
        port_vel = self.get_port_velocity_lvlh()
        print(f"  [Chief tumble t={t:.0f}s]"
              f"  |omega|={np.degrees(np.linalg.norm(self.omega)):.3f}deg/s"
              f"  port_lvlh=[{port[0]:.2f},{port[1]:.2f},{port[2]:.2f}]m"
              f"  port_vel=[{port_vel[0]*1e3:.1f},{port_vel[1]*1e3:.1f},{port_vel[2]*1e3:.1f}]mm/s")

    # ─────────────────────────────────────────────────────────────────
    # Utility
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _q_to_R(q: np.ndarray) -> np.ndarray:
        """Quaternion [w,x,y,z] to rotation matrix (body → LVLH)."""
        w, x, y, z = q
        return np.array([
            [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
            [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]
        ])


# ─────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Chief Tumble Validation ===\n")
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

    # Test 1: Fixed-axis tumble, angular momentum conserved
    print("Test 1: Angular momentum conservation (no external torque)")
    I = np.diag([5000., 8000., 3000.])
    omega0 = np.radians(np.array([1.0, 0.2, 0.5]))   # deg/s
    chief = ChiefTumble(omega0=omega0, inertia=I)
    L0 = I @ chief.omega

    dt = 0.1
    for _ in range(6000):   # 10 minutes
        q, omega = chief.step(dt)

    L1 = I @ chief.omega
    dL = np.linalg.norm(L1 - L0) / np.linalg.norm(L0)
    print(f"  Angular momentum drift: {dL*100:.4f}%  "
          f"({'PASS' if dL < 1e-6 else 'FAIL'})")
    print(f"  |omega| change: {np.degrees(np.linalg.norm(omega0)):.4f} → "
          f"{np.degrees(np.linalg.norm(chief.omega)):.4f} deg/s")

    # Test 2: Port position tracks with rotation
    print("\nTest 2: Docking port tracks correctly")
    chief2 = ChiefTumble(omega0=np.array([0., 0., np.radians(1.0)]),
                          q0=np.array([1., 0., 0., 0.]), inertia=I)
    port0 = chief2.get_port_lvlh()

    # After 90 degrees of rotation about Z: port should rotate
    for _ in range(int(90.0 / (1.0 * 0.1))):   # 90s at 1 deg/s
        chief2.step(0.1)

    port1 = chief2.get_port_lvlh()
    print(f"  Port at t=0:   [{port0[0]:.3f}, {port0[1]:.3f}, {port0[2]:.3f}]m")
    print(f"  Port at t=90s: [{port1[0]:.3f}, {port1[1]:.3f}, {port1[2]:.3f}]m")
    print(f"  Port distance from CoM: {np.linalg.norm(port0):.3f}m (const)  "
          f"{np.linalg.norm(port1):.3f}m")
    dist_ok = abs(np.linalg.norm(port0) - np.linalg.norm(port1)) < 1e-3
    print(f"  {'PASS' if dist_ok else 'FAIL'} — port distance conserved")

    # Test 3: Approach window
    print("\nTest 3: Docking cone check")
    chief3 = ChiefTumble(omega0=np.zeros(3), q0=np.array([1.,0.,0.,0.]), inertia=I)
    # Port is at [0,0,1.2], normal is +Z_lvlh. Deputy approaching from -Z.
    dep_on_axis  = np.array([0., 0., -2.0])    # on approach axis
    dep_off_axis = np.array([0., 5., 0.])      # perpendicular — outside cone
    print(f"  Approach on-axis:  window={'OPEN' if chief3.is_approach_window_open(dep_on_axis) else 'CLOSED'} (expect OPEN)")
    print(f"  Approach off-axis: window={'OPEN' if chief3.is_approach_window_open(dep_off_axis) else 'CLOSED'} (expect CLOSED)")