"""
Spacecraft rigid-body dynamics — 3U CubeSat
Corrected RW momentum coupling.

The equation of motion for a spacecraft with reaction wheels is:

    (I_sc + I_rw) · ω̇  =  τ_ext  -  τ_rw_cmd  -  ω × (I_sc·ω + h_rw)

Where:
    I_sc     = spacecraft bus inertia (constant, wheels are part of the body)
    τ_ext    = external torques (MTQ, drag, GG, SRP)
    τ_rw_cmd = torque commanded TO the wheels (reaction torque ON body = -τ_cmd)
    h_rw     = total angular momentum stored in wheels [N·m·s]
    ω × h_rw = gyroscopic coupling term (critical for accuracy)

The key sign convention:
    Wheels exert -τ_cmd on the body (Newton 3rd law).
    This is the tau_rw subtraction in omega_dot.
    Do NOT also add τ_cmd to tau_ext — that double-counts it.
    
    Correct call:  sc.step(torque_mtq, disturbance, dt, tau_rw=torque_rw)
    Wrong call:    sc.step(torque_rw + torque_mtq, disturbance, dt)  ← old code

Reference:
    Markley & Crassidis, "Fundamentals of Spacecraft Attitude
    Determination and Control", §7.5 — Eq. 7.108
    
    Wie, "Space Vehicle Dynamics and Control", §7.3
"""

import numpy as np
from utils.quaternion import normalize


class Spacecraft:

    def __init__(self, inertia):
        self.I     = inertia
        self.I_inv = np.linalg.inv(inertia)
        self.q     = np.array([1., 0., 0., 0.])
        self.omega = np.zeros(3)

    def omega_dot(self, omega, tau_ext, tau_rw_cmd, h_rw):
        gyro_sc = np.cross(omega, self.I @ omega)
        gyro_rw = np.cross(omega, h_rw)
        return self.I_inv @ (tau_ext + tau_rw_cmd - gyro_sc - gyro_rw)

    def quat_dot(self, q, omega):
        wx, wy, wz = omega
        Omega = np.array([
            [ 0,  -wx, -wy, -wz],
            [ wx,  0,   wz, -wy],
            [ wy, -wz,  0,   wx],
            [ wz,  wy, -wx,  0 ]
        ])
        return 0.5 * Omega @ q

    def step(self, tau_ext, disturbance, dt, tau_rw=None, h_rw=None):
        """
        Integrate spacecraft dynamics one timestep.

        Parameters
        ----------
        tau_ext    : external control torques — MTQ ONLY [N·m]
                     Do NOT include reaction wheel torque here.
        disturbance: environmental disturbance torques [N·m]
        dt         : timestep [s]
        tau_rw     : reaction wheel command torque [N·m]
                     Applied as -tau_rw on spacecraft body.
        h_rw       : current wheel momentum [N·m·s] for gyroscopic coupling.
                     If None, gyroscopic RW term is skipped (small effect at low h).
        """
        if tau_rw is None:
            tau_rw = np.zeros(3)
        if h_rw is None:
            h_rw = np.zeros(3)

        # ── Safety guards ──────────────────────────────────────────────
        if (np.any(np.isnan(self.omega)) or
                np.any(np.isinf(self.omega)) or
                np.linalg.norm(self.omega) > 500):
            self.omega = np.zeros(3)

        if (np.any(np.isnan(self.q)) or np.any(np.isinf(self.q))):
            self.q = np.array([1., 0., 0., 0.])

        tau_rw = np.clip(tau_rw, -1.0, 1.0)

        # Combined external: MTQ + disturbance
        tau_total_ext = tau_ext + disturbance

        def f(q, omega):
            qd = self.quat_dot(q, omega)
            wd = self.omega_dot(omega, tau_total_ext, tau_rw, h_rw)
            return qd, wd

        k1_q, k1_w = f(self.q, self.omega)
        k2_q, k2_w = f(self.q + 0.5*dt*k1_q, self.omega + 0.5*dt*k1_w)
        k3_q, k3_w = f(self.q + 0.5*dt*k2_q, self.omega + 0.5*dt*k2_w)
        k4_q, k4_w = f(self.q + dt*k3_q,     self.omega + dt*k3_w)

        self.q     += (dt/6) * (k1_q + 2*k2_q + 2*k3_q + k4_q)
        self.omega += (dt/6) * (k1_w + 2*k2_w + 2*k3_w + k4_w)
        self.q      = normalize(self.q)

        return self.q.copy(), self.omega.copy()