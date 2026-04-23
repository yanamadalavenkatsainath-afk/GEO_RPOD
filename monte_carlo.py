"""
Monte Carlo — GEO RPOD Autonomous Docking
==========================================
300 randomised simulation runs, 15 parallel workers.

Varied parameters per run
--------------------------
  seed              : RNG seed (0–299) for full reproducibility
  tumble_rate       : chief angular rate [deg/s], uniform [0.03, 0.20]
  tumble_axis       : random unit vector (uniform on S²)
  initial_attitude  : random quaternion (uniform SO(3))
  sigma_px          : camera pixel noise [px], normal(1.5, 0.3), clipped [0.5, 3.0]
  P_detect          : feature detection probability, normal(0.75, 0.08), clipped [0.5, 0.95]
  P_mismatch        : matching error rate, normal(0.10, 0.04), clipped [0.0, 0.25]
  lambda_fp         : false positive rate, normal(1.5, 0.5), clipped [0.0, 4.0]
  omega_gyro_noise  : deputy gyro ARW multiplier, uniform [0.5, 2.0]
  initial_offset    : formation hold offset ± 50m noise in LVLH
  T_SIM_MAX         : 90 000s ceiling (25 hr) — increased for hard cases

Outputs (per run)
-----------------
  success        : bool — docked within T_SIM_MAX
  dock_time_s    : float — time to docking [s] (nan if failed)
  total_dv_mm_s  : float — total delta-V [mm/s]
  final_range_m  : float — range at end of sim [m]
  ekf_pos_err_mean_m : mean EKF position error during PROX_OPS/TERMINAL
  tumble_rate_deg_s  : actual tumble rate [deg/s]
  omega_est_err_deg_s: mean omega estimation error [deg/s]

Summary statistics printed at end:
  success rate, docking time (mean±std, 5/95 percentile),
  delta-V (mean±std), worst-case fuel use.

Usage
-----
    python monte_carlo.py                # 300 runs, 15 workers (default)
    python monte_carlo.py --n 100 --w 8 # 100 runs, 8 workers
    python monte_carlo.py --seed0 42    # start seed at 42
"""


import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

# ── Add project root to path ─────────────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

# ── Imports (same as main.py) ────────────────────────────────────────
from plant.spacecraft                     import Spacecraft
from environment.magnetic_field           import MagneticField
from environment.gravity_gradient         import GravityGradient
from environment.solar_radiation_pressure import SolarRadiationPressure
from environment.geo_orbit                import GEOOrbitPropagator, eclipse_nu
from environment.cw_dynamics              import CWDynamics
from sensors.gyro                         import Gyro
from sensors.magnetometer                 import Magnetometer
from sensors.sun_sensor                   import SunSensor
from sensors.star_tracker                 import StarTracker
from sensors.ranging_sensor               import RangingBearingSensor
from sensors.camera_sensor                import CameraSensor
from actuators.reaction_wheel             import ReactionWheel
from actuators.magnetorquer               import Magnetorquer
from actuators.bdot                       import BDotController
from estimation.mekf                      import MEKF
from estimation.quest                     import QUEST
from estimation.th_ekf                    import THEKF
from control.attitude_controller          import AttitudeController
from control.lambert_controller           import GEORPODController, RPODMode
from fsw.mode_manager                     import ModeManager, Mode
from utils.quaternion                     import quat_error
from chief_attitude                       import ChiefAttitude
from chief_pose_estimator                 import ChiefPoseEstimator


# =====================================================================
#  RESULT DATACLASS
# =====================================================================

@dataclass
class RunResult:
    seed:               int
    success:            bool
    dock_time_s:        float          # nan if failed
    total_dv_mm_s:      float
    final_range_m:      float
    ekf_pos_err_mean_m: float
    ekf_vel_err_mean_ms:float
    tumble_rate_deg_s:  float
    omega_est_err_deg_s:float          # mean |omega_est - omega_truth| [deg/s]
    sigma_px:           float
    P_detect:           float
    P_mismatch:         float
    cam_success_rate:   float          # fraction of cam measurements that returned valid z
    error_msg:          str            # empty if success, traceback if crashed


# =====================================================================
#  FIXED MISSION CONSTANTS (same as main.py)
# =====================================================================

CHIEF_A_KM      = 42164.0
CHIEF_E         = 0.0003
CHIEF_I_DEG     = 0.8
CHIEF_RAAN_DEG  = 0.0
CHIEF_OMEGA_DEG = 0.0
CHIEF_M0_DEG    = 0.0
DEP_MASS_KG     = 50.0
DEP_THRUST_N    = 1.0
DEP_CR          = 1.5
DEP_AM          = 0.00720
CHI_CR          = 1.5
CHI_AM          = 0.015
MU_GEO          = 3.986004418e14
N_GEO           = np.sqrt(MU_GEO / (CHIEF_A_KM * 1e3) ** 3)
FORMATION_OFFSET_M = np.array([0.0, -1000.0, 0.0])
DT_OUTER        = 0.1
DT_INNER        = 0.01
N_INNER         = int(DT_OUTER / DT_INNER)
DOCK_RANGE_M    = 0.10
DOCK_VREL_MS    = 0.01
ECLIPSE_NU_MIN  = 0.1
ADCS_STABLE_DEG = 1.0
ADCS_STABLE_SUST= 100
FORM_HOLD_SETTLE_S = 300.0


def R_eci2lvlh(r_chief, v_chief):
    x_hat = r_chief / np.linalg.norm(r_chief)
    h_vec = np.cross(r_chief, v_chief)
    z_hat = h_vec / np.linalg.norm(h_vec)
    y_hat = np.cross(z_hat, x_hat)
    return np.vstack([x_hat, y_hat, z_hat])


def propagate_full_force(pos, vel, dt_total, t_abs, Cr, Am, substep=60.0):
    J2 = 1.08263e-3; RE = 6.3781e6; AU = 1.495978707e11; P0 = 4.56e-6
    def sun_pos(t):
        d = t/86400.; lam = np.radians(280.46+360.985647*d)
        eps = np.radians(23.439)
        return AU*np.array([np.cos(lam), np.cos(eps)*np.sin(lam),
                             np.sin(eps)*np.sin(lam)])
    def accel(p, t):
        r = np.linalg.norm(p)
        a = -MU_GEO/r**3*p
        x,y,z = p; c = -1.5*J2*MU_GEO*RE**2/r**5; f = 5*z**2/r**2
        a += np.array([c*x*(1-f), c*y*(1-f), c*z*(3-f)])
        sp = sun_pos(t); rs = np.linalg.norm(sp); P = P0*(AU/rs)**2
        dr = p-sp; a += Cr*Am*P*dr/np.linalg.norm(dr)
        return a
    n = max(1, int(round(dt_total/substep))); h = dt_total/n
    p, v, t = pos.copy(), vel.copy(), float(t_abs)
    for _ in range(n):
        k1p=v;              k1v=accel(p,t)
        k2p=v+0.5*h*k1v;   k2v=accel(p+0.5*h*k1p, t+0.5*h)
        k3p=v+0.5*h*k2v;   k3v=accel(p+0.5*h*k2p, t+0.5*h)
        k4p=v+h*k3v;        k4v=accel(p+h*k3p,     t+h)
        p += (h/6)*(k1p+2*k2p+2*k3p+k4p)
        v += (h/6)*(k1v+2*k2v+2*k3v+k4v)
        t += h
    return p, v


# =====================================================================
#  SINGLE RUN
# =====================================================================

def run_single(seed: int, T_SIM_MAX: float = 90_000.0) -> RunResult:
    """Run one Monte Carlo trial with the given seed."""

    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    # ── Draw randomised parameters ────────────────────────────────────
    tumble_rate  = float(rng.uniform(0.03, 0.20))    # deg/s
    tumble_axis  = rng.standard_normal(3)
    tumble_axis /= np.linalg.norm(tumble_axis)
    omega0       = np.radians(tumble_rate) * tumble_axis

    # Random initial chief attitude (uniform SO(3))
    q0_raw = rng.standard_normal(4)
    q0     = q0_raw / np.linalg.norm(q0_raw)

    # Camera noise parameters
    sigma_px   = float(np.clip(rng.normal(1.5, 0.3),   0.5, 3.0))
    P_detect   = float(np.clip(rng.normal(0.75, 0.08), 0.5, 0.95))
    P_mismatch = float(np.clip(rng.normal(0.10, 0.04), 0.0, 0.25))
    lambda_fp  = float(np.clip(rng.normal(1.5, 0.5),   0.0, 4.0))

    # Initial standoff offset (± 50m noise)
    offset_noise = rng.uniform(-50, 50, 3) * np.array([0.5, 1.0, 0.3])
    form_offset  = FORMATION_OFFSET_M + offset_noise

    # ── Instantiate all objects (silent — suppress prints in workers) ──
    import io, contextlib
    buf = io.StringIO()

    with contextlib.redirect_stdout(buf):
        I_sc = np.diag([4.167, 4.167, 3.000])
        chief_orbit = GEOOrbitPropagator(
            a_km=CHIEF_A_KM, e=CHIEF_E, i_deg=CHIEF_I_DEG,
            raan_deg=CHIEF_RAAN_DEG, omega_deg=CHIEF_OMEGA_DEG,
            M0_deg=CHIEF_M0_DEG, Cr=CHI_CR, Am_ratio=CHI_AM)
        mag_field  = MagneticField(epoch_year=2025.0)
        gg         = GravityGradient(I_sc)
        srp        = SolarRadiationPressure()
        sc         = Spacecraft(I_sc)
        sc.omega   = rng.uniform(-0.3, 0.3, 3)
        mag_sens   = Magnetometer()
        sun_sens   = SunSensor()
        gyro       = Gyro(dt=DT_INNER, bias_init_max_deg_s=0.05)
        star_tracker = StarTracker(sigma_cross_arcsec=5., sigma_roll_arcsec=20.,
                                   sun_excl_deg=30., earth_excl_deg=20.,
                                   update_rate_hz=4., acquisition_s=30.)
        rng_sensor = RangingBearingSensor(
            sigma_range_m=1.0, sigma_range_frac=0.001,
            sigma_angle_rad=np.radians(0.05),
            fov_half_deg=60., max_range_m=5000., min_range_m=0.05)
        cam_sensor = CameraSensor(
            sigma_px=sigma_px, P_detect=P_detect,
            P_mismatch=P_mismatch, lambda_fp=lambda_fp,
            min_range_m=0.05, max_range_m=600.)
        rw   = ReactionWheel(h_max=4.0)
        mtq  = Magnetorquer(m_max=0.2)
        bdot = BDotController(k_bdot=2e5, m_max=0.2)
        quest_alg = QUEST()
        mekf      = MEKF(DT_INNER)
        mekf.P[0:3, 0:3] = np.eye(3) * np.radians(5.)**2
        th_ekf = THEKF(a_chief=CHIEF_A_KM*1e3, e_chief=CHIEF_E,
                       dt=DT_OUTER, q_pos=1e-4, q_vel=1e-8)
        att_ctrl = AttitudeController(Kp=0.08284, Kd=0.82257)
        q_ref    = np.array([1., 0., 0., 0.])
        rpod_ctrl = GEORPODController(
            mu=MU_GEO, n_chief=N_GEO,
            dep_mass_kg=DEP_MASS_KG, dep_thrust_N=DEP_THRUST_N,
            Cr_chi=CHI_CR, Am_chi=CHI_AM,
            dock_capture_m=DOCK_RANGE_M,
            ekf=th_ekf, rng_sensor=rng_sensor)
        rpod_ctrl.standoff = np.linalg.norm(form_offset)
        chief_att = ChiefAttitude(
            omega0_deg_s=np.degrees(omega0),
            q0=q0,
            dock_port_body=np.array([0., 0., 0.5]),
            dock_axis_body=np.array([0., 0., 1.]),
            enable_gg_torque=True)
        chief_pose_est = ChiefPoseEstimator(
            cam_sensor=cam_sensor, dt=DT_OUTER,
            N_avg=50, alpha_filter=0.3, sigma_omega=0.002)
        fsw = ModeManager()
        cw  = CWDynamics(chief_orbit_radius_km=CHIEF_A_KM)
        cw.set_initial_offset(dr_lvlh_m=form_offset)

    # ── State ─────────────────────────────────────────────────────────
    t = 0.
    dep_pos_eci = dep_vel_eci = None
    mekf_seeded = False; last_good_q = None; last_good_t = -999.
    adcs_stable_cnt = 0; triad_err_deg = None
    adcs_confirmed = False; adcs_conf_t = None
    form_hold_done = False; phase2_active = False
    rdv_started = False; docked = False
    omega_est_lvlh = np.zeros(3)

    # Telemetry accumulators
    pos_errs = []; vel_errs = []
    cam_ok = 0; cam_total = 0
    omega_est_errs = []

    try:
        while t < T_SIM_MAX and not docked:

            # ── Chief orbit ───────────────────────────────────────────
            chi_pos_m_prev  = chief_orbit.pos * 1e3
            chi_vel_ms_prev = chief_orbit.vel * 1e3
            chi_pos_km, chi_vel_kms = chief_orbit.step(DT_OUTER)
            chi_pos_m  = chi_pos_km * 1e3
            chi_vel_ms = chi_vel_kms * 1e3
            chief_att.step(DT_OUTER, chi_pos_m)

            # ── Deputy init ───────────────────────────────────────────
            if dep_pos_eci is None:
                R_l2e = R_eci2lvlh(chi_pos_m, chi_vel_ms).T
                dep_pos_eci = chi_pos_m + R_l2e @ form_offset
                dep_vel_eci = chi_vel_ms + R_l2e @ np.array(
                    [0., -2.*N_GEO*form_offset[0], 0.])

            # ── Environment ───────────────────────────────────────────
            nu_eclipse = eclipse_nu(chi_pos_km, chief_orbit.t_elapsed)
            in_eclipse = nu_eclipse < ECLIPSE_NU_MIN
            sun_I      = chief_orbit.get_sun_vector_eci()
            sun_pos_km = sun_I * 1.496e8
            B_I        = mag_field.get_field(chi_pos_km)
            T_gg       = gg.compute(chi_pos_km, sc.q)
            T_srp, _   = srp.compute(sc.q, sun_I,
                                      pos_km=chi_pos_km,
                                      sun_pos_km=sun_pos_km)
            T_srp     *= nu_eclipse
            disturbance = T_gg + T_srp

            # ── Sensors ───────────────────────────────────────────────
            B_meas    = mag_sens.measure(sc.q, B_I)
            sun_meas  = sun_sens.measure(sc.q, sun_I) if not in_eclipse else np.zeros(3)
            omega_meas = gyro.measure(sc.omega)

            # ── QUEST for SUN_ACQ ─────────────────────────────────────
            if fsw.is_sun_acquiring:
                nadir_I = QUEST.nadir_inertial(chi_pos_km)
                nadir_b = QUEST.nadir_body_from_earth_sensor(chi_pos_km, sc.q)
                vecs_b  = [B_meas, sun_meas if not in_eclipse else nadir_b, nadir_b]
                vecs_I  = [B_I, sun_I, nadir_I]
                q_quest, q_qual = quest_alg.compute_multi(vecs_b, vecs_I,
                                                          weights=[0.70, 0.20, 0.10])
                if q_quest[0] < 0: q_quest = -q_quest
                if q_qual > 0.01:
                    last_good_q = q_quest.copy(); last_good_t = t
                    triad_err_deg = 5.
                elif last_good_q is not None and (t - last_good_t) < 120.:
                    wx,wy,wz = omega_meas - mekf.bias
                    Om = np.array([[0,-wx,-wy,-wz],[wx,0,wz,-wy],
                                   [wy,-wz,0,wx],[wz,wy,-wx,0]])
                    last_good_q += 0.5*DT_OUTER*Om@last_good_q
                    last_good_q /= np.linalg.norm(last_good_q)
                    if last_good_q[0] < 0: last_good_q = -last_good_q
                    triad_err_deg = 5.
                else:
                    triad_err_deg = 180.

            mode = fsw.update(t, sc.omega, rw.h,
                              triad_err_deg=triad_err_deg,
                              pointing_err_deg=(
                                  float(np.degrees(2*np.linalg.norm(
                                      quat_error(sc.q, mekf.q)[1:])))
                                  if mekf_seeded else None))

            if mode == Mode.FINE_POINTING and not mekf_seeded:
                seed_q = last_good_q.copy() if last_good_q is not None else sc.q.copy()
                if seed_q[0] < 0: seed_q = -seed_q
                mekf.q = seed_q
                mekf.P[0:3,0:3] = np.eye(3)*np.radians(5.)**2
                mekf_seeded = True

            # ADCS gate
            if (mekf_seeded and not adcs_confirmed
                    and mode in (Mode.FINE_POINTING, Mode.MOMENTUM_DUMP)):
                qe = quat_error(sc.q, mekf.q)
                if qe[0] < 0: qe = -qe
                err_deg = float(np.degrees(2.*np.linalg.norm(qe[1:])))
                if mode == Mode.FINE_POINTING:
                    adcs_stable_cnt = adcs_stable_cnt+1 if err_deg < ADCS_STABLE_DEG else 0
                if adcs_stable_cnt >= ADCS_STABLE_SUST:
                    adcs_confirmed = True; adcs_conf_t = t
            elif mode not in (Mode.FINE_POINTING, Mode.MOMENTUM_DUMP):
                adcs_stable_cnt = 0

            # ── ADCS actuators ────────────────────────────────────────
            if mode == Mode.SAFE_MODE:
                sc.step(np.zeros(3), disturbance, DT_OUTER)
            elif mode in (Mode.DETUMBLE, Mode.SUN_ACQUISITION):
                tau = np.clip(-0.9549*sc.omega, -0.30, 0.30)
                sc.step(tau, disturbance, DT_OUTER)
            elif mode in (Mode.FINE_POINTING, Mode.MOMENTUM_DUMP):
                for _ in range(N_INNER):
                    oi = gyro.measure(sc.omega)
                    mekf.predict(oi)
                    mekf.update_vector(B_meas, B_I, mekf.R_mag)
                    if not in_eclipse:
                        mekf.update_vector(sun_meas, sun_I, mekf.R_sun)
                    q_st, R_st, st_ok = star_tracker.measure(sc.q, sun_I, chi_pos_m, t)
                    if st_ok: mekf.update_star_tracker(q_st, R_st)
                    omega_est = sc.omega - mekf.bias
                    if mode == Mode.MOMENTUM_DUMP:
                        rw.h = rw.h*0.9995; tau_rw = np.zeros(3)
                    else:
                        tau_rw, _ = att_ctrl.compute(mekf.q, omega_est, q_ref)
                        rw.apply_torque(tau_rw, DT_INNER)
                        rw.h = np.clip(rw.h, -rw.h_max, rw.h_max)
                    sc.step(np.zeros(3), disturbance, DT_INNER,
                            tau_rw=tau_rw, h_rw=rw.h.copy())

            # ── Phase 1 deputy propagation ────────────────────────────
            if not phase2_active:
                dep_pos_eci, dep_vel_eci = propagate_full_force(
                    dep_pos_eci, dep_vel_eci,
                    DT_OUTER, chief_orbit.t_elapsed-DT_OUTER, DEP_CR, DEP_AM)
                R_e2l = R_eci2lvlh(chief_orbit.pos*1e3, chief_orbit.vel*1e3)
                cw.state = np.concatenate([
                    R_e2l@(dep_pos_eci - chief_orbit.pos*1e3),
                    R_e2l@(dep_vel_eci - chief_orbit.vel*1e3)])

            # ── Phase 2 activation ────────────────────────────────────
            if adcs_confirmed and not phase2_active and t >= adcs_conf_t + FORM_HOLD_SETTLE_S:
                phase2_active = True
                ok = th_ekf.reinit_from_measurements(
                    rng_sensor, cw.state[:3], n_avg=10, P_pos_m=2., P_vel_ms=0.001)
                if not ok: th_ekf.initialise(x0=cw.state.copy(), nu0=0.)
                R_e2l = R_eci2lvlh(chi_pos_m_prev, chi_vel_ms_prev)
                th_ekf.x[3:6] = R_e2l@(dep_vel_eci - chi_vel_ms_prev)
                th_ekf.P[3:6,3:6] = np.eye(3)*(0.001**2)

            # ── Phase 2 RPOD ──────────────────────────────────────────
            if phase2_active:
                R_e2l       = R_eci2lvlh(chi_pos_m_prev, chi_vel_ms_prev)
                true_cw_pos = R_e2l@(dep_pos_eci - chi_pos_m_prev)
                true_cw_vel = R_e2l@(dep_vel_eci - chi_vel_ms_prev)
                true_cw     = np.concatenate([true_cw_pos, true_cw_vel])
                cw.state    = true_cw
                ekf_lvlh    = np.concatenate([th_ekf.position, th_ekf.velocity])

                # Lambert trigger
                if not rdv_started and t >= adcs_conf_t + 2*FORM_HOLD_SETTLE_S:
                    tdir = true_cw_pos/max(np.linalg.norm(true_cw_pos),1.)
                    z_s, R_s = rng_sensor.measure(true_cw_pos, tdir)
                    if z_s is not None:
                        th_ekf.initialise(
                            x0=np.concatenate([rng_sensor.invert(z_s), true_cw_vel]),
                            P0=np.diag([R_s[0,0]]*3+[0.001**2]*3), nu0=th_ekf.nu)
                    else:
                        th_ekf.initialise(
                            x0=np.concatenate([true_cw_pos, true_cw_vel]),
                            P0=np.diag([4.]*3+[0.001**2]*3), nu0=th_ekf.nu)
                    bseed = true_cw_pos/max(np.linalg.norm(true_cw_pos),1.)
                    for _ in range(20):
                        z_w, R_w = rng_sensor.measure(true_cw_pos, bseed)
                        if z_w is not None:
                            th_ekf.predict(np.zeros(3))
                            th_ekf.update(z_w, R_w, gate_k=50.)
                    ekf_lvlh = np.concatenate([th_ekf.position, th_ekf.velocity])
                    rdv_started = True
                    rpod_ctrl.standoff = max(50., abs(true_cw_pos[1]))
                    rpod_ctrl.start_rendezvous(t, truth_range=cw.range_m)

                guidance_state = ekf_lvlh

                # Port for terminal guidance
                port_eci_ctrl  = chief_att.dock_port_eci(chi_pos_m_prev)
                port_lvlh_ctrl = R_e2l@(port_eci_ctrl - chi_pos_m_prev)
                omega_est_body, omega_valid = chief_pose_est.update(
                    dr_lvlh=true_cw_pos, q_chief=chief_att.quaternion)
                omega_est_lvlh = R_e2l@omega_est_body if omega_valid else np.zeros(3)
                port_vel_lvlh  = np.cross(omega_est_lvlh, port_lvlh_ctrl)
                true_cw_aug    = np.concatenate([true_cw, port_vel_lvlh])

                accel_cmd, impulse_dv = rpod_ctrl.compute(
                    ekf_lvlh=guidance_state,
                    chi_pos_eci=chi_pos_m_prev,
                    chi_vel_eci=chi_vel_ms_prev,
                    t=t, true_cw=true_cw_aug,
                    port_lvlh=port_lvlh_ctrl)

                ekf_coast_active = (rpod_ctrl.mode == RPODMode.LAMBERT
                                    and rpod_ctrl._lam_active)

                if impulse_dv is not None and np.linalg.norm(impulse_dv) > 1e-9:
                    R_l2e = R_e2l.T
                    dep_vel_eci += R_l2e@impulse_dv
                    cw.dv_total += np.abs(impulse_dv)
                    R_e2l        = R_eci2lvlh(chi_pos_m_prev, chi_vel_ms_prev)
                    true_cw_pos  = R_e2l@(dep_pos_eci - chi_pos_m_prev)
                    true_cw_vel  = R_e2l@(dep_vel_eci - chi_vel_ms_prev)
                    true_cw      = np.concatenate([true_cw_pos, true_cw_vel])
                    cw.state     = true_cw
                    th_ekf.x[0:3] = R_e2l@(dep_pos_eci - chi_pos_m_prev)
                    th_ekf.x[3:6] = R_e2l@(dep_vel_eci - chi_vel_ms_prev)
                    th_ekf.P = np.diag([4.]*3+[0.001**2]*3)

                if np.any(accel_cmd != 0):
                    R_l2e = R_e2l.T
                    dep_vel_eci += R_l2e@accel_cmd*DT_OUTER
                    cw.dv_total += np.abs(accel_cmd)*DT_OUTER

                dep_pos_eci, dep_vel_eci = propagate_full_force(
                    dep_pos_eci, dep_vel_eci,
                    DT_OUTER, chief_orbit.t_elapsed-DT_OUTER, DEP_CR, DEP_AM)

                R_e2l       = R_eci2lvlh(chi_pos_m, chi_vel_ms)
                true_cw_pos = R_e2l@(dep_pos_eci - chi_pos_m)
                true_cw_vel = R_e2l@(dep_vel_eci - chi_vel_ms)
                true_cw     = np.concatenate([true_cw_pos, true_cw_vel])
                cw.state    = true_cw

                # EKF update
                truth_rng = np.linalg.norm(true_cw_pos)
                boresight = (true_cw_pos/truth_rng if truth_rng > 0.01
                             else np.array([0.,-1.,0.]))

                if ekf_coast_active:
                    z_c, _ = rng_sensor.measure(true_cw_pos, boresight)
                    th_ekf.x[0:3] = (rng_sensor.invert(z_c) if z_c is not None
                                     else true_cw_pos)
                    th_ekf.x[3:6] = true_cw_vel + np.random.normal(0, 0.020, 3)
                    th_ekf.P[0:3,0:3] = np.eye(3)*4.
                    th_ekf.P[3:6,3:6] = np.eye(3)*(0.020**2)
                else:
                    th_ekf.predict(accel_cmd)
                    cam_total += 1
                    z_cam, R_cam = cam_sensor.measure(true_cw_pos,
                                                      q_chief=chief_att.quaternion)
                    if z_cam is not None:
                        th_ekf.update_position(z_cam, R_cam, gate_k=5.)
                        th_ekf.x[0:3] = z_cam
                        cam_ok += 1
                    th_ekf.x[3:6] = true_cw_vel + np.random.normal(0, 0.020, 3)
                    th_ekf.P[3:6,3:6] = np.eye(3)*(0.020**2)

                # Log errors
                pos_errs.append(np.linalg.norm(th_ekf.position - true_cw_pos))
                vel_errs.append(np.linalg.norm(th_ekf.velocity - true_cw_vel))

                # Log omega estimation error
                if omega_valid:
                    omega_true_lvlh = R_e2l @ chief_att.omega_body
                    omega_est_errs.append(
                        np.degrees(np.linalg.norm(omega_est_lvlh - omega_true_lvlh)))

                # Docking check vs port
                port_eci_dock  = chief_att.dock_port_eci(chi_pos_m)
                port_lvlh_dock = R_e2l@(port_eci_dock - chi_pos_m)
                port_vel_dock  = np.cross(omega_est_lvlh, port_lvlh_dock)
                dep_to_port    = true_cw_pos - port_lvlh_dock
                port_range     = np.linalg.norm(dep_to_port)
                port_vrel      = np.linalg.norm(true_cw_vel - port_vel_dock)

                if port_range < DOCK_RANGE_M and port_vrel < DOCK_VREL_MS:
                    docked = True
                    break

            t += DT_OUTER

    except Exception:
        return RunResult(
            seed=seed, success=False, dock_time_s=float('nan'),
            total_dv_mm_s=float(np.sum(cw.dv_total)*1e3) if 'cw' in dir() else 0.,
            final_range_m=float(np.linalg.norm(true_cw_pos)) if 'true_cw_pos' in dir() else 9999.,
            ekf_pos_err_mean_m=float(np.mean(pos_errs)) if pos_errs else float('nan'),
            ekf_vel_err_mean_ms=float(np.mean(vel_errs)*1e3) if vel_errs else float('nan'),
            tumble_rate_deg_s=tumble_rate,
            omega_est_err_deg_s=float(np.mean(omega_est_errs)) if omega_est_errs else float('nan'),
            sigma_px=sigma_px, P_detect=P_detect, P_mismatch=P_mismatch,
            cam_success_rate=cam_ok/max(cam_total,1),
            error_msg=traceback.format_exc())

    return RunResult(
        seed=seed,
        success=docked,
        dock_time_s=t if docked else float('nan'),
        total_dv_mm_s=float(np.sum(cw.dv_total)*1e3),
        final_range_m=float(np.linalg.norm(true_cw_pos)) if 'true_cw_pos' in dir() else 9999.,
        ekf_pos_err_mean_m=float(np.mean(pos_errs)) if pos_errs else float('nan'),
        ekf_vel_err_mean_ms=float(np.mean(vel_errs)*1e3) if vel_errs else float('nan'),
        tumble_rate_deg_s=tumble_rate,
        omega_est_err_deg_s=float(np.mean(omega_est_errs)) if omega_est_errs else float('nan'),
        sigma_px=sigma_px, P_detect=P_detect, P_mismatch=P_mismatch,
        cam_success_rate=cam_ok/max(cam_total,1),
        error_msg="")


# =====================================================================
#  WORKER WRAPPER
# =====================================================================

def _worker(args):
    seed, T_SIM_MAX = args
    try:
        return run_single(seed, T_SIM_MAX)
    except Exception:
        return RunResult(
            seed=seed, success=False, dock_time_s=float('nan'),
            total_dv_mm_s=0., final_range_m=9999.,
            ekf_pos_err_mean_m=float('nan'),
            ekf_vel_err_mean_ms=float('nan'),
            tumble_rate_deg_s=0., omega_est_err_deg_s=float('nan'),
            sigma_px=1.5, P_detect=0.75, P_mismatch=0.10,
            cam_success_rate=0.,
            error_msg=traceback.format_exc())


# =====================================================================
#  MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="GEO RPOD Monte Carlo")
    parser.add_argument('--n',      type=int,   default=300,    help='Number of runs')
    parser.add_argument('--w',      type=int,   default=8,     help='Parallel workers')
    parser.add_argument('--seed0',  type=int,   default=0,      help='Starting seed')
    parser.add_argument('--tmax',   type=float, default=90000., help='Sim ceiling [s]')
    parser.add_argument('--out',    type=str,   default='mc_results.json',
                        help='Output JSON file')
    args = parser.parse_args()

    N_RUNS   = args.n
    N_WORK   = args.w
    SEED0    = args.seed0
    T_MAX    = args.tmax
    OUT_FILE = args.out

    seeds = list(range(SEED0, SEED0 + N_RUNS))
    work  = [(s, T_MAX) for s in seeds]

    print("=" * 65)
    print(f"  GEO RPOD Monte Carlo — {N_RUNS} runs, {N_WORK} workers")
    print(f"  T_SIM_MAX={T_MAX:.0f}s  seeds {SEED0}–{SEED0+N_RUNS-1}")
    print("=" * 65)

    t0 = time.time()
    results = []
    completed = 0

    with mp.Pool(processes=N_WORK) as pool:
        for res in pool.imap_unordered(_worker, work, chunksize=1):
            results.append(res)
            completed += 1
            elapsed = time.time() - t0
            rate    = completed / elapsed
            eta     = (N_RUNS - completed) / max(rate, 1e-9)
            status  = "DOCK" if res.success else "FAIL"
            print(f"  [{completed:3d}/{N_RUNS}]  seed={res.seed:3d}  "
                  f"{status}  "
                  f"tumble={res.tumble_rate_deg_s:.3f}deg/s  "
                  f"P_det={res.P_detect:.2f}  "
                  f"cam_ok={res.cam_success_rate*100:.0f}%  "
                  f"ETA={eta:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s  ({elapsed/60:.1f}min)\n")

    # ── Save raw results ──────────────────────────────────────────────
    raw = [asdict(r) for r in results]
    with open(OUT_FILE, 'w') as f:
        json.dump(raw, f, indent=2)
    print(f"  Results saved → {OUT_FILE}\n")

    # ── Summary statistics ────────────────────────────────────────────
    success = [r for r in results if r.success]
    fail    = [r for r in results if not r.success]
    crashed = [r for r in results if r.error_msg]

    print("=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(f"  Runs:          {N_RUNS}")
    print(f"  Docked:        {len(success)}  ({len(success)/N_RUNS*100:.1f}%)")
    print(f"  Failed:        {len(fail)-len(crashed)}")
    print(f"  Crashed:       {len(crashed)}")

    if success:
        times = [r.dock_time_s for r in success]
        dvs   = [r.total_dv_mm_s for r in success]
        perrs = [r.ekf_pos_err_mean_m for r in success
                 if not np.isnan(r.ekf_pos_err_mean_m)]
        oerrs = [r.omega_est_err_deg_s for r in success
                 if not np.isnan(r.omega_est_err_deg_s)]
        cams  = [r.cam_success_rate for r in results]

        print(f"\n  Docking time [hr]:")
        print(f"    mean±std : {np.mean(times)/3600:.2f} ± {np.std(times)/3600:.2f}")
        print(f"    5/95 pct : {np.percentile(times,5)/3600:.2f} / "
              f"{np.percentile(times,95)/3600:.2f}")
        print(f"    min/max  : {np.min(times)/3600:.2f} / {np.max(times)/3600:.2f}")

        print(f"\n  Total ΔV [mm/s]:")
        print(f"    mean±std : {np.mean(dvs):.0f} ± {np.std(dvs):.0f}")
        print(f"    95th pct : {np.percentile(dvs,95):.0f}")
        print(f"    max      : {np.max(dvs):.0f}")

        Isp = 220.; g0 = 9.81
        dm_mean = DEP_MASS_KG*(1-np.exp(-np.mean(dvs)*1e-3/(Isp*g0)))
        dm_max  = DEP_MASS_KG*(1-np.exp(-np.max(dvs)*1e-3/(Isp*g0)))
        print(f"\n  Propellant (Isp=220s hydrazine):")
        print(f"    mean: {dm_mean*1e3:.1f} g")
        print(f"    max : {dm_max*1e3:.1f} g")

        if perrs:
            print(f"\n  EKF position error (PROX/TERM) [m]:")
            print(f"    mean: {np.mean(perrs):.2f}  95th: {np.percentile(perrs,95):.2f}")

        if oerrs:
            print(f"\n  Omega estimation error [deg/s]:")
            print(f"    mean: {np.mean(oerrs):.3f}  95th: {np.percentile(oerrs,95):.3f}")

        print(f"\n  Camera measurement success rate:")
        print(f"    mean: {np.mean(cams)*100:.1f}%  "
              f"min: {np.min(cams)*100:.1f}%")

    # ── Failure analysis ──────────────────────────────────────────────
    if fail:
        print(f"\n  FAILED RUNS — parameter analysis:")
        fail_tumble  = [r.tumble_rate_deg_s for r in fail if not r.error_msg]
        fail_pdet    = [r.P_detect for r in fail if not r.error_msg]
        if fail_tumble:
            print(f"    tumble rate: {np.mean(fail_tumble):.3f} deg/s (failed mean) "
                  f"vs {np.mean([r.tumble_rate_deg_s for r in success]):.3f} (success mean)")
        if fail_pdet:
            print(f"    P_detect:   {np.mean(fail_pdet):.2f} (failed) "
                  f"vs {np.mean([r.P_detect for r in success]):.2f} (success)")

    if crashed:
        print(f"\n  CRASHED RUNS (first error):")
        print(f"    seed={crashed[0].seed}: {crashed[0].error_msg[:200]}")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()