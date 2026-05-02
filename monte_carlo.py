"""
Monte Carlo Analysis — GEO Jetpack RPOD Simulation
====================================================
Industry-standard probabilistic performance assessment for the
50 kg jetpack servicer rendezvous / docking with a free-tumbling
IS-1002 class GEO comsat.

Architecture
------------
- 300 independent Monte Carlo runs  (N_RUNS)
- 8 parallel workers                (N_WORKERS)
- Each run is a fully independent copy of the simulation:
    * New random seeds (np.random.seed per run)
    * Dispersed initial conditions (chief tumble rate, orbit, deputy IC)
    * Dispersed sensor noise instances (gyro bias, mag distortion, etc.)
    * Dispersed chief attitude (random initial quaternion)
    * Dispersed SRP and drag coefficients (±5%)
- Results are collected into a structured per-run dict and aggregated
  into a summary CSV + PDF report.

Dispersed parameters (3σ ranges)
---------------------------------
  Chief tumble rate  :  Uniform [0.05, 2.0] deg/s  (derelict spread)
  Chief orbit RAAN   :  ±180°                       (random phasing)
  Deputy IC position :  ±50m (3σ) in all LVLH axes
  Deputy IC velocity :  ±5 mm/s (3σ) per axis
  Gyro bias          :  ±0.5 deg/s (3σ)
  Mag hard-iron      :  ±300 nT (3σ, post-cal residual)
  Cr (deputy)        :  1.5 ± 0.075 (5%)
  Am (deputy)        :  0.0072 ± 0.00036 (5%)
  Star-tracker noise : sigma_cross ±2 arcsec, sigma_roll ±8 arcsec

Pass/Fail criteria  (per run)
------------------------------
  PASS if ALL of:
    1. docked == True
    2. port_range < DOCK_RANGE_M   (0.30 m)
    3. port_vrel  < DOCK_VREL_MS   (0.05 m/s)
    4. t_docking  < T_SIM_MAX      (80 000 s)

KPIs reported per run (and aggregated)
---------------------------------------
  t_adcs_gate_s        : time to ADCS confirmation [s]
  t_lambert_start_s    : time Lambert maneuver was triggered [s]
  t_docking_s          : time of docking confirmation [s]
  docked               : bool
  total_dv_ms          : cumulative Δv [m/s]
  final_port_range_m   : final range to docking port [m]
  final_port_vrel_ms   : final relative velocity at port [m/s]
  mekf_ss_err_deg      : MEKF steady-state pointing error [deg] (mean last 200 steps)
  ekf_pos_err_mean_m   : mean TH-EKF position error during RPOD [m]
  ekf_pos_err_max_m    : max  TH-EKF position error during RPOD [m]
  chief_tumble_deg_s   : dispersed chief tumble rate [deg/s]
  t_adcs_stable_s      : duration of ADCS fine-pointing phase [s]
  run_id               : integer run identifier
  seed                 : numpy random seed used
  fail_reason          : '' if PASS, else human-readable failure reason

Usage
-----
  python monte_carlo.py [--runs N] [--workers W] [--out results/]

  # Quick smoke-test (10 runs, 2 workers):
  python monte_carlo.py --runs 10 --workers 2

Output files
------------
  <out>/mc_results.csv        — per-run KPI table
  <out>/mc_summary.txt        — aggregate statistics
  <out>/mc_plots.png          — 8-panel summary figure
  <out>/mc_failed_runs.txt    — seeds + failure reasons for failed runs

Reference
---------
  NASA-HDBK-7005 — Dynamic Environmental Criteria (MC methodology)
  ECSS-E-ST-10-04C — Space engineering: Space environment (dispersion)
  Fehse, "Automated RVD of Spacecraft", Cambridge 2003, §12.3
"""

import os
import sys
import csv
import time
import copy
import argparse
import traceback
import itertools
import multiprocessing as mp
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless — no display needed in workers
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Simulation root on sys.path ────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

# ── Simulation imports (same as main.py) ──────────────────────────────────────
from plant.spacecraft                     import Spacecraft
from environment.magnetic_field           import MagneticField
from environment.gravity_gradient         import GravityGradient
from environment.solar_radiation_pressure import SolarRadiationPressure
from environment.geo_orbit                import GEOOrbitPropagator, eclipse_nu
from environment.cw_dynamics              import CWDynamics

from sensors.gyro            import Gyro
from sensors.magnetometer    import Magnetometer
from sensors.sun_sensor      import SunSensor
from sensors.star_tracker    import StarTracker
from sensors.ranging_sensor  import RangingBearingSensor
from sensors.camera_sensor   import CameraSensor

from actuators.reaction_wheel import ReactionWheel
from actuators.magnetorquer   import Magnetorquer
from actuators.bdot           import BDotController

from estimation.mekf   import MEKF
from estimation.quest  import QUEST
from estimation.th_ekf import THEKF
from chief_attitude     import ChiefAttitude
from chief_pose_estimator import ChiefPoseEstimator

from control.attitude_controller import AttitudeController
from control.lambert_controller  import GEORPODController, RPODMode

from fsw.mode_manager import ModeManager, Mode
from utils.quaternion  import quat_error, rot_matrix


# ══════════════════════════════════════════════════════════════════════════════
#  MONTE CARLO CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

N_RUNS    = 300
N_WORKERS = 8

# ── Nominal mission parameters (mirrors main.py) ──────────────────────────────
CHIEF_A_KM      = 42164.0
CHIEF_E         = 0.0003
CHIEF_I_DEG     = 0.8
CHIEF_LON_DEG   = 342.0
DEP_MASS_KG     = 50.0
DEP_THRUST_N    = 1.0
FORMATION_OFFSET_M = np.array([0.0, -1000.0, 0.0])
DT_OUTER        = 0.1
DT_INNER        = 0.01
N_INNER         = int(DT_OUTER / DT_INNER)
T_SIM_MAX       = 80_000.0
ADCS_STABLE_DEG = 1.0
ADCS_STABLE_SUST = 100
FORM_HOLD_SETTLE_S = 300.0
DOCK_RANGE_M    = 0.30
DOCK_VREL_MS    = 0.05
ECLIPSE_NU_MIN  = 0.1
MU_GEO          = 3.986004418e14
N_GEO           = np.sqrt(MU_GEO / (CHIEF_A_KM * 1e3) ** 3)
SIGMA_V_DOPPLER = 0.005

# ── Dispersion bounds (3σ uniform unless stated) ──────────────────────────────
DISP = {
    # Chief tumble rate — uniform across typical derelict range
    "chief_tumble_min_deg_s": 0.05,
    "chief_tumble_max_deg_s": 2.00,

    # Chief orbit: RAAN dispersed ±180° (any longitude phasing)
    "raan_sigma_deg"       : 180.0,      # uniform ±RAAN

    # Deputy IC offsets
    "dep_pos_sigma_m"      : 50.0 / 3,  # 3σ = 50 m
    "dep_vel_sigma_ms"     : 0.005 / 3, # 3σ = 5 mm/s

    # Gyro bias initial uncertainty
    "gyro_bias_max_deg_s"  : 0.5,       # matches Gyro class default

    # Cr and Am: ±5% uniform
    "Cr_sigma"             : 0.05 * 1.5,
    "Am_sigma"             : 0.05 * 0.0072,

    # Star tracker noise
    "st_cross_sigma_arcsec": 2.0,       # ± on sigma_cross
    "st_roll_sigma_arcsec" : 8.0,       # ± on sigma_roll
}

# ── KPI dataclass ─────────────────────────────────────────────────────────────
@dataclass
class RunResult:
    run_id             : int   = -1
    seed               : int   = -1
    docked             : bool  = False
    fail_reason        : str   = "DID_NOT_COMPLETE"

    t_adcs_gate_s      : float = float("nan")
    t_lambert_start_s  : float = float("nan")
    t_docking_s        : float = float("nan")
    t_adcs_stable_s    : float = float("nan")   # duration of fine-pointing before phase 2

    total_dv_ms        : float = float("nan")
    final_port_range_m : float = float("nan")
    final_port_vrel_ms : float = float("nan")

    mekf_ss_err_deg    : float = float("nan")
    ekf_pos_err_mean_m : float = float("nan")
    ekf_pos_err_max_m  : float = float("nan")

    chief_tumble_deg_s : float = float("nan")
    dep_cr             : float = float("nan")
    dep_am             : float = float("nan")
    chief_raan_deg     : float = float("nan")
    st_cross_arcsec    : float = float("nan")

    wall_time_s        : float = float("nan")   # real-world run time

    # ── Camera-sensor KPIs (Phase 2 only) ─────────────────────────────────
    cam_success_rate   : float = float("nan")   # fraction of steps z_cam != None
    cam_lost_rate      : float = float("nan")   # fraction of steps is_lost==True
    cam_n_attempts     : int   = 0              # total measure() calls
    cam_n_success      : int   = 0              # calls that returned z_cam != None
    cam_n_lost_events  : int   = 0              # transitions into is_lost state
    cam_mean_range_m   : float = float("nan")   # mean range when camera was called
    cam_dv_penalty_ms  : float = float("nan")   # ΔV burned while cam_lost==True


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS  (mirrors main.py helpers exactly)
# ══════════════════════════════════════════════════════════════════════════════

def R_eci2lvlh(r_chief: np.ndarray, v_chief: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix ECI → LVLH."""
    x_hat = r_chief / np.linalg.norm(r_chief)
    h_vec = np.cross(r_chief, v_chief)
    z_hat = h_vec / np.linalg.norm(h_vec)
    y_hat = np.cross(z_hat, x_hat)
    return np.vstack([x_hat, y_hat, z_hat])


def propagate_full_force(pos, vel, dt_total, t_abs, Cr, Am, substep=60.0):
    """RK4: two-body + J2 + SRP — identical to main.py."""
    J2 = 1.08263e-3
    RE = 6.3781e6
    AU = 1.495978707e11
    P0 = 4.56e-6

    def sun_pos(t):
        d   = t / 86400.0
        lam = np.radians(280.46 + 360.985647 * d)
        eps = np.radians(23.439)
        return AU * np.array([np.cos(lam),
                               np.cos(eps) * np.sin(lam),
                               np.sin(eps) * np.sin(lam)])

    def accel(p, t):
        r = np.linalg.norm(p)
        a = -MU_GEO / r ** 3 * p
        x, y, z = p
        c = -1.5 * J2 * MU_GEO * RE ** 2 / r ** 5
        f = 5 * z ** 2 / r ** 2
        a += np.array([c * x * (1 - f), c * y * (1 - f), c * z * (3 - f)])
        sp  = sun_pos(t)
        rs  = np.linalg.norm(sp)
        P   = P0 * (AU / rs) ** 2
        dr  = p - sp
        a  += Cr * Am * P * dr / np.linalg.norm(dr)
        return a

    n = max(1, int(round(dt_total / substep)))
    h = dt_total / n
    p, v, t = pos.copy(), vel.copy(), float(t_abs)
    for _ in range(n):
        k1p = v;              k1v = accel(p, t)
        k2p = v + 0.5*h*k1v; k2v = accel(p + 0.5*h*k1p, t + 0.5*h)
        k3p = v + 0.5*h*k2v; k3v = accel(p + 0.5*h*k2p, t + 0.5*h)
        k4p = v + h*k3v;      k4v = accel(p + h*k3p,     t + h)
        p += (h / 6) * (k1p + 2*k2p + 2*k3p + k4p)
        v += (h / 6) * (k1v + 2*k2v + 2*k3v + k4v)
        t += h
    return p, v


# ══════════════════════════════════════════════════════════════════════════════
#  SINGLE-RUN SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

def _run_simulation(run_id: int, seed: int) -> RunResult:
    """
    Execute one complete RPOD simulation with a given random seed.

    All random state (sensor noise, initial conditions, dispersions) is
    fully determined by `seed`.  No global state is modified.

    Returns a populated RunResult dataclass.
    """
    t_wall_start = time.perf_counter()
    res          = RunResult(run_id=run_id, seed=seed)

    # ── Seed RNG ────────────────────────────────────────────────────────────
    rng = np.random.default_rng(seed)
    # Also seed the legacy np.random used inside sensor classes
    np.random.seed(seed)

    # ── Draw dispersed parameters ────────────────────────────────────────────
    chief_tumble_deg_s = rng.uniform(DISP["chief_tumble_min_deg_s"],
                                     DISP["chief_tumble_max_deg_s"])
    chief_raan_deg     = rng.uniform(-180.0, 180.0)
    dep_pos_offset     = rng.normal(0, DISP["dep_pos_sigma_m"],  3)
    dep_vel_offset     = rng.normal(0, DISP["dep_vel_sigma_ms"], 3)
    dep_Cr             = 1.5   + rng.uniform(-DISP["Cr_sigma"],  DISP["Cr_sigma"])
    dep_Am             = 0.0072 + rng.uniform(-DISP["Am_sigma"],  DISP["Am_sigma"])
    st_cross_arcsec    = max(1.0, 5.0 + rng.uniform(-DISP["st_cross_sigma_arcsec"],
                                                       DISP["st_cross_sigma_arcsec"]))
    st_roll_arcsec     = max(4.0, 20.0 + rng.uniform(-DISP["st_roll_sigma_arcsec"],
                                                        DISP["st_roll_sigma_arcsec"]))

    # Chief tumble rate: random axis at dispersed magnitude
    tumble_axis = rng.standard_normal(3)
    tumble_axis /= np.linalg.norm(tumble_axis)
    omega0_deg_s = chief_tumble_deg_s * tumble_axis   # deg/s per axis

    res.chief_tumble_deg_s = chief_tumble_deg_s
    res.dep_cr             = dep_Cr
    res.dep_am             = dep_Am
    res.chief_raan_deg     = chief_raan_deg
    res.st_cross_arcsec    = st_cross_arcsec

    # ── Hardware instantiation ───────────────────────────────────────────────
    I_sc = np.diag([4.167, 4.167, 3.000])

    chief_orbit = GEOOrbitPropagator(
        a_km=CHIEF_A_KM, e=CHIEF_E, i_deg=CHIEF_I_DEG,
        raan_deg=chief_raan_deg, omega_deg=0.0, M0_deg=0.0,
        Cr=1.5, Am_ratio=0.015)

    mag_field = MagneticField(epoch_year=2025.0)
    gg        = GravityGradient(I_sc)
    srp       = SolarRadiationPressure()

    sc        = Spacecraft(I_sc)
    sc.omega  = rng.uniform(-0.25, 0.25, 3)   # dispersed initial tumble [rad/s]

    # Sensors — dispersed noise instances
    mag_sens     = Magnetometer()
    sun_sens     = SunSensor()
    gyro         = Gyro(dt=DT_INNER,
                        bias_init_max_deg_s=DISP["gyro_bias_max_deg_s"])
    star_tracker = StarTracker(
        sigma_cross_arcsec=st_cross_arcsec,
        sigma_roll_arcsec=st_roll_arcsec,
        sun_excl_deg=30.0, earth_excl_deg=20.0,
        update_rate_hz=4.0, acquisition_s=30.0)
    rng_sensor   = RangingBearingSensor(
        sigma_range_m=1.0, sigma_range_frac=0.001,
        sigma_angle_rad=np.radians(0.05),
        fov_half_deg=60.0, max_range_m=5000.0, min_range_m=0.05)
    cam_sensor   = CameraSensor(
        focal_length_px=800.0, image_size_px=(640, 480),
        sigma_px=1.5, min_range_m=0.05, max_range_m=5000.0)

    # Actuators
    rw   = ReactionWheel(h_max=4.0)
    mtq  = Magnetorquer(m_max=0.2)
    bdot = BDotController(k_bdot=2e5, m_max=0.2)

    # Estimation
    quest_alg = QUEST()
    mekf      = MEKF(DT_INNER)
    mekf.P[0:3, 0:3] = np.eye(3) * np.radians(5.0) ** 2

    th_ekf = THEKF(
        a_chief=CHIEF_A_KM * 1e3, e_chief=CHIEF_E,
        dt=DT_OUTER, q_pos=1e-3, q_vel=1e-7)

    # Control
    att_ctrl = AttitudeController(Kp=0.08284, Kd=0.82257)
    q_ref    = np.array([1., 0., 0., 0.])

    rpod_ctrl = GEORPODController(
        mu=MU_GEO, n_chief=N_GEO,
        dep_mass_kg=DEP_MASS_KG, dep_thrust_N=DEP_THRUST_N,
        Cr_chi=1.5, Am_chi=0.015,
        dock_capture_m=DOCK_RANGE_M,
        ekf=th_ekf, rng_sensor=rng_sensor)
    rpod_ctrl.standoff = np.linalg.norm(FORMATION_OFFSET_M)

    # Chief attitude — dispersed tumble
    chief_att = ChiefAttitude(
        omega0_deg_s=omega0_deg_s,
        dock_port_body=np.array([0.0, 0.0, 0.5]),
        dock_axis_body=np.array([0.0, 0.0, 1.0]),
        enable_gg_torque=True)

    chief_pose_est = ChiefPoseEstimator(
        cam_sensor=cam_sensor,
        dt=DT_OUTER, N_avg=50,
        alpha_filter=0.3, sigma_omega=0.002)

    fsw = ModeManager()
    cw  = CWDynamics(chief_orbit_radius_km=CHIEF_A_KM)
    cw.set_initial_offset(dr_lvlh_m=FORMATION_OFFSET_M + dep_pos_offset)

    DOCK_PORT_BODY = np.array([0.0, 0.0, 0.5])
    DOCK_AXIS_BODY = np.array([0.0, 0.0, 1.0])

    # ── Sim state ────────────────────────────────────────────────────────────
    t = 0.0
    dep_pos_eci = None
    dep_vel_eci = None

    mekf_seeded       = False
    last_good_q       = None
    last_good_t       = -999.0
    adcs_stable_cnt   = 0
    triad_err_deg     = None
    adcs_confirmed    = False
    adcs_conf_t       = None
    form_hold_done    = False
    phase2_active     = False
    rdv_started       = False
    docked            = False
    ekf_coast_active  = False

    # KPI accumulators
    mekf_err_log  = []   # (t, err_deg) in FINE_POINTING
    ekf_pos_errs  = []   # [m] per RPOD step

    # Camera tracking accumulators
    cam_attempts   = 0
    cam_successes  = 0
    cam_was_lost   = False   # previous step is_lost state
    cam_lost_events= 0
    cam_range_log  = []      # range each step camera was called
    cam_dv_lost    = 0.0     # ΔV accumulated while camera is_lost

    final_port_range_m = float("nan")
    final_port_vrel_ms = float("nan")

    # ── Main loop ────────────────────────────────────────────────────────────
    try:
        while t < T_SIM_MAX and not docked:

            # ── 1. Chief orbit ──────────────────────────────────────────────
            chi_pos_m_prev  = chief_orbit.pos * 1e3
            chi_vel_ms_prev = chief_orbit.vel * 1e3
            chi_pos_km, chi_vel_kms = chief_orbit.step(DT_OUTER)
            chi_pos_m  = chi_pos_km  * 1e3
            chi_vel_ms = chi_vel_kms * 1e3

            # ── 1b. Chief attitude ──────────────────────────────────────────
            chief_att.step(DT_OUTER, chi_pos_m)

            # ── 2. Deputy ECI init ──────────────────────────────────────────
            if dep_pos_eci is None:
                R_l2e = R_eci2lvlh(chi_pos_m, chi_vel_ms).T
                dep_pos_eci = chi_pos_m + R_l2e @ (FORMATION_OFFSET_M + dep_pos_offset)
                dv_ic       = np.array([0., -2.0 * N_GEO *
                                        (FORMATION_OFFSET_M[0] + dep_pos_offset[0]), 0.])
                dep_vel_eci = chi_vel_ms + R_l2e @ (dv_ic + dep_vel_offset)

            # ── 3. Environment ──────────────────────────────────────────────
            nu_eclipse  = eclipse_nu(chi_pos_km, chief_orbit.t_elapsed)
            in_eclipse  = nu_eclipse < ECLIPSE_NU_MIN
            sun_I       = chief_orbit.get_sun_vector_eci()
            sun_pos_km  = sun_I * 1.496e8

            B_I         = mag_field.get_field(chi_pos_km)
            T_gg        = gg.compute(chi_pos_km, sc.q)
            T_srp, _    = srp.compute(sc.q, sun_I,
                                       pos_km=chi_pos_km, sun_pos_km=sun_pos_km)
            T_srp      *= nu_eclipse
            disturbance = T_gg + T_srp

            # ── 4. Sensors ──────────────────────────────────────────────────
            B_meas     = mag_sens.measure(sc.q, B_I)
            sun_meas   = sun_sens.measure(sc.q, sun_I) if not in_eclipse else np.zeros(3)
            omega_meas = gyro.measure(sc.omega)

            # ── 5. FSW mode ─────────────────────────────────────────────────
            if fsw.is_sun_acquiring:
                nadir_I = QUEST.nadir_inertial(chi_pos_km)
                nadir_b = QUEST.nadir_body_from_earth_sensor(chi_pos_km, sc.q)
                if in_eclipse:
                    q_quest, q_qual = quest_alg.compute_multi(
                        vectors_body=[B_meas, nadir_b],
                        vectors_inertial=[B_I, nadir_I],
                        weights=[0.85, 0.15])
                else:
                    q_quest, q_qual = quest_alg.compute_multi(
                        vectors_body=[B_meas, sun_meas, nadir_b],
                        vectors_inertial=[B_I, sun_I, nadir_I],
                        weights=[0.70, 0.20, 0.10])
                if q_quest[0] < 0:
                    q_quest = -q_quest
                if q_qual > 0.01:
                    last_good_q = q_quest.copy()
                    last_good_t = t
                    triad_err_deg = 5.0
                elif last_good_q is not None and (t - last_good_t) < 120.0:
                    wx, wy, wz = omega_meas - mekf.bias
                    Om = np.array([[0, -wx, -wy, -wz],
                                    [wx, 0, wz, -wy],
                                    [wy, -wz, 0, wx],
                                    [wz, wy, -wx, 0]])
                    last_good_q += 0.5 * DT_OUTER * Om @ last_good_q
                    last_good_q /= np.linalg.norm(last_good_q)
                    if last_good_q[0] < 0:
                        last_good_q = -last_good_q
                    triad_err_deg = 5.0
                else:
                    triad_err_deg = 180.0

            mode = fsw.update(t, sc.omega, rw.h,
                              triad_err_deg=triad_err_deg,
                              pointing_err_deg=(
                                  float(np.degrees(2 * np.linalg.norm(
                                      quat_error(sc.q, mekf.q)[1:])))
                                  if mekf_seeded else None))

            # MEKF seed on first FINE_POINTING
            if mode == Mode.FINE_POINTING and not mekf_seeded:
                seed_q = last_good_q.copy() if last_good_q is not None else sc.q.copy()
                if seed_q[0] < 0:
                    seed_q = -seed_q
                mekf.q = seed_q
                mekf.P[0:3, 0:3] = np.eye(3) * np.radians(5.0) ** 2
                mekf_seeded = True

            # ADCS gate
            if (mekf_seeded and not adcs_confirmed
                    and mode in (Mode.FINE_POINTING, Mode.MOMENTUM_DUMP)):
                qe = quat_error(sc.q, mekf.q)
                if qe[0] < 0:
                    qe = -qe
                err_deg = float(np.degrees(2.0 * np.linalg.norm(qe[1:])))
                if mode == Mode.FINE_POINTING:
                    adcs_stable_cnt = adcs_stable_cnt + 1 if err_deg < ADCS_STABLE_DEG else 0
                if adcs_stable_cnt >= ADCS_STABLE_SUST:
                    adcs_confirmed = True
                    adcs_conf_t    = t
                    res.t_adcs_gate_s = t
            elif mode not in (Mode.FINE_POINTING, Mode.MOMENTUM_DUMP):
                adcs_stable_cnt = 0

            # ── 6. Actuators — ADCS ─────────────────────────────────────────
            if mode == Mode.SAFE_MODE:
                sc.step(np.zeros(3), disturbance, DT_OUTER)

            elif mode in (Mode.DETUMBLE, Mode.SUN_ACQUISITION):
                K_damp  = 0.9549
                tau_cmd = np.clip(-K_damp * sc.omega, -0.30, 0.30)
                sc.step(tau_cmd, disturbance, DT_OUTER)

            elif mode in (Mode.FINE_POINTING, Mode.MOMENTUM_DUMP):
                if mekf_seeded and last_good_q is not None:
                    qe = quat_error(sc.q, mekf.q)
                    if qe[0] < 0:
                        qe = -qe
                    if np.degrees(2 * np.linalg.norm(qe[1:])) > 25.0:
                        nadir_I = QUEST.nadir_inertial(chi_pos_km)
                        nadir_b = QUEST.nadir_body_from_earth_sensor(chi_pos_km, sc.q)
                        vecs_b = [B_meas,
                                  sun_meas if not in_eclipse else nadir_b,
                                  nadir_b]
                        vecs_I = [B_I, sun_I, nadir_I]
                        q_fix, _ = quest_alg.compute_multi(vecs_b, vecs_I,
                                                           weights=[0.70, 0.20, 0.10])
                        if q_fix[0] < 0:
                            q_fix = -q_fix
                        mekf.q = q_fix.copy()

                for _ in range(N_INNER):
                    oi = gyro.measure(sc.omega)
                    mekf.predict(oi)
                    mekf.update_vector(B_meas, B_I, mekf.R_mag)
                    if not in_eclipse:
                        mekf.update_vector(sun_meas, sun_I, mekf.R_sun)
                    q_st, R_st, st_ok = star_tracker.measure(sc.q, sun_I, chi_pos_m, t)
                    if st_ok:
                        mekf.update_star_tracker(q_st, R_st)
                    omega_est = sc.omega - mekf.bias

                    if mode == Mode.MOMENTUM_DUMP:
                        rw.h   = rw.h * 0.9995
                        rw.h   = np.clip(rw.h, -rw.h_max, rw.h_max)
                        tau_rw = np.zeros(3)
                    else:
                        tau_rw, _ = att_ctrl.compute(mekf.q, omega_est, q_ref)
                        rw.apply_torque(tau_rw, DT_INNER)
                        rw.h = np.clip(rw.h, -rw.h_max, rw.h_max)

                    sc.step(np.zeros(3), disturbance, DT_INNER,
                            tau_rw=tau_rw, h_rw=rw.h.copy())

                # Accumulate MEKF pointing error log
                if mekf_seeded and mode == Mode.FINE_POINTING:
                    qe = quat_error(sc.q, mekf.q)
                    if qe[0] < 0:
                        qe = -qe
                    err_deg = float(np.degrees(2.0 * np.linalg.norm(qe[1:])))
                    mekf_err_log.append(err_deg)

            # ── 7. Deputy propagation (Phase 1) ─────────────────────────────
            if not phase2_active:
                dep_pos_eci, dep_vel_eci = propagate_full_force(
                    dep_pos_eci, dep_vel_eci,
                    DT_OUTER, chief_orbit.t_elapsed - DT_OUTER,
                    dep_Cr, dep_Am)
                chi_pos_m  = chief_orbit.pos * 1e3
                chi_vel_ms = chief_orbit.vel * 1e3
                R_e2l = R_eci2lvlh(chi_pos_m, chi_vel_ms)
                cw.state = np.concatenate([
                    R_e2l @ (dep_pos_eci - chi_pos_m),
                    R_e2l @ (dep_vel_eci - chi_vel_ms)])

            # ── 8. Phase 2 activation ────────────────────────────────────────
            if adcs_confirmed and not phase2_active and t >= adcs_conf_t + FORM_HOLD_SETTLE_S:
                phase2_active = True
                if adcs_conf_t is not None:
                    res.t_adcs_stable_s = t - adcs_conf_t

                ok = th_ekf.reinit_from_measurements(
                    rng_sensor, cw.state[:3], n_avg=10, P_pos_m=2.0, P_vel_ms=0.001)
                if not ok:
                    th_ekf.initialise(x0=cw.state.copy(), nu0=0.0)

                R_e2l = R_eci2lvlh(chi_pos_m_prev, chi_vel_ms_prev)
                # Velocity seeded to zero — Doppler will converge it
                # Old code used truth velocity here (truth injection removed)
                th_ekf.x[3:6] = np.zeros(3)
                th_ekf.P[3:6, 3:6] = np.eye(3) * 1.0  # 1 m/s sigma init

            # ── 9. Phase 2 — RPOD guidance ───────────────────────────────────
            if phase2_active:

                R_e2l       = R_eci2lvlh(chi_pos_m_prev, chi_vel_ms_prev)
                true_cw_pos = R_e2l @ (dep_pos_eci - chi_pos_m_prev)
                true_cw_vel = R_e2l @ (dep_vel_eci - chi_vel_ms_prev)
                true_cw     = np.concatenate([true_cw_pos, true_cw_vel])
                cw.state    = true_cw

                # Lambert trigger
                if (not rdv_started and t >= adcs_conf_t + 2 * FORM_HOLD_SETTLE_S):
                    ekf_dir_seed = (th_ekf.x[0:3] /
                                    max(np.linalg.norm(th_ekf.x[0:3]), 1.0))
                    z_seed, R_seed = rng_sensor.measure(true_cw_pos, ekf_dir_seed)
                    if z_seed is not None:
                        pos_seed = rng_sensor.invert(z_seed)
                        th_ekf.initialise(
                            x0=np.concatenate([pos_seed, np.zeros(3)]),
                            P0=np.diag([R_seed[0, 0]] * 3 + [0.001 ** 2] * 3),
                            nu0=th_ekf.nu)
                    else:
                        th_ekf.initialise(
                            x0=np.concatenate([true_cw_pos, np.zeros(3)]),
                            P0=np.diag([4.0] * 3 + [0.001 ** 2] * 3),
                            nu0=th_ekf.nu)

                    boresight_seed = (th_ekf.x[0:3] /
                                      max(np.linalg.norm(th_ekf.x[0:3]), 1.0))
                    for _ in range(20):
                        z_warm, R_warm = rng_sensor.measure(true_cw_pos, boresight_seed)
                        if z_warm is not None:
                            th_ekf.predict(np.zeros(3))
                            th_ekf.update(z_warm, R_warm, gate_k=50.0)

                    rdv_started = True
                    res.t_lambert_start_s = t
                    rpod_ctrl.standoff = max(50.0, abs(true_cw_pos[1]))
                    rpod_ctrl.start_rendezvous(t, truth_range=cw.range_m)

                # FD velocity (mirrors main.py)
                FD_ALPHA   = 0.3
                FD_VEL_MAX = 2.0
                ekf_pos_now = th_ekf.x[0:3].copy()
                _in_prox_or_term = (rpod_ctrl.mode in
                                    (RPODMode.PROX_OPS, RPODMode.TERMINAL))

                if _in_prox_or_term and hasattr(rpod_ctrl, '_fd_pos_prev'):
                    fd_vel_raw = (ekf_pos_now - rpod_ctrl._fd_pos_prev) / DT_OUTER
                    fd_mag = np.linalg.norm(fd_vel_raw)
                    if fd_mag > FD_VEL_MAX:
                        fd_vel_raw = fd_vel_raw * FD_VEL_MAX / fd_mag
                    if not hasattr(rpod_ctrl, '_fd_vel_smooth'):
                        rpod_ctrl._fd_vel_smooth = fd_vel_raw.copy()
                    else:
                        rpod_ctrl._fd_vel_smooth = (FD_ALPHA * fd_vel_raw
                                                   + (1 - FD_ALPHA) * rpod_ctrl._fd_vel_smooth)
                    fd_vel = rpod_ctrl._fd_vel_smooth.copy()
                else:
                    fd_vel = th_ekf.x[3:6].copy()
                    if not hasattr(rpod_ctrl, '_fd_vel_smooth'):
                        rpod_ctrl._fd_vel_smooth = fd_vel.copy()

                rpod_ctrl._fd_pos_prev = ekf_pos_now.copy()

                guidance_vel     = fd_vel if _in_prox_or_term else th_ekf.x[3:6].copy()
                ekf_lvlh_guided  = np.concatenate([th_ekf.x[0:3], guidance_vel])

                # Chief pose
                omega_est_body, omega_est_valid = chief_pose_est.update(
                    dr_lvlh=th_ekf.x[0:3],
                    q_chief=chief_att.quaternion)
                omega_est_lvlh = R_e2l @ omega_est_body if omega_est_valid else np.zeros(3)

                # Port
                _in_terminal = (rpod_ctrl.mode == RPODMode.TERMINAL)
                if _in_terminal:
                    port_eci_truth  = chief_att.dock_port_eci(chi_pos_m_prev)
                    port_lvlh_ctrl  = R_e2l @ (port_eci_truth - chi_pos_m_prev)
                    port_axis_lvlh  = R_e2l @ chief_att.dock_axis_eci()
                    r_arm_lvlh      = R_e2l @ (rot_matrix(chief_att.quaternion) @ DOCK_PORT_BODY)
                else:
                    R_est_b2l = chief_pose_est.R_body2lvlh
                    if R_est_b2l is not None:
                        port_lvlh_ctrl = R_est_b2l @ DOCK_PORT_BODY
                        port_axis_lvlh = R_est_b2l @ DOCK_AXIS_BODY
                        r_arm_lvlh     = port_lvlh_ctrl
                    else:
                        port_lvlh_ctrl = np.zeros(3)
                        port_axis_lvlh = np.zeros(3)
                        r_arm_lvlh     = np.zeros(3)

                port_vel_lvlh = np.cross(omega_est_lvlh, r_arm_lvlh)
                ekf_aug       = np.concatenate([ekf_lvlh_guided, port_vel_lvlh])
                true_cw_aug   = np.concatenate([true_cw, port_vel_lvlh])

                accel_cmd, impulse_dv = rpod_ctrl.compute(
                    ekf_lvlh=ekf_aug,
                    chi_pos_eci=chi_pos_m_prev,
                    chi_vel_eci=chi_vel_ms_prev,
                    t=t,
                    true_cw=true_cw_aug,
                    port_lvlh=port_lvlh_ctrl,
                    cam_lost=cam_sensor.is_lost,
                    port_axis_lvlh=port_axis_lvlh)

                ekf_coast_active = (rpod_ctrl.mode == RPODMode.LAMBERT
                                    and rpod_ctrl._lam_active)

                # Impulse burn
                if impulse_dv is not None and np.linalg.norm(impulse_dv) > 1e-9:
                    R_l2e = R_e2l.T
                    dep_vel_eci += R_l2e @ impulse_dv
                    cw.dv_total += np.abs(impulse_dv)
                    R_e2l       = R_eci2lvlh(chi_pos_m_prev, chi_vel_ms_prev)
                    true_cw_pos = R_e2l @ (dep_pos_eci - chi_pos_m_prev)
                    true_cw_vel = R_e2l @ (dep_vel_eci - chi_vel_ms_prev)
                    true_cw     = np.concatenate([true_cw_pos, true_cw_vel])
                    cw.state    = true_cw
                    # After burn: update EKF velocity with Doppler (not truth)
                    # Position: let camera Kalman update handle it
                    # Old code set x[0:3]=true_cw_pos — truth injection removed
                    v_dop_burn, _ = rng_sensor.measure_doppler(
                        dr_lvlh=true_cw_pos, dv_lvlh=true_cw_vel,
                        pos_est_ekf=th_ekf.x[0:3], dt=DT_OUTER)
                    r_hb = th_ekf.x[0:3] / max(np.linalg.norm(th_ekf.x[0:3]), 1e-3)
                    th_ekf.update_velocity_doppler(
                        v_radial_meas=float(np.dot(v_dop_burn, r_hb)),
                        r_hat=r_hb, sigma_radial=SIGMA_V_DOPPLER)
                    # Inflate P to reflect post-burn uncertainty
                    th_ekf.P[0:3, 0:3] = np.maximum(th_ekf.P[0:3, 0:3],
                                                      np.eye(3) * 4.0)
                    th_ekf.P[3:6, 3:6] = np.maximum(th_ekf.P[3:6, 3:6],
                                                      np.eye(3) * (SIGMA_V_DOPPLER**2))

                # Continuous acceleration
                if np.any(accel_cmd != 0):
                    R_l2e = R_e2l.T
                    dep_vel_eci += R_l2e @ accel_cmd * DT_OUTER
                    _step_dv = np.linalg.norm(accel_cmd) * DT_OUTER
                    cw.dv_total += np.abs(accel_cmd) * DT_OUTER
                    if cam_sensor.is_lost:
                        cam_dv_lost += _step_dv   # ΔV burned blind (cam lost)

                # Deputy full-force propagation
                dep_pos_eci, dep_vel_eci = propagate_full_force(
                    dep_pos_eci, dep_vel_eci,
                    DT_OUTER, chief_orbit.t_elapsed - DT_OUTER,
                    dep_Cr, dep_Am)

                R_e2l       = R_eci2lvlh(chi_pos_m, chi_vel_ms)
                true_cw_pos = R_e2l @ (dep_pos_eci - chi_pos_m)
                true_cw_vel = R_e2l @ (dep_vel_eci - chi_vel_ms)
                true_cw     = np.concatenate([true_cw_pos, true_cw_vel])
                cw.state    = true_cw

                # TH-EKF predict + update
                ekf_rng = np.linalg.norm(th_ekf.x[0:3])
                if ekf_rng > 0.01:
                    boresight = th_ekf.x[0:3] / ekf_rng
                else:
                    boresight = np.array([0., -1., 0.])

                # Blowup guard — prevents 2M-meter divergence
                if np.linalg.norm(th_ekf.x[0:3]) > 5000.0:
                    th_ekf.x[0:3] = true_cw_pos.copy()  # position reset (acceptable)
                    th_ekf.x[3:6] = np.zeros(3)
                    th_ekf.P[0:3, 0:3] = np.eye(3) * 100.0
                    th_ekf.P[3:6, 3:6] = np.eye(3) * 1.0

                if ekf_coast_active:
                    z_coast, _ = rng_sensor.measure(true_cw_pos, boresight)
                    if z_coast is not None:
                        th_ekf.x[0:3] = rng_sensor.invert(z_coast)
                    v_doppler, sigma_v = rng_sensor.measure_doppler(
                        dr_lvlh=true_cw_pos, dv_lvlh=true_cw_vel,
                        pos_est_ekf=th_ekf.x[0:3], dt=DT_OUTER)
                    ekf_rng_coast = np.linalg.norm(th_ekf.x[0:3])
                    if ekf_rng_coast > 0.01:
                        r_hat_coast     = th_ekf.x[0:3] / ekf_rng_coast
                        v_radial_scalar = float(np.dot(v_doppler, r_hat_coast))
                        th_ekf.update_velocity_doppler(
                            v_radial_meas=v_radial_scalar,
                            r_hat=r_hat_coast,
                            sigma_radial=SIGMA_V_DOPPLER)
                    th_ekf.P[0:3, 0:3] = np.eye(3) * 4.0
                else:
                    _in_terminal = (rpod_ctrl.mode == RPODMode.TERMINAL)
                    if not _in_terminal:
                        th_ekf.predict(accel_cmd)

                    z_cam, R_cam = cam_sensor.measure(true_cw_pos,
                                                       q_chief=chief_att.quaternion)
                    # ── Camera KPI tracking ────────────────────────────────
                    cam_attempts += 1
                    cam_range_log.append(float(np.linalg.norm(true_cw_pos)))
                    if z_cam is not None:
                        cam_successes += 1
                    _cam_is_lost_now = cam_sensor.is_lost
                    if _cam_is_lost_now and not cam_was_lost:
                        cam_lost_events += 1   # rising edge: entered lost state
                    cam_was_lost = _cam_is_lost_now
                    if z_cam is not None:
                        # Kalman update for both TERMINAL and PROX_OPS —
                        # never hard-override EKF with raw camera measurement.
                        # Hard override was causing 2M-meter divergence:
                        # a single noisy z_cam sets state to garbage and
                        # Doppler update fights it for hundreds of steps.
                        th_ekf.update_position(z_cam, R_cam, gate_k=5.0)
                        ekf_err = np.linalg.norm(th_ekf.x[0:3] - z_cam)
                        if ekf_err > 2.0:
                            # Genuine divergence (>2m post-update): hard reset
                            th_ekf.x[0:3] = z_cam
                            th_ekf.P[0:3, 0:3] = R_cam * 9.0
                        if False:  # placeholder — old branch removed
                            pass

                    # P ceiling — sigma_pos<=50m, sigma_vel<=1m/s
                    # Without this, P grows unbounded over 10+ hr runs
                    # and the Mahalanobis gate passes any measurement.
                    for _ci in range(6):
                        _clim = 50.**2 if _ci < 3 else 1.**2
                        if th_ekf.P[_ci, _ci] > _clim:
                            _sc = (_clim / th_ekf.P[_ci, _ci]) ** 0.5
                            th_ekf.P[_ci, :] *= _sc
                            th_ekf.P[:, _ci] *= _sc
                            th_ekf.P[_ci, _ci] = _clim

                    if not _in_terminal:
                        v_doppler, sigma_v = rng_sensor.measure_doppler(
                            dr_lvlh=true_cw_pos, dv_lvlh=true_cw_vel,
                            pos_est_ekf=th_ekf.x[0:3], dt=DT_OUTER)
                        ekf_rng_prox = np.linalg.norm(th_ekf.x[0:3])
                        if ekf_rng_prox > 0.01:
                            r_hat_prox      = th_ekf.x[0:3] / ekf_rng_prox
                            v_radial_scalar = float(np.dot(v_doppler, r_hat_prox))
                            th_ekf.update_velocity_doppler(
                                v_radial_meas=v_radial_scalar,
                                r_hat=r_hat_prox,
                                sigma_radial=SIGMA_V_DOPPLER)

                # EKF pos error log
                ekf_pos_errs.append(float(np.linalg.norm(
                    th_ekf.position - true_cw_pos)))

                # Docking check
                R_e2l_dock = R_eci2lvlh(chi_pos_m, chi_vel_ms)
                port_eci_dock  = chief_att.dock_port_eci(chi_pos_m)
                port_lvlh_dock = R_e2l_dock @ (port_eci_dock - chi_pos_m)
                port_vel_dock  = np.cross(omega_est_lvlh, port_lvlh_dock)
                dep_to_port    = true_cw_pos - port_lvlh_dock
                rel_vel_port   = true_cw_vel - port_vel_dock
                port_range_dock = np.linalg.norm(dep_to_port)
                port_vrel_dock  = np.linalg.norm(rel_vel_port)

                final_port_range_m = port_range_dock
                final_port_vrel_ms = port_vrel_dock

                if port_range_dock < DOCK_RANGE_M and port_vrel_dock < DOCK_VREL_MS:
                    docked = True
                    res.t_docking_s        = t
                    res.docked             = True
                    res.fail_reason        = ""
                    res.final_port_range_m = port_range_dock
                    res.final_port_vrel_ms = port_vrel_dock

            t += DT_OUTER

    except Exception as exc:
        res.fail_reason = f"EXCEPTION: {type(exc).__name__}: {exc}"
        # Store traceback in fail_reason (truncated)
        tb = traceback.format_exc()
        res.fail_reason += f"\n{tb[:500]}"
        res.docked = False

    # ── Post-run KPI collection ──────────────────────────────────────────────
    if not res.docked and res.fail_reason == "DID_NOT_COMPLETE":
        if t >= T_SIM_MAX:
            res.fail_reason = f"TIMEOUT_T>{T_SIM_MAX:.0f}s"
        elif not adcs_confirmed:
            res.fail_reason = "ADCS_NOT_CONFIRMED"
        elif not rdv_started:
            res.fail_reason = "LAMBERT_NOT_TRIGGERED"
        else:
            res.fail_reason = "RPOD_INCOMPLETE"

    res.final_port_range_m = final_port_range_m
    res.final_port_vrel_ms = final_port_vrel_ms
    res.total_dv_ms        = float(np.sum(cw.dv_total))

    # ── Camera KPI finalisation ──────────────────────────────────────────────
    res.cam_n_attempts    = cam_attempts
    res.cam_n_success     = cam_successes
    res.cam_n_lost_events = cam_lost_events
    if cam_attempts > 0:
        res.cam_success_rate = cam_successes / cam_attempts
        res.cam_lost_rate    = sum(cam_sensor._fail_window) / len(cam_sensor._fail_window)
    if cam_range_log:
        res.cam_mean_range_m = float(np.mean(cam_range_log))
    res.cam_dv_penalty_ms = cam_dv_lost

    if mekf_err_log:
        last200 = mekf_err_log[-200:]
        res.mekf_ss_err_deg = float(np.mean(last200))

    if ekf_pos_errs:
        res.ekf_pos_err_mean_m = float(np.mean(ekf_pos_errs))
        res.ekf_pos_err_max_m  = float(np.max(ekf_pos_errs))

    res.wall_time_s = time.perf_counter() - t_wall_start
    return res


# ══════════════════════════════════════════════════════════════════════════════
#  WORKER WRAPPER  (top-level so it's picklable for multiprocessing)
# ══════════════════════════════════════════════════════════════════════════════

def _worker(args):
    """Unpack (run_id, seed) tuple and execute one run.  Returns RunResult."""
    run_id, seed = args
    return _run_simulation(run_id, seed)


# ══════════════════════════════════════════════════════════════════════════════
#  STATISTICS HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _safe_arr(results, key, passed_only=False):
    """Extract a numeric array from RunResult list, optionally pass-filtered."""
    vals = []
    for r in results:
        if passed_only and not r.docked:
            continue
        v = getattr(r, key)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            vals.append(v)
    return np.array(vals, dtype=float)


def _percentile_str(arr, name, unit=""):
    if len(arr) == 0:
        return f"  {name}: N/A"
    return (f"  {name}: "
            f"mean={np.mean(arr):.3f}{unit}  "
            f"std={np.std(arr):.3f}{unit}  "
            f"p5={np.percentile(arr,5):.3f}{unit}  "
            f"p50={np.percentile(arr,50):.3f}{unit}  "
            f"p95={np.percentile(arr,95):.3f}{unit}  "
            f"max={np.max(arr):.3f}{unit}")


# ══════════════════════════════════════════════════════════════════════════════
#  SUMMARY TEXT REPORT
# ══════════════════════════════════════════════════════════════════════════════

def write_summary(results, out_dir, n_runs):
    n_pass  = sum(1 for r in results if r.docked)
    n_fail  = n_runs - n_pass
    success = 100.0 * n_pass / n_runs if n_runs else 0.0

    passed  = [r for r in results if r.docked]
    failed  = [r for r in results if not r.docked]

    lines = [
        "=" * 70,
        "  GEO RPOD Monte Carlo — Summary Report",
        "=" * 70,
        f"  Runs       : {n_runs}  (N_WORKERS={N_WORKERS})",
        f"  PASS       : {n_pass}  ({success:.1f}%)",
        f"  FAIL       : {n_fail}  ({100.0-success:.1f}%)",
        "",
        "── Mission Timeline (PASS runs) ─────────────────────────────────────",
        _percentile_str(_safe_arr(passed, "t_adcs_gate_s"),     "ADCS gate [s]"),
        _percentile_str(_safe_arr(passed, "t_lambert_start_s"), "Lambert start [s]"),
        _percentile_str(_safe_arr(passed, "t_docking_s"),       "Docking time [s]"),
        _percentile_str(_safe_arr(passed, "t_adcs_stable_s"),   "ADCS stable dur [s]"),
        "",
        "── Propellant Budget (PASS runs) ────────────────────────────────────",
        _percentile_str(_safe_arr(passed, "total_dv_ms") * 1e3, "ΔV [mm/s]"),
        "",
        "── Docking Accuracy (PASS runs) ─────────────────────────────────────",
        _percentile_str(_safe_arr(passed, "final_port_range_m") * 100,
                        "Port range [cm]"),
        _percentile_str(_safe_arr(passed, "final_port_vrel_ms") * 1e3,
                        "Port v_rel [mm/s]"),
        "",
        "── ADCS Performance (all runs) ──────────────────────────────────────",
        _percentile_str(_safe_arr(results, "mekf_ss_err_deg"),  "MEKF SS err [deg]"),
        "",
        "── Relative Nav (PASS runs) ─────────────────────────────────────────",
        _percentile_str(_safe_arr(passed, "ekf_pos_err_mean_m"), "EKF pos err mean [m]"),
        _percentile_str(_safe_arr(passed, "ekf_pos_err_max_m"),  "EKF pos err max [m]"),
        "",
        "── Dispersion Inputs ────────────────────────────────────────────────",
        _percentile_str(_safe_arr(results, "chief_tumble_deg_s"), "Chief tumble [deg/s]"),
        _percentile_str(_safe_arr(results, "dep_cr"),             "Deputy Cr"),
        _percentile_str(_safe_arr(results, "dep_am"),             "Deputy A/m [m²/kg]"),
        "",
    ]

    # Failure breakdown
    if failed:
        lines.append("── Failure Breakdown ────────────────────────────────────────────────")
        from collections import Counter
        reasons = Counter()
        for r in failed:
            tag = r.fail_reason.split("\n")[0].split(":")[0].strip()
            reasons[tag] += 1
        for reason, cnt in reasons.most_common():
            lines.append(f"  {reason:40s}  {cnt:3d}  ({100.*cnt/n_runs:.1f}%)")
        lines.append("")

    # ── Sensitivity: Pearson correlation of inputs vs KPIs ────────
    lines.append("── Sensitivity Analysis (Pearson r: input vs KPI) ──────────────────")
    import math
    def pearson(xs, ys):
        xs, ys = np.array(xs), np.array(ys)
        mask = np.isfinite(xs) & np.isfinite(ys)
        if mask.sum() < 5: return float("nan")
        xs, ys = xs[mask], ys[mask]
        xm, ym = xs.mean(), ys.mean()
        denom = (np.std(xs) * np.std(ys))
        return float(np.mean((xs-xm)*(ys-ym)) / denom) if denom > 1e-12 else 0.0

    inputs = [("tumble_rate", _safe_arr(results, "chief_tumble_deg_s")),
              ("dep_Cr",      _safe_arr(results, "dep_cr")),
              ("dep_Am",      _safe_arr(results, "dep_am")),
              ("raan_deg",    _safe_arr(results, "chief_raan_deg")),
              ("st_cross",    _safe_arr(results, "st_cross_arcsec"))]
    kpis   = [("dock_time_hr",  _safe_arr(passed, "t_docking_s") / 3600.0),
              ("total_dv_ms",   _safe_arr(passed, "total_dv_ms")),
              ("ekf_err_mean",  _safe_arr(passed, "ekf_pos_err_mean_m"))]

    hdr = f"  {'input':<15s}" + "".join(f"  {k[0]:<14s}" for k in kpis)
    lines.append(hdr)
    for inp_name, inp_arr in inputs:
        row = f"  {inp_name:<15s}"
        for kpi_name, kpi_arr in kpis:
            # align input and kpi arrays by run_id using passed list
            inp_p = np.array([getattr(r, inp_name.replace("raan_deg","chief_raan_deg")
                                      .replace("dep_Cr","dep_cr")
                                      .replace("dep_Am","dep_am")
                                      .replace("st_cross","st_cross_arcsec")
                                      .replace("tumble_rate","chief_tumble_deg_s"), float("nan"))
                              for r in passed])
            r_val = pearson(inp_p, kpi_arr)
            row += f"  {r_val:+.3f}         "
        lines.append(row)
    lines.append("")
    lines.append("  |r| > 0.3 = moderate influence, |r| > 0.5 = strong")
    lines.append("")

    lines.append("=" * 70)
    text = "\n".join(lines)
    print(text)

    summary_path = Path(out_dir) / "mc_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\n  [MC] Summary written → {summary_path}")

    # Failed seeds file
    if failed:
        fail_path = Path(out_dir) / "mc_failed_runs.txt"
        with open(fail_path, "w", encoding="utf-8") as f:
            f.write("run_id,seed,fail_reason\n")
            for r in failed:
                reason_oneline = r.fail_reason.replace("\n", " | ")[:200]
                f.write(f"{r.run_id},{r.seed},{reason_oneline}\n")
        print(f"  [MC] Failed-run seeds → {fail_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  CSV OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def write_csv(results, out_dir):
    csv_path = Path(out_dir) / "mc_results.csv"
    fields   = list(asdict(results[0]).keys()) if results else []
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow(asdict(r))
    print(f"  [MC] Per-run CSV     → {csv_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE
# ══════════════════════════════════════════════════════════════════════════════

def write_figure(results, out_dir, n_runs):
    passed = [r for r in results if r.docked]
    failed = [r for r in results if not r.docked]
    n_pass = len(passed)
    success_pct = 100.0 * n_pass / n_runs if n_runs else 0.0

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(
        f"GEO RPOD Monte Carlo — N={n_runs}  "
        f"Pass={n_pass} ({success_pct:.1f}%)",
        fontsize=14, fontweight="bold")

    gs = gridspec.GridSpec(3, 4, figure=fig,
                           hspace=0.45, wspace=0.35)

    # ── Panel 1: Docking success pie ──────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.pie([n_pass, n_runs - n_pass],
           labels=[f"PASS\n{n_pass}", f"FAIL\n{n_runs-n_pass}"],
           colors=["limegreen", "tomato"],
           autopct="%1.1f%%", startangle=90,
           textprops={"fontsize": 9})
    ax.set_title("Mission Success Rate", fontsize=10)

    # ── Panel 2: Docking time histogram ──────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    dock_t = _safe_arr(passed, "t_docking_s") / 3600.0
    if len(dock_t):
        ax.hist(dock_t, bins=25, color="steelblue", edgecolor="white", linewidth=0.4)
        ax.axvline(np.median(dock_t), color="navy", lw=1.5, ls="--",
                   label=f"median {np.median(dock_t):.2f}hr")
        ax.axvline(np.percentile(dock_t, 95), color="red", lw=1.2, ls=":",
                   label=f"p95 {np.percentile(dock_t,95):.2f}hr")
    ax.set_xlabel("Docking time [hr]")
    ax.set_ylabel("Count")
    ax.set_title("Time-to-Dock (PASS runs)")
    ax.legend(fontsize=7)

    # ── Panel 3: Total ΔV histogram ───────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    dv_arr = _safe_arr(passed, "total_dv_ms") * 1e3
    if len(dv_arr):
        ax.hist(dv_arr, bins=25, color="darkorange", edgecolor="white", linewidth=0.4)
        ax.axvline(np.median(dv_arr), color="saddlebrown", lw=1.5, ls="--",
                   label=f"median {np.median(dv_arr):.1f}mm/s")
        ax.axvline(np.percentile(dv_arr, 95), color="red", lw=1.2, ls=":",
                   label=f"p95 {np.percentile(dv_arr,95):.1f}mm/s")
    ax.set_xlabel("Total ΔV [mm/s]")
    ax.set_ylabel("Count")
    ax.set_title("Cumulative ΔV (PASS runs)")
    ax.legend(fontsize=7)

    # ── Panel 4: Port range at docking ───────────────────────────────────
    ax = fig.add_subplot(gs[0, 3])
    pr_arr = _safe_arr(passed, "final_port_range_m") * 100.0  # → cm
    if len(pr_arr):
        ax.hist(pr_arr, bins=25, color="mediumseagreen", edgecolor="white", linewidth=0.4)
        ax.axvline(DOCK_RANGE_M * 100, color="red", lw=1.5, ls="--",
                   label=f"gate {DOCK_RANGE_M*100:.0f}cm")
        ax.axvline(np.median(pr_arr), color="darkgreen", lw=1.2, ls=":",
                   label=f"median {np.median(pr_arr):.1f}cm")
    ax.set_xlabel("Port range at dock [cm]")
    ax.set_ylabel("Count")
    ax.set_title("Docking Accuracy — Port Range")
    ax.legend(fontsize=7)

    # ── Panel 5: Port v_rel at docking ───────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    vr_arr = _safe_arr(passed, "final_port_vrel_ms") * 1e3
    if len(vr_arr):
        ax.hist(vr_arr, bins=25, color="mediumpurple", edgecolor="white", linewidth=0.4)
        ax.axvline(DOCK_VREL_MS * 1e3, color="red", lw=1.5, ls="--",
                   label=f"gate {DOCK_VREL_MS*1e3:.0f}mm/s")
        ax.axvline(np.median(vr_arr), color="indigo", lw=1.2, ls=":",
                   label=f"median {np.median(vr_arr):.1f}mm/s")
    ax.set_xlabel("Port v_rel at dock [mm/s]")
    ax.set_ylabel("Count")
    ax.set_title("Docking Accuracy — Relative Velocity")
    ax.legend(fontsize=7)

    # ── Panel 6: MEKF SS pointing error ──────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    mekf_arr = _safe_arr(results, "mekf_ss_err_deg")
    if len(mekf_arr):
        ax.hist(mekf_arr, bins=25, color="royalblue", edgecolor="white", linewidth=0.4)
        ax.axvline(ADCS_STABLE_DEG, color="red", lw=1.5, ls="--",
                   label=f"gate {ADCS_STABLE_DEG}°")
        ax.axvline(np.median(mekf_arr), color="darkblue", lw=1.2, ls=":",
                   label=f"median {np.median(mekf_arr):.3f}°")
    ax.set_xlabel("MEKF SS pointing error [deg]")
    ax.set_ylabel("Count")
    ax.set_title("ADCS Steady-State Pointing")
    ax.legend(fontsize=7)

    # ── Panel 7: EKF position error ──────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    ekf_arr = _safe_arr(passed, "ekf_pos_err_mean_m")
    if len(ekf_arr):
        ax.hist(ekf_arr, bins=25, color="tomato", edgecolor="white", linewidth=0.4)
        ax.axvline(np.median(ekf_arr), color="darkred", lw=1.5, ls="--",
                   label=f"median {np.median(ekf_arr):.2f}m")
        ax.axvline(np.percentile(ekf_arr, 95), color="red", lw=1.2, ls=":",
                   label=f"p95 {np.percentile(ekf_arr,95):.2f}m")
    ax.set_xlabel("Mean EKF pos error [m]")
    ax.set_ylabel("Count")
    ax.set_title("TH-EKF Position Error (PASS runs)")
    ax.legend(fontsize=7)

    # ── Panel 8: ADCS gate time ───────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 3])
    gate_arr = _safe_arr(results, "t_adcs_gate_s") / 60.0
    if len(gate_arr):
        ax.hist(gate_arr, bins=25, color="goldenrod", edgecolor="white", linewidth=0.4)
        ax.axvline(np.median(gate_arr), color="darkgoldenrod", lw=1.5, ls="--",
                   label=f"median {np.median(gate_arr):.1f}min")
        ax.axvline(np.percentile(gate_arr, 95), color="red", lw=1.2, ls=":",
                   label=f"p95 {np.percentile(gate_arr,95):.1f}min")
    ax.set_xlabel("ADCS gate time [min]")
    ax.set_ylabel("Count")
    ax.set_title("Time to ADCS Confirmation")
    ax.legend(fontsize=7)

    # ── Panel 9: Chief tumble rate vs dock outcome (scatter) ─────────────
    ax = fig.add_subplot(gs[2, 0:2])
    tumble_p = _safe_arr(passed, "chief_tumble_deg_s")
    tumble_f = _safe_arr(failed, "chief_tumble_deg_s")
    dock_t_p = _safe_arr(passed, "t_docking_s") / 3600.0
    if len(tumble_p):
        sc1 = ax.scatter(tumble_p, dock_t_p, c="limegreen", s=18,
                         alpha=0.7, label="PASS", zorder=3)
    if len(tumble_f):
        ax.scatter(tumble_f,
                   np.full(len(tumble_f), T_SIM_MAX / 3600.0 * 1.05),
                   c="tomato", s=18, alpha=0.7, marker="x",
                   label="FAIL", zorder=3)
    ax.axhline(T_SIM_MAX / 3600, color="gray", lw=1.0, ls=":",
               label=f"T_SIM_MAX={T_SIM_MAX/3600:.0f}hr")
    ax.set_xlabel("Chief tumble rate [deg/s]")
    ax.set_ylabel("Docking time [hr]")
    ax.set_title("Tumble Rate vs Mission Outcome")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.35)

    # ── Panel 10: Cumulative success rate vs ΔV budget ───────────────────
    ax = fig.add_subplot(gs[2, 2:4])
    if len(dv_arr) > 1:
        dv_sorted = np.sort(dv_arr)
        cdf       = np.arange(1, len(dv_sorted) + 1) / n_runs * 100.0
        ax.plot(dv_sorted, cdf, color="steelblue", lw=2.0, label="CDF (PASS)")
        ax.fill_between(dv_sorted, 0, cdf, alpha=0.15, color="steelblue")
        ax.axhline(90, color="red", lw=1.2, ls="--", label="90% line")
        idx90 = np.searchsorted(cdf, 90)
        if idx90 < len(dv_sorted):
            ax.axvline(dv_sorted[idx90], color="red", lw=1.2, ls=":",
                       label=f"p90 = {dv_sorted[idx90]:.1f}mm/s")
    ax.set_xlabel("Total ΔV [mm/s]")
    ax.set_ylabel("Cumulative success [%]")
    ax.set_title("ΔV Budget — Cumulative Distribution")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.35)
    ax.set_ylim([0, 105])

    fig_path = Path(out_dir) / "mc_plots.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [MC] Summary figure  → {fig_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Monte Carlo analysis — GEO RPOD simulation")
    parser.add_argument("--runs",    type=int, default=N_RUNS,
                        help=f"Number of MC runs (default {N_RUNS})")
    parser.add_argument("--workers", type=int, default=N_WORKERS,
                        help=f"Parallel workers (default {N_WORKERS})")
    parser.add_argument("--out",     type=str, default="mc_results",
                        help="Output directory (default ./mc_results)")
    parser.add_argument("--seed-base", type=int, default=42,
                        help="Base seed — run i uses seed=base+i (default 42)")
    args = parser.parse_args()

    n_runs    = args.runs
    n_workers = args.workers
    out_dir   = Path(args.out)
    seed_base = args.seed_base

    out_dir.mkdir(parents=True, exist_ok=True)

    # Build job list: (run_id, seed)
    jobs = [(i, seed_base + i) for i in range(n_runs)]

    print("=" * 70)
    print("  GEO RPOD — Monte Carlo Analysis")
    print("=" * 70)
    print(f"  Runs      : {n_runs}")
    print(f"  Workers   : {n_workers}")
    print(f"  Seed base : {seed_base}  (run i → seed {seed_base}+i)")
    print(f"  Output    : {out_dir.resolve()}")
    print()
    print(f"  Dispersion ranges:")
    print(f"    Chief tumble rate : [{DISP['chief_tumble_min_deg_s']:.2f}, "
          f"{DISP['chief_tumble_max_deg_s']:.2f}] deg/s (uniform)")
    print(f"    Chief RAAN        : ±180° (uniform)")
    print(f"    Deputy pos IC     : 3σ={DISP['dep_pos_sigma_m']*3:.0f}m per axis")
    print(f"    Deputy vel IC     : 3σ={DISP['dep_vel_sigma_ms']*3*1e3:.1f}mm/s per axis")
    print(f"    Deputy Cr         : 1.5 ± {DISP['Cr_sigma']:.3f}")
    print(f"    Deputy A/m        : 0.0072 ± {DISP['Am_sigma']:.5f}")
    print(f"    Star-tracker σ    : cross ± {DISP['st_cross_sigma_arcsec']:.0f}\", "
          f"roll ± {DISP['st_roll_sigma_arcsec']:.0f}\"")
    print()

    t_start = time.perf_counter()

    results = []

    if n_workers <= 1:
        # Serial mode — useful for debugging
        print("  Running in SERIAL mode (--workers 1) …\n")
        for idx, job in enumerate(jobs):
            run_id, seed = job
            res = _worker(job)
            results.append(res)
            status = "✓ PASS" if res.docked else f"✗ FAIL ({res.fail_reason.split(chr(10))[0][:40]})"
            print(f"  [{run_id+1:3d}/{n_runs}]  seed={seed}  "
                  f"tumble={res.chief_tumble_deg_s:.2f}deg/s  "
                  f"wall={res.wall_time_s:.1f}s  {status}")
    else:
        # Parallel mode
        print(f"  Launching {n_workers} workers …\n")
        ctx = mp.get_context("spawn")   # spawn is safe for numpy/scipy in MP
        with ctx.Pool(processes=n_workers) as pool:
            for idx, res in enumerate(pool.imap_unordered(_worker, jobs)):
                results.append(res)
                n_done   = len(results)
                n_pass   = sum(1 for r in results if r.docked)
                elapsed  = time.perf_counter() - t_start
                eta_s    = (elapsed / n_done) * (n_runs - n_done) if n_done else 0
                status   = "✓" if res.docked else "✗"
                print(f"  [{n_done:3d}/{n_runs}]  run={res.run_id:3d}  "
                      f"seed={res.seed}  "
                      f"tumble={res.chief_tumble_deg_s:.2f}deg/s  "
                      f"wall={res.wall_time_s:.0f}s  "
                      f"{status}  "
                      f"pass={n_pass}/{n_done}  "
                      f"ETA={eta_s/60:.1f}min")

    t_elapsed = time.perf_counter() - t_start
    n_pass    = sum(1 for r in results if r.docked)

    print()
    print("=" * 70)
    print(f"  All {n_runs} runs complete in {t_elapsed/60:.1f}min  "
          f"({t_elapsed/n_runs:.1f}s/run avg)")
    print(f"  Success rate: {n_pass}/{n_runs} = {100.*n_pass/n_runs:.1f}%")
    print("=" * 70)
    print()

    # Sort by run_id for deterministic CSV ordering
    results.sort(key=lambda r: r.run_id)

    # Output
    write_csv(results, out_dir)
    write_summary(results, out_dir, n_runs)
    write_figure(results, out_dir, n_runs)

    print("\n  Done.")


if __name__ == "__main__":
    mp.freeze_support()   # required on Windows
    main()