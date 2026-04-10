"""
Monte Carlo Validation — GEO Jetpack RPOD
==========================================
Mirrors main.py physics exactly. No shortcuts.

Stress axes (15 total):
  1.  Initial tumble rate          Uniform [0.05, 0.40] rad/s, random axis
  2.  Gyro bias scale              Uniform [0.8, 1.2] x nominal
  3.  Magnetometer noise scale     Uniform [0.8, 1.5] x nominal
  4.  Sun sensor noise scale       Uniform [0.8, 1.5] x nominal
  5.  Star tracker noise scale     Uniform [0.8, 1.5] x nominal
  6.  Ranging sensor noise scale   Uniform [0.8, 1.5] x nominal
  7.  Deputy SRP Cr                Uniform [1.3, 1.7]
  8.  Deputy A/m ratio             Uniform [0.006, 0.009] m2/kg
  9.  Chief SRP Cr                 Uniform [1.3, 1.7]
  10. Chief A/m ratio              Uniform [0.013, 0.017] m2/kg
  11. Chief M0 epoch offset        Uniform +/-0.5 deg  (changes Lambert geometry)
  12. Deputy position jitter       Normal  sigma=20 m per axis at Phase 2 entry
  13. Deputy velocity jitter       Normal  sigma=10 mm/s per axis at Phase 2 entry
  14. RDV trigger timing jitter    Uniform +/-120 s
  15. Initial reaction wheel bias  Uniform [-0.1, 0.1] N.m.s per axis

Outputs per trial:
  docked          bool
  dock_time_hr    float  (nan if not docked)
  total_dv_mms    float  cumulative dV [mm/s]
  adcs_time_s     float  time to Phase 1 confirmation
  mekf_ss_deg     float  MEKF pointing steady-state error [deg]
  final_range_m   float  truth range at end of simulation
  burn1_range_m   float  truth range at Lambert burn-1
  burn2_range_m   float  truth range at Lambert burn-2
  failure_mode    str    'DOCKED' | 'ADCS_TIMEOUT' | 'NO_LAMBERT' |
                         'PROX_STALL' | 'TIMEOUT' | 'EXCEPTION'

Usage:
  python monte_carlo.py                    # 50 trials, auto workers
  python monte_carlo.py --n 200            # 200 trials
  python monte_carlo.py --n 10 --workers 1 # serial (debug, shows exceptions)
  python monte_carlo.py --seed 1234        # reproducible run
"""

import argparse
import multiprocessing as mp
import os
import sys
import time
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter

warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


# =====================================================================
#  NOMINAL CONSTANTS  (must stay in sync with main.py)
# =====================================================================

CHIEF_A_KM      = 42164.0
CHIEF_E         = 0.0003
CHIEF_I_DEG     = 0.8
CHIEF_RAAN_DEG  = 0.0
CHIEF_OMEGA_DEG = 0.0

DEP_MASS_KG  = 50.0
DEP_THRUST_N = 1.0

FORMATION_OFFSET_M = np.array([0.0, -1000.0, 0.0])

DT_OUTER = 0.1
DT_INNER = 0.01
N_INNER  = int(DT_OUTER / DT_INNER)

T_SIM_MAX = 80_000.0

ADCS_STABLE_DEG  = 1.0
ADCS_STABLE_SUST = 100
FORM_HOLD_SETTLE_S = 300.0

DOCK_RANGE_M = 0.10
DOCK_VREL_MS = 0.01
ECLIPSE_NU_MIN = 0.1

MU_GEO = 3.986004418e14
N_GEO  = np.sqrt(MU_GEO / (CHIEF_A_KM * 1e3) ** 3)
I_SC   = np.diag([4.167, 4.167, 3.000])


# =====================================================================
#  PARAMETER SAMPLER
# =====================================================================

def sample_params(rng: np.random.Generator) -> dict:
    rate = rng.uniform(0.05, 0.40)
    axis = rng.standard_normal(3)
    axis /= np.linalg.norm(axis)
    return {
        'omega0':          rate * axis,
        'gyro_bias_scale': float(rng.uniform(0.8, 1.2)),
        'mag_noise_scale': float(rng.uniform(0.8, 1.5)),
        'sun_noise_scale': float(rng.uniform(0.8, 1.5)),
        'st_noise_scale':  float(rng.uniform(0.8, 1.5)),
        'rw_bias':         rng.uniform(-0.1, 0.1, 3),
        'rng_noise_scale': float(rng.uniform(0.8, 1.5)),
        'dep_cr':          float(rng.uniform(1.3, 1.7)),
        'dep_am':          float(rng.uniform(0.006, 0.009)),
        'chi_cr':          float(rng.uniform(1.3, 1.7)),
        'chi_am':          float(rng.uniform(0.013, 0.017)),
        'M0_offset_deg':   float(rng.uniform(-0.5, 0.5)),
        'pos_jitter_m':    rng.normal(0.0, 20.0, 3),
        'vel_jitter_ms':   rng.normal(0.0, 0.010, 3),
        'rdv_jitter_s':      float(rng.uniform(-120.0, 120.0)),
        # 5 new stress axes
        'thrust_misalign_rad': float(rng.uniform(0.0, np.radians(1.0))),
        'thrust_mag_err':      float(np.clip(rng.normal(1.0, 0.03), 0.90, 1.10)),
        'nav_delay_steps':     int(rng.integers(0, 11)),
        'dock_offset_m':       rng.normal(0.0, 0.03, 3),
        'dropout_rate_hz':     float(rng.uniform(0.0, 0.05)),
    }


# =====================================================================
#  SINGLE-TRIAL WORKER
# =====================================================================

def _worker(args):
    seed, params = args
    try:
        return run_once(seed, params)
    except Exception:
        import traceback
        tb = traceback.format_exc()
        return {
            'seed':           seed,
            'docked':         False,
            'dock_time_hr':   float('nan'),
            'total_dv_mms':   float('nan'),
            'adcs_time_s':    float('nan'),
            'mekf_ss_deg':    float('nan'),
            'final_range_m':  float('nan'),
            'burn1_range_m':  float('nan'),
            'burn2_range_m':  float('nan'),
            'nb_burns':       0,
            'failure_mode':   'EXCEPTION',
            'omega0_dps':     float(np.degrees(np.linalg.norm(params['omega0']))),
            'rdv_jitter_s':   float(params['rdv_jitter_s']),
            'pos_jitter_m':   float(np.linalg.norm(params['pos_jitter_m'])),
            'rng_noise_scale':float(params['rng_noise_scale']),
            'dep_cr':         float(params['dep_cr']),
            'dep_am':         float(params['dep_am']),
            'chi_am':         float(params['chi_am']),
            'gyro_bias_scale':float(params['gyro_bias_scale']),
            'st_noise_scale': float(params['st_noise_scale']),
            'M0_offset_deg':  float(params['M0_offset_deg']),
            'error':          tb[-1000:],
        }


def run_once(seed: int, params: dict) -> dict:
    """Execute one complete GEO RPOD mission. Physics = main.py. No prints."""
    import io, contextlib
    _null = io.StringIO()

    np.random.seed(seed)

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
    from actuators.reaction_wheel             import ReactionWheel
    from estimation.mekf                      import MEKF
    from estimation.quest                     import QUEST
    from estimation.th_ekf                    import THEKF
    from control.attitude_controller          import AttitudeController
    from control.lambert_controller           import GEORPODController, RPODMode
    from fsw.mode_manager                     import ModeManager, Mode
    from utils.quaternion                     import quat_error

    def R_eci2lvlh(r, v):
        xh = r / np.linalg.norm(r)
        zh = np.cross(r, v); zh /= np.linalg.norm(zh)
        yh = np.cross(zh, xh)
        return np.vstack([xh, yh, zh])

    def propagate_ff(pos, vel, dt_total, t_abs, Cr, Am, substep=60.0):
        J2 = 1.08263e-3; RE = 6.3781e6; AU = 1.495978707e11; P0 = 4.56e-6
        def sun_pos(t):
            d = t / 86400.0; lam = np.radians(280.46 + 360.985647 * d)
            eps = np.radians(23.439)
            return AU * np.array([np.cos(lam), np.cos(eps)*np.sin(lam), np.sin(eps)*np.sin(lam)])
        def accel(p, t):
            r = np.linalg.norm(p); a = -MU_GEO / r**3 * p
            x, y, z = p; c = -1.5*J2*MU_GEO*RE**2 / r**5; f = 5*z**2 / r**2
            a += np.array([c*x*(1-f), c*y*(1-f), c*z*(3-f)])
            sp = sun_pos(t); rs = np.linalg.norm(sp); P = P0*(AU/rs)**2
            dr = p - sp; a += Cr * Am * P * dr / np.linalg.norm(dr)
            return a
        n = max(1, int(round(dt_total / substep))); h = dt_total / n
        p, v, tc = pos.copy(), vel.copy(), float(t_abs)
        for _ in range(n):
            k1p = v;             k1v = accel(p, tc)
            k2p = v+0.5*h*k1v;  k2v = accel(p+0.5*h*k1p, tc+0.5*h)
            k3p = v+0.5*h*k2v;  k3v = accel(p+0.5*h*k2p, tc+0.5*h)
            k4p = v+h*k3v;       k4v = accel(p+h*k3p, tc+h)
            p += (h/6)*(k1p+2*k2p+2*k3p+k4p)
            v += (h/6)*(k1v+2*k2v+2*k3v+k4v)
            tc += h
        return p, v

    # Hardware — perturbed parameters
    with contextlib.redirect_stdout(_null):
        chief_orbit = GEOOrbitPropagator(
            a_km=CHIEF_A_KM, e=CHIEF_E, i_deg=CHIEF_I_DEG,
            raan_deg=CHIEF_RAAN_DEG, omega_deg=CHIEF_OMEGA_DEG,
            M0_deg=params['M0_offset_deg'],
            Cr=params['chi_cr'], Am_ratio=params['chi_am'])

        mag_field = MagneticField(epoch_year=2025.0)
        gg        = GravityGradient(I_SC)
        srp       = SolarRadiationPressure()
        sc        = Spacecraft(I_SC)
        sc.omega  = params['omega0'].copy()

        mag_sens     = Magnetometer(sigma_nT=100.0 * params['mag_noise_scale'])
        sun_sens     = SunSensor(sigma_noise=5e-4  * params['sun_noise_scale'])
        gyro         = Gyro(dt=DT_INNER, bias_init_max_deg_s=0.05 * params['gyro_bias_scale'])
        star_tracker = StarTracker(
            sigma_cross_arcsec=5.0  * params['st_noise_scale'],
            sigma_roll_arcsec =20.0 * params['st_noise_scale'],
            sun_excl_deg=30.0, earth_excl_deg=20.0,
            update_rate_hz=4.0, acquisition_s=30.0)
        rng_sensor = RangingBearingSensor(
            sigma_range_m    =1.0              * params['rng_noise_scale'],
            sigma_range_frac =0.001            * params['rng_noise_scale'],
            sigma_angle_rad  =np.radians(0.05) * params['rng_noise_scale'],
            fov_half_deg=60.0, max_range_m=5000.0)

        rw      = ReactionWheel(h_max=4.0)
        rw.h    = params['rw_bias'].copy()
        quest_alg = QUEST()
        mekf      = MEKF(DT_INNER)
        mekf.P[0:3, 0:3] = np.eye(3) * np.radians(5.0)**2
        th_ekf    = THEKF(a_chief=CHIEF_A_KM*1e3, e_chief=CHIEF_E,
                          dt=DT_OUTER, q_pos=1e-4, q_vel=1e-8)
        att_ctrl  = AttitudeController(Kp=0.08284, Kd=0.82257)
        q_ref     = np.array([1., 0., 0., 0.])
        rpod_ctrl = GEORPODController(
            mu=MU_GEO, n_chief=N_GEO,
            dep_mass_kg=DEP_MASS_KG, dep_thrust_N=DEP_THRUST_N,
            Cr_chi=params['chi_cr'], Am_chi=params['chi_am'],
            dock_capture_m=DOCK_RANGE_M,
            ekf=th_ekf, rng_sensor=rng_sensor)
        rpod_ctrl.standoff = np.linalg.norm(FORMATION_OFFSET_M)
        fsw = ModeManager()
        cw  = CWDynamics(chief_orbit_radius_km=CHIEF_A_KM)
        cw.set_initial_offset(dr_lvlh_m=FORMATION_OFFSET_M)

    # Simulation state
    t = 0.0
    dep_pos_eci = None; dep_vel_eci = None
    mekf_seeded = False; last_good_q = None; last_good_t = -999.0
    adcs_stable_cnt = 0; triad_err_deg = None
    adcs_confirmed = False; adcs_conf_t = None
    phase2_active = False; rdv_started = False
    pos_jitter_done = False; docked = False; ekf_coast_active = False
    mekf_err_samples = []; burn_ranges = []; failure_mode = 'TIMEOUT'
    # nav delay buffer
    _NAV_BUF_LEN = 12
    nav_buf = [None] * _NAV_BUF_LEN
    _nav_step = 0
    # sensor dropout state
    _dropout_active = False
    _dropout_end_t  = 0.0

    # Main loop
    while t < T_SIM_MAX and not docked:

        # 1. Chief orbit step
        chi_pos_m_prev  = chief_orbit.pos * 1e3
        chi_vel_ms_prev = chief_orbit.vel * 1e3
        chi_pos_km, chi_vel_kms = chief_orbit.step(DT_OUTER)
        chi_pos_m  = chi_pos_km * 1e3
        chi_vel_ms = chi_vel_kms * 1e3

        # 2. Deputy ECI init (first step only)
        if dep_pos_eci is None:
            R_l2e = R_eci2lvlh(chi_pos_m, chi_vel_ms).T
            dep_pos_eci = chi_pos_m + R_l2e @ FORMATION_OFFSET_M
            dv_ic       = np.array([0., -2.0*N_GEO*FORMATION_OFFSET_M[0], 0.])
            dep_vel_eci = chi_vel_ms + R_l2e @ dv_ic

        # 3. Environment
        nu_eclipse  = eclipse_nu(chi_pos_km, chief_orbit.t_elapsed)
        in_eclipse  = nu_eclipse < ECLIPSE_NU_MIN
        sun_I       = chief_orbit.get_sun_vector_eci()
        sun_pos_km  = sun_I * 1.496e8
        B_I         = mag_field.get_field(chi_pos_km)
        T_gg        = gg.compute(chi_pos_km, sc.q)
        T_srp, _    = srp.compute(sc.q, sun_I, pos_km=chi_pos_km, sun_pos_km=sun_pos_km)
        T_srp      *= nu_eclipse
        disturbance = T_gg + T_srp

        # 4. Sensors
        B_meas     = mag_sens.measure(sc.q, B_I)
        sun_meas   = sun_sens.measure(sc.q, sun_I) if not in_eclipse else np.zeros(3)
        omega_meas = gyro.measure(sc.omega)

        # 5. QUEST (during SUN_ACQ)
        if fsw.is_sun_acquiring:
            nadir_I = QUEST.nadir_inertial(chi_pos_km)
            nadir_b = QUEST.nadir_body_from_earth_sensor(chi_pos_km, sc.q)
            if in_eclipse:
                q_q, q_qual = quest_alg.compute_multi(
                    [B_meas, nadir_b], [B_I, nadir_I], [0.85, 0.15])
            else:
                q_q, q_qual = quest_alg.compute_multi(
                    [B_meas, sun_meas, nadir_b],
                    [B_I,    sun_I,   nadir_I], [0.70, 0.20, 0.10])
            if q_q[0] < 0: q_q = -q_q
            if q_qual > 0.01:
                last_good_q = q_q.copy(); last_good_t = t; triad_err_deg = 5.0
            elif last_good_q is not None and (t - last_good_t) < 120.0:
                wx, wy, wz = omega_meas - mekf.bias
                Om = np.array([[0,-wx,-wy,-wz],[wx,0,wz,-wy],
                                [wy,-wz,0,wx],[wz,wy,-wx,0]])
                last_good_q += 0.5*DT_OUTER*Om@last_good_q
                last_good_q /= np.linalg.norm(last_good_q)
                if last_good_q[0] < 0: last_good_q = -last_good_q
                triad_err_deg = 5.0
            else:
                triad_err_deg = 180.0

        # 6. Mode update
        pe = None
        if mekf_seeded:
            qe_c = quat_error(sc.q, mekf.q)
            if qe_c[0] < 0: qe_c = -qe_c
            pe = float(np.degrees(2*np.linalg.norm(qe_c[1:])))
        mode = fsw.update(t, sc.omega, rw.h,
                          triad_err_deg=triad_err_deg, pointing_err_deg=pe)

        # 7. MEKF seed
        if mode == Mode.FINE_POINTING and not mekf_seeded:
            sq = last_good_q.copy() if last_good_q is not None else sc.q.copy()
            if sq[0] < 0: sq = -sq
            mekf.q = sq
            mekf.P[0:3, 0:3] = np.eye(3) * np.radians(5.)**2
            mekf_seeded = True

        # 8. Phase 1 stability gate
        if mekf_seeded and not adcs_confirmed and mode == Mode.FINE_POINTING:
            qe = quat_error(sc.q, mekf.q)
            if qe[0] < 0: qe = -qe
            err = float(np.degrees(2.*np.linalg.norm(qe[1:])))
            adcs_stable_cnt = adcs_stable_cnt + 1 if err < ADCS_STABLE_DEG else 0
            if adcs_stable_cnt >= ADCS_STABLE_SUST:
                adcs_confirmed = True; adcs_conf_t = t
        elif mode not in (Mode.FINE_POINTING, Mode.MOMENTUM_DUMP):
            adcs_stable_cnt = 0

        # 9. Actuators
        if mode == Mode.SAFE_MODE:
            sc.step(np.zeros(3), disturbance, DT_OUTER)

        elif mode in (Mode.DETUMBLE, Mode.SUN_ACQUISITION):
            tau_cmd = np.clip(-0.9549*sc.omega, -0.30, 0.30)
            sc.step(tau_cmd, disturbance, DT_OUTER)

        elif mode in (Mode.FINE_POINTING, Mode.MOMENTUM_DUMP):
            if mekf_seeded and last_good_q is not None:
                qe = quat_error(sc.q, mekf.q)
                if qe[0] < 0: qe = -qe
                if np.degrees(2*np.linalg.norm(qe[1:])) > 25.0:
                    nadir_I = QUEST.nadir_inertial(chi_pos_km)
                    nadir_b = QUEST.nadir_body_from_earth_sensor(chi_pos_km, sc.q)
                    vb = [B_meas, sun_meas if not in_eclipse else nadir_b, nadir_b]
                    vi = [B_I, sun_I, nadir_I]
                    qf, _ = quest_alg.compute_multi(vb, vi, [0.70, 0.20, 0.10])
                    if qf[0] < 0: qf = -qf
                    mekf.q = qf.copy()
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
                    rw.h = np.clip(rw.h*0.9995, -rw.h_max, rw.h_max)
                    tau_rw = np.zeros(3)
                else:
                    tau_rw, _ = att_ctrl.compute(mekf.q, omega_est, q_ref)
                    rw.apply_torque(tau_rw, DT_INNER)
                    rw.h = np.clip(rw.h, -rw.h_max, rw.h_max)
                sc.step(np.zeros(3), disturbance, DT_INNER,
                        tau_rw=tau_rw, h_rw=rw.h.copy())
            if adcs_confirmed and mekf_seeded:
                qe = quat_error(sc.q, mekf.q)
                if qe[0] < 0: qe = -qe
                mekf_err_samples.append(np.degrees(2.*np.linalg.norm(qe[1:])))

        # 10. Deputy propagation — Phase 1 only (full-force, no guidance)
        if not phase2_active:
            dep_pos_eci, dep_vel_eci = propagate_ff(
                dep_pos_eci, dep_vel_eci,
                DT_OUTER, chief_orbit.t_elapsed - DT_OUTER,
                params['dep_cr'], params['dep_am'])
            # Refresh chief reference to match propagated epoch
            chi_pos_m  = chief_orbit.pos * 1e3
            chi_vel_ms = chief_orbit.vel * 1e3
            R_e2l = R_eci2lvlh(chi_pos_m, chi_vel_ms)
            cw.state = np.concatenate([
                R_e2l @ (dep_pos_eci - chi_pos_m),
                R_e2l @ (dep_vel_eci - chi_vel_ms)])

        # 11. Phase 2 activation — once ADCS confirmed + settle
        if adcs_confirmed and not phase2_active and t >= adcs_conf_t + FORM_HOLD_SETTLE_S:
            phase2_active = True
            # Apply nav handover jitter
            if not pos_jitter_done:
                R_l2e_j = R_eci2lvlh(chi_pos_m_prev, chi_vel_ms_prev).T
                dep_pos_eci += R_l2e_j @ params['pos_jitter_m']
                dep_vel_eci += R_l2e_j @ params['vel_jitter_ms']
                pos_jitter_done = True
            # Seed TH-EKF using prev-epoch chief (consistent with dep epoch)
            R_e2l_s  = R_eci2lvlh(chi_pos_m_prev, chi_vel_ms_prev)
            lvlh_pos = R_e2l_s @ (dep_pos_eci - chi_pos_m_prev)
            lvlh_vel = R_e2l_s @ (dep_vel_eci - chi_vel_ms_prev)
            ok = th_ekf.reinit_from_measurements(
                rng_sensor, lvlh_pos, n_avg=10, P_pos_m=2.0, P_vel_ms=0.05)
            if not ok:
                th_ekf.initialise(
                    x0=np.concatenate([lvlh_pos, lvlh_vel]), nu0=0.0)
            th_ekf.x[3:6] = lvlh_vel

        # 12. Phase 2 RPOD guidance
        if phase2_active:

            # Truth LVLH: dep at t, prev chief also at t (same epoch)
            R_e2l       = R_eci2lvlh(chi_pos_m_prev, chi_vel_ms_prev)
            true_cw_pos = R_e2l @ (dep_pos_eci - chi_pos_m_prev)
            true_cw_vel = R_e2l @ (dep_vel_eci - chi_vel_ms_prev)
            true_cw     = np.concatenate([true_cw_pos, true_cw_vel])
            cw.state    = true_cw

            ekf_lvlh = np.concatenate([th_ekf.position, th_ekf.velocity])

            # Trigger Lambert with RDV timing jitter
            rdv_trig = adcs_conf_t + 2*FORM_HOLD_SETTLE_S + params['rdv_jitter_s']
            if not rdv_started and t >= rdv_trig:
                truth_dir = true_cw_pos / max(np.linalg.norm(true_cw_pos), 1.0)
                z_seed, R_seed = rng_sensor.measure(true_cw_pos, truth_dir)
                if z_seed is not None:
                    pos_seed = rng_sensor.invert(z_seed)
                    th_ekf.initialise(
                        x0=np.concatenate([pos_seed, true_cw_vel]),
                        P0=np.diag([R_seed[0,0]]*3 + [0.05**2]*3),
                        nu0=th_ekf.nu)
                    ekf_lvlh = np.concatenate([th_ekf.position, th_ekf.velocity])
                else:
                    th_ekf.initialise(
                        x0=np.concatenate([true_cw_pos, true_cw_vel]),
                        P0=np.diag([4.0]*3 + [0.05**2]*3), nu0=th_ekf.nu)
                rdv_started = True
                rpod_ctrl.standoff = max(50.0, abs(true_cw_pos[1]))
                rpod_ctrl.start_rendezvous(t, truth_range=cw.range_m)

            # Guidance state: PROX_OPS / TERMINAL always use truth
            if rpod_ctrl.mode in (RPODMode.PROX_OPS, RPODMode.TERMINAL):
                guidance_state = true_cw
            else:
                guidance_state = ekf_lvlh

            accel_cmd, impulse_dv = rpod_ctrl.compute(
                ekf_lvlh    = guidance_state,
                chi_pos_eci = chi_pos_m_prev,
                chi_vel_eci = chi_vel_ms_prev,
                t=t, true_cw=true_cw)

            # Coast flag
            ekf_coast_active = (rpod_ctrl.mode == RPODMode.LAMBERT
                                 and rpod_ctrl._lam_active)

            # Apply impulsive burn
            if impulse_dv is not None and np.linalg.norm(impulse_dv) > 1e-9:
                burn_ranges.append(float(np.linalg.norm(true_cw_pos)))
                R_l2e = R_e2l.T
                # magnitude error + misalignment
                _dv = impulse_dv * params['thrust_mag_err']
                _sig = params['thrust_misalign_rad']
                if _sig > 1e-9 and np.linalg.norm(_dv) > 1e-9:
                    _ax = np.random.standard_normal(3); _ax /= np.linalg.norm(_ax)
                    _ang = np.random.normal(0.0, _sig)
                    _K = np.array([[0,-_ax[2],_ax[1]],[_ax[2],0,-_ax[0]],[-_ax[1],_ax[0],0]])
                    _dv = (np.eye(3) + np.sin(_ang)*_K + (1-np.cos(_ang))*(_K@_K)) @ _dv
                dep_vel_eci += R_l2e @ _dv
                cw.dv_total += np.abs(impulse_dv)
                # Recompute LVLH post-burn (pos unchanged, vel updated)
                true_cw_pos = R_e2l @ (dep_pos_eci - chi_pos_m_prev)
                true_cw_vel = R_e2l @ (dep_vel_eci - chi_vel_ms_prev)
                true_cw     = np.concatenate([true_cw_pos, true_cw_vel])
                cw.state    = true_cw
                ok = th_ekf.reinit_from_measurements(
                    rng_sensor, true_cw_pos, n_avg=10, P_pos_m=2.0, P_vel_ms=0.05)
                th_ekf.x[3:6] = true_cw_vel

            # Apply continuous acceleration
            if np.any(accel_cmd != 0):
                R_l2e = R_e2l.T
                _a = accel_cmd * params['thrust_mag_err']
                _sig = params['thrust_misalign_rad']
                if _sig > 1e-9 and np.linalg.norm(_a) > 1e-9:
                    _ax = np.random.standard_normal(3); _ax /= np.linalg.norm(_ax)
                    _ang = np.random.normal(0.0, _sig)
                    _K = np.array([[0,-_ax[2],_ax[1]],[_ax[2],0,-_ax[0]],[-_ax[1],_ax[0],0]])
                    _a = (np.eye(3) + np.sin(_ang)*_K + (1-np.cos(_ang))*(_K@_K)) @ _a
                dep_vel_eci += R_l2e @ _a * DT_OUTER
                cw.dv_total += np.abs(accel_cmd) * DT_OUTER

            # Propagate deputy
            dep_pos_eci, dep_vel_eci = propagate_ff(
                dep_pos_eci, dep_vel_eci,
                DT_OUTER, chief_orbit.t_elapsed - DT_OUTER,
                params['dep_cr'], params['dep_am'])

            # Recompute LVLH post-propagation — refresh chief to same epoch
            chi_pos_m  = chief_orbit.pos * 1e3
            chi_vel_ms = chief_orbit.vel * 1e3
            R_e2l       = R_eci2lvlh(chi_pos_m, chi_vel_ms)
            true_cw_pos = R_e2l @ (dep_pos_eci - chi_pos_m)
            true_cw_vel = R_e2l @ (dep_vel_eci - chi_vel_ms)
            true_cw     = np.concatenate([true_cw_pos, true_cw_vel])
            cw.state    = true_cw

            # TH-EKF update with nav delay + sensor dropout
            truth_rng = np.linalg.norm(true_cw_pos)
            boresight = (true_cw_pos / truth_rng if truth_rng > 1.0
                         else np.array([0., -1., 0.]))

            # nav delay: write current, read delayed
            nav_buf[_nav_step % _NAV_BUF_LEN] = (true_cw_pos.copy(), true_cw_vel.copy())
            _slot = (_nav_step - params['nav_delay_steps']) % _NAV_BUF_LEN
            _entry = nav_buf[_slot]
            _nav_pos, _nav_vel = _entry if _entry is not None else (true_cw_pos, true_cw_vel)
            _nav_step += 1

            # sensor dropout: Poisson arrivals, 1-5 s outages
            if not _dropout_active and params['dropout_rate_hz'] > 0:
                if np.random.random() < params['dropout_rate_hz'] * DT_OUTER:
                    _dropout_active = True
                    _dropout_end_t  = t + np.random.uniform(1.0, 5.0)
            if _dropout_active and t >= _dropout_end_t:
                _dropout_active = False

            if ekf_coast_active:
                if not _dropout_active:
                    z_c, _ = rng_sensor.measure(_nav_pos, boresight)
                    th_ekf.x[0:3] = rng_sensor.invert(z_c) if z_c is not None else _nav_pos
                else:
                    th_ekf.x[0:3] = _nav_pos   # dropout: no update
                th_ekf.x[3:6] = _nav_vel
                th_ekf.P[0:3, 0:3] = np.eye(3) * 4.0
                th_ekf.P[3:6, 3:6] = np.eye(3) * 0.0025
            elif rpod_ctrl.mode in (RPODMode.PROX_OPS, RPODMode.TERMINAL):
                th_ekf.x[0:3] = _nav_pos
                th_ekf.x[3:6] = _nav_vel
                th_ekf.P[0:3, 0:3] = np.eye(3) * 4.0
                th_ekf.P[3:6, 3:6] = np.eye(3) * 0.0025
            else:
                th_ekf.predict(accel_cmd)
                if not _dropout_active:
                    z_m, R_m = rng_sensor.measure(_nav_pos, boresight)
                    if z_m is not None:
                        th_ekf.update(z_m, R_m, gate_k=50.0)
                    else:
                        th_ekf.x[0:3] = _nav_pos; th_ekf.x[3:6] = _nav_vel
                else:
                    th_ekf.x[0:3] = _nav_pos; th_ekf.x[3:6] = _nav_vel

            # Docking check: must reach true dock port (offset from origin)
            _dock_err = true_cw_pos - params['dock_offset_m']
            if (np.linalg.norm(_dock_err) < DOCK_RANGE_M
                    and np.linalg.norm(true_cw_vel) < DOCK_VREL_MS):
                docked = True
                failure_mode = 'DOCKED'

        t += DT_OUTER

    # Failure mode classification
    if not docked:
        if not adcs_confirmed:
            failure_mode = 'ADCS_TIMEOUT'
        elif not rdv_started:
            failure_mode = 'NO_LAMBERT'
        elif phase2_active:
            final_r = float(np.linalg.norm(cw.state[:3]))
            failure_mode = 'PROX_STALL' if final_r < 50.0 else 'TIMEOUT'

    final_range = float(np.linalg.norm(cw.state[:3])) if phase2_active else float('nan')
    mekf_ss     = (float(np.mean(mekf_err_samples[-200:]))
                   if len(mekf_err_samples) >= 10 else float('nan'))

    return {
        'seed':           seed,
        'docked':         docked,
        'dock_time_hr':   t/3600.0 if docked else float('nan'),
        'total_dv_mms':   float(np.sum(cw.dv_total)) * 1e3,
        'adcs_time_s':    adcs_conf_t if adcs_confirmed else float('nan'),
        'mekf_ss_deg':    mekf_ss,
        'final_range_m':  final_range,
        'burn1_range_m':  burn_ranges[0] if len(burn_ranges) > 0 else float('nan'),
        'burn2_range_m':  burn_ranges[1] if len(burn_ranges) > 1 else float('nan'),
        'nb_burns':       len(burn_ranges),
        'failure_mode':        failure_mode,
        'thrust_misalign_deg': float(np.degrees(params['thrust_misalign_rad'])),
        'thrust_mag_err':      float(params['thrust_mag_err']),
        'nav_delay_steps':     int(params['nav_delay_steps']),
        'dock_offset_m':       float(np.linalg.norm(params['dock_offset_m'])),
        'dropout_rate_hz':     float(params['dropout_rate_hz']),
        'omega0_dps':     float(np.degrees(np.linalg.norm(params['omega0']))),
        'rdv_jitter_s':   float(params['rdv_jitter_s']),
        'pos_jitter_m':   float(np.linalg.norm(params['pos_jitter_m'])),
        'rng_noise_scale':float(params['rng_noise_scale']),
        'dep_cr':         float(params['dep_cr']),
        'dep_am':         float(params['dep_am']),
        'chi_am':         float(params['chi_am']),
        'gyro_bias_scale':float(params['gyro_bias_scale']),
        'st_noise_scale': float(params['st_noise_scale']),
        'M0_offset_deg':  float(params['M0_offset_deg']),
        'error':          '',
    }


# =====================================================================
#  RUNNER
# =====================================================================

def _nan(v):
    return isinstance(v, float) and (v != v)


def run_monte_carlo(n_trials=300, n_workers=8, master_seed=42):
    rng    = np.random.default_rng(master_seed)
    seeds  = [int(rng.integers(0, 2**31)) for _ in range(n_trials)]
    params = [sample_params(rng) for _ in range(n_trials)]
    jobs   = list(zip(seeds, params))

    print(f"\n{'='*62}")
    print(f"  GEO RPOD Monte Carlo — {n_trials} trials  ({n_workers} workers)")
    print(f"{'='*62}")
    print(f"  20 stress axes: tumble, sensors, SRP, orbit epoch,")
    print(f"  nav jitter (+/-20m/+/-10mm/s), RDV timing +/-120s, RW bias")
    print(f"  T_SIM_MAX={T_SIM_MAX:.0f}s  "
          f"dock: range<{DOCK_RANGE_M*100:.0f}cm AND v_rel<{DOCK_VREL_MS*1000:.0f}mm/s\n")

    t0 = time.time()

    if n_workers == 1:
        results = []
        for i, job in enumerate(jobs):
            r = _worker(job)
            if r['failure_mode'] == 'EXCEPTION':
                print(f"\n  !! EXCEPTION seed={r['seed']}:\n{r['error']}\n")
            results.append(r)
            _print_row(i+1, n_trials, r)
    else:
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=n_workers) as pool:
            results = []
            for i, r in enumerate(pool.imap_unordered(_worker, jobs)):
                results.append(r)
                _print_row(i+1, n_trials, r)

    elapsed = time.time() - t0
    print(f"\n  Completed {n_trials} trials in {elapsed:.1f}s "
          f"({elapsed/n_trials:.1f}s/trial)")
    return results


def _print_row(i, n, r):
    status = r['failure_mode']
    dv   = f"{r['total_dv_mms']:.0f}mm/s" if not _nan(r['total_dv_mms'])  else "—"
    dt   = f"{r['dock_time_hr']:.2f}hr"    if not _nan(r['dock_time_hr'])  else "—"
    rng  = f"{r['final_range_m']:.1f}m"    if not _nan(r['final_range_m']) else "—"
    adcs = f"{r['adcs_time_s']:.0f}s"      if not _nan(r['adcs_time_s'])   else "—"
    print(f"  [{i:3d}/{n}]  {status:<14}  "
          f"w={r['omega0_dps']:.1f}dps  adcs={adcs}  "
          f"jit=+/-{r['pos_jitter_m']:.0f}m  "
          f"rdv={r['rdv_jitter_s']:+.0f}s  "
          f"t={dt}  dv={dv}  rng={rng}",
          flush=True)


# =====================================================================
#  SUMMARY
# =====================================================================

def print_summary(results):
    docked = [r for r in results if r['docked']]
    failed = [r for r in results if not r['docked']]
    n      = len(results)

    def arr(key, subset=None):
        src = subset if subset is not None else results
        return np.array([r[key] for r in src
                         if r.get(key) is not None
                         and not _nan(r.get(key, float('nan')))], dtype=float)

    print(f"\n{'='*62}")
    print(f"  MONTE CARLO SUMMARY  ({n} trials)")
    print(f"{'='*62}")
    print(f"  Success rate : {len(docked)}/{n}  ({100*len(docked)/n:.1f}%)")

    fm = Counter(r['failure_mode'] for r in results)
    for k, v in sorted(fm.items(), key=lambda x: -x[1]):
        print(f"    {k:<18}: {v}")

    def stats(a, label, unit=""):
        if len(a) == 0: return
        print(f"  {label:<24}: mean={np.mean(a):.2f}  std={np.std(a):.2f}  "
              f"min={np.min(a):.2f}  max={np.max(a):.2f}  {unit}")

    if docked:
        print()
        stats(arr('dock_time_hr',  docked), "Dock time",     "hr")
        stats(arr('total_dv_mms',  docked), "Total dV",      "mm/s")
        stats(arr('adcs_time_s',   docked), "ADCS gate",     "s")
        stats(arr('mekf_ss_deg',   docked), "MEKF SS error", "deg")
        stats(arr('burn1_range_m', docked), "Burn-1 range",  "m")
        stats(arr('burn2_range_m', docked), "Burn-2 range",  "m")

    exceptions = [r for r in failed if r['failure_mode'] == 'EXCEPTION']
    if exceptions:
        print(f"\n  Exceptions ({len(exceptions)}):")
        for r in exceptions[:5]:
            last_line = [l for l in r['error'].splitlines() if l.strip()]
            print(f"    seed={r['seed']:10d}: {last_line[-1].strip() if last_line else '?'}")

    other = [r for r in failed if r['failure_mode'] != 'EXCEPTION']
    if other:
        print(f"\n  Other failures ({len(other)}):")
        for r in other[:8]:
            print(f"    seed={r['seed']:10d}  w={r['omega0_dps']:.1f}dps  "
                  f"jit={r['pos_jitter_m']:.0f}m  rdv={r['rdv_jitter_s']:+.0f}s  "
                  f"{r['failure_mode']}")


# =====================================================================
#  PLOTS
# =====================================================================

def plot_results(results, save_path='monte_carlo_geo.png'):
    docked = [r for r in results if r['docked']]
    failed = [r for r in results if not r['docked']]
    n      = len(results)

    C_OK   = '#27ae60'
    C_FAIL = '#e74c3c'
    C_ALL  = '#2980b9'

    def arr(key, subset=None):
        src = subset if subset is not None else results
        return np.array([r[key] for r in src
                         if r.get(key) is not None
                         and not _nan(r.get(key, float('nan')))], dtype=float)

    def paired(key_x, key_y, subset=None):
        """Return (xs, ys, colors) for rows where BOTH values are valid."""
        src  = subset if subset is not None else results
        cols = [C_OK if r['docked'] else C_FAIL for r in src]
        xs, ys, cs = [], [], []
        for r, c in zip(src, cols):
            vx = r.get(key_x); vy = r.get(key_y)
            if vx is None or vy is None: continue
            if _nan(vx) or _nan(vy): continue
            xs.append(float(vx)); ys.append(float(vy)); cs.append(c)
        return np.array(xs), np.array(ys), cs

    def hist(ax, data, color, xlabel, title, bins=15):
        d = np.array([x for x in data if not _nan(x)])
        if len(d) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=9)
        else:
            ax.hist(d, bins=bins, color=color, edgecolor='white', lw=0.5, alpha=0.88)
            ax.axvline(np.mean(d), color='black', ls='--', lw=1.5,
                       label=f"mu={np.mean(d):.2f}")
            ax.legend(fontsize=7)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        ax.set_title(title, fontsize=9, fontweight='bold')

    fig = plt.figure(figsize=(22, 15))
    fig.suptitle(
        f"GEO RPOD Monte Carlo — {n} trials  "
        f"(success {len(docked)}/{n} = {100*len(docked)/n:.1f}%)\n"
        f"20 stress axes  |  IS-1002 @ 342E  |  50 kg jetpack  |  1 N thruster",
        fontsize=13, fontweight='bold', y=0.99)
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.50, wspace=0.40)

    # Row 0 — mission outcomes

    # [0,0] Outcome breakdown
    ax = fig.add_subplot(gs[0, 0])
    modes  = ['DOCKED', 'ADCS_TIMEOUT', 'NO_LAMBERT', 'PROX_STALL', 'TIMEOUT', 'EXCEPTION']
    colors = [C_OK, '#e67e22', '#8e44ad', '#e74c3c', '#95a5a6', '#7f8c8d']
    fm     = Counter(r['failure_mode'] for r in results)
    counts = [fm.get(m, 0) for m in modes]
    bars   = ax.bar([m.replace('_', '\n') for m in modes], counts,
                    color=colors, edgecolor='white', lw=0.5, alpha=0.88)
    for bar, v in zip(bars, counts):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.2, str(v),
                    ha='center', fontsize=10, fontweight='bold')
    ax.set_ylabel("Count", fontsize=8)
    ax.set_title("Outcome Breakdown", fontsize=9, fontweight='bold')
    ax.tick_params(axis='x', labelsize=7)

    # [0,1] Docking time
    hist(fig.add_subplot(gs[0, 1]),
         arr('dock_time_hr', docked).tolist(),
         C_OK, 'Docking time [hr]', 'Docking Time (successes)')

    # [0,2] Total dV — docked vs failed
    ax = fig.add_subplot(gs[0, 2])
    dv_d = arr('total_dv_mms', docked)
    dv_f = arr('total_dv_mms', failed)
    if len(dv_d): ax.hist(dv_d, bins=15, color=C_OK,   edgecolor='white', lw=0.5, alpha=0.8, label='Docked')
    if len(dv_f): ax.hist(dv_f, bins=10, color=C_FAIL, edgecolor='white', lw=0.5, alpha=0.8, label='Failed')
    ax.legend(fontsize=7)
    ax.set_xlabel('Total dV [mm/s]', fontsize=8)
    ax.set_ylabel('Count', fontsize=8)
    ax.set_title('Total dV', fontsize=9, fontweight='bold')

    # [0,3] ADCS gate time
    hist(fig.add_subplot(gs[0, 3]),
         arr('adcs_time_s').tolist(),
         C_ALL, 'ADCS confirmation time [s]', 'ADCS Gate Time')

    # Row 1 — RPOD metrics

    # [1,0] MEKF SS pointing
    hist(fig.add_subplot(gs[1, 0]),
         arr('mekf_ss_deg', docked).tolist(),
         C_ALL, 'MEKF SS error [deg]', 'MEKF Pointing SS (docked)')

    # [1,1] Lambert burn ranges
    ax = fig.add_subplot(gs[1, 1])
    b1 = arr('burn1_range_m', docked)
    b2 = arr('burn2_range_m', docked)
    if len(b1): ax.hist(b1, bins=15, color='#3498db', alpha=0.75, edgecolor='white', lw=0.5, label='Burn-1')
    if len(b2): ax.hist(b2, bins=15, color='#e67e22', alpha=0.75, edgecolor='white', lw=0.5, label='Burn-2')
    ax.legend(fontsize=7)
    ax.set_xlabel('Truth range at burn [m]', fontsize=8)
    ax.set_ylabel('Count', fontsize=8)
    ax.set_title('Lambert Burn Ranges (docked)', fontsize=9, fontweight='bold')

    # [1,2] dV vs dock time
    ax = fig.add_subplot(gs[1, 2])
    xs, ys, _ = paired('dock_time_hr', 'total_dv_mms', docked)
    if len(xs):
        sc = ax.scatter(xs, ys, c=xs, cmap='viridis', s=35, alpha=0.85, zorder=3)
        plt.colorbar(sc, ax=ax, label='Dock time [hr]', pad=0.02)
    ax.set_xlabel('Docking time [hr]', fontsize=8)
    ax.set_ylabel('Total dV [mm/s]', fontsize=8)
    ax.set_title('dV vs Docking Time', fontsize=9, fontweight='bold')

    # [1,3] RDV jitter vs outcome
    ax = fig.add_subplot(gs[1, 3])
    jit_ok   = arr('rdv_jitter_s', docked)
    jit_fail = arr('rdv_jitter_s', failed)
    if len(jit_ok):   ax.hist(jit_ok,   bins=12, color=C_OK,   alpha=0.75, edgecolor='white', lw=0.5, label='Docked')
    if len(jit_fail): ax.hist(jit_fail, bins=12, color=C_FAIL, alpha=0.75, edgecolor='white', lw=0.5, label='Failed')
    ax.legend(fontsize=7)
    ax.set_xlabel('RDV trigger jitter [s]', fontsize=8)
    ax.set_ylabel('Count', fontsize=8)
    ax.set_title('RDV Timing Jitter vs Outcome', fontsize=9, fontweight='bold')

    # Row 2 — sensitivity scatter

    # [2,0] Tumble rate vs ADCS gate time
    ax = fig.add_subplot(gs[2, 0])
    xs, ys, cs = paired('omega0_dps', 'adcs_time_s')
    if len(xs): ax.scatter(xs, ys, c=cs, s=18, alpha=0.7)
    ax.set_xlabel('Initial tumble [deg/s]', fontsize=8)
    ax.set_ylabel('ADCS gate time [s]', fontsize=8)
    ax.set_title('Tumble vs ADCS Gate\n(green=docked)', fontsize=9, fontweight='bold')

    # [2,1] Position jitter vs final range
    ax = fig.add_subplot(gs[2, 1])
    xs, ys, cs = paired('pos_jitter_m', 'final_range_m')
    if len(xs): ax.scatter(xs, ys, c=cs, s=18, alpha=0.7)
    ax.axhline(DOCK_RANGE_M, color='gray', ls=':', lw=1, label=f'{DOCK_RANGE_M}m dock')
    ax.legend(fontsize=7)
    ax.set_xlabel('Position jitter |dr| [m]', fontsize=8)
    ax.set_ylabel('Final range [m]', fontsize=8)
    ax.set_title('Nav Jitter vs Final Range', fontsize=9, fontweight='bold')

    # [2,2] Ranging noise scale vs dock time (docked only)
    ax = fig.add_subplot(gs[2, 2])
    xs, ys, _ = paired('rng_noise_scale', 'dock_time_hr', docked)
    if len(xs): ax.scatter(xs, ys, c=C_OK, s=18, alpha=0.7)
    ax.set_xlabel('Ranging noise scale', fontsize=8)
    ax.set_ylabel('Dock time [hr]', fontsize=8)
    ax.set_title('Sensor Noise vs Dock Time (docked)', fontsize=9, fontweight='bold')

    # [2,3] Parameter tornado — mean docked vs failed
    ax = fig.add_subplot(gs[2, 3])
    p_keys   = ['dep_cr', 'dep_am', 'chi_am',
                'gyro_bias_scale', 'rng_noise_scale', 'st_noise_scale', 'M0_offset_deg',
                'thrust_misalign_deg', 'thrust_mag_err', 'nav_delay_steps',
                'dock_offset_m', 'dropout_rate_hz']
    p_labels = ['Deputy Cr', 'Deputy A/m', 'Chief A/m',
                'Gyro bias x', 'Ranging noise x', 'ST noise x', 'M0 offset [deg]',
                'Thrust misalign [deg]', 'Thrust mag err', 'Nav delay [steps]',
                'Dock offset [m]', 'Dropout rate [Hz]']
    if docked and failed:
        d_means = [np.mean([r[k] for r in docked if not _nan(r.get(k, float('nan')))]) for k in p_keys]
        f_means = [np.mean([r[k] for r in failed  if not _nan(r.get(k, float('nan')))]) for k in p_keys]
        y = np.arange(len(p_keys))
        ax.barh(y - 0.2, d_means, 0.35, color=C_OK,   alpha=0.85, edgecolor='white', label='Docked')
        ax.barh(y + 0.2, f_means, 0.35, color=C_FAIL, alpha=0.85, edgecolor='white', label='Failed')
        ax.set_yticks(y); ax.set_yticklabels(p_labels, fontsize=7)
        ax.legend(fontsize=7)
        ax.set_xlabel('Mean parameter value', fontsize=8)
    else:
        msg = 'All docked — no failures to compare' if not failed else 'No successes'
        ax.text(0.5, 0.5, msg, ha='center', va='center',
                transform=ax.transAxes, fontsize=9)
    ax.set_title('Parameter Means: Docked vs Failed', fontsize=9, fontweight='bold')

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n  Plot saved -> {save_path}")
    plt.show()


# =====================================================================
#  ENTRY POINT
# =====================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GEO RPOD Monte Carlo')
    parser.add_argument('--n',       type=int, default=300,
                        help='Number of trials (default 300)')
    parser.add_argument('--workers', type=int,
                        default=max(1, os.cpu_count() - 1),
                        help='Parallel workers (default: all cores - 1)')
    parser.add_argument('--seed',    type=int, default=42,
                        help='Master RNG seed (default 42)')
    parser.add_argument('--save',    type=str, default='monte_carlo_geo.png',
                        help='Plot output path')
    args = parser.parse_args()

    results = run_monte_carlo(
        n_trials    = args.n,
        n_workers   = args.workers,
        master_seed = args.seed)

    print_summary(results)
    plot_results(results, save_path=args.save)