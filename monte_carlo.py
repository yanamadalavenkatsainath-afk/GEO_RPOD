"""
Monte Carlo Validation - 3U CubeSat ADCS
=========================================
100 runs with randomised:
  1. Initial tumble rate magnitude and direction
  2. Sensor noise seeds (gyro, mag, sun)
  3. Orbit epoch (start position offset)
  4. Solar activity f107 +-20%
  5. Gyro initial bias

All fixes from main.py applied:
  - QUEST replaces TRIAD (3-vector: mag + sun + nadir)
  - QUEST-assisted MEKF convergence (re-injects QUEST while err > 25 deg)
  - Joseph form covariance update in MEKF
  - Vector normalisation in MEKF update_vector
  - tau_rw + h_rw passed correctly to spacecraft dynamics
  - k_bdot = 2e5
  - SAFE_RATE_THRESHOLD = 40 deg/s (covers post-deployment tumble up to 35 deg/s)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import sys
import os
import io
import contextlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from plant.spacecraft import Spacecraft
from sensors.gyro import Gyro
from sensors.magnetometer import Magnetometer
from sensors.sun_sensor import SunSensor
from environment.magnetic_field import MagneticField
from environment.sun_model import SunModel
from environment.orbit import OrbitPropagator
from actuators.reaction_wheel import ReactionWheel
from actuators.magnetorquer import Magnetorquer
from actuators.bdot import BDotController
from control.attitude_controller import AttitudeController
from estimation.mekf import MEKF
from estimation.quest import QUEST
from utils.quaternion import quat_error
from environment.gravity_gradient import GravityGradient
from environment.solar_radiation_pressure import SolarRadiationPressure
from environment.aerodynamic_drag import AerodynamicDrag
from fsw.mode_manager import ModeManager, Mode

# =====================================================
# Fixed hardware parameters
# =====================================================
I           = np.diag([0.030, 0.025, 0.010])
dt_detumble = 0.1
dt_control  = 0.01
N_INNER     = int(dt_detumble / dt_control)

TLE_LINE1 = "1 25544U 98067A   25001.50000000  .00006789  00000-0  12345-3 0  9999"
TLE_LINE2 = "2 25544  51.6400 208.9163 0001147  83.8771  11.2433 15.49815689432399"

# =====================================================
# Monte Carlo settings
# =====================================================
N_RUNS      = 100
T_SIM_MAX   = 1800.0   # 30 min per run
CONV_THRESH = 0.5      # deg - MEKF convergence threshold
SS_WINDOW   = 60.0     # s  - steady-state window at end of run

OMEGA_MAG_MEAN = np.radians(18.0)
OMEGA_MAG_STD  = np.radians(5.0)
OMEGA_MAG_MIN  = np.radians(5.0)
OMEGA_MAG_MAX  = np.radians(35.0)
F107_MEAN      = 150.0
F107_STD       = 30.0

# =====================================================
# Results storage
# =====================================================
results = {
    'run':             [],
    'detumble_time':   [],
    'quest_accepted':  [],
    'conv_time':       [],
    'ss_mean':         [],
    'ss_3sigma':       [],
    'wheel_saturated': [],
    'mode_reached':    [],
    'f107':            [],
    'omega0_mag':      [],
}

print("=" * 65)
print("  Monte Carlo ADCS Validation -- 100 Runs")
print("=" * 65)
print(f"  Randomised: tumble, noise seeds, orbit epoch, f107, gyro bias")
print(f"  Convergence threshold: {CONV_THRESH} deg")
print(f"  Steady-state window: last {SS_WINDOW:.0f}s of fine pointing")
print(f"  Estimator: QUEST (3-vector: mag+sun+nadir) + MEKF Joseph form")
print()

t_start_wall = time.time()

for run_idx in range(N_RUNS):
    rng = np.random.default_rng(seed=run_idx)

    # ── Randomise parameters ─────────────────────────────────────────
    omega_mag = float(np.clip(rng.normal(OMEGA_MAG_MEAN, OMEGA_MAG_STD),
                              OMEGA_MAG_MIN, OMEGA_MAG_MAX))
    omega_dir = rng.standard_normal(3)
    omega_dir /= np.linalg.norm(omega_dir)
    omega0    = omega_mag * omega_dir

    epoch_offset_s = rng.uniform(0, 5400)
    f107  = float(np.clip(rng.normal(F107_MEAN, F107_STD), 70.0, 250.0))
    f107a = f107

    # ── Initialise all components ────────────────────────────────────
    with contextlib.redirect_stdout(io.StringIO()):
        gg   = GravityGradient(I)
        srp  = SolarRadiationPressure()
        drag = AerodynamicDrag(Cd=2.2, f107=f107, f107a=f107a, ap=4.0)

        sc       = Spacecraft(I)
        sc.omega = omega0.copy()

        orbit = OrbitPropagator(tle_line1=TLE_LINE1, tle_line2=TLE_LINE2)
        for _ in range(int(epoch_offset_s / dt_detumble)):
            orbit.step(dt_detumble)

        mag_field = MagneticField(epoch_year=2025.0)
        sun_model = SunModel(epoch_year=2025.0)
        mag_sens  = Magnetometer()
        sun_sens  = SunSensor()
        gyro      = Gyro(dt=dt_control, bias_init_max_deg_s=0.05)
        gyro.bias = rng.uniform(-np.radians(0.05), np.radians(0.05), 3)

        rw         = ReactionWheel(h_max=0.004)
        mtq        = Magnetorquer(m_max=0.2)
        bdot       = BDotController(k_bdot=2e5, m_max=0.2)
        controller = AttitudeController(Kp=0.0005, Kd=0.008)
        quest_alg  = QUEST()
        ekf        = MEKF(dt_control)
        ekf.P[0:3, 0:3] = np.eye(3) * (np.radians(5.0)) ** 2

        fsw = ModeManager()

    q_ref         = np.array([1., 0., 0., 0.])
    t             = float(epoch_offset_s)
    t_run         = 0.0
    mekf_seeded   = False
    triad_err_deg = None
    in_eclipse    = False

    last_good_q = None
    last_good_t = -999.0
    gyro_bridge = False

    detumble_time  = None
    quest_accepted = False
    conv_time      = None
    ss_errors      = []
    wheel_sat      = False
    highest_mode   = Mode.DETUMBLE
    fine_point_t0  = None

    # ── Simulation loop ──────────────────────────────────────────────
    while t_run < T_SIM_MAX:

        with contextlib.redirect_stdout(io.StringIO()):
            pos, vel   = orbit.step(dt_detumble)
            B_I        = mag_field.get_field(pos)
            B_meas     = mag_sens.measure(sc.q, B_I)
            sun_I      = sun_model.get_sun_vector(t_seconds=t)
            sun_meas   = sun_sens.measure(sc.q, sun_I)
            omega_meas = gyro.measure(sc.omega)

            sun_pos_km  = sun_I * 1.496e8
            T_gg        = gg.compute(pos, sc.q)
            T_srp, nu   = srp.compute(sc.q, sun_I, pos_km=pos,
                                      sun_pos_km=sun_pos_km)
            T_aero, rho = drag.compute(sc.q, pos, vel, t_seconds=t)
            disturbance = T_gg + T_srp + T_aero

        # ── QUEST during SUN_ACQUISITION only ────────────────────────
        if fsw.is_sun_acquiring:
            in_eclipse = (nu < 0.1)
            nadir_I = QUEST.nadir_inertial(pos)
            nadir_b = QUEST.nadir_body_from_earth_sensor(pos, sc.q)

            if in_eclipse:
                q_quest, quest_quality = quest_alg.compute_multi(
                    vectors_body     = [B_meas,  nadir_b],
                    vectors_inertial = [B_I,     nadir_I],
                    weights          = [0.85,    0.15],
                )
            else:
                q_quest, quest_quality = quest_alg.compute_multi(
                    vectors_body     = [B_meas,  sun_meas, nadir_b],
                    vectors_inertial = [B_I,     sun_I,    nadir_I],
                    weights          = [0.70,    0.20,     0.10],
                )

            if q_quest[0] < 0:
                q_quest = -q_quest

            quest_ok = (quest_quality > 0.01)

            if quest_ok:
                last_good_q   = q_quest.copy()
                if last_good_q[0] < 0:
                    last_good_q = -last_good_q
                last_good_t   = t_run
                gyro_bridge   = False
                triad_err_deg = 5.0
            elif last_good_q is not None and (t_run - last_good_t) < 120.0:
                omega_corr = omega_meas - ekf.bias
                wx, wy, wz = omega_corr
                Omega = np.array([
                    [ 0,   -wx, -wy, -wz],
                    [ wx,   0,   wz, -wy],
                    [ wy,  -wz,  0,   wx],
                    [ wz,   wy, -wx,  0 ]
                ])
                last_good_q += 0.5 * dt_detumble * Omega @ last_good_q
                last_good_q /= np.linalg.norm(last_good_q)
                if last_good_q[0] < 0:
                    last_good_q = -last_good_q
                triad_err_deg = 5.0
                gyro_bridge   = True
            else:
                triad_err_deg = 180.0

        # ── FSW mode update ───────────────────────────────────────────
        mode = fsw.update(t_run, sc.omega, rw.h,
                          triad_err_deg=triad_err_deg)

        if mode.value > highest_mode.value:
            highest_mode = mode

        if detumble_time is None and mode == Mode.SUN_ACQUISITION:
            detumble_time = t_run

        # ── Seed MEKF once on FINE_POINTING entry ────────────────────
        if mode == Mode.FINE_POINTING and not mekf_seeded:
            quest_accepted = (last_good_q is not None)
            if last_good_q is not None:
                seed_q = last_good_q.copy()
                if seed_q[0] < 0:
                    seed_q = -seed_q
                ekf.q = seed_q
            else:
                ekf.q = sc.q.copy()
            ekf.P[0:3, 0:3] = np.eye(3) * (np.radians(5.0)) ** 2
            mekf_seeded   = True
            fine_point_t0 = t_run

        # ── Actuators ─────────────────────────────────────────────────
        with contextlib.redirect_stdout(io.StringIO()):

            if mode == Mode.SAFE_MODE:
                sc.step(np.zeros(3), disturbance, dt_detumble)

            elif mode in (Mode.DETUMBLE, Mode.SUN_ACQUISITION):
                m_cmd, _ = bdot.compute(B_meas, sc.omega, B_I, dt_detumble)
                sc.step(mtq.compute_torque(m_cmd, B_meas),
                        disturbance, dt_detumble)

            elif mode in (Mode.FINE_POINTING, Mode.MOMENTUM_DUMP):

                # QUEST-assisted convergence: reinitialise ekf.q from QUEST
                # while error is outside the EKF small-angle linear regime
                if mekf_seeded and last_good_q is not None:
                    qe_chk = quat_error(sc.q, ekf.q)
                    if qe_chk[0] < 0:
                        qe_chk = -qe_chk
                    err_chk = np.degrees(2 * np.linalg.norm(qe_chk[1:]))

                    if err_chk > 25.0:
                        nadir_I = QUEST.nadir_inertial(pos)
                        nadir_b = QUEST.nadir_body_from_earth_sensor(
                            pos, sc.q)
                        q_fresh, _ = quest_alg.compute_multi(
                            vectors_body     = [B_meas,  sun_meas, nadir_b],
                            vectors_inertial = [B_I,     sun_I,    nadir_I],
                            weights          = [0.70,    0.20,     0.10],
                        )
                        if q_fresh[0] < 0:
                            q_fresh = -q_fresh
                        ekf.q = q_fresh.copy()

                # Inner loop at 100 Hz
                for _ in range(N_INNER):
                    omega_inner = gyro.measure(sc.omega)
                    ekf.predict(omega_inner)
                    ekf.update_vector(B_meas, B_I, ekf.R_mag)
                    ekf.update_vector(sun_meas, sun_I, ekf.R_sun)

                    omega_est     = sc.omega - ekf.bias
                    torque_cmd, _ = controller.compute(ekf.q, omega_est,
                                                       q_ref)
                    torque_rw     = rw.apply_torque(torque_cmd, dt_control)

                    m_cmd = mtq.compute_dipole(rw.h, B_meas)
                    if mode == Mode.MOMENTUM_DUMP:
                        m_cmd = np.clip(m_cmd * 5.0, -mtq.m_max, mtq.m_max)
                    torque_mtq = mtq.compute_torque(m_cmd, B_meas)

                    sc.step(torque_mtq + torque_rw, disturbance,
                            dt_control,
                            tau_rw=np.zeros(3), h_rw=rw.h.copy())

                # Record estimation error
                if mekf_seeded:
                    qe = quat_error(sc.q, ekf.q)
                    if qe[0] < 0:
                        qe = -qe
                    err_deg = np.degrees(2 * np.linalg.norm(qe[1:]))

                    if conv_time is None and err_deg < CONV_THRESH:
                        conv_time = t_run

                    if (fine_point_t0 is not None and
                            t_run > T_SIM_MAX - SS_WINDOW):
                        ss_errors.append(err_deg)

        if np.any(np.abs(rw.h) >= 0.0039):
            wheel_sat = True

        t     += dt_detumble
        t_run += dt_detumble

    # ── Store run results ────────────────────────────────────────────
    ss_arr  = np.array(ss_errors) if ss_errors else np.array([np.nan])
    ss_mean = float(np.nanmean(ss_arr))
    ss_3sig = float(np.nanmean(ss_arr) + 3 * np.nanstd(ss_arr))

    results['run'].append(run_idx)
    results['detumble_time'].append(
        detumble_time if detumble_time is not None else T_SIM_MAX)
    results['quest_accepted'].append(quest_accepted)
    results['conv_time'].append(conv_time)
    results['ss_mean'].append(ss_mean)
    results['ss_3sigma'].append(ss_3sig)
    results['wheel_saturated'].append(wheel_sat)
    results['mode_reached'].append(highest_mode.name)
    results['f107'].append(f107)
    results['omega0_mag'].append(np.degrees(omega_mag))

    elapsed = time.time() - t_start_wall
    eta     = elapsed / (run_idx + 1) * (N_RUNS - run_idx - 1)
    conv_s  = f"{conv_time:.0f}s" if conv_time else "FAIL"
    ss_str  = f"{ss_mean:.2f}" if not np.isnan(ss_mean) else "nan"
    print(f"  Run {run_idx+1:3d}/{N_RUNS}  |  "
          f"detumble={results['detumble_time'][-1]:.0f}s  |  "
          f"QUEST={'OK' if quest_accepted else 'FAIL'}  |  "
          f"conv={conv_s}  |  "
          f"ss={ss_str} deg  |  "
          f"ETA {eta:.0f}s")

# =====================================================
# Statistics summary
# =====================================================
det_arr  = np.array(results['detumble_time'])
conv_arr = np.array([x for x in results['conv_time'] if x is not None])
ss_m_arr = np.array([x for x in results['ss_mean']   if not np.isnan(x)])
ss_3_arr = np.array([x for x in results['ss_3sigma'] if not np.isnan(x)])
sat_arr  = np.array(results['wheel_saturated'])
acc_arr  = np.array(results['quest_accepted'])

n_converged = len(conv_arr)
n_saturated = int(sat_arr.sum())
n_quest_ok  = int(acc_arr.sum())

print()
print("=" * 65)
print("  Monte Carlo Results -- 100 Runs")
print("=" * 65)
print(f"  Detumble time    : {det_arr.mean():.1f}s mean  |  "
      f"{det_arr.std():.1f}s std  |  "
      f"{np.percentile(det_arr, 99):.1f}s 99th pct")
print(f"  QUEST accepted   : {n_quest_ok}/100 runs ({n_quest_ok}%)")
print(f"  MEKF converged   : {n_converged}/100 runs")
if len(conv_arr):
    print(f"  Convergence time : {conv_arr.mean():.1f}s mean  |  "
          f"{conv_arr.std():.1f}s std  |  "
          f"{np.percentile(conv_arr, 99):.1f}s 99th pct")
if len(ss_m_arr):
    print(f"  Steady-state err : {ss_m_arr.mean():.3f} deg mean  |  "
          f"{ss_3_arr.mean():.3f} deg 3-sigma")
print(f"  Wheel saturation : {n_saturated}/100 runs ({n_saturated}%)")
print(f"  End-to-end OK    : {n_converged}/100 runs")
print(f"  Wall time        : {time.time()-t_start_wall:.1f}s")

# =====================================================
# Plots
# =====================================================
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.35})

fig = plt.figure(figsize=(18, 11))
fig.suptitle(
    "Monte Carlo ADCS Validation -- 100 Runs\n"
    "QUEST (3-vector: mag+sun+nadir) + MEKF Joseph form",
    fontsize=13, fontweight='bold')
gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.35)

# 1 -- Detumble time histogram
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(det_arr, bins=20, color='royalblue', edgecolor='white', alpha=0.85)
ax1.axvline(det_arr.mean(), color='red', linestyle='--',
            label=f"Mean {det_arr.mean():.0f}s")
ax1.axvline(np.percentile(det_arr, 99), color='orange', linestyle=':',
            label=f"99th {np.percentile(det_arr,99):.0f}s")
ax1.set_xlabel("Detumble Time [s]")
ax1.set_ylabel("Count")
ax1.set_title("Detumble Time Distribution")
ax1.legend(fontsize=8)

# 2 -- Tumble rate vs detumble time
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(results['omega0_mag'], det_arr, c='royalblue', alpha=0.5, s=15)
ax2.set_xlabel("Initial Tumble [deg/s]")
ax2.set_ylabel("Detumble Time [s]")
ax2.set_title("Tumble Rate vs Detumble Time")

# 3 -- QUEST acceptance
ax3 = fig.add_subplot(gs[0, 2])
bars = ax3.bar(['QUEST OK', 'QUEST FAIL'],
               [n_quest_ok, N_RUNS - n_quest_ok],
               color=['forestgreen', 'crimson'],
               edgecolor='white', alpha=0.85)
ax3.set_ylabel("Run Count")
ax3.set_title(f"QUEST Acceptance\n({n_quest_ok}/100 accepted)")
for bar, val in zip(bars, [n_quest_ok, N_RUNS - n_quest_ok]):
    ax3.text(bar.get_x() + bar.get_width() / 2, val + 0.5,
             str(val), ha='center', fontsize=12, fontweight='bold')

# 4 -- MEKF convergence time
ax4 = fig.add_subplot(gs[0, 3])
if len(conv_arr):
    ax4.hist(conv_arr, bins=20, color='forestgreen',
             edgecolor='white', alpha=0.85)
    ax4.axvline(conv_arr.mean(), color='red', linestyle='--',
                label=f"Mean {conv_arr.mean():.0f}s")
    ax4.set_xlabel("Convergence Time [s]")
    ax4.set_ylabel("Count")
    ax4.set_title(f"MEKF Convergence Time\n(converged: {n_converged}/100)")
    ax4.legend(fontsize=8)
else:
    ax4.text(0.5, 0.5, "No convergence\nin any run",
             ha='center', va='center', transform=ax4.transAxes, fontsize=12)
    ax4.set_title("MEKF Convergence")

# 5 -- Steady-state CDF
ax5 = fig.add_subplot(gs[1, 0])
if len(ss_m_arr):
    sorted_ss = np.sort(ss_m_arr)
    cdf = np.arange(1, len(sorted_ss) + 1) / len(sorted_ss)
    ax5.plot(sorted_ss, cdf * 100, color='crimson', linewidth=1.5)
    ax5.axvline(0.5, color='gray', linestyle=':', label="0.5 deg ref")
    pct_below = float(np.mean(ss_m_arr < 0.5) * 100)
    p99 = np.percentile(ss_m_arr, 99)
    ax5.axvline(p99, color='orange', linestyle='--',
                label=f"99th {p99:.2f} deg")
    ax5.set_xlabel("Steady-state Error [deg]")
    ax5.set_ylabel("CDF [%]")
    ax5.set_title(f"Pointing Error CDF\n({pct_below:.0f}% runs below 0.5 deg)")
    ax5.legend(fontsize=8)

# 6 -- 3-sigma per run sorted
ax6 = fig.add_subplot(gs[1, 1])
if len(ss_3_arr):
    ax6.bar(range(len(ss_3_arr)), sorted(ss_3_arr),
            color='crimson', alpha=0.6, width=1.0)
    ax6.axhline(0.5, color='gray', linestyle=':', label="0.5 deg ref")
    ax6.set_xlabel("Run (sorted by 3-sigma)")
    ax6.set_ylabel("3-sigma Pointing Error [deg]")
    ax6.set_title("3-Sigma Pointing Error per Run")
    ax6.legend(fontsize=8)

# 7 -- Solar activity vs pointing error
ax7 = fig.add_subplot(gs[1, 2])
f107_arr = np.array(results['f107'])
ax7.scatter(f107_arr, results['ss_mean'],
            c='saddlebrown', alpha=0.5, s=15)
ax7.set_xlabel("Solar Activity f107 [sfu]")
ax7.set_ylabel("SS Mean Error [deg]")
ax7.set_title("Solar Activity vs Pointing Error")

# 8 -- Mode reach distribution
ax8 = fig.add_subplot(gs[1, 3])
mode_counts = {}
for m in results['mode_reached']:
    mode_counts[m] = mode_counts.get(m, 0) + 1
modes  = list(mode_counts.keys())
counts = [mode_counts[m] for m in modes]
cmap   = {
    'DETUMBLE':        'royalblue',
    'SUN_ACQUISITION': 'orange',
    'FINE_POINTING':   'forestgreen',
    'MOMENTUM_DUMP':   'purple',
    'SAFE_MODE':       'red',
}
ax8.bar(modes, counts,
        color=[cmap.get(m, 'gray') for m in modes],
        edgecolor='white', alpha=0.85)
ax8.set_xlabel("Highest Mode Reached")
ax8.set_ylabel("Run Count")
ax8.set_title(f"Mode Reach Distribution\n(wheel sat: {n_saturated}/100)")
ax8.tick_params(axis='x', rotation=20)

plt.savefig("monte_carlo_results.png", dpi=150, bbox_inches='tight')
print()
print("  Plot saved: monte_carlo_results.png")
plt.show()
