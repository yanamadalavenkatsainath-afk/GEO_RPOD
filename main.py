import numpy as np
import matplotlib.pyplot as plt
import sys

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
from telemetry.logger import Logger
from utils.quaternion import quat_error
from environment.gravity_gradient import GravityGradient
from environment.solar_radiation_pressure import SolarRadiationPressure
from environment.aerodynamic_drag import AerodynamicDrag
from fsw.mode_manager import ModeManager, Mode

modules_to_clear = [key for key in sys.modules.keys()
                    if 'estimation' in key or 'sensors' in key
                    or 'actuators' in key or 'plant' in key]
for mod in modules_to_clear:
    sys.modules.pop(mod, None)

# =====================================================
# Hardware Parameters — 3U CubeSat
# =====================================================
I = np.diag([0.030, 0.025, 0.010])   # kg·m²
dt_detumble  = 0.1    # 10 Hz — detumble and sun acquisition
dt_control   = 0.01   # 100 Hz — MEKF + PD fine pointing

TLE_LINE1 = "1 25544U 98067A   25001.50000000  .00006789  00000-0  12345-3 0  9999"
TLE_LINE2 = "2 25544  51.6400 208.9163 0001147  83.8771  11.2433 15.49815689432399"

# =====================================================
# Disturbance environment
# =====================================================
gg   = GravityGradient(I)
srp  = SolarRadiationPressure()
drag = AerodynamicDrag(Cd=2.2, f107=150.0, f107a=150.0, ap=4.0)

# =====================================================
# Spacecraft & Sensors — single set, whole mission
# =====================================================
sc         = Spacecraft(I)
sc.omega   = np.array([0.18, -0.14, 0.22])   # ~18 deg/s post-deployment

orbit      = OrbitPropagator(tle_line1=TLE_LINE1, tle_line2=TLE_LINE2)
mag_field  = MagneticField(epoch_year=2025.0)
sun_model  = SunModel(epoch_year=2025.0)
mag_sens   = Magnetometer()
sun_sens   = SunSensor()
gyro       = Gyro(dt=dt_control, bias_init_max_deg_s=0.05)  # 100 Hz — 0.05 deg/s bias matches old model

rw         = ReactionWheel(h_max=0.004)
mtq        = Magnetorquer(m_max=0.2)
bdot       = BDotController(k_bdot=2e5, m_max=0.2)
controller = AttitudeController(Kp=0.0005, Kd=0.008)
quest_alg  = QUEST()
in_eclipse = False
ekf        = MEKF(dt_control)
ekf.P[0:3, 0:3] = np.eye(3) * np.radians(5.0) ** 2

fsw        = ModeManager()
logger     = Logger()

q_ref      = np.array([1., 0., 0., 0.])
t_sim_max  = 3600.0   # 60 min hard cap

# =====================================================
# Telemetry storage
# =====================================================
tel_t, tel_mode                     = [], []
tel_wx, tel_wy, tel_wz, tel_rate    = [], [], [], []
tel_B, tel_rho                      = [], []
tel_T_aero, tel_T_gg, tel_T_srp    = [], [], []
tel_hx, tel_hy, tel_hz             = [], [], []
tel_err_deg                         = []

print("=" * 60)
print("  3U CubeSat ADCS — State Machine Driven Simulation")
print("=" * 60)
print(f"  Initial tumble: {np.degrees(np.linalg.norm(sc.omega)):.1f} deg/s")
print()

# =====================================================
# Main simulation loop — state machine driven
# =====================================================
t = 0.0
triad_err_deg = None   # updated each cycle during SUN_ACQUISITION
mekf_seeded   = False  # True after MEKF is seeded on FINE_POINTING entry
last_good_q      = None   # last TRIAD result with good geometry
last_good_t      = -999.0 # time of last good TRIAD
gyro_bridge      = False  # True when bridging with gyro propagation
mekf_seed_t = 0.0
while t < t_sim_max:

    # ── Sensors ──────────────────────────────────────────────────────
    pos, vel   = orbit.step(dt_detumble)
    B_I        = mag_field.get_field(pos)
    B_meas     = mag_sens.measure(sc.q, B_I)
    sun_I      = sun_model.get_sun_vector(t_seconds=t)
    sun_meas   = sun_sens.measure(sc.q, sun_I)
    omega_meas = gyro.measure(sc.omega)

    # ── Disturbances ─────────────────────────────────────────────────
    sun_pos_km  = sun_I * 1.496e8
    T_gg        = gg.compute(pos, sc.q)
    T_srp, nu   = srp.compute(sc.q, sun_I, pos_km=pos, sun_pos_km=sun_pos_km)
    T_aero, rho = drag.compute(sc.q, pos, vel, t_seconds=t)
    disturbance = T_gg + T_srp + T_aero

    # ── QUEST — only during SUN_ACQUISITION ──────────────────────────
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
            last_good_q = q_quest.copy()
            if last_good_q[0] < 0:
                last_good_q = -last_good_q
            last_good_t = t
            gyro_bridge = False
            triad_err_deg = 5.0
        elif last_good_q is not None and (t - last_good_t) < 120.0:
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
            gyro_bridge = True
        else:
            triad_err_deg = 180.0

    # ── FSW mode update ───────────────────────────────────────────────
    mode = fsw.update(t, sc.omega, rw.h, triad_err_deg=triad_err_deg)

    # ── Seed MEKF ONCE on transition into FINE_POINTING ──────────────
    if mode == Mode.FINE_POINTING and not mekf_seeded:
        if last_good_q is not None:
            seed_q = last_good_q.copy()
            if seed_q[0] < 0:
                seed_q = -seed_q
            ekf.q = seed_q
            ekf.P[0:3, 0:3] = np.eye(3) * (np.radians(5.0)) ** 2
            print(f"  MEKF seeded from QUEST (5° uncertainty)")
        else:
            ekf.q = sc.q.copy()
            ekf.P[0:3, 0:3] = np.eye(3) * (np.radians(5.0)) ** 2
            print(f"  MEKF seeded from sc.q")
        mekf_seeded = True
        mekf_seed_t = t
    # ── Actuators ─────────────────────────────────────────────────────
    if mode == Mode.SAFE_MODE:
        sc.step(np.zeros(3), disturbance, dt_detumble)

    elif mode in (Mode.DETUMBLE, Mode.SUN_ACQUISITION):
        m_cmd, _     = bdot.compute(B_meas, sc.omega, B_I, dt_detumble)
        total_torque = mtq.compute_torque(m_cmd, B_meas)
        sc.step(total_torque, disturbance, dt_detumble)

    elif mode in (Mode.FINE_POINTING, Mode.MOMENTUM_DUMP):
        # If MEKF attitude error still large, inject QUEST correction
        # to pull ekf.q into the linear regime before relying on EKF updates
        q_err_check = quat_error(sc.q, ekf.q)
        if q_err_check[0] < 0:
            q_err_check = -q_err_check
        err_check = np.degrees(2 * np.linalg.norm(q_err_check[1:]))

        if err_check > 25.0 and last_good_q is not None:
            # Re-run QUEST with current measurements to get fresh seed
            nadir_I  = QUEST.nadir_inertial(pos)
            nadir_b  = QUEST.nadir_body_from_earth_sensor(pos, sc.q)
            q_fresh, q_qual = quest_alg.compute_multi(
                vectors_body     = [B_meas,  sun_meas, nadir_b],
                vectors_inertial = [B_I,     sun_I,    nadir_I],
                weights          = [0.70,    0.20,     0.10],
            )
            if q_fresh[0] < 0:
                q_fresh = -q_fresh
            ekf.q = q_fresh.copy()

        n_inner = int(dt_detumble / dt_control)
        for _ in range(n_inner):
            omega_meas_inner = gyro.measure(sc.omega)
            ekf.predict(omega_meas_inner)
            ekf.update_vector(B_meas, B_I, ekf.R_mag)
            ekf.update_vector(sun_meas, sun_I, ekf.R_sun)

            omega_est     = sc.omega - ekf.bias
            torque_cmd, _ = controller.compute(ekf.q, omega_est, q_ref)
            torque_rw     = rw.apply_torque(torque_cmd, dt_control)

            m_cmd = mtq.compute_dipole(rw.h, B_meas)
            if mode == Mode.MOMENTUM_DUMP:
                m_cmd = np.clip(m_cmd * 5.0, -mtq.m_max, mtq.m_max)
            torque_mtq = mtq.compute_torque(m_cmd, B_meas)

            sc.step(torque_mtq + torque_rw, disturbance, dt_control,
                    tau_rw=np.zeros(3), h_rw=rw.h.copy())

        q_err_vec = quat_error(sc.q, ekf.q)
        if q_err_vec[0] < 0:
            q_err_vec = -q_err_vec
        err_deg = np.degrees(2 * np.linalg.norm(q_err_vec[1:]))
        tel_err_deg.append((t, err_deg))

        if int(t) % 200 == 0:
            bias_deg = np.degrees(np.linalg.norm(ekf.bias))
            P_att    = np.sqrt(np.diag(ekf.P[0:3, 0:3]))
            P_bias   = np.sqrt(np.diag(ekf.P[3:6, 3:6]))
            print(f"  t={t:.0f} err={err_deg:.2f}° "
                  f"bias={bias_deg:.4f}°/s "
                  f"P_att={np.degrees(P_att.mean()):.3f}° "
                  f"P_bias={np.degrees(P_bias.mean()):.4f}°/s")

    # ── Telemetry ─────────────────────────────────────────────────────
    tel_t.append(t)
    tel_mode.append(mode.value)
    tel_wx.append(np.degrees(sc.omega[0]))
    tel_wy.append(np.degrees(sc.omega[1]))
    tel_wz.append(np.degrees(sc.omega[2]))
    tel_rate.append(np.degrees(np.linalg.norm(sc.omega)))
    tel_B.append(np.linalg.norm(B_I) * 1e9)
    tel_rho.append(rho)
    tel_T_aero.append(np.linalg.norm(T_aero) * 1e9)
    tel_T_gg.append(np.linalg.norm(T_gg) * 1e9)
    tel_T_srp.append(np.linalg.norm(T_srp) * 1e9)
    tel_hx.append(rw.h[0] * 1000)
    tel_hy.append(rw.h[1] * 1000)
    tel_hz.append(rw.h[2] * 1000)

    t += dt_detumble

print(f"t={t:.0f} P_cross={ekf.P[0:3, 3:6]}")
# =====================================================
# Summary
# =====================================================
print()
print("=" * 60)
print("  Simulation complete")
print("=" * 60)
print(f"  Total sim time : {t:.1f}s ({t/60:.1f} min)")
print(f"  Mode history:")
for t_trans, m in fsw.mode_history:
    print(f"    t={t_trans:7.1f}s  →  {m.name}")
print()
print(f"  Final disturbance means:")
print(f"    |T_aero| : {np.mean(tel_T_aero):.3f} nN·m")
print(f"    |T_gg|   : {np.mean(tel_T_gg):.3f} nN·m")
print(f"    |T_srp|  : {np.mean(tel_T_srp):.3f} nN·m")
if tel_err_deg:
    errs = [e for _, e in tel_err_deg]
    print(f"  Estimation error (fine pointing): "
          f"mean={np.mean(errs):.3f}°, max={np.max(errs):.3f}°")

# =====================================================
# Plots
# =====================================================
plt.rcParams.update({"font.size": 11, "axes.grid": True,
                      "grid.alpha": 0.35, "lines.linewidth": 1.2})

# Mode colour bands for all plots
MODE_COLORS = {
    Mode.SAFE_MODE.value:       ('red',       'SAFE'),
    Mode.DETUMBLE.value:        ('royalblue', 'DETUMBLE'),
    Mode.SUN_ACQUISITION.value: ('orange',    'SUN ACQ'),
    Mode.FINE_POINTING.value:   ('green',     'FINE POINT'),
    Mode.MOMENTUM_DUMP.value:   ('purple',    'MTM DUMP'),
}

def add_mode_bands(ax, t_arr, mode_arr):
    """Shade background by FSW mode."""
    t_arr    = np.array(t_arr)
    mode_arr = np.array(mode_arr)
    changes  = np.where(np.diff(mode_arr))[0]
    segments = np.concatenate([[0], changes + 1, [len(mode_arr)]])
    for i in range(len(segments) - 1):
        s, e = segments[i], segments[i+1]
        m    = mode_arr[s]
        col, _ = MODE_COLORS.get(m, ('gray', '?'))
        ax.axvspan(t_arr[s], t_arr[e-1], alpha=0.08, color=col, linewidth=0)

# ── Figure 1: Full mission overview ──────────────────────────────────
fig1, axes1 = plt.subplots(2, 3, figsize=(18, 9))
fig1.suptitle("3U CubeSat ADCS — Full Mission (State Machine)",
              fontsize=13, fontweight='bold')

t_arr = np.array(tel_t)

ax = axes1[0, 0]
ax.plot(t_arr, tel_wx, label="ωx", color='royalblue', alpha=0.8)
ax.plot(t_arr, tel_wy, label="ωy", color='darkorange', alpha=0.8)
ax.plot(t_arr, tel_wz, label="ωz", color='green', alpha=0.8)
add_mode_bands(ax, tel_t, tel_mode)
ax.set_xlabel("Time [s]"); ax.set_ylabel("Angular Rate [deg/s]")
ax.set_title("Angular Rate Components"); ax.legend(fontsize=8)

ax = axes1[0, 1]
ax.plot(t_arr, tel_rate, color='purple')
add_mode_bands(ax, tel_t, tel_mode)
ax.set_xlabel("Time [s]"); ax.set_ylabel("Total Rate [deg/s]")
ax.set_title("Total Angular Rate")

ax = axes1[0, 2]
ax.semilogy(t_arr, tel_T_aero, label="Aero drag",    color='saddlebrown')
ax.semilogy(t_arr, tel_T_gg,   label="Gravity grad", color='steelblue')
ax.semilogy(t_arr, tel_T_srp,  label="SRP",          color='goldenrod')
add_mode_bands(ax, tel_t, tel_mode)
ax.set_xlabel("Time [s]"); ax.set_ylabel("|Torque| [nN·m]")
ax.set_title("Disturbance Torques"); ax.legend(fontsize=8)

ax = axes1[1, 0]
ax.plot(t_arr, tel_hx, label="h_x", color='royalblue')
ax.plot(t_arr, tel_hy, label="h_y", color='darkorange')
ax.plot(t_arr, tel_hz, label="h_z", color='green')
ax.axhline( 4.0, color='red', linestyle=':', linewidth=1, label="±h_max")
ax.axhline(-4.0, color='red', linestyle=':', linewidth=1)
add_mode_bands(ax, tel_t, tel_mode)
ax.set_xlabel("Time [s]"); ax.set_ylabel("Momentum [mN·m·s]")
ax.set_title("Reaction Wheel Momentum"); ax.legend(fontsize=8)

ax = axes1[1, 1]
ax.semilogy(t_arr, tel_rho, color='saddlebrown')
add_mode_bands(ax, tel_t, tel_mode)
ax.set_xlabel("Time [s]"); ax.set_ylabel("Density [kg/m³]")
ax.set_title("Atmospheric Density (NRLMSISE-00)")

ax = axes1[1, 2]
# Mode timeline
mode_arr = np.array(tel_mode)
ax.step(t_arr, mode_arr, color='black', linewidth=1.5)
for val, (col, label) in MODE_COLORS.items():
    mask = mode_arr == val
    if mask.any():
        ax.fill_between(t_arr, val-0.4, val+0.4,
                        where=mask, alpha=0.4, color=col, label=label)
ax.set_xlabel("Time [s]"); ax.set_ylabel("Mode")
ax.set_yticks([m.value for m in Mode])
ax.set_yticklabels([m.name for m in Mode], fontsize=7)
ax.set_title("FSW Mode Timeline"); ax.legend(fontsize=7, loc='right')

fig1.tight_layout()

# ── Figure 2: Estimation error (fine pointing only) ──────────────────
if tel_err_deg:
    t_err  = [x[0] for x in tel_err_deg]
    e_err  = [x[1] for x in tel_err_deg]
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(t_err, e_err, color='crimson', label="Attitude estimation error")
    ax2.axhline(0.5, color='gray', linestyle=':', label="0.5° ref")
    ax2.set_xlabel("Time [s]"); ax2.set_ylabel("Error [deg]")
    ax2.set_title("MEKF Attitude Estimation Error — Fine Pointing Phase")
    ax2.legend(fontsize=9)
    fig2.tight_layout()

plt.show()
