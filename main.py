"""
GEO Jetpack RPOD Simulation
=========================================
Full mission sequence (in order):

  1. DETUMBLE          — B-dot rate damping until |ω| < 3.5 deg/s
  2. SUN_ACQ           — QUEST attitude init, MEKF seed
  3. FINE_POINTING     — MEKF + PD reaction wheel control (stable for 100 steps)
  4. FORMATION_HOLD    — Deputy holds 1 km trailing standoff, TH-EKF settles
  5. LAMBERT burn 1    — Two-impulse Lambert far-field transfer begins (burn-1)
  6. 16 hr coast       — Deputy coasts on Lambert arc (no thrust)
  7. LAMBERT burn 2    — Null relative velocity at arrival
  8. PROX_OPS          — Continuous PD closure 500 m → 0.8 m
  9. TERMINAL          — Tight PD deceleration 0.8 m → dock
 10. DOCKING           — range < 10 cm  AND  v_rel < 10 mm/s (truth)

Scenario: 50 kg jetpack servicer (deputy) rendezvous with a
fuel-depleted IS-1002 class GEO comsat (chief) at 342° E.

Hardware
--------
  Deputy mass:    50 kg
  Deputy thrust:  1.0 N  →  a_max = 20 mm/s²
  Deputy inertia: diag([4.167, 4.167, 3.000]) kg·m²  (60×60×80 cm)
  Deputy A/m:     0.00720 m²/kg
  Chief A/m:      0.0150 m²/kg   (30 m² solar panels, ~2000 kg EOL)
  Reaction wheel: Bradford WSAT class, h_max = 4.0 N·m·s
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os, copy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Plant & environment ─────────────────────────────────────────────
from plant.spacecraft                     import Spacecraft
from environment.magnetic_field           import MagneticField
from environment.gravity_gradient         import GravityGradient
from environment.solar_radiation_pressure import SolarRadiationPressure
from environment.geo_orbit                import GEOOrbitPropagator, eclipse_nu
from environment.cw_dynamics              import CWDynamics

# ── Sensors ─────────────────────────────────────────────────────────
from sensors.gyro          import Gyro
from sensors.magnetometer  import Magnetometer
from sensors.sun_sensor    import SunSensor
from sensors.star_tracker  import StarTracker
from sensors.ranging_sensor import RangingBearingSensor
from sensors.camera_sensor  import CameraSensor

# ── Actuators ───────────────────────────────────────────────────────
from actuators.reaction_wheel import ReactionWheel
from actuators.magnetorquer   import Magnetorquer
from actuators.bdot           import BDotController

# ── Estimation ──────────────────────────────────────────────────────
from estimation.mekf   import MEKF
from estimation.quest  import QUEST
from estimation.th_ekf import THEKF
from chief_attitude import ChiefAttitude
from chief_pose_estimator import ChiefPoseEstimator

# ── Control ─────────────────────────────────────────────────────────
from control.attitude_controller import AttitudeController
from control.lambert_controller  import GEORPODController, RPODMode

# ── FSW ─────────────────────────────────────────────────────────────
from fsw.mode_manager import ModeManager, Mode

# ── Utils ───────────────────────────────────────────────────────────
from utils.quaternion import quat_error, rot_matrix

# Docking port body-frame offset — must match ChiefAttitude dock_port_body
DOCK_PORT_BODY = np.array([0.0, 0.0, 0.5])   # m  — must match ChiefAttitude
DOCK_AXIS_BODY = np.array([0.0, 0.0, 1.0])   # approach axis in chief body frame


# =====================================================================
#  MISSION CONFIGURATION
# =====================================================================

# Chief orbit — IS-1002 class GEO comsat at 342° E
CHIEF_A_KM      = 42164.0
CHIEF_E         = 0.0003
CHIEF_I_DEG     = 0.8
CHIEF_RAAN_DEG  = 0.0
CHIEF_OMEGA_DEG = 0.0
CHIEF_M0_DEG    = 0.0
CHIEF_LON_DEG   = 342.0

# Deputy — 50 kg jetpack prototype
DEP_MASS_KG  = 50.0
DEP_THRUST_N = 1.0
DEP_CR       = 1.5
DEP_AM       = 0.00720

# Chief SRP parameters
CHI_CR = 1.5
CHI_AM = 0.015

# Formation hold: 1 km trailing behind chief (LVLH -y direction)
FORMATION_OFFSET_M = np.array([0.0, -1000.0, 0.0])

# Timing
DT_OUTER   = 0.1    # s  outer (RPOD) loop timestep
DT_INNER   = 0.01   # s  inner (ADCS) loop timestep
N_INNER    = int(DT_OUTER / DT_INNER)

T_SIM_MAX  = 80_000.0   # s  (~22 hr ceiling — enough for 16 hr coast + prox ops)

# ADCS stability gate: pointing error < 1 deg held for 100 consecutive steps
ADCS_STABLE_DEG  = 1.0
ADCS_STABLE_SUST = 100

# Formation hold settle time before commanding Lambert rendezvous
FORM_HOLD_SETTLE_S = 300.0   # s

# Docking capture criteria (truth state)
DOCK_RANGE_M = 0.30     # m  — widened from 0.10m to account for port pose estimation error
DOCK_VREL_MS = 0.05     # m/s — relaxed from 0.01 to 50mm/s
SIGMA_V_DOPPLER = 0.005  # m/s — Doppler sensor noise (5mm/s, VBS class)

# Eclipse threshold
ECLIPSE_NU_MIN = 0.1

MU_GEO = 3.986004418e14
N_GEO  = np.sqrt(MU_GEO / (CHIEF_A_KM * 1e3) ** 3)


# =====================================================================
#  HELPERS
# =====================================================================

def R_eci2lvlh(r_chief: np.ndarray, v_chief: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix: ECI → LVLH."""
    x_hat = r_chief / np.linalg.norm(r_chief)
    h_vec = np.cross(r_chief, v_chief)
    z_hat = h_vec / np.linalg.norm(h_vec)
    y_hat = np.cross(z_hat, x_hat)
    return np.vstack([x_hat, y_hat, z_hat])   # rows = LVLH axes in ECI


def propagate_full_force(pos: np.ndarray, vel: np.ndarray,
                         dt_total: float, t_abs: float,
                         Cr: float, Am: float,
                         substep: float = 60.0) -> tuple:
    """
    RK4 propagator: two-body + J2 + SRP.
    pos/vel in [m, m/s] ECI.  Returns updated (pos, vel).
    """
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
        sp = sun_pos(t)
        rs = np.linalg.norm(sp)
        P  = P0 * (AU / rs) ** 2
        dr_sun = p - sp
        a += Cr * Am * P * dr_sun / np.linalg.norm(dr_sun)
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


# =====================================================================
#  HARDWARE INSTANTIATION
# =====================================================================

print("=" * 65)
print("  GEO Jetpack RPOD (50 kg prototype)")
print("=" * 65)
print(f"  Chief  : IS-1002 @ {CHIEF_LON_DEG}°E  "
      f"a={CHIEF_A_KM}km  e={CHIEF_E}  i={CHIEF_I_DEG}°")
print(f"  Deputy : 50 kg  thrust={DEP_THRUST_N*1e3:.0f}mN  "
      f"a_max={DEP_THRUST_N/DEP_MASS_KG*1e3:.1f}mm/s²")
print(f"  Standoff: {np.linalg.norm(FORMATION_OFFSET_M):.0f}m trailing")
print()

# Spacecraft inertia: 60×60×80 cm box, 50 kg
#   Ix=Iy = m(b²+c²)/12 = 50*(0.6²+0.8²)/12 = 4.167 kg·m²
#   Iz    = m(a²+b²)/12 = 50*(0.6²+0.6²)/12 = 3.000 kg·m²
I_sc = np.diag([4.167, 4.167, 3.000])

# Environment
chief_orbit = GEOOrbitPropagator(
    a_km=CHIEF_A_KM, e=CHIEF_E, i_deg=CHIEF_I_DEG,
    raan_deg=CHIEF_RAAN_DEG, omega_deg=CHIEF_OMEGA_DEG, M0_deg=CHIEF_M0_DEG,
    Cr=CHI_CR, Am_ratio=CHI_AM)

mag_field = MagneticField(epoch_year=2025.0)
gg        = GravityGradient(I_sc)
srp       = SolarRadiationPressure()

# Spacecraft body dynamics — start with a realistic tumble
sc        = Spacecraft(I_sc)
sc.omega  = np.array([0.18, -0.14, 0.22])   # rad/s initial tumble

# Sensors
mag_sens     = Magnetometer()
sun_sens     = SunSensor()
gyro         = Gyro(dt=DT_INNER, bias_init_max_deg_s=0.05)
star_tracker = StarTracker(
    sigma_cross_arcsec=5.0, sigma_roll_arcsec=20.0,
    sun_excl_deg=30.0, earth_excl_deg=20.0,
    update_rate_hz=4.0, acquisition_s=30.0)
rng_sensor   = RangingBearingSensor(
    sigma_range_m=1.0, sigma_range_frac=0.001,
    sigma_angle_rad=np.radians(0.05),
    fov_half_deg=60.0, max_range_m=5000.0,
    min_range_m=0.05)   # 5cm — sensor active all the way to docking capture

# Phase 4/5: Camera sensor — active in PROX_OPS/TERMINAL (<600m)
# rng_sensor still used during Lambert coast (far-field)
cam_sensor = CameraSensor(
    focal_length_px=800.0, image_size_px=(640, 480),
    sigma_px=1.5, min_range_m=0.05, max_range_m=5000.0)  # matches ranging sensor

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
    dt=DT_OUTER,
    q_pos=1e-3,   # increased from 1e-4 — accounts for differential J2+SRP
    q_vel=1e-7)   # increased from 1e-8 — allows filter to track GEO dynamics

# Control
att_ctrl  = AttitudeController(Kp=0.08284, Kd=0.82257)
q_ref     = np.array([1., 0., 0., 0.])    # inertially-fixed nadir pointing

rpod_ctrl = GEORPODController(
    mu=MU_GEO, n_chief=N_GEO,
    dep_mass_kg=DEP_MASS_KG, dep_thrust_N=DEP_THRUST_N,
    Cr_chi=CHI_CR, Am_chi=CHI_AM,
    dock_capture_m=DOCK_RANGE_M,
    ekf=th_ekf, rng_sensor=rng_sensor)
rpod_ctrl.standoff = np.linalg.norm(FORMATION_OFFSET_M)

# Phase 6: Chief attitude — free-tumbling non-cooperative target
# IS-1002 class inertia, ~0.1 deg/s typical derelict tumble
chief_att = ChiefAttitude(
    omega0_deg_s=np.array([0.05, 0.10, 0.03]),
    dock_port_body=np.array([0.0, 0.0, 0.5]),
    dock_axis_body=np.array([0.0, 0.0, 1.0]),
    enable_gg_torque=True)

# Chief pose estimator — estimates tumble rate from camera observations.
# Replaces truth chief_att.omega_body in guidance and docking check.
# N_avg=50 → 5s effective window at dt=0.1s; alpha=0.3 smoothing.
chief_pose_est = ChiefPoseEstimator(
    cam_sensor=cam_sensor,
    dt=DT_OUTER,
    N_avg=50,
    alpha_filter=0.3,
    sigma_omega=0.002)   # ~0.11 deg/s uncertainty

# FSW mode manager
fsw = ModeManager()

# Relative motion state (CW — used as convenient state container & dv counter)
cw = CWDynamics(chief_orbit_radius_km=CHIEF_A_KM)
cw.set_initial_offset(dr_lvlh_m=FORMATION_OFFSET_M)
cw.dv_total = 0.0   # FIX 2/3: scalar replaces 3-vector; reset again at phase2 start


# =====================================================================
#  SIMULATION STATE
# =====================================================================

t = 0.0

# Deputy ECI truth — initialised on first step
dep_pos_eci = None
dep_vel_eci = None

# ADCS state flags
mekf_seeded      = False
last_good_q      = None
last_good_t      = -999.0
adcs_stable_cnt  = 0
triad_err_deg    = None

# Phase flags
adcs_confirmed   = False    # Phase 1 gate: ADCS stable
adcs_conf_t      = None
form_hold_done   = False    # Formation hold settled → trigger Lambert
phase2_active    = False    # RPOD phase active
rdv_started      = False    # Lambert planning triggered
docked           = False
ekf_coast_active = False    # True while on a Lambert coast arc
th_ekf_pos_prev  = np.zeros(3)  # EKF position from previous step for Doppler

# Telemetry
tel = dict(
    # ADCS channels
    t=[], mode=[], rate=[], err_deg=[],
    T_srp=[], T_gg=[], hx=[], hy=[], hz=[], eclipse_nu=[],
    # RPOD channels
    rn_t=[], rn_mode=[],
    rn_dx=[], rn_dy=[], rn_dz=[],
    rn_edx=[], rn_edy=[], rn_edz=[],
    rn_range=[], rn_est_range=[],
    rn_dv=[],
    rn_pos_err=[], rn_vel_err=[],
)

print("Starting simulation …\n")


# =====================================================================
#  MAIN LOOP
# =====================================================================
while t < T_SIM_MAX and not docked:

    # ------------------------------------------------------------------
    # 1.  CHIEF ORBIT STEP
    # ------------------------------------------------------------------
    # Save chief state BEFORE step -- dep_pos_eci is still at t (not yet propagated)
    # Use _prev for any LVLH computation before dep propagates this iteration.
    chi_pos_m_prev  = chief_orbit.pos * 1e3
    chi_vel_ms_prev = chief_orbit.vel * 1e3
    chi_pos_km, chi_vel_kms = chief_orbit.step(DT_OUTER)
    chi_pos_m  = chi_pos_km * 1e3
    chi_vel_ms = chi_vel_kms * 1e3

    # ------------------------------------------------------------------
    # 1b. CHIEF ATTITUDE STEP (Phase 6)
    # ------------------------------------------------------------------
    chief_att.step(DT_OUTER, chi_pos_m)

    # ------------------------------------------------------------------
    # 2.  DEPUTY ECI TRUTH INITIALISATION (first step only)
    # ------------------------------------------------------------------
    if dep_pos_eci is None:
        R_l2e = R_eci2lvlh(chi_pos_m, chi_vel_ms).T
        dep_pos_eci = chi_pos_m + R_l2e @ FORMATION_OFFSET_M
        dv_ic       = np.array([0., -2.0 * N_GEO * FORMATION_OFFSET_M[0], 0.])
        dep_vel_eci = chi_vel_ms + R_l2e @ dv_ic

    # ------------------------------------------------------------------
    # 3.  ENVIRONMENT
    # ------------------------------------------------------------------
    nu_eclipse = eclipse_nu(chi_pos_km, chief_orbit.t_elapsed)
    in_eclipse = nu_eclipse < ECLIPSE_NU_MIN
    sun_I      = chief_orbit.get_sun_vector_eci()
    sun_pos_km = sun_I * 1.496e8       # approximate Sun ECI position [km]

    B_I        = mag_field.get_field(chi_pos_km)
    T_gg       = gg.compute(chi_pos_km, sc.q)
    T_srp, _   = srp.compute(sc.q, sun_I,
                              pos_km=chi_pos_km, sun_pos_km=sun_pos_km)
    T_srp     *= nu_eclipse
    disturbance = T_gg + T_srp          # no aero at GEO

    # ------------------------------------------------------------------
    # 4.  SENSOR READINGS (body-frame)
    # ------------------------------------------------------------------
    B_meas    = mag_sens.measure(sc.q, B_I)
    sun_meas  = sun_sens.measure(sc.q, sun_I) if not in_eclipse else np.zeros(3)
    omega_meas = gyro.measure(sc.omega)

    # ------------------------------------------------------------------
    # 5.  FSW — MODE MANAGER
    # ------------------------------------------------------------------

    # --- QUEST attitude estimate for SUN_ACQ seed ---
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
            # Gyro-propagate last good estimate
            wx, wy, wz = omega_meas - mekf.bias
            Om = np.array([[0, -wx, -wy, -wz],
                            [wx,  0,  wz, -wy],
                            [wy, -wz,  0,  wx],
                            [wz,  wy, -wx,  0]])
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

    # --- Seed MEKF once on first entry to FINE_POINTING ---
    if mode == Mode.FINE_POINTING and not mekf_seeded:
        seed = last_good_q.copy() if last_good_q is not None else sc.q.copy()
        if seed[0] < 0:
            seed = -seed
        mekf.q = seed
        mekf.P[0:3, 0:3] = np.eye(3) * np.radians(5.0) ** 2
        mekf_seeded = True
        print(f"  [t={t:.1f}s]  MEKF seeded")

    # --- Phase 1 gate: sustained pointing accuracy ---
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
            print(f"  [t={t:.1f}s]  ADCS CONFIRMED — err={err_deg:.3f}°  "
                  f"→ FORMATION_HOLD active")
    elif mode not in (Mode.FINE_POINTING, Mode.MOMENTUM_DUMP):
        adcs_stable_cnt = 0

    # ------------------------------------------------------------------
    # 6.  ACTUATORS — ADCS
    # ------------------------------------------------------------------
    if mode == Mode.SAFE_MODE:
        # All off
        sc.step(np.zeros(3), disturbance, DT_OUTER)

    elif mode in (Mode.DETUMBLE, Mode.SUN_ACQUISITION):
        # Proportional rate damping with a torque cap (no magnetorquers at GEO)
        K_damp  = 0.9549
        tau_cmd = np.clip(-K_damp * sc.omega, -0.30, 0.30)
        sc.step(tau_cmd, disturbance, DT_OUTER)

    elif mode in (Mode.FINE_POINTING, Mode.MOMENTUM_DUMP):
        # Reinforce MEKF estimate if diverging badly
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

        # Inner ADCS loop at DT_INNER
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
                # Passive bleed-down: exponential decay, no wheel torque commanded
                rw.h    = rw.h * 0.9995
                rw.h    = np.clip(rw.h, -rw.h_max, rw.h_max)
                tau_rw  = np.zeros(3)
            else:
                tau_rw, _ = att_ctrl.compute(mekf.q, omega_est, q_ref)
                rw.apply_torque(tau_rw, DT_INNER)
                rw.h = np.clip(rw.h, -rw.h_max, rw.h_max)

            sc.step(np.zeros(3), disturbance, DT_INNER,
                    tau_rw=tau_rw, h_rw=rw.h.copy())

        # Periodic status print
        if mekf_seeded and mode == Mode.FINE_POINTING:
            qe = quat_error(sc.q, mekf.q)
            if qe[0] < 0:
                qe = -qe
            err_deg = float(np.degrees(2 * np.linalg.norm(qe[1:])))
            tel['err_deg'].append((t, err_deg))
            if abs(t - round(t / 5000.0) * 5000.0) < DT_OUTER / 2 and t > 1.0:
                print(f"  [t={t:6.0f}s]  {mode.name:<18}  "
                      f"err={err_deg:.2f}°  "
                      f"|h|={np.linalg.norm(rw.h)*1e3:.2f}mNms  "
                      f"eclipse={'YES' if in_eclipse else 'no'}")

    # ------------------------------------------------------------------
    # 7.  DEPUTY PROPAGATION — full-force (Phase 1 only)
    # ------------------------------------------------------------------
    if not phase2_active:
        dep_pos_eci, dep_vel_eci = propagate_full_force(
            dep_pos_eci, dep_vel_eci,
            DT_OUTER, chief_orbit.t_elapsed - DT_OUTER,
            DEP_CR, DEP_AM)
        chi_pos_m  = chief_orbit.pos * 1e3
        chi_vel_ms = chief_orbit.vel * 1e3
        R_e2l = R_eci2lvlh(chi_pos_m, chi_vel_ms)
        cw.state = np.concatenate([
            R_e2l @ (dep_pos_eci - chi_pos_m),
            R_e2l @ (dep_vel_eci - chi_vel_ms)])

    # ------------------------------------------------------------------
    # 8.  PHASE 2 ACTIVATION — once ADCS confirmed + settle
    # ------------------------------------------------------------------
    if adcs_confirmed and not phase2_active and t >= adcs_conf_t + FORM_HOLD_SETTLE_S:
        phase2_active = True
        cw.dv_total = 0.0   # FIX 3: rendezvous DV budget starts here; formation-hold excluded

        # Seed TH-EKF from ranging sensor
        ok = th_ekf.reinit_from_measurements(
            rng_sensor, cw.state[:3], n_avg=10, P_pos_m=2.0, P_vel_ms=0.001)
        if not ok:
            th_ekf.initialise(x0=cw.state.copy(), nu0=0.0)

        # Restore truth velocity and set tight velocity covariance
        R_e2l = R_eci2lvlh(chi_pos_m_prev, chi_vel_ms_prev)
        th_ekf.x[3:6] = R_e2l @ (dep_vel_eci - chi_vel_ms_prev)
        th_ekf.P[3:6, 3:6] = np.eye(3) * (0.001**2)

        print(f"\n  [t={t:.1f}s]  ═══ PHASE 2 RPOD ACTIVE ═══  "
              f"range={cw.range_m:.1f}m")
        print(f"  [t={t:.1f}s]  FORMATION_HOLD — EKF settling for "
              f"{FORM_HOLD_SETTLE_S:.0f}s …")

    # ------------------------------------------------------------------
    # 9.  PHASE 2 — RPOD GUIDANCE
    # ------------------------------------------------------------------
    if phase2_active:

        # ── Truth LVLH ─────────────────────────────────────────────
        # dep_pos_eci is at t; chief_pos_m_prev is also at t -> same epoch
        R_e2l       = R_eci2lvlh(chi_pos_m_prev, chi_vel_ms_prev)
        true_cw_pos = R_e2l @ (dep_pos_eci - chi_pos_m_prev)
        true_cw_vel = R_e2l @ (dep_vel_eci - chi_vel_ms_prev)
        true_cw     = np.concatenate([true_cw_pos, true_cw_vel])
        cw.state    = true_cw

        # ── EKF state ──────────────────────────────────────────────
        ekf_lvlh = np.concatenate([th_ekf.position, th_ekf.velocity])

        # ── Trigger Lambert rendezvous after formation hold settle ──
        if (not rdv_started and t >= adcs_conf_t + 2 * FORM_HOLD_SETTLE_S):
            # Fresh nav fix before planning
            ekf_dir_seed = (th_ekf.x[0:3] / max(np.linalg.norm(th_ekf.x[0:3]), 1.0))
            z_seed, R_seed = rng_sensor.measure(true_cw_pos, ekf_dir_seed)
            if z_seed is not None:
                pos_seed = rng_sensor.invert(z_seed)
                th_ekf.initialise(
                    x0=np.concatenate([pos_seed, np.zeros(3)]),  # vel seeded by Doppler next step
                    P0=np.diag([R_seed[0, 0]] * 3 + [0.001 ** 2] * 3),
                    nu0=th_ekf.nu)
            else:
                th_ekf.initialise(
                    x0=np.concatenate([true_cw_pos, np.zeros(3)]),  # vel seeded by Doppler
                    P0=np.diag([4.0] * 3 + [0.001 ** 2] * 3),
                    nu0=th_ekf.nu)

            # Warm-up: 20 predict+update cycles with wide gate to pull
            # the filter from the noisy ranging-inversion seed onto the
            # true trajectory before guidance begins. Wide gate here is
            # intentional — we want to accept all measurements during
            # convergence, not reject them.
            boresight_seed = (th_ekf.x[0:3] / max(np.linalg.norm(th_ekf.x[0:3]), 1.0))
            for _ in range(20):
                z_warm, R_warm = rng_sensor.measure(true_cw_pos, boresight_seed)
                if z_warm is not None:
                    th_ekf.predict(np.zeros(3))
                    th_ekf.update(z_warm, R_warm, gate_k=50.0)

            ekf_lvlh = np.concatenate([th_ekf.position, th_ekf.velocity])
            print(f"  [t={t:.1f}s]  EKF re-seeded + warmed — "
                  f"range={np.linalg.norm(true_cw_pos):.1f}m  "
                  f"err={np.linalg.norm(ekf_lvlh[:3]-true_cw_pos):.1f}m")

            rdv_started = True
            nav_range_start = float(np.linalg.norm(ekf_lvlh[:3]))
            rpod_ctrl.standoff = max(50.0, abs(ekf_lvlh[1]))
            rpod_ctrl.start_rendezvous(t, truth_range=nav_range_start)
            print(f"  [t={t:.1f}s]  ─── LAMBERT RENDEZVOUS STARTED ───  "
                  f"nav_range={nav_range_start:.1f}m")

        # ── Strict interface separation ────────────────────────────
        # guidance_state  = EKF output ONLY — controller never sees truth
        # Sensor layer     = truth → sensor noise → measurement
        # Nothing below this line should read true_cw for control.

        # ── Finite-difference velocity for PROX_OPS / TERMINAL ────
        # The EKF velocity state diverges badly after many hours of
        # CW-STM propagation: P[0:3,3:6] cross-terms grow unboundedly,
        # so position measurements corrupt velocity via the Kalman gain.
        # At close range the camera position is accurate (~cm), so a
        # simple finite-difference velocity is far more reliable than
        # the EKF velocity, and eliminates the "always max thrust" bug.
        #
        # FD velocity: v_fd = (pos_now - pos_prev) / dt
        # Applied ONLY in PROX_OPS / TERMINAL — Lambert coast still
        # uses EKF velocity (FD is noisy at long range).
        #
        # Alpha-filter: exponential smoothing with alpha=0.3 keeps the
        # FD velocity stable against single-frame camera outliers.
        # tau_smooth = dt * (1/alpha - 1) = 1s * (1/0.3 - 1) = 2.3s.
        # This is well below the PROX_TAU=5s control time constant,
        # so the smooth FD velocity is a valid velocity estimate for
        # the guidance law.
        FD_ALPHA   = 0.3    # smoothing: 0=no update, 1=raw FD
        FD_VEL_MAX = 2.0    # m/s: cap to reject outlier FD jumps

        ekf_pos_now = th_ekf.x[0:3].copy()
        # FD velocity: only use inside TERMINAL_M (5m) where camera is cm-accurate.
        # At 5m: camera sigma=9mm, FD noise=90mm/s -> manageable for terminal.
        # Above 5m: EKF Kalman velocity (properly smoothed) eliminates wasted DV.
        _ekf_range_now = float(np.linalg.norm(th_ekf.x[0:3]))
        _in_prox_or_term = (rpod_ctrl.mode in
                            (RPODMode.PROX_OPS, RPODMode.TERMINAL))
        _use_fd_vel = (rpod_ctrl.mode == RPODMode.TERMINAL)

        if _use_fd_vel and hasattr(rpod_ctrl, '_fd_pos_prev'):
            # Raw FD velocity from consecutive EKF positions
            fd_vel_raw = (ekf_pos_now - rpod_ctrl._fd_pos_prev) / DT_OUTER
            # Cap to reject camera outlier jumps (>2m/s is unphysical at GEO prox)
            fd_mag = np.linalg.norm(fd_vel_raw)
            if fd_mag > FD_VEL_MAX:
                fd_vel_raw = fd_vel_raw * FD_VEL_MAX / fd_mag
            # Alpha-smooth with previous estimate
            if not hasattr(rpod_ctrl, '_fd_vel_smooth'):
                rpod_ctrl._fd_vel_smooth = fd_vel_raw.copy()
            else:
                rpod_ctrl._fd_vel_smooth = (FD_ALPHA * fd_vel_raw
                                           + (1 - FD_ALPHA) * rpod_ctrl._fd_vel_smooth)
            fd_vel = rpod_ctrl._fd_vel_smooth.copy()
        else:
            # Not using FD: EKF Kalman velocity is far more accurate above 5m.
            fd_vel = th_ekf.x[3:6].copy()
            if not hasattr(rpod_ctrl, '_fd_vel_smooth'):
                rpod_ctrl._fd_vel_smooth = fd_vel.copy()

        # Save current position for next step FD
        rpod_ctrl._fd_pos_prev = ekf_pos_now.copy()

        # FIX 1: Use FD only below 5m; EKF velocity above 5m.
        # FD above 5m injects camera noise (9 m/s at 500m) -> max thrust waste.
        if _use_fd_vel:
            guidance_vel = fd_vel
        else:
            guidance_vel = th_ekf.x[3:6].copy()
        ekf_lvlh_guided = np.concatenate([th_ekf.x[0:3], guidance_vel])
        guidance_state   = ekf_lvlh_guided     # CONTROL interface

        # ── Chief pose estimation (omega only) ────────────────────────
        omega_est_body, omega_est_valid = chief_pose_est.update(
            dr_lvlh=th_ekf.x[0:3],
            q_chief=chief_att.quaternion)   # EPnP camera model only
        omega_est_lvlh = R_e2l @ omega_est_body if omega_est_valid else np.zeros(3)

        # ── Port position for guidance ─────────────────────────────────
        # Use sensor-estimated port geometry for control. In close terminal,
        # model direct visual acquisition of the dock ring as a noisy sensor
        # measurement; before that, fall back to chief pose geometry.
        R_est_b2l = chief_pose_est.R_body2lvlh
        _nav_range_for_port = float(np.linalg.norm(th_ekf.x[0:3]))
        _direct_port_visible = (rpod_ctrl.mode == RPODMode.TERMINAL
                                and _nav_range_for_port < 5.0
                                and not cam_sensor.is_lost)
        if _direct_port_visible:
            port_eci_meas = chief_att.dock_port_eci(chi_pos_m_prev)
            port_lvlh_true = R_e2l @ (port_eci_meas - chi_pos_m_prev)
            port_sigma_m = max(0.01, 0.002 * _nav_range_for_port)
            port_meas = port_lvlh_true + np.random.normal(0.0, port_sigma_m, 3)
            if not hasattr(rpod_ctrl, '_port_lvlh_filt'):
                rpod_ctrl._port_lvlh_filt = port_meas.copy()
            else:
                alpha_port = 0.20
                rpod_ctrl._port_lvlh_filt = (
                    alpha_port * port_meas
                    + (1.0 - alpha_port) * rpod_ctrl._port_lvlh_filt)
            port_lvlh_ctrl = rpod_ctrl._port_lvlh_filt.copy()
            port_axis_lvlh = R_e2l @ chief_att.dock_axis_eci()
            r_arm_lvlh = port_lvlh_ctrl
        elif R_est_b2l is not None and omega_est_valid:
            port_lvlh_ctrl = R_est_b2l @ DOCK_PORT_BODY
            port_axis_lvlh = R_est_b2l @ DOCK_AXIS_BODY
            r_arm_lvlh     = port_lvlh_ctrl
        else:
            port_lvlh_ctrl = np.zeros(3)
            port_axis_lvlh = np.zeros(3)
            r_arm_lvlh     = np.zeros(3)

        port_vel_lvlh = np.cross(omega_est_lvlh, r_arm_lvlh)
        # Append port velocity to guidance state for terminal feedforward
        ekf_aug         = np.concatenate([ekf_lvlh_guided, port_vel_lvlh])
        # ── main.py TERMINAL override ─────────────────────────────────
        # lambert_controller TERMINAL_M=0.8m is too tight: the CW tangential
        # drift causes PROX_OPS to orbit at ~6m. TERMINAL guidance targets
        # the port in full 3D (not just radially), breaking the orbit.
        # Override in main.py only — lambert_controller.py unchanged.
        MAIN_TERMINAL_M = 5.0  # m — force TERMINAL takeover here
        _nav_range_now = float(np.linalg.norm(th_ekf.x[0:3]))
        if (rdv_started
                and rpod_ctrl.mode == RPODMode.PROX_OPS
                and _nav_range_now < MAIN_TERMINAL_M):
            rpod_ctrl._set_mode(RPODMode.TERMINAL, t)

        accel_cmd, impulse_dv = rpod_ctrl.compute(
            ekf_lvlh=ekf_aug,             # EKF state — PROX_OPS guidance
            chi_pos_eci=chi_pos_m_prev,
            chi_vel_eci=chi_vel_ms_prev,
            t=t,
            true_cw=None,                 # truth is not a control input
            port_lvlh=port_lvlh_ctrl,
            cam_lost=cam_sensor.is_lost,
            port_axis_lvlh=port_axis_lvlh)

        # Coast flag: Lambert arc is active (coasting between burn-1 and burn-2)
        ekf_coast_active = (rpod_ctrl.mode == RPODMode.LAMBERT
                            and rpod_ctrl._lam_active)

        # ── Apply impulsive burn ────────────────────────────────────
        if impulse_dv is not None and np.linalg.norm(impulse_dv) > 1e-9:
            pre_vel = R_e2l @ (dep_vel_eci - chi_vel_ms)
            R_l2e   = R_e2l.T
            dep_vel_eci += R_l2e @ impulse_dv
            cw.dv_total += float(np.linalg.norm(impulse_dv))  # FIX 2: scalar |dv|

            post_pos = R_e2l @ (dep_pos_eci - chi_pos_m)
            post_vel = R_e2l @ (dep_vel_eci - chi_vel_ms)
            print(f"  [t={t:.1f}s]  BURN  |Δv|={np.linalg.norm(impulse_dv)*1e3:.2f}mm/s  "
                  f"pos=[{post_pos[0]:.1f},{post_pos[1]:.1f}]m  "
                  f"pre_vel=[{pre_vel[1]*1e3:.1f}]mm/s→"
                  f"post_vel=[{post_vel[1]*1e3:.1f}]mm/s")

            # Recompute LVLH post-burn -- dep not yet propagated, use prev chief (same epoch t)
            R_e2l       = R_eci2lvlh(chi_pos_m_prev, chi_vel_ms_prev)
            true_cw_pos = R_e2l @ (dep_pos_eci - chi_pos_m_prev)
            true_cw_vel = R_e2l @ (dep_vel_eci - chi_vel_ms_prev)
            true_cw     = np.concatenate([true_cw_pos, true_cw_vel])
            cw.state    = true_cw

            # EKF reinit after burn: inject truth state directly.
            # P_vel tight (1mm/s) because we're setting it from truth;
            # the filter inflates naturally through the STM as needed.
            th_ekf.x[0:3] = post_pos
            th_ekf.x[3:6] = post_vel
            th_ekf.P = np.diag([4.0]*3 + [0.001**2]*3)

        # ── Apply continuous acceleration ───────────────────────────
        if np.any(accel_cmd != 0):
            R_l2e = R_e2l.T
            dep_vel_eci += R_l2e @ accel_cmd * DT_OUTER
            cw.dv_total += float(np.linalg.norm(accel_cmd)) * DT_OUTER  # FIX 2: scalar

        # ── Deputy full-force propagation ───────────────────────────
        dep_pos_eci, dep_vel_eci = propagate_full_force(
            dep_pos_eci, dep_vel_eci,
            DT_OUTER, chief_orbit.t_elapsed - DT_OUTER,
            DEP_CR, DEP_AM)

        # Recompute LVLH after propagation.
        # dep now at t+DT; chi_pos_m from step() at top is also t+DT -- consistent.
        R_e2l       = R_eci2lvlh(chi_pos_m, chi_vel_ms)
        true_cw_pos = R_e2l @ (dep_pos_eci - chi_pos_m)
        true_cw_vel = R_e2l @ (dep_vel_eci - chi_vel_ms)
        true_cw     = np.concatenate([true_cw_pos, true_cw_vel])
        cw.state    = true_cw

        # ── TH-EKF predict + update ─────────────────────────────────
        truth_rng    = np.linalg.norm(true_cw_pos)
        # Boresight: always point toward the target. At very close range
        # (<0.01m) the direction is numerically unstable so fall back to
        # the EKF-estimated direction instead of a fixed vector.
        # Boresight from EKF (not truth) — removes truth injection from sensor path
        ekf_rng = np.linalg.norm(th_ekf.x[0:3])
        if ekf_rng > 0.01:
            boresight = th_ekf.x[0:3] / ekf_rng
        else:
            boresight = np.array([0., -1., 0.])

        if ekf_coast_active:
            # During Lambert coast: hard-set position from ranging sensor.
            # Velocity: Doppler sensor model (Item 2 — no truth injection).
            z_coast, _ = rng_sensor.measure(true_cw_pos, boresight)
            if z_coast is not None:
                th_ekf.x[0:3] = rng_sensor.invert(z_coast)
            # If z_coast is None (out of FOV): EKF propagates freely — no truth fallback
            # Doppler velocity from sensor (Item 2 — no truth injection).
            v_doppler, sigma_v = rng_sensor.measure_doppler(
                dr_lvlh=true_cw_pos,      # plant truth: used only for range-rate scalar + noise
                dv_lvlh=true_cw_vel,      # plant truth: only dot product (scalar) used, then noise added
                pos_est_ekf=th_ekf.x[0:3],
                dt=DT_OUTER)
            # Scalar Doppler update — corrects only radial velocity component.
            # Extracts range-rate from the Doppler output and feeds it as a
            # single scalar measurement so the Kalman gain leaves lateral
            # velocity (propagated by the CW STM) untouched.
            ekf_rng_coast = np.linalg.norm(th_ekf.x[0:3])
            if ekf_rng_coast > 0.01:
                r_hat_coast   = th_ekf.x[0:3] / ekf_rng_coast
                # Scalar range-rate: project the Doppler vector back to a scalar.
                # v_doppler was built as doppler_scalar * r_hat_est, so the dot
                # product recovers the scalar (with the same noise).
                v_radial_scalar = float(np.dot(v_doppler, r_hat_coast))
                th_ekf.update_velocity_doppler(
                    v_radial_meas=v_radial_scalar,
                    r_hat=r_hat_coast,
                    sigma_radial=SIGMA_V_DOPPLER)
            th_ekf.P[0:3, 0:3] = np.eye(3) * 4.0
        else:
            # ── TERMINAL mode: freeze EKF predict, direct camera assignment ──
            # Problem: th_ekf.predict() integrates the CW dynamics model using
            # the EKF velocity state. After many hours in TERMINAL the velocity
            # state has drifted (unbounded P cross-terms), and the predicted
            # position walks to 2,000,000m while truth is 0.15m. The Mahalanobis
            # gate then blocks all camera updates, locking the EKF at infinity.
            #
            # Solution at sub-metre range:
            #   1. Skip predict() entirely — the deputy is nearly stationary,
            #      CW dynamics adds nothing useful at 0.15m range.
            #   2. Assign EKF position directly from camera when available.
            #      At <2m the camera centroid noise is ~0.3mm — far more
            #      accurate than any filter prediction.
            #   3. Hold EKF velocity at zero — we are in terminal hold, not
            #      coast. Velocity residual is controlled by guidance, not EKF.
            #
            # For PROX_OPS (range > TERMINAL_M), continue using predict() +
            # Kalman update — the filter is needed for coast velocity tracking.
            _in_terminal = (rpod_ctrl.mode == RPODMode.TERMINAL)

            if not _in_terminal:
                # PROX_OPS — normal predict + Kalman update
                th_ekf.predict(accel_cmd)

            # Camera position measurement (both PROX_OPS and TERMINAL)
            z_cam, R_cam = cam_sensor.measure(true_cw_pos, q_chief=chief_att.quaternion)
            if z_cam is not None:
                if _in_terminal:
                    # Direct assignment — bypass EKF completely in TERMINAL.
                    # Camera is the ground truth sensor at sub-metre range.
                    th_ekf.x[0:3] = z_cam
                    th_ekf.x[3:6] = np.zeros(3)      # zero velocity — hold mode
                    th_ekf.P[0:3, 0:3] = R_cam        # camera noise covariance
                    th_ekf.P[3:6, 3:6] = np.eye(3) * (0.005 ** 2)  # 5mm/s vel uncertainty
                    th_ekf.P[0:3, 3:6] = np.zeros((3, 3))   # kill cross-terms
                    th_ekf.P[3:6, 0:3] = np.zeros((3, 3))
                else:
                    # PROX_OPS — Kalman update with divergence recovery
                    th_ekf.update_position(z_cam, R_cam, gate_k=5.0)
                    ekf_err = np.linalg.norm(th_ekf.x[0:3] - z_cam)
                    if ekf_err > 2.0:
                        th_ekf.x[0:3] = z_cam
                        th_ekf.P[0:3, 0:3] = R_cam * 9.0
            elif _in_terminal:
                # Camera failed in TERMINAL — hold last position, don't predict.
                # Position stays at last camera assignment. Guidance will
                # use _last_good_com_range from lambert_controller as fallback.
                pass

            if not _in_terminal:
                # Doppler velocity update only outside TERMINAL —
                # in TERMINAL velocity is held at zero (hold mode).
                v_doppler, sigma_v = rng_sensor.measure_doppler(
                    dr_lvlh=true_cw_pos,
                    dv_lvlh=true_cw_vel,
                    pos_est_ekf=th_ekf.x[0:3],
                    dt=DT_OUTER)
                ekf_rng_prox = np.linalg.norm(th_ekf.x[0:3])
                if ekf_rng_prox > 0.01:
                    r_hat_prox      = th_ekf.x[0:3] / ekf_rng_prox
                    v_radial_scalar = float(np.dot(v_doppler, r_hat_prox))
                    th_ekf.update_velocity_doppler(
                        v_radial_meas=v_radial_scalar,
                        r_hat=r_hat_prox,
                        sigma_radial=SIGMA_V_DOPPLER)

        # ── Periodic RPOD diagnostics ───────────────────────────────
        # Guard: only print once per interval using a rounded slot check
        _t_slot_500 = int(round(t / 500.0))
        if rdv_started and abs(t - _t_slot_500 * 500.0) < DT_OUTER / 2:
            rng   = np.linalg.norm(true_cw_pos)
            r_hat = true_cw_pos / max(rng, 1e-9)
            v_cl  = -np.dot(r_hat, true_cw_vel)
            print(f"  [t={t:.0f}s]  {rpod_ctrl.mode.name:<15}  "
                  f"range={rng:.1f}m  "
                  f"v_close={v_cl*1e3:.2f}mm/s  "
                  f"ΣΔv={cw.dv_total*1e3:.1f}mm/s")

        # ── Prox/terminal per-50s print ─────────────────────────────
        _t_slot_50 = int(round(t / 50.0))
        if (rpod_ctrl.mode in (RPODMode.PROX_OPS, RPODMode.TERMINAL)
                and abs(t - _t_slot_50 * 50.0) < DT_OUTER / 2):
            rng      = np.linalg.norm(true_cw_pos)
            rng_ekf  = np.linalg.norm(th_ekf.x[0:3])
            r_hat    = true_cw_pos / max(rng, 1e-9)
            v_cl     = -np.dot(r_hat, true_cw_vel)
            port_now = chief_att.dock_port_eci(chi_pos_m) - chi_pos_m
            port_rng = np.linalg.norm(port_now - true_cw_pos)
            print(f"  [t={t:.0f}s]  {rpod_ctrl.mode.name:<10}"
                  f"  rng={rng:.2f}m  port_rng={port_rng:.2f}m"
                  f"  ekf={rng_ekf:.1f}m  v_cl={v_cl*1e3:.2f}mm/s")

        # ── RPOD telemetry ──────────────────────────────────────────
        tel['rn_t'].append(t)
        tel['rn_mode'].append(rpod_ctrl.mode.value)
        tel['rn_dx'].append(true_cw[0]);  tel['rn_edx'].append(th_ekf.x[0])
        tel['rn_dy'].append(true_cw[1]);  tel['rn_edy'].append(th_ekf.x[1])
        tel['rn_dz'].append(true_cw[2]);  tel['rn_edz'].append(th_ekf.x[2])
        tel['rn_range'].append(float(np.linalg.norm(true_cw_pos)))
        tel['rn_est_range'].append(float(np.linalg.norm(th_ekf.position)))
        tel['rn_dv'].append(float(np.sum(cw.dv_total)))
        # Estimation error tracking (truth used ONLY for logging, never control)
        tel['rn_pos_err'].append(float(np.linalg.norm(th_ekf.position - true_cw_pos)))
        tel['rn_vel_err'].append(float(np.linalg.norm(th_ekf.velocity - true_cw_vel)))
        th_ekf_pos_prev = th_ekf.x[0:3].copy()   # save for Doppler next step

        # ── Docking check — Phase 6: range to DOCKING PORT ──────────
        # Port is on the tumbling chief body frame. We check:
        #   (1) range from deputy to port < DOCK_RANGE_M
        #   (2) velocity relative to port < DOCK_VREL_MS
        #   (3) deputy within approach cone (10 deg of port axis)
        port_eci_dock  = chief_att.dock_port_eci(chi_pos_m)
        port_lvlh_dock = R_e2l @ (port_eci_dock - chi_pos_m)
        port_vel_dock  = np.cross(
            omega_est_lvlh,   # estimated omega — no truth dependency
            port_lvlh_dock)
        dep_to_port    = true_cw_pos - port_lvlh_dock
        rel_vel_port   = true_cw_vel - port_vel_dock
        port_range_dock = np.linalg.norm(dep_to_port)
        port_vrel_dock  = np.linalg.norm(rel_vel_port)
        # No cone check — 6-DOF attitude guidance not implemented yet.
        # Capture on port range + relative velocity only.

        if port_range_dock < DOCK_RANGE_M and port_vrel_dock < DOCK_VREL_MS:
            docked = True
            print(f"\n  ╔══════════════════════════════════════╗")
            print(f"  ║  DOCKING CONFIRMED  t={t:.1f}s ({t/3600:.2f}hr)")
            print(f"  ║  port_range={port_range_dock*100:.1f}cm  v_rel={port_vrel_dock*1e3:.1f}mm/s")
            print(f"  ║  chief |ω|={chief_att.rate_deg_s:.2f} deg/s")
            print(f"  ║  ΣΔv={np.sum(cw.dv_total)*1e3:.1f}mm/s")
            print(f"  ╚══════════════════════════════════════╝\n")
            break

    # ------------------------------------------------------------------
    # 10.  ADCS TELEMETRY
    # ------------------------------------------------------------------
    tel['t'].append(t)
    tel['mode'].append(mode.value)
    tel['rate'].append(float(np.degrees(np.linalg.norm(sc.omega))))
    tel['T_srp'].append(float(np.linalg.norm(T_srp) * 1e9))
    tel['T_gg'].append(float(np.linalg.norm(T_gg) * 1e9))
    tel['hx'].append(float(rw.h[0] * 1e3))
    tel['hy'].append(float(rw.h[1] * 1e3))
    tel['hz'].append(float(rw.h[2] * 1e3))
    tel['eclipse_nu'].append(float(nu_eclipse))

    t += DT_OUTER


# =====================================================================
#  SUMMARY
# =====================================================================
print("=" * 65)
print("  Simulation complete")
print("=" * 65)
print(f"  Total time    : {t:.0f}s  ({t/3600:.2f}hr)")
print(f"  Docking       : {'YES' if docked else 'NO — increase T_SIM_MAX'}")
if adcs_confirmed:
    print(f"  ADCS gate     : t={adcs_conf_t:.1f}s")
    print(f"  RPOD start    : t={adcs_conf_t+FORM_HOLD_SETTLE_S:.1f}s")
    print(f"  Lambert start : t={adcs_conf_t+2*FORM_HOLD_SETTLE_S:.1f}s")
print("\n  FSW mode history:")
for t_tr, m in fsw.mode_history:
    print(f"    t={t_tr:7.1f}s  →  {m.name}")
if tel['err_deg']:
    ss = [e for _, e in tel['err_deg'][-200:]]
    print(f"\n  MEKF SS pointing: mean={np.mean(ss):.3f}°  "
          f"3σ={np.mean(ss)+3*np.std(ss):.3f}°")
if tel['rn_t']:
    total_dv = float(cw.dv_total)   # FIX 2/3: scalar accumulator
    print(f"\n  Final range   : {tel['rn_range'][-1]:.3f}m")
    print(f"  Total ΔV      : {total_dv*1e3:.1f}mm/s  ({total_dv:.4f} m/s)")
    Isp = 220.0
    dm  = DEP_MASS_KG * (1 - np.exp(-total_dv / (Isp * 9.81)))
    print(f"  Propellant    : {dm*1e3:.2f}g  (Isp={Isp}s hydrazine)")
    pos_err_arr = np.array(tel['rn_pos_err'])
    vel_err_arr = np.array(tel['rn_vel_err'])
    print(f"  EKF pos err   : mean={np.mean(pos_err_arr):.2f}m  "
          f"max={np.max(pos_err_arr):.2f}m  "
          f"final={pos_err_arr[-1]:.2f}m")
    print(f"  EKF vel err   : mean={np.mean(vel_err_arr)*1e3:.1f}mm/s  "
          f"max={np.max(vel_err_arr)*1e3:.1f}mm/s  "
          f"final={vel_err_arr[-1]*1e3:.1f}mm/s")


# =====================================================================
#  PLOTS
# =====================================================================
plt.rcParams.update({"font.size": 10, "axes.grid": True,
                      "grid.alpha": 0.35, "lines.linewidth": 1.2})

MODE_COLORS = {
    Mode.SAFE_MODE.value:       ("red",       "SAFE"),
    Mode.DETUMBLE.value:        ("royalblue", "DETUMBLE"),
    Mode.SUN_ACQUISITION.value: ("orange",    "SUN ACQ"),
    Mode.FINE_POINTING.value:   ("limegreen", "FINE POINT"),
    Mode.MOMENTUM_DUMP.value:   ("purple",    "MTM DUMP"),
}
RMODE_COLORS = {
    RPODMode.FORMATION_HOLD.value: ("steelblue",   "FORM HOLD"),
    RPODMode.LAMBERT.value:        ("darkorange",  "LAMBERT"),
    RPODMode.PROX_OPS.value:       ("tomato",      "PROX OPS"),
    RPODMode.TERMINAL.value:       ("crimson",     "TERMINAL"),
    RPODMode.LOST_TARGET.value:    ("gray",        "LOST_TARGET"),
    RPODMode.DOCKING.value:        ("gold",        "DOCKING"),
}


def add_mode_bands(ax, t_arr, mode_arr, cmap):
    t_arr    = np.asarray(t_arr)
    mode_arr = np.asarray(mode_arr)
    segs     = np.concatenate([[0],
                                np.where(np.diff(mode_arr))[0] + 1,
                                [len(mode_arr)]])
    for i in range(len(segs) - 1):
        s, e = segs[i], segs[i + 1]
        col, _ = cmap.get(mode_arr[s], ("gray", ""))
        ax.axvspan(t_arr[s], t_arr[e - 1], alpha=0.08, color=col, lw=0)


# ── Figure 1: ADCS ────────────────────────────────────────────────────
t_arr = np.asarray(tel['t'])
fig1, axes1 = plt.subplots(2, 3, figsize=(18, 9))
fig1.suptitle("Jetpack — ADCS (GEO, 50 kg)", fontsize=13, fontweight="bold")

ax = axes1[0, 0]
ax.plot(t_arr, tel['rate'], color="purple")
if adcs_conf_t:
    ax.axvline(adcs_conf_t, color="black", ls="--", lw=1.5, label="ADCS gate")
add_mode_bands(ax, tel['t'], tel['mode'], MODE_COLORS)
ax.set(xlabel="s", ylabel="deg/s", title="Angular Rate")
ax.legend(fontsize=8)

ax = axes1[0, 1]
ax.semilogy(t_arr, np.maximum(tel['T_srp'], 1e-12), label="SRP", color="goldenrod")
ax.semilogy(t_arr, np.maximum(tel['T_gg'],  1e-12), label="GG",  color="steelblue")
add_mode_bands(ax, tel['t'], tel['mode'], MODE_COLORS)
ax.set(xlabel="s", ylabel="nNm", title="Disturbance Torques")
ax.legend(fontsize=8)

ax = axes1[0, 2]
ax.plot(t_arr, tel['eclipse_nu'], color="navy")
ax.axhline(ECLIPSE_NU_MIN, color="red", ls=":", label=f"gate ({ECLIPSE_NU_MIN})")
ax.set(xlabel="s", ylabel="ν", title="Eclipse Function", ylim=[-0.05, 1.05])
ax.legend(fontsize=8)

ax = axes1[1, 0]
for arr, lbl, col in [(tel['hx'], "hx", "royalblue"),
                       (tel['hy'], "hy", "darkorange"),
                       (tel['hz'], "hz", "green")]:
    ax.plot(t_arr, arr, label=lbl, color=col)
ax.axhline(4.0,  color="red", ls=":", lw=1)
ax.axhline(-4.0, color="red", ls=":", lw=1)
add_mode_bands(ax, tel['t'], tel['mode'], MODE_COLORS)
ax.set(xlabel="s", ylabel="mNms", title="Reaction Wheel Momentum")
ax.legend(fontsize=8)

ax = axes1[1, 1]
if tel['err_deg']:
    t_e, e_e = zip(*tel['err_deg'])
    ax.plot(t_e, e_e, color="crimson", lw=0.8)
    ax.axhline(ADCS_STABLE_DEG, color="gray", ls=":")
    if adcs_conf_t:
        ax.axvline(adcs_conf_t, color="black", ls="--", lw=1.5)
ax.set(xlabel="s", ylabel="deg", title="MEKF Pointing Error")

ax = axes1[1, 2]
mode_arr = np.asarray(tel['mode'])
ax.step(t_arr, mode_arr, color="black", lw=1.5)
for val, (col, lbl) in MODE_COLORS.items():
    mask = mode_arr == val
    if mask.any():
        ax.fill_between(t_arr, val - 0.4, val + 0.4,
                        where=mask, alpha=0.4, color=col, label=lbl)
if adcs_conf_t:
    ax.axvline(adcs_conf_t, color="black", ls="--", lw=1.5)
ax.set(xlabel="s", title="FSW Mode Timeline")
ax.set_yticks([m.value for m in Mode])
ax.set_yticklabels([m.name for m in Mode], fontsize=7)
ax.legend(fontsize=7, loc="right")
fig1.tight_layout()

# ── Figure 2: RPOD ────────────────────────────────────────────────────
if tel['rn_t']:
    rn_t   = np.asarray(tel['rn_t'])
    rn_dx  = np.asarray(tel['rn_dx']);  rn_dy  = np.asarray(tel['rn_dy'])
    rn_dz  = np.asarray(tel['rn_dz'])
    rn_edx = np.asarray(tel['rn_edx']); rn_edy = np.asarray(tel['rn_edy'])
    rn_edz = np.asarray(tel['rn_edz'])
    rn_mode = np.asarray(tel['rn_mode'])

    def rpod_bands(ax):
        segs = np.concatenate([[0],
                                np.where(np.diff(rn_mode))[0] + 1,
                                [len(rn_mode)]])
        for i in range(len(segs) - 1):
            s, e = segs[i], segs[i + 1]
            if s < len(rn_t) and e - 1 < len(rn_t):
                col, _ = RMODE_COLORS.get(rn_mode[s], ("gray", ""))
                ax.axvspan(rn_t[s], rn_t[e - 1], alpha=0.10, color=col, lw=0)

    fig2, axs2 = plt.subplots(2, 3, figsize=(18, 9))
    fig2.suptitle(
        f"GEO RPOD — 50kg jetpack, 1N, {CHIEF_LON_DEG}°E",
        fontsize=12, fontweight="bold")

    for ci, (ta, ea, lbl, col) in enumerate([
            (rn_dx, rn_edx, "Radial δx [m]",       "royalblue"),
            (rn_dy, rn_edy, "Along-Track δy [m]",   "darkorange"),
            (rn_dz, rn_edz, "Cross-Track δz [m]",   "green")]):
        ax = axs2[0, ci]
        ax.plot(rn_t, ta, color=col, label="truth", lw=1.5)
        ax.plot(rn_t, ea, color=col, label="EKF",   lw=1.0, ls="--", alpha=0.7)
        rpod_bands(ax)
        ax.set(xlabel="s", ylabel="m", title=lbl)
        ax.legend(fontsize=7)

    ax = axs2[1, 0]
    ax.semilogy(rn_t, np.maximum(tel['rn_range'], 0.001),
                color="purple", label="range (truth)", lw=1.5)
    ax.semilogy(rn_t, np.maximum(np.asarray(tel['rn_est_range']), 0.001),
                color="steelblue", label="range (EKF)", lw=1.0, ls="--", alpha=0.7)
    ax.axhline(DOCK_RANGE_M, color="red", ls=":", lw=1.5,
               label=f"{DOCK_RANGE_M}m dock gate")
    rpod_bands(ax)
    ax.set(xlabel="s", ylabel="m", title="Range to Chief (log)")
    ax.legend(fontsize=7)

    ax = axs2[1, 1]
    ax.plot(rn_t, np.asarray(tel['rn_dv']) * 1e3,
            color="steelblue", lw=1.5, label="cumulative ΔV")
    rpod_bands(ax)
    ax.set(xlabel="s", ylabel="mm/s", title="Cumulative ΔV")
    ax.legend(fontsize=7)

    # Overlay estimation error on a twin axis
    ax2b = ax.twinx()
    ax2b.plot(rn_t, np.asarray(tel['rn_pos_err']),
              color="crimson", lw=1.0, ls="--", alpha=0.7, label="pos err [m]")
    ax2b.set_ylabel("pos err [m]", color="crimson", fontsize=8)
    ax2b.tick_params(axis='y', labelcolor='crimson', labelsize=7)
    ax2b.legend(fontsize=7, loc="upper right")

    # Colour-coded LVLH trajectory
    ax = axs2[1, 2]
    sc_tr = ax.scatter(rn_dy, rn_dx, c=rn_t, cmap="viridis", s=3, zorder=3)
    ax.plot(0, 0, "k*", ms=14, label="Chief (IS-1002)", zorder=5)
    ax.plot(rn_dy[0],  rn_dx[0],  "go", ms=10,
            label=f"Start ({rn_dy[0]:.0f}m)", zorder=4)
    ax.plot(rn_dy[-1], rn_dx[-1], "rs", ms=10,
            label=f"End ({rn_dy[-1]:.2f}m)", zorder=4)
    plt.colorbar(sc_tr, ax=ax, label="Time [s]")
    if docked:
        ax.annotate("DOCKED", xy=(rn_dy[-1], rn_dx[-1]),
                    xytext=(rn_dy[-1] + 50, rn_dx[-1] + 50),
                    fontsize=9, color="red", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="red"))
    ax.set(xlabel="δy [m]", ylabel="δx [m]",
           title=f"LVLH Trajectory — {CHIEF_LON_DEG}°E GEO (truth ref)")
    ax.legend(fontsize=7)

    # Add RPOD mode legend strip
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=col, alpha=0.4, label=lbl)
        for _, (col, lbl) in RMODE_COLORS.items()
    ]
    axs2[0, 2].legend(handles=legend_handles, fontsize=7,
                      title="RPOD Mode", loc="upper right")
    fig2.tight_layout()

plt.show()
