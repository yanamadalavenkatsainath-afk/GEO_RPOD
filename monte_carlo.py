"""
Monte Carlo — GEO Jetpack RPOD
==============================
Runs N trials of the full mission simulation (main.py logic) with
randomised initial conditions to characterise DV and docking performance.

Varied per trial
----------------
  chief_omega0_deg_s  : tumble rate magnitude (uniform 0.05-0.20 deg/s)
                        and direction (random unit vector on sphere)
  chief_M0_deg        : mean anomaly at epoch (uniform 0-360 deg)
                        -> changes initial relative geometry + tumble phase
  sc_omega0_rad_s     : deputy initial tumble (uniform magnitude 0.1-0.3 rad/s)
  gyro_bias_seed      : numpy seed for gyro bias draw
  sensor_noise_seed   : numpy seed for rng/camera noise

Fixed per trial (same as main.py)
-----------------------------------
  All hardware parameters, gains, thresholds, DV accounting fixes

Outputs
-------
  monte_carlo_results.npz    — raw per-trial arrays
  monte_carlo_summary.txt    — statistics table
  monte_carlo_plots.png      — DV histogram + timeline scatter
"""

import numpy as np
import sys, os, time, traceback, warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
warnings.filterwarnings("ignore")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Imports (same as main.py) ────────────────────────────────────────
from plant.spacecraft                     import Spacecraft
from plant.contact_dynamics               import DockingContactModel
from plant.thruster_layout                import ThrusterLayout
from plant.finite_body                    import BoxBody, FiniteBodyPair
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
from sensors.body_camera                  import BodyMountedCamera
from actuators.reaction_wheel             import ReactionWheel
from actuators.magnetorquer               import Magnetorquer
from actuators.bdot                       import BDotController
from estimation.mekf                      import MEKF
from estimation.quest                     import QUEST
from estimation.th_ekf                    import THEKF
from estimation.port_tracker              import PortTracker
from estimation.terminal_nav_filter       import TerminalNavFilter
from chief_attitude                       import ChiefAttitude
from chief_pose_estimator                 import ChiefPoseEstimator
from control.attitude_controller          import AttitudeController
from control.lambert_controller           import GEORPODController, RPODMode
from control.keepout_planner              import KeepoutAvoidancePlanner
from control.spin_sync_controller         import SpinSyncController
from fsw.mode_manager                     import ModeManager, Mode
from utils.quaternion                     import quat_error, rot_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =====================================================================
#  FIXED MISSION CONFIG  (identical to main.py)
# =====================================================================
CHIEF_A_KM      = 42164.0
CHIEF_E         = 0.0003
CHIEF_I_DEG     = 0.8
CHIEF_RAAN_DEG  = 0.0
CHIEF_OMEGA_DEG = 0.0
DEP_MASS_KG     = 50.0
DEP_THRUST_N    = 1.0
DEP_CR          = 1.5
DEP_AM          = 0.00720
CHI_CR          = 1.5
CHI_AM          = 0.015
FORMATION_OFFSET_M = np.array([0.0, -1000.0, 0.0])
DT_OUTER        = 0.1
DT_INNER        = 0.01
N_INNER         = int(DT_OUTER / DT_INNER)
T_SIM_MAX       = 80_000.0
ADCS_STABLE_DEG = 1.0
ADCS_STABLE_SUST= 100
FORM_HOLD_SETTLE_S = 300.0
DOCK_RANGE_M    = 0.30
DOCK_VREL_MS    = 0.05
SOFT_CAPTURE_RANGE_M = DOCK_RANGE_M
SOFT_CAPTURE_VREL_MS = DOCK_VREL_MS
HARD_CAPTURE_RANGE_M = 0.08
HARD_CAPTURE_VREL_MS = 0.010
HARD_CAPTURE_HOLD_S  = 5.0
SOFT_CAPTURE_HOLD_S  = 5.0
SOFT_CAPTURE_LATCH_VREL_MS = 0.030
SOFT_CAPTURE_MAX_HOLD_S = 480.0   # FIX
CHIEF_BODY_HALF_EXTENTS_M = np.array([0.80, 0.80, 0.50])
DOCK_PORT_APERTURE_M = 0.15
DOCK_CONE_HALF_ANGLE_DEG = 15.0
DOCK_CONE_MIN_RANGE_M = 0.05
DOCK_FACE_TOL_M = 0.05
DOCK_ALIGN_MAX_DEG = 10.0
SOFT_CAPTURE_CORE_ALIGN_MAX_DEG = 20.0
SOFT_CAPTURE_ENTRY_ALIGN_MAX_DEG = 30.0  # FIX
SOFT_CAPTURE_ATTITUDE_TORQUE_SCALE = 0.4   # FIX
SOFT_CAPTURE_RESTITUTION = 0.10
SOFT_CAPTURE_TANGENTIAL_DAMPING = 0.30
ENABLE_PHYSICAL_THRUSTER_LAYOUT = False
THRUSTER_MAX_FORCE_N = 0.25
ENABLE_FINITE_BODY_COLLISION = False
DEPUTY_BODY_HALF_EXTENTS_M = np.array([0.30, 0.30, 0.40])
ENABLE_COUPLED_CONTACT_DYNAMICS = False
ENABLE_BODY_MOUNTED_CAMERA_FOV = False
ENABLE_KEEP_OUT_AVOIDANCE = False
ENABLE_SPIN_SYNC = False
CHIEF_MASS_KG = 3000.0
SIGMA_V_DOPPLER = 0.005
TERM_NAV_ALPHA   = 0.35
TERM_NAV_BETA    = 0.06
TERM_NAV_VMAX_MS = 0.10
TERM_NAV_GATE_M  = 0.25
PORT_TRACK_ALPHA  = 0.40
PORT_TRACK_GATE_M = 0.25
ECLIPSE_NU_MIN  = 0.1
DOCK_PORT_BODY  = np.array([0.0, 0.0, 0.5])
DOCK_AXIS_BODY  = np.array([0.0, 0.0, 1.0])
DEP_DOCK_AXIS_BODY = np.array([0.0, 0.0, 1.0])
MU_GEO          = 3.986004418e14
N_GEO           = np.sqrt(MU_GEO / (CHIEF_A_KM * 1e3)**3)
I_SC            = np.diag([4.167, 4.167, 3.000])
MAIN_TERMINAL_M = 5.0    # main.py override threshold

MC_STRESS_CASES = (
    "nominal",
    "range_dropout",
    "camera_dropout",
    "gyro_bias",
    "high_pose_noise",
    "slow_detumble",
    "weak_thruster",
)

MC_STRESS_WEIGHTS = np.array([0.52, 0.10, 0.10, 0.10, 0.08, 0.05, 0.05])
MC_STRESS_WEIGHTS = MC_STRESS_WEIGHTS / np.sum(MC_STRESS_WEIGHTS)

STRESS_PROFILES = {
    "nominal": {
        "thrust_scale": 1.00,
        "sc_omega_scale": 1.00,
        "port_noise_scale": 1.00,
        "gyro_bias_rad_s": np.zeros(3),
        "range_dropout_s": None,
        "camera_dropout_s": None,
    },
    "range_dropout": {
        "thrust_scale": 1.00,
        "sc_omega_scale": 1.00,
        "port_noise_scale": 1.00,
        "gyro_bias_rad_s": np.zeros(3),
        "range_dropout_s": (120.0, 420.0),
        "camera_dropout_s": None,
    },
    "camera_dropout": {
        "thrust_scale": 1.00,
        "sc_omega_scale": 1.00,
        "port_noise_scale": 1.00,
        "gyro_bias_rad_s": np.zeros(3),
        "range_dropout_s": None,
        "camera_dropout_s": (15.0, 135.0),
    },
    "gyro_bias": {
        "thrust_scale": 1.00,
        "sc_omega_scale": 1.00,
        "port_noise_scale": 1.00,
        "gyro_bias_rad_s": np.radians(np.array([0.035, -0.020, 0.015])),
        "range_dropout_s": None,
        "camera_dropout_s": None,
    },
    "high_pose_noise": {
        "thrust_scale": 1.00,
        "sc_omega_scale": 1.00,
        "port_noise_scale": 3.00,
        "gyro_bias_rad_s": np.zeros(3),
        "range_dropout_s": None,
        "camera_dropout_s": None,
    },
    "slow_detumble": {
        "thrust_scale": 1.00,
        "sc_omega_scale": 1.60,
        "port_noise_scale": 1.00,
        "gyro_bias_rad_s": np.zeros(3),
        "range_dropout_s": None,
        "camera_dropout_s": None,
    },
    "weak_thruster": {
        "thrust_scale": 0.65,
        "sc_omega_scale": 1.00,
        "port_noise_scale": 1.00,
        "gyro_bias_rad_s": np.zeros(3),
        "range_dropout_s": None,
        "camera_dropout_s": None,
    },
}


# =====================================================================
#  HELPERS
# =====================================================================
def R_eci2lvlh(r, v):
    x = r / np.linalg.norm(r)
    h = np.cross(r, v);  z = h / np.linalg.norm(h)
    return np.vstack([x, np.cross(z, x), z])


def quat_from_rot_matrix(R):
    tr = float(np.trace(R))
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        q = np.array([0.25 * s,
                      (R[2, 1] - R[1, 2]) / s,
                      (R[0, 2] - R[2, 0]) / s,
                      (R[1, 0] - R[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                          (R[0, 1] + R[1, 0]) / s,
                          (R[0, 2] + R[2, 0]) / s])
        elif i == 1:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            q = np.array([(R[0, 2] - R[2, 0]) / s,
                          (R[0, 1] + R[1, 0]) / s, 0.25 * s,
                          (R[1, 2] + R[2, 1]) / s])
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            q = np.array([(R[1, 0] - R[0, 1]) / s,
                          (R[0, 2] + R[2, 0]) / s,
                          (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    q = q / max(np.linalg.norm(q), 1e-12)
    return q if q[0] >= 0.0 else -q


def q_ref_align_axis(q_current, body_axis, desired_axis_eci):
    R_current = rot_matrix(q_current)
    body_axis = body_axis / max(np.linalg.norm(body_axis), 1e-12)
    desired_axis = desired_axis_eci / max(np.linalg.norm(desired_axis_eci), 1e-12)
    current_axis = R_current @ body_axis
    current_axis /= max(np.linalg.norm(current_axis), 1e-12)

    v = np.cross(current_axis, desired_axis)
    c = float(np.clip(np.dot(current_axis, desired_axis), -1.0, 1.0))
    s = float(np.linalg.norm(v))

    if s < 1e-9:
        if c > 0.0:
            R_delta = np.eye(3)
        else:
            tmp = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(tmp, current_axis)) > 0.9:
                tmp = np.array([0.0, 1.0, 0.0])
            axis = np.cross(current_axis, tmp)
            axis /= max(np.linalg.norm(axis), 1e-12)
            K = np.array([[0.0, -axis[2], axis[1]],
                          [axis[2], 0.0, -axis[0]],
                          [-axis[1], axis[0], 0.0]])
            R_delta = np.eye(3) + 2.0 * (K @ K)
    else:
        axis = v / s
        K = np.array([[0.0, -axis[2], axis[1]],
                      [axis[2], 0.0, -axis[0]],
                      [-axis[1], axis[0], 0.0]])
        R_delta = np.eye(3) + K * s + (K @ K) * (1.0 - c)

    return quat_from_rot_matrix(R_delta @ R_current)


def docking_alignment_metrics(R_dep_body_to_lvlh, port_axis_lvlh):
    dep_axis = R_dep_body_to_lvlh @ DEP_DOCK_AXIS_BODY
    dep_axis /= max(np.linalg.norm(dep_axis), 1e-12)
    desired_axis = -port_axis_lvlh
    desired_axis /= max(np.linalg.norm(desired_axis), 1e-12)
    align_deg = float(np.degrees(np.arccos(
        np.clip(np.dot(dep_axis, desired_axis), -1.0, 1.0))))
    return {
        "ok": bool(align_deg <= DOCK_ALIGN_MAX_DEG),
        "align_deg": align_deg,
    }


def docking_geometry_metrics(dep_lvlh, R_body_to_lvlh):
    axis_body = DOCK_AXIS_BODY / max(np.linalg.norm(DOCK_AXIS_BODY), 1e-12)
    dep_body = R_body_to_lvlh.T @ dep_lvlh
    port_to_dep_body = dep_body - DOCK_PORT_BODY
    port_range = float(np.linalg.norm(port_to_dep_body))
    axial = float(np.dot(port_to_dep_body, axis_body))
    lateral_vec = port_to_dep_body - axial * axis_body
    lateral = float(np.linalg.norm(lateral_vec))

    inside_body = bool(np.all(np.abs(dep_body) < CHIEF_BODY_HALF_EXTENTS_M))
    on_dock_face = dep_body[2] > CHIEF_BODY_HALF_EXTENTS_M[2] - DOCK_FACE_TOL_M
    in_aperture = lateral <= DOCK_PORT_APERTURE_M
    body_clear = (not inside_body) or (on_dock_face and in_aperture)
    capture_core = port_range <= DOCK_CONE_MIN_RANGE_M and in_aperture

    if port_range <= 1e-9:
        cone_angle_deg = 0.0
    else:
        cos_ang = np.clip(axial / port_range, -1.0, 1.0)
        cone_angle_deg = float(np.degrees(np.arccos(cos_ang)))

    cone_ok = capture_core or (axial > 0.0
                               and cone_angle_deg <= DOCK_CONE_HALF_ANGLE_DEG)
    cone_error_deg = max(0.0, cone_angle_deg - DOCK_CONE_HALF_ANGLE_DEG)
    if capture_core:
        cone_error_deg = 0.0

    return {
        "ok": bool(body_clear and cone_ok and in_aperture),
        "body_clear": bool(body_clear),
        "cone_ok": bool(cone_ok),
        "in_aperture": bool(in_aperture),
        "capture_core": bool(capture_core),
        "inside_body": inside_body,
        "lateral_m": lateral,
        "axial_m": axial,
        "cone_angle_deg": cone_angle_deg,
        "cone_error_deg": cone_error_deg,
    }


def propagate_full_force(pos, vel, dt_total, t_abs, Cr, Am, substep=60.0):
    J2 = 1.08263e-3;  RE = 6.3781e6
    AU = 1.495978707e11;  P0 = 4.56e-6
    def sun_pos(t):
        d = t/86400.0;  lam = np.radians(280.46 + 360.985647*d)
        eps = np.radians(23.439)
        return AU * np.array([np.cos(lam), np.cos(eps)*np.sin(lam),
                               np.sin(eps)*np.sin(lam)])
    def accel(p, t):
        r = np.linalg.norm(p);  a = -MU_GEO/r**3 * p
        x, y, z = p;  c = -1.5*J2*MU_GEO*RE**2/r**5;  f = 5*z**2/r**2
        a += np.array([c*x*(1-f), c*y*(1-f), c*z*(3-f)])
        sp = sun_pos(t);  rs = np.linalg.norm(sp);  P = P0*(AU/rs)**2
        dr = p - sp;  a += Cr*Am*P*dr/np.linalg.norm(dr)
        return a
    n = max(1, int(round(dt_total/substep)));  h = dt_total/n
    p, v_now, t = pos.copy(), vel.copy(), float(t_abs)
    for _ in range(n):
        k1p=v_now;               k1v=accel(p,t)
        k2p=v_now+.5*h*k1v;     k2v=accel(p+.5*h*k1p,t+.5*h)
        k3p=v_now+.5*h*k2v;     k3v=accel(p+.5*h*k2p,t+.5*h)
        k4p=v_now+h*k3v;         k4v=accel(p+h*k3p,t+h)
        p   += (h/6)*(k1p+2*k2p+2*k3p+k4p)
        v_now += (h/6)*(k1v+2*k2v+2*k3v+k4v)
        t += h
    return p, v_now


def _stress_profile(stress_case):
    return STRESS_PROFILES.get(stress_case, STRESS_PROFILES["nominal"])


def _in_relative_window(t_now, t_ref, window):
    if t_ref is None or window is None:
        return False
    start_s, stop_s = window
    return (t_now >= t_ref + start_s) and (t_now <= t_ref + stop_s)


def _failure_reason(docked, adcs_confirmed, phase2_active, rdv_started,
                    t_term_start, crashed=False):
    if crashed:
        return "CRASH"
    if docked:
        return "DOCKED"
    if not adcs_confirmed:
        return "ADCS_NOT_CONFIRMED"
    if not phase2_active:
        return "RPOD_NOT_ACTIVE"
    if not rdv_started:
        return "RENDEZVOUS_NOT_STARTED"
    if t_term_start is None:
        return "TERMINAL_NOT_REACHED"
    return "DOCK_TIMEOUT"


# =====================================================================
#  SINGLE TRIAL
# =====================================================================
def run_trial(trial_id,
              chief_omega0_deg_s,   # np.ndarray shape (3,)
              chief_M0_deg,         # float
              sc_omega0_rad_s,      # np.ndarray shape (3,)
              rng_seed,             # int for sensor noise
              stress_case="nominal"):
    """Run one full mission simulation. Returns dict of results."""

    rng = np.random.default_rng(rng_seed)
    np.random.seed(rng_seed)
    profile = _stress_profile(stress_case)

    # ── Environment ──────────────────────────────────────────────────
    chief_orbit = GEOOrbitPropagator(
        a_km=CHIEF_A_KM, e=CHIEF_E, i_deg=CHIEF_I_DEG,
        raan_deg=CHIEF_RAAN_DEG, omega_deg=CHIEF_OMEGA_DEG,
        M0_deg=chief_M0_deg,
        Cr=CHI_CR, Am_ratio=CHI_AM)

    mag_field = MagneticField(epoch_year=2025.0)
    gg        = GravityGradient(I_SC)
    srp       = SolarRadiationPressure()

    # ── Spacecraft ───────────────────────────────────────────────────
    sc       = Spacecraft(I_SC)
    sc.omega = sc_omega0_rad_s.copy() * profile["sc_omega_scale"]

    # ── Sensors ──────────────────────────────────────────────────────
    mag_sens     = Magnetometer()
    sun_sens     = SunSensor()
    gyro         = Gyro(dt=DT_INNER, bias_init_max_deg_s=0.05)
    star_tracker = StarTracker(sigma_cross_arcsec=5.0, sigma_roll_arcsec=20.0,
                               sun_excl_deg=30.0, earth_excl_deg=20.0,
                               update_rate_hz=4.0, acquisition_s=30.0)
    rng_sensor   = RangingBearingSensor(sigma_range_m=1.0, sigma_range_frac=0.001,
                                        sigma_angle_rad=np.radians(0.05),
                                        fov_half_deg=60.0, max_range_m=5000.0,
                                        min_range_m=0.05)
    cam_sensor   = CameraSensor(focal_length_px=800.0, image_size_px=(640, 480),
                                sigma_px=1.5, min_range_m=0.05, max_range_m=5000.0)

    # ── Actuators ────────────────────────────────────────────────────
    rw   = ReactionWheel(h_max=4.0)
    mtq  = Magnetorquer(m_max=0.2)
    bdot = BDotController(k_bdot=2e5, m_max=0.2)

    # ── Estimation ───────────────────────────────────────────────────
    quest_alg = QUEST()
    mekf      = MEKF(DT_INNER)
    mekf.P[0:3, 0:3] = np.eye(3) * np.radians(5.0)**2
    th_ekf    = THEKF(a_chief=CHIEF_A_KM*1e3, e_chief=CHIEF_E,
                      dt=DT_OUTER, q_pos=1e-3, q_vel=1e-7)

    # ── Control ──────────────────────────────────────────────────────
    att_ctrl  = AttitudeController(Kp=0.08284, Kd=0.82257)
    q_ref     = np.array([1., 0., 0., 0.])
    rpod_ctrl = GEORPODController(
        mu=MU_GEO, n_chief=N_GEO,
        dep_mass_kg=DEP_MASS_KG,
        dep_thrust_N=DEP_THRUST_N * profile["thrust_scale"],
        Cr_chi=CHI_CR, Am_chi=CHI_AM,
        dock_capture_m=DOCK_RANGE_M,
        ekf=th_ekf, rng_sensor=rng_sensor)
    rpod_ctrl.standoff = np.linalg.norm(FORMATION_OFFSET_M)
    contact_model = DockingContactModel(
        restitution=SOFT_CAPTURE_RESTITUTION,
        tangential_damping=SOFT_CAPTURE_TANGENTIAL_DAMPING,
        capture_vrel_ms=SOFT_CAPTURE_VREL_MS)
    thruster_layout = ThrusterLayout.box_16(
        half_extents_m=(0.30, 0.30, 0.40),
        max_force_n=THRUSTER_MAX_FORCE_N)
    body_pair = FiniteBodyPair(
        chief_body=BoxBody(CHIEF_BODY_HALF_EXTENTS_M, name="chief"),
        deputy_body=BoxBody(DEPUTY_BODY_HALF_EXTENTS_M, name="deputy"))
    body_camera = BodyMountedCamera(
        boresight_body=DEP_DOCK_AXIS_BODY,
        fov_half_angle_deg=35.0,
        max_range_m=5000.0)
    keepout_planner = KeepoutAvoidancePlanner(
        zones=KeepoutAvoidancePlanner.default_appendage_zones())
    spin_sync = SpinSyncController()

    # ── Chief attitude (varied tumble) ───────────────────────────────
    chief_att = ChiefAttitude(
        omega0_deg_s=chief_omega0_deg_s,
        dock_port_body=DOCK_PORT_BODY.copy(),
        dock_axis_body=DOCK_AXIS_BODY.copy(),
        enable_gg_torque=True)
    chief_pose_est = ChiefPoseEstimator(
        cam_sensor=cam_sensor, dt=DT_OUTER,
        N_avg=50, alpha_filter=0.3, sigma_omega=0.002)

    fsw = ModeManager()
    cw  = CWDynamics(chief_orbit_radius_km=CHIEF_A_KM)
    cw.set_initial_offset(dr_lvlh_m=FORMATION_OFFSET_M)
    cw.dv_total = 0.0    # scalar magnitude accumulator

    # ── State ────────────────────────────────────────────────────────
    t = 0.0
    dep_pos_eci = None
    dep_vel_eci = None
    mekf_seeded = False
    last_good_q = None;  last_good_t = -999.0
    adcs_stable_cnt = 0;  triad_err_deg = None
    adcs_confirmed = False;  adcs_conf_t = None
    phase2_active  = False;  rdv_started  = False
    docked         = False;  ekf_coast_active = False
    hard_capture_hold_s = 0.0
    soft_capture_seen = False
    soft_capture_t = None
    q_cmd_at_soft_capture = None
    soft_capture_align_entry_deg = np.nan
    soft_capture_align_min_deg = np.inf
    capture_timeout = False
    capture_timeout_detail = "NONE"
    max_capture_hold_s = 0.0
    thruster_torque_body_pending = np.zeros(3)
    last_capture_diag = {
        "align_deg": np.nan,
        "cone_err_deg": np.nan,
        "cone_angle_deg": np.nan,
        "lateral_m": np.nan,
        "geometry_ok": False,
        "soft_stable": False,
        "soft_certified": False,
        "hard_strict": False,
    }
    th_ekf_pos_prev = np.zeros(3)
    phase2_start_t = None
    range_drop_ticks = 0
    camera_drop_ticks = 0
    max_nav_err_m = 0.0
    min_port_range_m = np.inf
    final_range_m = np.nan
    final_port_range_m = np.nan
    final_port_vrel_ms = np.nan

    def apply_gyro_stress(omega, t_now):
        if t_now >= 30.0:
            return omega + profile["gyro_bias_rad_s"]
        return omega

    # Per-trial DV split tracking
    dv_at_prox_start  = 0.0
    dv_at_term_start  = 0.0
    t_prox_start      = None
    t_term_start      = None

    # ── Main loop ────────────────────────────────────────────────────
    while t < T_SIM_MAX and not docked:

        chi_pos_m_prev  = chief_orbit.pos * 1e3
        chi_vel_ms_prev = chief_orbit.vel * 1e3
        chi_pos_km, chi_vel_kms = chief_orbit.step(DT_OUTER)
        chi_pos_m  = chi_pos_km * 1e3
        chi_vel_ms = chi_vel_kms * 1e3
        chief_att.step(DT_OUTER, chi_pos_m)

        if dep_pos_eci is None:
            R_l2e = R_eci2lvlh(chi_pos_m, chi_vel_ms).T
            dep_pos_eci = chi_pos_m + R_l2e @ FORMATION_OFFSET_M
            dv_ic = np.array([0., -2.0*N_GEO*FORMATION_OFFSET_M[0], 0.])
            dep_vel_eci = chi_vel_ms + R_l2e @ dv_ic

        nu_eclipse = eclipse_nu(chi_pos_km, chief_orbit.t_elapsed)
        in_eclipse = nu_eclipse < ECLIPSE_NU_MIN
        sun_I      = chief_orbit.get_sun_vector_eci()
        sun_pos_km = sun_I * 1.496e8
        B_I        = mag_field.get_field(chi_pos_km)
        T_gg       = gg.compute(chi_pos_km, sc.q)
        T_srp, _   = srp.compute(sc.q, sun_I, pos_km=chi_pos_km, sun_pos_km=sun_pos_km)
        T_srp     *= nu_eclipse
        disturbance = T_gg + T_srp

        B_meas     = mag_sens.measure(sc.q, B_I)
        sun_meas   = sun_sens.measure(sc.q, sun_I) if not in_eclipse else np.zeros(3)
        omega_meas = apply_gyro_stress(gyro.measure(sc.omega), t)

        if fsw.is_sun_acquiring:
            nadir_I = QUEST.nadir_inertial(chi_pos_km)
            nadir_b = QUEST.nadir_body_from_earth_sensor(chi_pos_km, sc.q)
            if in_eclipse:
                q_quest, q_qual = quest_alg.compute_multi(
                    [B_meas, nadir_b], [B_I, nadir_I], [0.85, 0.15])
            else:
                q_quest, q_qual = quest_alg.compute_multi(
                    [B_meas, sun_meas, nadir_b], [B_I, sun_I, nadir_I],
                    [0.70, 0.20, 0.10])
            if q_quest[0] < 0: q_quest = -q_quest
            if q_qual > 0.01:
                last_good_q = q_quest.copy();  last_good_t = t;  triad_err_deg = 5.0
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

        mode = fsw.update(t, sc.omega, rw.h,
                          triad_err_deg=triad_err_deg,
                          pointing_err_deg=(
                              float(np.degrees(2*np.linalg.norm(
                                  quat_error(sc.q, mekf.q)[1:])))
                              if mekf_seeded else None))

        if mode == Mode.FINE_POINTING and not mekf_seeded:
            seed = last_good_q.copy() if last_good_q is not None else sc.q.copy()
            if seed[0] < 0: seed = -seed
            mekf.q = seed
            mekf.P[0:3, 0:3] = np.eye(3) * np.radians(5.0)**2
            mekf_seeded = True

        if (mekf_seeded and not adcs_confirmed
                and mode in (Mode.FINE_POINTING, Mode.MOMENTUM_DUMP)):
            qe = quat_error(sc.q, mekf.q)
            if qe[0] < 0: qe = -qe
            err_deg = float(np.degrees(2.0*np.linalg.norm(qe[1:])))
            if mode == Mode.FINE_POINTING:
                adcs_stable_cnt = adcs_stable_cnt+1 if err_deg < ADCS_STABLE_DEG else 0
            if adcs_stable_cnt >= ADCS_STABLE_SUST:
                adcs_confirmed = True;  adcs_conf_t = t
        elif mode not in (Mode.FINE_POINTING, Mode.MOMENTUM_DUMP):
            adcs_stable_cnt = 0

        # ── ADCS actuators ───────────────────────────────────────────
        if mode == Mode.SAFE_MODE:
            sc.step(np.zeros(3), disturbance, DT_OUTER)
        elif mode in (Mode.DETUMBLE, Mode.SUN_ACQUISITION):
            K_damp  = 0.9549
            tau_cmd = np.clip(-K_damp*sc.omega, -0.30, 0.30)
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
                    q_fix, _ = quest_alg.compute_multi(vb, vi, [0.70, 0.20, 0.10])
                    if q_fix[0] < 0: q_fix = -q_fix
                    mekf.q = q_fix.copy()
            for _ in range(N_INNER):
                oi = apply_gyro_stress(gyro.measure(sc.omega), t)
                mekf.predict(oi)
                mekf.update_vector(B_meas, B_I, mekf.R_mag)
                if not in_eclipse:
                    mekf.update_vector(sun_meas, sun_I, mekf.R_sun)
                q_st, R_st, st_ok = star_tracker.measure(sc.q, sun_I, chi_pos_m, t)
                if st_ok: mekf.update_star_tracker(q_st, R_st)
                omega_est = sc.omega - mekf.bias
                if mode == Mode.MOMENTUM_DUMP:
                    rw.h    = rw.h * 0.9995
                    rw.h    = np.clip(rw.h, -rw.h_max, rw.h_max)
                    tau_rw  = np.zeros(3)
                else:
                    q_cmd = q_ref
                    if rpod_ctrl.mode == RPODMode.TERMINAL:
                        q_cmd = q_ref_align_axis(
                            mekf.q, DEP_DOCK_AXIS_BODY, -chief_att.dock_axis_eci())
                    elif rpod_ctrl.mode == RPODMode.SOFT_CAPTURE:
                        q_cmd = (q_cmd_at_soft_capture
                                 if q_cmd_at_soft_capture is not None
                                 else q_ref_align_axis(
                                     mekf.q, DEP_DOCK_AXIS_BODY,
                                     -chief_att.dock_axis_eci()))
                    omega_for_ctrl = omega_est
                    if ENABLE_SPIN_SYNC and rpod_ctrl.mode in (RPODMode.TERMINAL,
                                                                RPODMode.SOFT_CAPTURE):
                        omega_chief_lvlh = (R_e2l @ rot_matrix(chief_att.quaternion)
                                            @ chief_att.omega_body)
                        omega_sync_body = spin_sync.compute_rate_command(
                            omega_chief_lvlh, (R_e2l @ rot_matrix(sc.q)).T)
                        omega_for_ctrl = omega_est - omega_sync_body
                    tau_rw, _ = att_ctrl.compute(mekf.q, omega_for_ctrl, q_cmd)
                    if rpod_ctrl.mode == RPODMode.SOFT_CAPTURE:
                        tau_rw *= SOFT_CAPTURE_ATTITUDE_TORQUE_SCALE
                    rw.apply_torque(tau_rw, DT_INNER)
                    rw.h = np.clip(rw.h, -rw.h_max, rw.h_max)
                sc.step(np.zeros(3), disturbance + thruster_torque_body_pending,
                        DT_INNER,
                        tau_rw=tau_rw, h_rw=rw.h.copy())

        # ── Deputy propagation (phase 1) ─────────────────────────────
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

        # ── Phase 2 activation ───────────────────────────────────────
        if adcs_confirmed and not phase2_active and t >= adcs_conf_t + FORM_HOLD_SETTLE_S:
            phase2_active = True
            phase2_start_t = t
            cw.dv_total   = 0.0   # rendezvous DV budget starts here
            ok = th_ekf.reinit_from_measurements(
                rng_sensor, cw.state[:3], n_avg=10, P_pos_m=2.0, P_vel_ms=0.001)
            if not ok:
                th_ekf.initialise(x0=cw.state.copy(), nu0=0.0)
            R_e2l = R_eci2lvlh(chi_pos_m_prev, chi_vel_ms_prev)
            th_ekf.x[3:6] = R_e2l @ (dep_vel_eci - chi_vel_ms_prev)
            th_ekf.P[3:6, 3:6] = np.eye(3) * (0.001**2)

        # ── Phase 2 RPOD ─────────────────────────────────────────────
        if phase2_active:
            R_e2l       = R_eci2lvlh(chi_pos_m_prev, chi_vel_ms_prev)
            true_cw_pos = R_e2l @ (dep_pos_eci - chi_pos_m_prev)
            true_cw_vel = R_e2l @ (dep_vel_eci - chi_vel_ms_prev)
            true_cw     = np.concatenate([true_cw_pos, true_cw_vel])
            cw.state    = true_cw
            ekf_lvlh    = np.concatenate([th_ekf.position, th_ekf.velocity])
            range_blocked = _in_relative_window(
                t, phase2_start_t, profile["range_dropout_s"])
            camera_blocked = _in_relative_window(
                t, t_term_start, profile["camera_dropout_s"])
            if range_blocked:
                range_drop_ticks += 1
            if camera_blocked:
                camera_drop_ticks += 1
            max_nav_err_m = max(
                max_nav_err_m,
                float(np.linalg.norm(th_ekf.x[0:3] - true_cw_pos)))

            # Lambert trigger
            if not rdv_started and t >= adcs_conf_t + 2*FORM_HOLD_SETTLE_S:
                ekf_dir_seed = (th_ekf.x[0:3] /
                                max(np.linalg.norm(th_ekf.x[0:3]), 1.0))
                if range_blocked:
                    z_seed, R_seed = None, None
                else:
                    z_seed, R_seed = rng_sensor.measure(true_cw_pos, ekf_dir_seed)
                if z_seed is not None:
                    pos_seed = rng_sensor.invert(z_seed)
                    th_ekf.initialise(
                        x0=np.concatenate([pos_seed, np.zeros(3)]),
                        P0=np.diag([R_seed[0,0]]*3 + [0.001**2]*3),
                        nu0=th_ekf.nu)
                else:
                    th_ekf.initialise(
                        x0=np.concatenate([true_cw_pos, np.zeros(3)]),
                        P0=np.diag([4.0]*3 + [0.001**2]*3), nu0=th_ekf.nu)
                boresight_seed = (th_ekf.x[0:3] /
                                  max(np.linalg.norm(th_ekf.x[0:3]), 1.0))
                for _ in range(20):
                    if range_blocked:
                        z_w, R_w = None, None
                    else:
                        z_w, R_w = rng_sensor.measure(true_cw_pos, boresight_seed)
                    if z_w is not None:
                        th_ekf.predict(np.zeros(3))
                        th_ekf.update(z_w, R_w, gate_k=50.0)
                ekf_lvlh = np.concatenate([th_ekf.position, th_ekf.velocity])
                rdv_started = True
                rpod_ctrl.standoff = max(50.0, abs(true_cw_pos[1]))
                rpod_ctrl.start_rendezvous(t, truth_range=cw.range_m)
                dv_at_prox_start = cw.dv_total
                t_prox_start     = t

            # Terminal alpha-beta velocity (EKF velocity elsewhere)
            ekf_pos_now = th_ekf.x[0:3].copy()
            _use_term_nav = (rpod_ctrl.mode in (RPODMode.TERMINAL, RPODMode.SOFT_CAPTURE))
            if _use_term_nav:
                if not hasattr(rpod_ctrl, '_terminal_nav'):
                    rpod_ctrl._terminal_nav = TerminalNavFilter(
                        alpha=TERM_NAV_ALPHA,
                        beta=TERM_NAV_BETA,
                        v_max_ms=TERM_NAV_VMAX_MS,
                        innovation_gate_m=TERM_NAV_GATE_M)
                guidance_pos, guidance_vel = rpod_ctrl._terminal_nav.update(
                    ekf_pos_now, DT_OUTER,
                    measurement_valid=not (cam_sensor.is_lost or camera_blocked),
                    vel_seed=th_ekf.x[3:6])
            else:
                if hasattr(rpod_ctrl, '_terminal_nav'):
                    rpod_ctrl._terminal_nav.reset()
                guidance_pos = th_ekf.x[0:3].copy()
                guidance_vel = th_ekf.x[3:6].copy()
            ekf_lvlh_guided = np.concatenate([guidance_pos, guidance_vel])

            # Chief pose
            omega_est_body, omega_est_valid = chief_pose_est.update(
                dr_lvlh=th_ekf.x[0:3], q_chief=chief_att.quaternion)
            omega_est_lvlh = (R_e2l @ omega_est_body
                              if omega_est_valid else np.zeros(3))

            # Port position
            R_est_b2l = chief_pose_est.R_body2lvlh
            _nav_range_for_port = float(np.linalg.norm(th_ekf.x[0:3]))
            _direct_port_visible = (rpod_ctrl.mode in (RPODMode.TERMINAL, RPODMode.SOFT_CAPTURE)
                                    and _nav_range_for_port < 5.0
                                    and not cam_sensor.is_lost
                                    and not camera_blocked)
            if _direct_port_visible:
                port_eci_meas = chief_att.dock_port_eci(chi_pos_m_prev)
                port_lvlh_true = R_e2l @ (port_eci_meas - chi_pos_m_prev)
                port_sigma_m = (max(0.01, 0.002 * _nav_range_for_port)
                                * profile["port_noise_scale"])
                port_meas = port_lvlh_true + rng.normal(0.0, port_sigma_m, 3)
                if not hasattr(rpod_ctrl, '_port_tracker'):
                    rpod_ctrl._port_tracker = PortTracker(
                        alpha=PORT_TRACK_ALPHA,
                        innovation_gate_m=PORT_TRACK_GATE_M)
                port_lvlh_ctrl, _ = rpod_ctrl._port_tracker.update(
                    port_meas, DT_OUTER, measurement_valid=True)
                port_axis_lvlh = R_e2l @ chief_att.dock_axis_eci()
                r_arm_lvlh     = port_lvlh_ctrl
            elif R_est_b2l is not None and omega_est_valid:
                if hasattr(rpod_ctrl, '_port_tracker'):
                    rpod_ctrl._port_tracker.update(
                        np.zeros(3), DT_OUTER, measurement_valid=False)
                port_lvlh_ctrl = R_est_b2l @ DOCK_PORT_BODY
                port_axis_lvlh = R_est_b2l @ DOCK_AXIS_BODY
                r_arm_lvlh     = port_lvlh_ctrl
            else:
                if hasattr(rpod_ctrl, '_port_tracker'):
                    rpod_ctrl._port_tracker.update(
                        np.zeros(3), DT_OUTER, measurement_valid=False)
                port_lvlh_ctrl = np.zeros(3)
                port_axis_lvlh = np.zeros(3)
                r_arm_lvlh     = np.zeros(3)

            port_vel_lvlh = np.cross(omega_est_lvlh, r_arm_lvlh)
            ekf_aug   = np.concatenate([ekf_lvlh_guided, port_vel_lvlh])
            if np.linalg.norm(port_axis_lvlh) > 1e-9 and mekf_seeded:
                R_dep_est_b2l = R_e2l @ rot_matrix(mekf.q)
                attitude_align_deg_cmd = docking_alignment_metrics(
                    R_dep_est_b2l, port_axis_lvlh)["align_deg"]
            else:
                attitude_align_deg_cmd = None

            # TERMINAL override (main.py fix — unchanged)
            _nav_range_now = float(np.linalg.norm(th_ekf.x[0:3]))
            if (rdv_started
                    and rpod_ctrl.mode == RPODMode.PROX_OPS
                    and _nav_range_now < MAIN_TERMINAL_M):
                rpod_ctrl._set_mode(RPODMode.TERMINAL, t)
                if t_term_start is None:
                    dv_at_term_start = cw.dv_total
                    t_term_start     = t

            accel_cmd, impulse_dv = rpod_ctrl.compute(
                ekf_lvlh=ekf_aug,
                chi_pos_eci=chi_pos_m_prev,
                chi_vel_eci=chi_vel_ms_prev,
                t=t,
                true_cw=None,
                port_lvlh=port_lvlh_ctrl,
                cam_lost=(cam_sensor.is_lost or camera_blocked),
                port_axis_lvlh=port_axis_lvlh,
                attitude_align_deg=attitude_align_deg_cmd)
            if ENABLE_KEEP_OUT_AVOIDANCE:
                keepout = keepout_planner.compute(
                    true_cw_pos, R_e2l @ rot_matrix(chief_att.quaternion))
                accel_cmd = accel_cmd + keepout["accel"]

            ekf_coast_active = (rpod_ctrl.mode == RPODMode.LAMBERT
                                and rpod_ctrl._lam_active)

            # Track when TERMINAL first entered (for DV split)
            if (rpod_ctrl.mode in (RPODMode.TERMINAL, RPODMode.SOFT_CAPTURE)
                    and t_term_start is None):
                dv_at_term_start = cw.dv_total
                t_term_start     = t

            # Apply impulse
            if impulse_dv is not None and np.linalg.norm(impulse_dv) > 1e-9:
                R_l2e = R_e2l.T
                dep_vel_eci += R_l2e @ impulse_dv
                cw.dv_total += float(np.linalg.norm(impulse_dv))
                R_e2l       = R_eci2lvlh(chi_pos_m_prev, chi_vel_ms_prev)
                true_cw_pos = R_e2l @ (dep_pos_eci - chi_pos_m_prev)
                true_cw_vel = R_e2l @ (dep_vel_eci - chi_vel_ms_prev)
                true_cw     = np.concatenate([true_cw_pos, true_cw_vel])
                cw.state    = true_cw
                th_ekf.x[0:3] = R_e2l @ (dep_pos_eci - chi_pos_m_prev)
                th_ekf.x[3:6] = R_e2l @ (dep_vel_eci - chi_vel_ms_prev)
                th_ekf.P = np.diag([4.0]*3 + [0.001**2]*3)

            # Apply continuous accel
            if np.any(accel_cmd != 0):
                R_l2e = R_e2l.T
                if ENABLE_PHYSICAL_THRUSTER_LAYOUT:
                    R_body_to_lvlh = R_e2l @ rot_matrix(sc.q)
                    force_req_body = R_body_to_lvlh.T @ (DEP_MASS_KG * accel_cmd)
                    allocation = thruster_layout.allocate(force_req_body)
                    accel_applied = R_body_to_lvlh @ allocation.force_body_n / DEP_MASS_KG
                    thruster_torque_body_pending = allocation.torque_body_nm
                else:
                    accel_applied = accel_cmd
                    thruster_torque_body_pending = np.zeros(3)
                dep_vel_eci += R_l2e @ accel_applied * DT_OUTER
                cw.dv_total += float(np.linalg.norm(accel_applied)) * DT_OUTER
            else:
                thruster_torque_body_pending = np.zeros(3)

            # Propagate deputy
            dep_pos_eci, dep_vel_eci = propagate_full_force(
                dep_pos_eci, dep_vel_eci,
                DT_OUTER, chief_orbit.t_elapsed - DT_OUTER,
                DEP_CR, DEP_AM)
            R_e2l       = R_eci2lvlh(chi_pos_m, chi_vel_ms)
            true_cw_pos = R_e2l @ (dep_pos_eci - chi_pos_m)
            true_cw_vel = R_e2l @ (dep_vel_eci - chi_vel_ms)
            true_cw     = np.concatenate([true_cw_pos, true_cw_vel])
            cw.state    = true_cw

            # EKF predict + update
            truth_rng = np.linalg.norm(true_cw_pos)
            ekf_rng   = np.linalg.norm(th_ekf.x[0:3])
            boresight = (th_ekf.x[0:3]/ekf_rng if ekf_rng > 0.01
                         else np.array([0., -1., 0.]))

            if ekf_coast_active:
                if range_blocked:
                    z_c = None
                else:
                    z_c, _ = rng_sensor.measure(true_cw_pos, boresight)
                if z_c is not None:
                    th_ekf.x[0:3] = rng_sensor.invert(z_c)
                v_dop, sig_v = rng_sensor.measure_doppler(
                    dr_lvlh=true_cw_pos, dv_lvlh=true_cw_vel,
                    pos_est_ekf=th_ekf.x[0:3], dt=DT_OUTER)
                ekf_r_c = np.linalg.norm(th_ekf.x[0:3])
                if ekf_r_c > 0.01:
                    rh = th_ekf.x[0:3]/ekf_r_c
                    th_ekf.update_velocity_doppler(
                        float(np.dot(v_dop, rh)), rh, SIGMA_V_DOPPLER)
                th_ekf.P[0:3, 0:3] = np.eye(3)*4.0
            else:
                _in_term = (rpod_ctrl.mode in (RPODMode.TERMINAL, RPODMode.SOFT_CAPTURE))
                if not _in_term:
                    th_ekf.predict(accel_cmd)
                R_dep_body_to_lvlh_nav = R_e2l @ rot_matrix(sc.q)
                cam_view = body_camera.visibility(
                    target_from_deputy_world=-true_cw_pos,
                    R_body_to_world=R_dep_body_to_lvlh_nav)
                if camera_blocked or (ENABLE_BODY_MOUNTED_CAMERA_FOV
                                      and not cam_view["visible"]):
                    z_cam, R_cam = None, None
                    if ENABLE_BODY_MOUNTED_CAMERA_FOV and not cam_view["visible"]:
                        cam_sensor._mark(False)
                else:
                    z_cam, R_cam = cam_sensor.measure(
                        true_cw_pos, q_chief=chief_att.quaternion)
                if z_cam is not None:
                    if _in_term:
                        th_ekf.x[0:3] = z_cam
                        th_ekf.x[3:6] = np.zeros(3)
                        th_ekf.P[0:3, 0:3] = R_cam
                        th_ekf.P[3:6, 3:6] = np.eye(3)*(0.005**2)
                        th_ekf.P[0:3, 3:6] = np.zeros((3, 3))
                        th_ekf.P[3:6, 0:3] = np.zeros((3, 3))
                    else:
                        th_ekf.update_position(z_cam, R_cam, gate_k=5.0)
                        if np.linalg.norm(th_ekf.x[0:3]-z_cam) > 2.0:
                            th_ekf.x[0:3] = z_cam
                            th_ekf.P[0:3, 0:3] = R_cam*9.0
                if not _in_term:
                    v_dop, sig_v = rng_sensor.measure_doppler(
                        dr_lvlh=true_cw_pos, dv_lvlh=true_cw_vel,
                        pos_est_ekf=th_ekf.x[0:3], dt=DT_OUTER)
                    ekf_r_p = np.linalg.norm(th_ekf.x[0:3])
                    if ekf_r_p > 0.01:
                        rh = th_ekf.x[0:3]/ekf_r_p
                        th_ekf.update_velocity_doppler(
                            float(np.dot(v_dop, rh)), rh, SIGMA_V_DOPPLER)

            # Docking check
            port_eci_d  = chief_att.dock_port_eci(chi_pos_m)
            port_lvlh_d = R_e2l @ (port_eci_d - chi_pos_m)
            port_vel_d  = np.cross(omega_est_lvlh, port_lvlh_d)
            dep_to_port = true_cw_pos - port_lvlh_d
            rel_vel_p   = true_cw_vel - port_vel_d
            if (rpod_ctrl.mode == RPODMode.SOFT_CAPTURE
                    and not ENABLE_COUPLED_CONTACT_DYNAMICS):
                latch_pos_dv_lvlh, latch_dv_lvlh, _ = contact_model.ideal_latch(
                    dep_to_port, rel_vel_p, DEP_MASS_KG)
                dep_pos_eci += R_e2l.T @ latch_pos_dv_lvlh
                true_cw_pos += latch_pos_dv_lvlh
                dep_to_port = np.zeros(3)
                dep_vel_eci += R_e2l.T @ latch_dv_lvlh
                true_cw_vel += latch_dv_lvlh
                rel_vel_p = np.zeros(3)
            port_range  = np.linalg.norm(dep_to_port)
            port_vrel   = np.linalg.norm(rel_vel_p)
            final_range_m = float(np.linalg.norm(true_cw_pos))
            final_port_range_m = float(port_range)
            final_port_vrel_ms = float(port_vrel)
            min_port_range_m = min(min_port_range_m, final_port_range_m)
            R_body_to_lvlh_d = R_e2l @ rot_matrix(chief_att.quaternion)
            dock_geom = docking_geometry_metrics(true_cw_pos, R_body_to_lvlh_d)
            port_axis_lvlh_d = R_body_to_lvlh_d @ DOCK_AXIS_BODY
            R_dep_body_to_lvlh_d = R_e2l @ rot_matrix(sc.q)
            align_geom = docking_alignment_metrics(
                R_dep_body_to_lvlh_d, port_axis_lvlh_d)
            finite_body = body_pair.clearance(
                chief_com_world=np.zeros(3),
                R_chief_to_world=R_body_to_lvlh_d,
                deputy_com_world=true_cw_pos,
                R_deputy_to_world=R_dep_body_to_lvlh_d)
            finite_body_ok = ((not ENABLE_FINITE_BODY_COLLISION)
                              or not finite_body["collision"])
            soft_core_ready = (dock_geom["capture_core"]
                               and align_geom["align_deg"] <= SOFT_CAPTURE_CORE_ALIGN_MAX_DEG)
            # Match main.py: soft capture is the compliant contact latch.
            # Strict cone/alignment certification is diagnostic here; do not
            # let it block a translationally clean port capture.
            soft_capture_ready = (port_range < SOFT_CAPTURE_RANGE_M
                                  and port_vrel < SOFT_CAPTURE_VREL_MS
                                  and finite_body_ok
                                  and align_geom.get("align_deg", 0.0)
                                  < SOFT_CAPTURE_ENTRY_ALIGN_MAX_DEG)
            hard_capture_ready = (port_range < HARD_CAPTURE_RANGE_M
                                  and port_vrel < HARD_CAPTURE_VREL_MS
                                  and dock_geom["ok"]
                                  and finite_body_ok
                                  and align_geom["ok"])

            if rpod_ctrl.mode == RPODMode.TERMINAL and soft_capture_ready:
                if port_range > 1e-6:
                    n_contact = dep_to_port / port_range
                else:
                    n_contact = port_axis_lvlh_d / max(
                        np.linalg.norm(port_axis_lvlh_d), 1e-12)
                if ENABLE_COUPLED_CONTACT_DYNAMICS:
                    contact = contact_model.resolve_coupled(
                        rel_vel_p, n_contact,
                        deputy_mass_kg=DEP_MASS_KG,
                        chief_mass_kg=CHIEF_MASS_KG,
                        deputy_I_body=I_SC,
                        chief_I_body=chief_att.I,
                        r_dep_contact_body=DEP_DOCK_AXIS_BODY * DEPUTY_BODY_HALF_EXTENTS_M[2],
                        r_chief_contact_body=DOCK_PORT_BODY,
                        R_dep_body_to_world=R_dep_body_to_lvlh_d,
                        R_chief_body_to_world=R_body_to_lvlh_d)
                    sc.omega += contact.deputy_delta_omega
                    chief_att.omega += contact.chief_delta_omega
                else:
                    # Idealized baseline latch: remove all port-relative
                    # motion. Restitution belongs to coupled contact dynamics.
                    _, _, contact = contact_model.ideal_latch(
                        dep_to_port, rel_vel_p, DEP_MASS_KG)
                impact_dv_lvlh = contact.rel_vel_after - rel_vel_p
                dep_vel_eci += R_e2l.T @ impact_dv_lvlh
                true_cw_vel += impact_dv_lvlh
                rpod_ctrl._set_mode(RPODMode.SOFT_CAPTURE, t)
                hard_capture_hold_s = 0.0
                soft_capture_seen = True
                if soft_capture_t is None:
                    soft_capture_t = t
                    soft_capture_align_entry_deg = align_geom["align_deg"]
                    soft_capture_align_min_deg = align_geom["align_deg"]
                    q_cmd_at_soft_capture = q_ref_align_axis(
                        mekf.q, DEP_DOCK_AXIS_BODY, -chief_att.dock_axis_eci())

            soft_capture_stable = (port_range < SOFT_CAPTURE_RANGE_M
                                   and port_vrel < SOFT_CAPTURE_LATCH_VREL_MS
                                   and finite_body_ok)
            if rpod_ctrl.mode == RPODMode.SOFT_CAPTURE:
                soft_capture_align_min_deg = min(soft_capture_align_min_deg,
                                                 align_geom["align_deg"])
            soft_capture_certified = (soft_capture_stable
                                      and soft_core_ready
                                      and align_geom["align_deg"] <= DOCK_ALIGN_MAX_DEG)
            capture_hold_ready = (rpod_ctrl.mode == RPODMode.SOFT_CAPTURE
                                  and (hard_capture_ready or soft_capture_certified))
            last_capture_diag = {
                "align_deg": align_geom["align_deg"],
                "cone_err_deg": dock_geom["cone_error_deg"],
                "cone_angle_deg": dock_geom["cone_angle_deg"],
                "lateral_m": dock_geom["lateral_m"],
                "geometry_ok": dock_geom["ok"],
                "soft_stable": soft_capture_stable,
                "soft_certified": soft_capture_certified,
                "hard_strict": hard_capture_ready,
                "soft_align_entry_deg": soft_capture_align_entry_deg,
                "soft_align_min_deg": soft_capture_align_min_deg,
            }

            if rpod_ctrl.mode == RPODMode.SOFT_CAPTURE and capture_hold_ready:
                hard_capture_hold_s += DT_OUTER
                max_capture_hold_s = max(max_capture_hold_s,
                                         hard_capture_hold_s)
            else:
                hard_capture_hold_s = 0.0

            if (rpod_ctrl.mode == RPODMode.SOFT_CAPTURE
                    and soft_capture_t is not None
                    and t - soft_capture_t > SOFT_CAPTURE_MAX_HOLD_S
                    and not (hard_capture_ready or soft_capture_certified)):
                capture_timeout = True
                if not soft_capture_stable:
                    capture_timeout_detail = "ESCAPED_SOFT_CAPTURE"
                elif align_geom["align_deg"] > DOCK_ALIGN_MAX_DEG:
                    capture_timeout_detail = "SOFT_ALIGN_TIMEOUT"
                elif not dock_geom["ok"]:
                    capture_timeout_detail = "BAD_DOCKING_GEOMETRY"
                else:
                    capture_timeout_detail = "CAPTURE_TIMEOUT"
                break

            if (rpod_ctrl.mode == RPODMode.SOFT_CAPTURE
                    and hard_capture_hold_s >= HARD_CAPTURE_HOLD_S):
                docked = True
                rpod_ctrl._set_mode(RPODMode.DOCKING, t)
                break

        t += DT_OUTER

    # ── Collect results ──────────────────────────────────────────────
    Isp = 220.0; g0 = 9.81
    total_dv = float(cw.dv_total)
    dm = DEP_MASS_KG * (1 - np.exp(-total_dv / (Isp*g0)))

    prox_dv = (dv_at_term_start - dv_at_prox_start
               if t_term_start is not None and t_prox_start is not None
               else total_dv)
    term_dv = (total_dv - dv_at_term_start
               if t_term_start is not None else 0.0)
    if not np.isfinite(min_port_range_m):
        min_port_range_m = np.nan
    failure_reason = _failure_reason(
        docked, adcs_confirmed, phase2_active, rdv_started, t_term_start)
    if capture_timeout:
        failure_reason = "CAPTURE_TIMEOUT"
    elif failure_reason == "DOCK_TIMEOUT" and soft_capture_seen:
        failure_reason = "CAPTURE_TIMEOUT"

    return {
        "trial":          trial_id,
        "stress_case":    stress_case,
        "failure_reason": failure_reason,
        "docked":         docked,
        "total_dv_ms":    total_dv,        # m/s
        "propellant_g":   dm * 1000,       # g
        "t_dock_s":       t if docked else None,
        "t_dock_hr":      t/3600 if docked else None,
        "t_adcs_s":       adcs_conf_t,
        "prox_dv_ms":     prox_dv,         # m/s
        "term_dv_ms":     term_dv,         # m/s
        "chief_omega_dps": float(np.linalg.norm(chief_omega0_deg_s)),
        "chief_M0_deg":   chief_M0_deg,
        "sc_omega_rads":  float(np.linalg.norm(sc_omega0_rad_s)
                                * profile["sc_omega_scale"]),
        "final_range_m":  final_range_m,
        "final_port_range_m": final_port_range_m,
        "final_port_vrel_ms": final_port_vrel_ms,
        "min_port_range_m": min_port_range_m,
        "soft_capture_seen": soft_capture_seen,
        "capture_timeout": capture_timeout,
        "capture_timeout_detail": capture_timeout_detail,
        "soft_capture_t_hr": (soft_capture_t / 3600.0
                              if soft_capture_t is not None else np.nan),
        "soft_capture_align_entry_deg": float(last_capture_diag.get("soft_align_entry_deg", np.nan)),
        "soft_capture_align_min_deg": float(last_capture_diag.get("soft_align_min_deg", np.nan)),
        "max_capture_hold_s": max_capture_hold_s,
        "final_align_deg": float(last_capture_diag["align_deg"]),
        "final_align_flip_deg": float(180.0 - last_capture_diag["align_deg"]),
        "final_cone_err_deg": float(last_capture_diag["cone_err_deg"]),
        "final_cone_angle_deg": float(last_capture_diag["cone_angle_deg"]),
        "final_lateral_m": float(last_capture_diag["lateral_m"]),
        "final_geometry_ok": bool(last_capture_diag["geometry_ok"]),
        "final_soft_stable": bool(last_capture_diag["soft_stable"]),
        "final_soft_certified": bool(last_capture_diag.get("soft_certified", False)),
        "final_hard_strict": bool(last_capture_diag["hard_strict"]),
        "max_nav_err_m": max_nav_err_m,
        "range_drop_s": range_drop_ticks * DT_OUTER,
        "camera_drop_s": camera_drop_ticks * DT_OUTER,
    }


# =====================================================================
#  MONTE CARLO RUNNER
# =====================================================================
def run_monte_carlo_serial_legacy(n_trials=30, master_seed=42, n_workers=1):
    rng_master = np.random.default_rng(master_seed)

    # Generate trial parameters
    trial_params = []
    for i in range(n_trials):
        # Chief tumble: random axis on sphere, magnitude uniform 0.05-0.25 deg/s
        omega_mag = rng_master.uniform(0.05, 0.25)
        omega_dir = rng_master.standard_normal(3)
        omega_dir /= np.linalg.norm(omega_dir)
        chief_omega0 = omega_dir * omega_mag

        # Chief M0: uniform 0-360 (changes arrival tumble phase)
        chief_M0 = rng_master.uniform(0, 360)

        # Deputy initial tumble: random axis, magnitude 0.1-0.3 rad/s
        sc_mag  = rng_master.uniform(0.10, 0.30)
        sc_dir  = rng_master.standard_normal(3)
        sc_dir /= np.linalg.norm(sc_dir)
        sc_omega0 = sc_dir * sc_mag

        # Sensor noise seed
        noise_seed = int(rng_master.integers(0, 2**31))

        trial_params.append((i, chief_omega0, chief_M0, sc_omega0, noise_seed))

    results = []
    print(f"\nRunning {n_trials} Monte Carlo trials...")
    print(f"{'Trial':>6} {'Docked':>7} {'DV(m/s)':>9} "
          f"{'T_dock(hr)':>11} {'Prox_DV':>9} {'Term_DV':>9} "
          f"{'Chief_w(d/s)':>13} {'M0(deg)':>8}")
    print("-" * 80)

    for i, chief_omega0, chief_M0, sc_omega0, noise_seed in trial_params:
        t0 = time.time()
        try:
            r = run_trial(i, chief_omega0, chief_M0, sc_omega0, noise_seed)
        except Exception as e:
            print(f"  Trial {i:3d}: CRASHED — {e}")
            traceback.print_exc()
            r = {
                "trial": i, "docked": False,
                "total_dv_ms": np.nan, "propellant_g": np.nan,
                "t_dock_s": None, "t_dock_hr": None, "t_adcs_s": None,
                "prox_dv_ms": np.nan, "term_dv_ms": np.nan,
                "chief_omega_dps": float(np.linalg.norm(chief_omega0)),
                "chief_M0_deg": chief_M0,
                "sc_omega_rads": float(np.linalg.norm(sc_omega0)),
            }
        results.append(r)
        wall = time.time() - t0
        dv_str   = f"{r['total_dv_ms']:.3f}" if r['docked'] else "  --   "
        dock_str = f"{r['t_dock_hr']:.2f}" if r['t_dock_hr'] else "  --  "
        prox_str = f"{r['prox_dv_ms']:.3f}" if r['docked'] else "  --   "
        term_str = f"{r['term_dv_ms']:.3f}" if r['docked'] else "  --   "
        print(f"  {i:3d}    {'YES' if r['docked'] else 'NO ':>5}  "
              f"{dv_str:>8}  {dock_str:>10}  {prox_str:>8}  {term_str:>8}  "
              f"{r['chief_omega_dps']:>11.3f}  {r['chief_M0_deg']:>7.1f}"
              f"  [{wall:.0f}s]")

    return results


def _run_trial_from_params(params):
    i, chief_omega0, chief_M0, sc_omega0, noise_seed, stress_case = params
    try:
        return run_trial(i, chief_omega0, chief_M0, sc_omega0, noise_seed,
                         stress_case=stress_case)
    except Exception as e:
        return {
            "trial": i, "stress_case": stress_case,
            "failure_reason": _failure_reason(
                False, False, False, False, None, crashed=True),
            "docked": False,
            "total_dv_ms": np.nan, "propellant_g": np.nan,
            "t_dock_s": None, "t_dock_hr": None, "t_adcs_s": None,
            "prox_dv_ms": np.nan, "term_dv_ms": np.nan,
            "chief_omega_dps": float(np.linalg.norm(chief_omega0)),
            "chief_M0_deg": chief_M0,
            "sc_omega_rads": float(np.linalg.norm(sc_omega0)),
            "final_range_m": np.nan,
            "final_port_range_m": np.nan,
            "final_port_vrel_ms": np.nan,
            "min_port_range_m": np.nan,
            "soft_capture_seen": False,
            "capture_timeout": False,
            "capture_timeout_detail": "CRASHED",
            "soft_capture_t_hr": np.nan,
            "soft_capture_align_entry_deg": np.nan,
            "soft_capture_align_min_deg": np.nan,
            "max_capture_hold_s": 0.0,
            "final_align_deg": np.nan,
            "final_align_flip_deg": np.nan,
            "final_cone_err_deg": np.nan,
            "final_cone_angle_deg": np.nan,
            "final_lateral_m": np.nan,
            "final_geometry_ok": False,
            "final_soft_stable": False,
            "final_soft_certified": False,
            "final_hard_strict": False,
            "max_nav_err_m": np.nan,
            "range_drop_s": 0.0,
            "camera_drop_s": 0.0,
            "error": repr(e),
            "traceback": traceback.format_exc(),
        }


def _print_trial_result(r, wall):
    dv_str   = f"{r['total_dv_ms']:.3f}" if r['docked'] else "  --   "
    dock_str = f"{r['t_dock_hr']:.2f}" if r['t_dock_hr'] else "  --  "
    prox_str = f"{r['prox_dv_ms']:.3f}" if r['docked'] else "  --   "
    term_str = f"{r['term_dv_ms']:.3f}" if r['docked'] else "  --   "
    print(f"  {r['trial']:3d}    {'YES' if r['docked'] else 'NO ':>5}  "
          f"{dv_str:>8}  {dock_str:>10}  {prox_str:>8}  {term_str:>8}  "
          f"{r['chief_omega_dps']:>11.3f}  {r['chief_M0_deg']:>7.1f}"
          f"  {r.get('stress_case', 'nominal')[:14]:>14}  [{wall:.0f}s]")
    if "error" in r:
        err = str(r["error"]).encode("ascii", "backslashreplace").decode("ascii")
        print(f"         crashed: {err}")


def _draw_stress_case(rng_master, stress_mode, trial_id):
    if stress_mode == "nominal":
        return "nominal"
    if stress_mode == "sweep":
        return MC_STRESS_CASES[trial_id % len(MC_STRESS_CASES)]
    return str(rng_master.choice(MC_STRESS_CASES, p=MC_STRESS_WEIGHTS))


def run_monte_carlo(n_trials=300, master_seed=42, n_workers=8,
                    stress_mode="mixed"):
    rng_master = np.random.default_rng(master_seed)
    trial_params = []
    for i in range(n_trials):
        omega_mag = rng_master.uniform(0.05, 0.25)
        omega_dir = rng_master.standard_normal(3)
        omega_dir /= np.linalg.norm(omega_dir)
        chief_omega0 = omega_dir * omega_mag
        chief_M0 = rng_master.uniform(0, 360)
        sc_mag = rng_master.uniform(0.10, 0.30)
        sc_dir = rng_master.standard_normal(3)
        sc_dir /= np.linalg.norm(sc_dir)
        sc_omega0 = sc_dir * sc_mag
        noise_seed = int(rng_master.integers(0, 2**31))
        stress_case = _draw_stress_case(rng_master, stress_mode, i)
        trial_params.append((i, chief_omega0, chief_M0, sc_omega0,
                             noise_seed, stress_case))

    print(f"\nRunning {n_trials} Monte Carlo trials with {n_workers} worker(s)...")
    print(f"Stress mode: {stress_mode}")
    print(f"{'Trial':>6} {'Docked':>7} {'DV(m/s)':>9} "
          f"{'T_dock(hr)':>11} {'Prox_DV':>9} {'Term_DV':>9} "
          f"{'Chief_w(d/s)':>13} {'M0(deg)':>8} {'Stress':>14}")
    print("-" * 98)

    results = []
    if n_workers <= 1:
        for params in trial_params:
            t0 = time.time()
            r = _run_trial_from_params(params)
            results.append(r)
            _print_trial_result(r, time.time() - t0)
    else:
        start_by_trial = {p[0]: time.time() for p in trial_params}
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            future_map = {ex.submit(_run_trial_from_params, p): p[0]
                          for p in trial_params}
            for fut in as_completed(future_map):
                trial_id = future_map[fut]
                r = fut.result()
                results.append(r)
                _print_trial_result(r, time.time() - start_by_trial[trial_id])

    return sorted(results, key=lambda r: r["trial"])


# =====================================================================
#  STATISTICS + PLOTS
# =====================================================================
def summarise(results, out_dir="."):
    docked_r = [r for r in results if r['docked']]
    n_total  = len(results)
    n_docked = len(docked_r)

    dvs  = np.array([r['total_dv_ms'] for r in docked_r], dtype=float)
    tdks = np.array([r['t_dock_hr']   for r in docked_r], dtype=float)
    pdvs = np.array([r['prox_dv_ms']  for r in docked_r], dtype=float)
    tdvs = np.array([r['term_dv_ms']  for r in docked_r], dtype=float)
    oms  = np.array([r['chief_omega_dps'] for r in docked_r], dtype=float)
    nav_err = np.array([r.get('max_nav_err_m', np.nan) for r in docked_r],
                       dtype=float)
    final_port = np.array([r.get('final_port_range_m', np.nan)
                           for r in docked_r], dtype=float)
    final_vrel = np.array([r.get('final_port_vrel_ms', np.nan)
                           for r in docked_r], dtype=float)
    stress_all = np.array([r.get('stress_case', 'nominal') for r in results])
    reason_all = np.array([r.get('failure_reason',
                                 'DOCKED' if r['docked'] else 'UNKNOWN')
                           for r in results])
    soft_seen_all = np.array([r.get('soft_capture_seen', False) for r in results],
                             dtype=bool)
    hard_strict_all = np.array([r.get('final_hard_strict', False) for r in results],
                               dtype=bool)
    soft_cert_all = np.array([r.get('final_soft_certified', False) for r in results],
                             dtype=bool)
    capture_timeout_all = np.array([r.get('capture_timeout', False) for r in results],
                                   dtype=bool)
    capture_detail_all = np.array([r.get('capture_timeout_detail', 'NONE')
                                   for r in results])
    Isp  = 220.0; g0 = 9.81; m0 = 50.0
    stress_counts = Counter(stress_all)
    stress_success = Counter(r.get('stress_case', 'nominal')
                             for r in results if r['docked'])
    reason_counts = Counter(reason_all)

    lines = []
    lines.append("=" * 65)
    lines.append("  GEO RPOD Monte Carlo Summary")
    lines.append("=" * 65)
    lines.append(f"  Trials total   : {n_total}")
    lines.append(f"  Docking success: {n_docked} / {n_total}  "
                 f"({100*n_docked/n_total:.1f}%)")
    lines.append(f"  Soft capture   : {int(soft_seen_all.sum())} / {n_total}  "
                 f"({100*soft_seen_all.sum()/max(n_total,1):.1f}%)")
    lines.append(f"  Hard strict    : {int(hard_strict_all.sum())} / {n_total}  "
                 f"({100*hard_strict_all.sum()/max(n_total,1):.1f}%)")
    lines.append(f"  Soft certified : {int(soft_cert_all.sum())} / {n_total}  "
                 f"({100*soft_cert_all.sum()/max(n_total,1):.1f}%)")
    lines.append(f"  Capture timeout: {int(capture_timeout_all.sum())} / {n_total}")
    lines.append("")
    lines.append("  Stress-case pass rate:")
    for case in sorted(stress_counts):
        passed = stress_success[case]
        total = stress_counts[case]
        lines.append(f"    {case:<20} {passed:>4} / {total:<4}"
                     f" ({100*passed/max(total, 1):>5.1f}%)")
    lines.append("")
    lines.append("  Outcome counts:")
    for reason, count in reason_counts.most_common():
        lines.append(f"    {reason:<24} {count:>4}")
    if capture_timeout_all.any():
        lines.append("")
        lines.append("  Capture timeout detail:")
        for detail, count in Counter(capture_detail_all[capture_timeout_all]).most_common():
            lines.append(f"    {detail:<24} {count:>4}")
    lines.append("")
    if n_docked > 0:
        lines.append(f"  {'Metric':<28} {'Mean':>8} {'Std':>8} "
                     f"{'5th%':>8} {'50th%':>8} {'95th%':>8}")
        lines.append("  " + "-"*62)
        for label, arr, unit in [
            ("Total DV",      dvs,  "m/s"),
            ("PROX_OPS DV",   pdvs, "m/s"),
            ("TERMINAL DV",   tdvs, "m/s"),
            ("Time to dock",  tdks, "hr"),
            ("Chief tumble",  oms,  "deg/s"),
            ("Max nav error", nav_err[np.isfinite(nav_err)], "m"),
            ("Final port range", final_port[np.isfinite(final_port)], "m"),
            ("Final port vrel", final_vrel[np.isfinite(final_vrel)], "m/s"),
        ]:
            if len(arr) == 0:
                continue
            lines.append(
                f"  {label+' ('+unit+')' :<28}"
                f"  {np.mean(arr):>7.3f}"
                f"  {np.std(arr):>7.3f}"
                f"  {np.percentile(arr,5):>7.3f}"
                f"  {np.percentile(arr,50):>7.3f}"
                f"  {np.percentile(arr,95):>7.3f}")
        lines.append("")

        # Propellant
        dm_arr = m0*(1-np.exp(-dvs/(Isp*g0)))
        lines.append(f"  Propellant (Isp={Isp}s hydrazine):")
        lines.append(f"    Mean  : {np.mean(dm_arr)*1000:.1f}g")
        lines.append(f"    95th% : {np.percentile(dm_arr,95)*1000:.1f}g")
        lines.append(f"    Worst : {np.max(dm_arr)*1000:.1f}g")
        lines.append("")

        worst_i = int(np.argmax(dvs))
        worst = docked_r[worst_i]
        lines.append("  Worst docked DV case:")
        lines.append(
            f"    trial={worst['trial']}  stress={worst.get('stress_case','nominal')}  "
            f"DV={worst['total_dv_ms']:.3f}m/s  "
            f"terminal={worst['term_dv_ms']:.3f}m/s  "
            f"align={worst.get('final_align_deg', np.nan):.1f}deg")
        lines.append("")

        # Correlation
        if len(dvs) > 3:
            corr_om = np.corrcoef(oms, dvs)[0, 1]
            corr_td = np.corrcoef(tdks, dvs)[0, 1]
            lines.append(f"  DV correlation with chief tumble rate : {corr_om:+.3f}")
            lines.append(f"  DV correlation with time-to-dock      : {corr_td:+.3f}")

    lines.append("=" * 65)
    summary = "\n".join(lines)
    print("\n" + summary)

    out_txt = os.path.join(out_dir, "monte_carlo_summary.txt")
    with open(out_txt, "w") as f:
        f.write(summary + "\n")

    # ── Save raw results ─────────────────────────────────────────────
    out_npz = os.path.join(out_dir, "monte_carlo_results.npz")
    np.savez(out_npz,
             docked        = np.array([r['docked']         for r in results]),
             total_dv_ms   = np.array([r['total_dv_ms']    for r in results], dtype=float),
             t_dock_hr     = np.array([r['t_dock_hr'] or np.nan for r in results], dtype=float),
             prox_dv_ms    = np.array([r['prox_dv_ms']     for r in results], dtype=float),
             term_dv_ms    = np.array([r['term_dv_ms']     for r in results], dtype=float),
             chief_omega_dps = np.array([r['chief_omega_dps'] for r in results], dtype=float),
             chief_M0_deg  = np.array([r['chief_M0_deg']   for r in results], dtype=float),
             stress_case   = stress_all,
             failure_reason = reason_all,
             final_range_m = np.array([r.get('final_range_m', np.nan) for r in results], dtype=float),
             final_port_range_m = np.array([r.get('final_port_range_m', np.nan) for r in results], dtype=float),
             final_port_vrel_ms = np.array([r.get('final_port_vrel_ms', np.nan) for r in results], dtype=float),
             min_port_range_m = np.array([r.get('min_port_range_m', np.nan) for r in results], dtype=float),
             soft_capture_seen = np.array([r.get('soft_capture_seen', False) for r in results]),
             capture_timeout = np.array([r.get('capture_timeout', False) for r in results]),
             capture_timeout_detail = capture_detail_all,
             soft_capture_t_hr = np.array([r.get('soft_capture_t_hr', np.nan) for r in results], dtype=float),
             soft_capture_align_entry_deg = np.array([r.get('soft_capture_align_entry_deg', np.nan) for r in results], dtype=float),
             soft_capture_align_min_deg = np.array([r.get('soft_capture_align_min_deg', np.nan) for r in results], dtype=float),
             max_capture_hold_s = np.array([r.get('max_capture_hold_s', 0.0) for r in results], dtype=float),
             final_align_deg = np.array([r.get('final_align_deg', np.nan) for r in results], dtype=float),
             final_align_flip_deg = np.array([r.get('final_align_flip_deg', np.nan) for r in results], dtype=float),
             final_cone_err_deg = np.array([r.get('final_cone_err_deg', np.nan) for r in results], dtype=float),
             final_cone_angle_deg = np.array([r.get('final_cone_angle_deg', np.nan) for r in results], dtype=float),
             final_lateral_m = np.array([r.get('final_lateral_m', np.nan) for r in results], dtype=float),
             final_geometry_ok = np.array([r.get('final_geometry_ok', False) for r in results]),
             final_soft_stable = np.array([r.get('final_soft_stable', False) for r in results]),
             final_soft_certified = np.array([r.get('final_soft_certified', False) for r in results]),
             final_hard_strict = np.array([r.get('final_hard_strict', False) for r in results]),
             max_nav_err_m = np.array([r.get('max_nav_err_m', np.nan) for r in results], dtype=float),
             range_drop_s = np.array([r.get('range_drop_s', 0.0) for r in results], dtype=float),
             camera_drop_s = np.array([r.get('camera_drop_s', 0.0) for r in results], dtype=float))
    print(f"\n  Raw data saved: {out_npz}")
    print(f"  Summary saved:  {out_txt}")

    # ── Plots ────────────────────────────────────────────────────────
    if n_docked >= 1:
        fig, axes = plt.subplots(2, 4, figsize=(18, 9))
        fig.suptitle(f"GEO RPOD Monte Carlo  (n={n_total}, "
                     f"{n_docked} docked)", fontsize=13, fontweight="bold")

        ax = axes[0, 0]
        ax.hist(dvs, bins=max(1, min(18, n_docked)), color="steelblue",
                edgecolor="white", linewidth=0.5)
        ax.axvline(np.mean(dvs), color="red", ls="--", lw=1.5,
                   label=f"Mean {np.mean(dvs):.2f} m/s")
        ax.axvline(np.percentile(dvs, 95), color="orange", ls=":", lw=1.5,
                   label=f"95th {np.percentile(dvs,95):.2f} m/s")
        ax.set(xlabel="Total DV (m/s)", ylabel="Count", title="DV Distribution")
        ax.legend(fontsize=8)

        ax = axes[0, 1]
        ax.hist(tdks, bins=max(1, min(18, n_docked)), color="seagreen",
                edgecolor="white", linewidth=0.5)
        ax.axvline(np.mean(tdks), color="black", ls="--", lw=1.2,
                   label=f"Mean {np.mean(tdks):.2f} hr")
        ax.set(xlabel="Time to dock (hr)", ylabel="Count",
               title="Dock Time Distribution")
        ax.legend(fontsize=8)

        ax = axes[0, 2]
        sc = ax.scatter(oms, dvs, c=tdks, cmap="plasma", s=45,
                        edgecolor="black", linewidth=0.2)
        plt.colorbar(sc, ax=ax, label="Time to dock (hr)")
        ax.set(xlabel="Chief tumble rate (deg/s)", ylabel="Total DV (m/s)",
               title="DV vs Chief Tumble Rate")

        ax = axes[0, 3]
        cases = sorted(stress_counts)
        x = np.arange(len(cases))
        mean_prox = []
        mean_term = []
        for case in cases:
            rr = [r for r in docked_r if r.get('stress_case', 'nominal') == case]
            mean_prox.append(np.nanmean([r['prox_dv_ms'] for r in rr])
                             if rr else 0.0)
            mean_term.append(np.nanmean([r['term_dv_ms'] for r in rr])
                             if rr else 0.0)
        ax.bar(x, mean_prox, label="PROX_OPS", color="steelblue")
        ax.bar(x, mean_term, bottom=mean_prox, label="TERMINAL", color="tomato")
        ax.set_xticks(x)
        ax.set_xticklabels(cases, rotation=35, ha="right", fontsize=8)
        ax.set(ylabel="Mean DV (m/s)", title="Mean DV Split by Stress Case")
        ax.legend(fontsize=8)

        ax = axes[1, 0]
        success_pct = [
            100.0 * stress_success[c] / max(stress_counts[c], 1)
            for c in cases
        ]
        ax.bar(x, success_pct, color="mediumseagreen")
        ax.set_ylim(0, 105)
        ax.set_xticks(x)
        ax.set_xticklabels(cases, rotation=35, ha="right", fontsize=8)
        ax.set(ylabel="Docking success (%)", title="Pass Rate by Stress Case")

        ax = axes[1, 1]
        reasons = [r for r, _ in reason_counts.most_common()]
        counts = [reason_counts[r] for r in reasons]
        ax.bar(np.arange(len(reasons)), counts, color="slategray")
        ax.set_xticks(np.arange(len(reasons)))
        ax.set_xticklabels(reasons, rotation=35, ha="right", fontsize=8)
        ax.set(ylabel="Trials", title="Outcome / Failure Mode Counts")

        ax = axes[1, 2]
        sc2 = ax.scatter(tdks, dvs, c=oms, cmap="viridis", s=45,
                         edgecolor="black", linewidth=0.2)
        plt.colorbar(sc2, ax=ax, label="Chief tumble (deg/s)")
        ax.set(xlabel="Time to dock (hr)", ylabel="Total DV (m/s)",
               title="DV vs Mission Duration")

        ax = axes[1, 3]
        ax.scatter(final_port, final_vrel * 1000.0, c=dvs, cmap="cividis",
                   s=45, edgecolor="black", linewidth=0.2)
        ax.axvline(DOCK_RANGE_M, color="red", ls="--", lw=1.2,
                   label=f"{DOCK_RANGE_M:.2f} m gate")
        ax.axhline(DOCK_VREL_MS * 1000.0, color="orange", ls="--", lw=1.2,
                   label=f"{DOCK_VREL_MS*1000:.0f} mm/s gate")
        ax.set(xlabel="Final port range (m)", ylabel="Final port vrel (mm/s)",
               title="Docking Gate Margin")
        ax.legend(fontsize=8)

        fig.tight_layout()
        out_png = os.path.join(out_dir, "monte_carlo_plots.png")
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Plots saved:    {out_png}")

    return summary


# =====================================================================
#  ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GEO RPOD Monte Carlo")
    parser.add_argument("--trials",  type=int, default=300,
                        help="Number of MC trials (default 300)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel worker processes (default 8)")
    parser.add_argument("--seed",    type=int, default=42,
                        help="Master random seed (default 42)")
    parser.add_argument("--outdir",  type=str,
                        default=os.path.dirname(os.path.abspath(__file__)),
                        help="Output directory for results")
    parser.add_argument("--stress-mode", type=str, default="mixed",
                        choices=("mixed", "nominal", "sweep"),
                        help="Stress-case selection: mixed, nominal, or sweep")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    t_wall_start = time.time()
    results = run_monte_carlo(n_trials=args.trials, master_seed=args.seed,
                              n_workers=args.workers,
                              stress_mode=args.stress_mode)
    summarise(results, out_dir=args.outdir)
    print(f"\n  Total wall time: {(time.time()-t_wall_start)/60:.1f} min")