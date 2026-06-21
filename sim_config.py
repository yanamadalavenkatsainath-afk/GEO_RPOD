"""
sim_config.py — Single source of truth for all GEO RPOD mission parameters.

Both main.py and monte_carlo.py import from here via `from sim_config import *`.
Script-local overrides are allowed for MC stress profiles, but canonical values
live here. Divergences between the two scripts should be fixed here, not patched
in both files independently.
"""

import numpy as np

# ── Chief orbit (IS-1002 class GEO comsat at 342° E) ─────────────────
CHIEF_A_KM      = 42164.0
CHIEF_E         = 0.0003
CHIEF_I_DEG     = 0.8
CHIEF_RAAN_DEG  = 0.0
CHIEF_OMEGA_DEG = 0.0
CHIEF_M0_DEG    = 0.0
CHIEF_LON_DEG   = 342.0

# ── Deputy hardware (50 kg jetpack) ──────────────────────────────────
DEP_MASS_KG  = 50.0
DEP_THRUST_N = 1.0
DEP_CR       = 1.5
DEP_AM       = 0.00720
I_SC         = np.diag([4.167, 4.167, 3.000])   # kg·m²

# ── Chief SRP / mass ──────────────────────────────────────────────────
CHI_CR        = 1.5
CHI_AM        = 0.015
CHIEF_MASS_KG = 3000.0
CHIEF_SOLAR_ARRAY_HALF_SPAN_M = 6.0   # IS-1002 class: ~12m tip-to-tip arrays

# ── Formation hold ────────────────────────────────────────────────────
FORMATION_OFFSET_M = np.array([0.0, -1000.0, 0.0])   # 1 km trailing (LVLH -y)

# ── Timing ────────────────────────────────────────────────────────────
DT_OUTER  = 0.1          # s  outer (RPOD) loop
DT_INNER  = 0.01         # s  inner (ADCS) loop
N_INNER   = int(DT_OUTER / DT_INNER)
T_SIM_MAX = 80_000.0     # s  (~22 hr ceiling)

# ── ADCS stability gate ───────────────────────────────────────────────
ADCS_STABLE_DEG  = 1.0
ADCS_STABLE_SUST = 100   # consecutive steps

# ── Formation hold settle before Lambert ─────────────────────────────
FORM_HOLD_SETTLE_S = 300.0   # s

# ── Docking geometry ──────────────────────────────────────────────────
DOCK_PORT_BODY            = np.array([0.0, 0.0, 0.5])   # m in chief body frame
DOCK_AXIS_BODY            = np.array([0.0, 0.0, 1.0])   # approach axis, chief body
DEP_DOCK_AXIS_BODY        = np.array([0.0, 0.0, 1.0])   # approach axis, deputy body
CHIEF_BODY_HALF_EXTENTS_M = np.array([0.80, 0.80, 0.50])

# ── LAE nozzle truth geometry (−Z face, anti-nadir) ──────────────────
# Used ONLY by sensor simulation and renderer — GNC never reads these.
# IS-1002 class bipropellant LAE: ~400N thrust, ~0.35m bell diameter.
LAE_NOZZLE_BASE_RADIUS_M = 0.28   # radius where nozzle meets -Z face
LAE_NOZZLE_EXIT_RADIUS_M = 0.16   # radius at nozzle exit (narrower tip)
LAE_NOZZLE_LENGTH_M      = 0.30   # protrusion below -Z face
LAE_NOZZLE_N_SEG         = 10     # polygon segments for frustum approximation

DOCK_PORT_APERTURE_M      = 0.15
DOCK_CONE_HALF_ANGLE_DEG  = 15.0
DOCK_CONE_MIN_RANGE_M     = 0.05
DOCK_FACE_TOL_M           = 0.05
DOCK_ALIGN_MAX_DEG        = 10.0

# ── Soft / hard capture thresholds ───────────────────────────────────
DOCK_RANGE_M                    = 0.30   # widened from 0.10m for pose-estimator error
DOCK_VREL_MS                    = 0.05   # relaxed from 0.01 to 50 mm/s
SOFT_CAPTURE_RANGE_M            = DOCK_RANGE_M
SOFT_CAPTURE_VREL_MS            = DOCK_VREL_MS
HARD_CAPTURE_RANGE_M            = 0.08
HARD_CAPTURE_VREL_MS            = 0.010
HARD_CAPTURE_HOLD_S             = 5.0
SOFT_CAPTURE_HOLD_S             = 5.0
SOFT_CAPTURE_LATCH_VREL_MS      = 0.030
SOFT_CAPTURE_MAX_HOLD_S         = 1200.0  # need ~870s from 76.8° entry (sin rate model)
SOFT_CAPTURE_CORE_ALIGN_MAX_DEG = 20.0
SOFT_CAPTURE_ENTRY_ALIGN_MAX_DEG = 30.0  # entry gate uses MEKF alignment (reliable) — 30° is correct
SOFT_CAPTURE_ATTITUDE_TORQUE_SCALE = 1.0  # restored: 0.4 couldn't despin 0.5 Nms residual
SOFT_CAPTURE_ATTITUDE_LOG_S     = 30.0
SOFT_CAPTURE_RESTITUTION        = 0.10
SOFT_CAPTURE_TANGENTIAL_DAMPING = 0.30

# ── Feature flags ─────────────────────────────────────────────────────
ENABLE_PHYSICAL_THRUSTER_LAYOUT = True
ENABLE_FINITE_BODY_COLLISION    = True
ENABLE_COUPLED_CONTACT_DYNAMICS = True
ENABLE_BODY_MOUNTED_CAMERA_FOV  = True
ENABLE_KEEP_OUT_AVOIDANCE       = True
ENABLE_SPIN_SYNC                = True

# ── Thruster / deputy body ────────────────────────────────────────────
THRUSTER_MAX_FORCE_N       = 0.25
THRUSTER_CANT_DEG          = 35.26   # isometric corner-pod cant angle
DEPUTY_BODY_HALF_EXTENTS_M = np.array([0.30, 0.30, 0.40])

# ── Navigation / estimation ───────────────────────────────────────────
SIGMA_V_DOPPLER        = 0.005   # m/s  Doppler noise (VBS class)
TERM_NAV_ALPHA         = 0.25
TERM_NAV_BETA          = 0.02
TERM_NAV_VMAX_MS       = 0.05
TERM_NAV_GATE_M        = 0.25
PORT_TRACK_ALPHA       = 0.40
PORT_TRACK_GATE_M      = 0.25
CLOSE_PROX_NAV_RANGE_M = 20.0   # m  dock-axis pre-alignment activates below this

# ── PROX_OPS → TERMINAL handoff ───────────────────────────────────────
# Raised from 5 m: CW orbit-trap at ~6 m prevented reaching 5 m when
# the deputy had to navigate around the port in full 3D.
MAIN_TERMINAL_M = 10.0

# ── Spin sync health gate ─────────────────────────────────────────────
SPIN_SYNC_MAX_OMEGA_RAD_S = np.radians(1.0)  # reject omega estimates > 1 deg/s

# ── Chief pose estimator model points ────────────────────────────────
# 3D feature model of the IS-1002 chief used by ChiefPoseEstimator EPnP.
# Points are in the chief body frame [m].
#
# Body: 8 corners of the IS-1002 bus (half-extents [0.8, 0.8, 0.5]).
# The bus has a square cross-section → equal X and Y PCA eigenvalues →
# EPnP dock-axis rotation is poorly conditioned, producing the ~60 deg
# orientation bias seen in telemetry.
#
# Solar stubs: ±X array root fittings at [±1.5, 0, 0].  IS-1002 carries
# two symmetric wings; both roots improve PROX_OPS PCA conditioning and
# give the EKF two robust off-axis landmarks.  Visible at range > ~4 m.
# Dock marker: [0, 0.4, 0.5] is a retroreflective patch offset +0.4 m in Y
# on the dock face.  Breaks the 4-fold Z symmetry so EPnP can resolve
# dock-axis roll at TERMINAL range (2–4 m) where the solar stubs are
# outside the camera FOV.
CHIEF_POSE_MODEL_PTS = np.array([
    [ 0.80,  0.80,  0.50], [ 0.80, -0.80,  0.50],
    [-0.80,  0.80,  0.50], [-0.80, -0.80,  0.50],
    [ 0.80,  0.80, -0.50], [ 0.80, -0.80, -0.50],
    [-0.80,  0.80, -0.50], [-0.80, -0.80, -0.50],
    [ 1.50,  0.00,  0.00],   # +X solar array root fitting
    [-1.50,  0.00,  0.00],   # -X solar array root fitting (IS-1002 has two wings)
    [ 0.00,  0.40,  0.50],   # dock-face marker (+Y offset, breaks 4-fold Z symmetry at close range)
], dtype=float)

# ── Phase timeouts ────────────────────────────────────────────────────
PROX_OPS_MAX_S   = 50_000.0
TERMINAL_MAX_S   = 20_000.0

# ── Hard capture hysteresis ───────────────────────────────────────────
HARD_CAPTURE_GRACE_S = 1.0   # brief misalignment allowed without resetting hold

# ── Engine-bell grapple geometry (uncooperative capture) ─────────────
# GNC-accessible parameters — NOT derived from truth LAE_NOZZLE_* geometry.
# These define what geometry must be certified before capture is declared.
BELL_SOFT_LATERAL_MAX_M  = 0.15   # max lateral miss from nozzle axis [m] — soft gate
BELL_SOFT_AXIAL_MAX_M    = 0.30   # max axial stand-off from nozzle exit [m] — soft gate
BELL_SOFT_VREL_MAX_MS    = 0.05   # max relative speed [m/s] — soft gate
BELL_SOFT_APPROACH_DEG   = 55.0   # max angle from nozzle approach axis [deg] — soft gate
BELL_HARD_LATERAL_MAX_M  = 0.08   # tighter lateral limit [m] — hard gate
BELL_HARD_AXIAL_MAX_M    = 0.15   # tighter axial stand-off [m] — hard gate
BELL_HARD_VREL_MAX_MS    = 0.02   # tighter velocity [m/s] — hard gate
BELL_HARD_APPROACH_DEG   = 25.0   # tighter axis alignment [deg] — hard gate
BELL_HARD_HOLD_S         = 5.0    # sustained hard-gate hold before DOCKING [s]

# ── Terminal commit stability gate (SURVEY → TERMINAL) ────────────────
# Prevents TERMINAL commit on a fleeting detection.
NOZZLE_STABILITY_S    = 3.0    # confidence must stay ≥ threshold for this long [s]
NOZZLE_DRIFT_MAX_M    = 0.80   # RANSAC shot noise is 10–70 cm; 80 cm catches only gross divergence [m]
NOZZLE_FP_SCORE_MAX   = 1.01   # fp_score is informational only — LAE frustum geometry always scores ≥ 0.67

# ── CNN pose estimator ────────────────────────────────────────────────
# Off by default — analytic DLT+GN path remains the baseline.
# Enable to add CNN-based orientation measurements in PROX_OPS/TERMINAL.
ENABLE_CNN_POSE_ESTIMATOR = False
CNN_POSE_UPDATE_HZ        = 1.0    # inference rate (keep ≤1 Hz — rendering is slow)
CNN_CHECKPOINT_PATH       = "pose_cnn/checkpoints/pose_net.pt"
CNN_POSE_SIGMA_DEG        = 5.0    # orientation noise assigned to CNN measurements

# ── Environment ───────────────────────────────────────────────────────
ECLIPSE_NU_MIN = 0.1
MU_GEO = 3.986004418e14
N_GEO  = np.sqrt(MU_GEO / (CHIEF_A_KM * 1e3) ** 3)
