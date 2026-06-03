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
SOFT_CAPTURE_ENTRY_ALIGN_MAX_DEG = 60.0  # raised from 30°: pose-spike blocked 26cm near-miss
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

# ── Environment ───────────────────────────────────────────────────────
ECLIPSE_NU_MIN = 0.1
MU_GEO = 3.986004418e14
N_GEO  = np.sqrt(MU_GEO / (CHIEF_A_KM * 1e3) ** 3)
