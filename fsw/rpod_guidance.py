"""
fsw/rpod_guidance.py — Pure RPOD guidance kernels.

These functions are the direct Python reference for the C port.

Rules enforced here (match C constraints):
  - No globals.  All config passed as arguments.
  - No Python objects as inputs/outputs — only np.ndarray, float, bool, int.
  - State is a TerminalState dataclass (maps to C struct passed by pointer).
  - No print statements.  Caller handles logging.
  - import only at module level.
  - Every function returns a value; state mutation is explicit via the struct.

C signatures will look like:
  void prox_ops_accel(const double pos[3], const double vel[3],
                      double accel_max, double accel_out[3]);

  void terminal_accel(const double pos[3], const double vel[3],
                      const double port_lvlh[3], int have_port,
                      double accel_max, double mode_entry_t,
                      TerminalState *state, double accel_out[3]);
"""

import math
import numpy as np
from typing import Optional
from spec.rpod_state import TerminalState

# ── PROX_OPS constants (match lambert_controller.py) ─────────────────
_PROX_TAU    = 5.0     # s   velocity-error time constant
_K_SQRT_NUM  = 0.200   # m/s numerator of sqrt-law (speed at 500m)
_K_SQRT_DEN  = 500.0   # m   denominator range
_V_CLOSE_MAX = 0.200   # m/s absolute cap
_V_CLOSE_10M = 0.005   # m/s cap below 10 m

# ── TERMINAL constants ────────────────────────────────────────────────
_TERM_V_MAX_MS     = 0.025   # m/s  25 mm/s terminal max speed
_TERM_V_CAPTURE_MS = 0.0015  # m/s   1.5 mm/s inside capture sphere
_TERM_DOCK_RANGE_M = 0.30    # m    capture zone radius
_TERM_PORT_SANITY_M = 2.0    # m    max credible port offset from EKF pos
_TERM_BRAKE_THRESH = 2.0     # ×V_MAX — brake if entry speed exceeds this
_TERM_BRAKE_DONE_MS = 10.0   # mm/s — stop braking below this speed


def prox_ops_accel(
    pos: np.ndarray,
    vel: np.ndarray,
    accel_max: float,
) -> np.ndarray:
    """
    PROX_OPS closing guidance.

    Sqrt-law desired speed toward chief, velocity P-controller.
    No state — fully stateless each tick.

    Parameters
    ----------
    pos        : [3] EKF LVLH position  [m]
    vel        : [3] EKF LVLH velocity  [m/s]
    accel_max  : scalar thrust authority [m/s²]

    Returns
    -------
    accel : [3] commanded acceleration LVLH [m/s²]
    """
    pos = np.asarray(pos, dtype=float)
    vel = np.asarray(vel, dtype=float)

    rng = float(np.linalg.norm(pos))
    if rng < 1e-3:
        pos_hat = np.array([0.0, -1.0, 0.0])
    else:
        pos_hat = pos / rng

    k = _K_SQRT_NUM / math.sqrt(_K_SQRT_DEN)
    v_close = min(k * math.sqrt(max(rng, 0.1)), _V_CLOSE_MAX)
    if rng < 10.0:
        v_close = min(v_close, _V_CLOSE_10M)

    vel_des = -pos_hat * v_close
    accel   = -(vel - vel_des) / _PROX_TAU

    mag = float(np.linalg.norm(accel))
    if mag > accel_max:
        accel *= accel_max / mag

    return accel


def terminal_accel(
    pos: np.ndarray,
    vel: np.ndarray,
    port_lvlh: Optional[np.ndarray],
    accel_max: float,
    mode_entry_t: float,
    state: TerminalState,
) -> np.ndarray:
    """
    TERMINAL guidance: port-targeting with sqrt speed law and entry brake.

    Mutates `state` in place (mirrors C pass-by-pointer).
    Returns commanded acceleration LVLH [m/s²].

    Parameters
    ----------
    pos          : [3] EKF LVLH position  [m]
    vel          : [3] EKF LVLH velocity  [m/s]
    port_lvlh    : [3] estimated dock port position LVLH [m], or None
    accel_max    : scalar thrust authority [m/s²]
    mode_entry_t : time TERMINAL was entered [s] — used to detect re-entry
    state        : TerminalState (mutated in place)

    Returns
    -------
    accel : [3] commanded acceleration LVLH [m/s²]
    """
    pos  = np.asarray(pos,  dtype=float)
    vel  = np.asarray(vel,  dtype=float)

    com_range = float(np.linalg.norm(pos))

    # ── Detect fresh TERMINAL entry ───────────────────────────────────
    entry_key = int(mode_entry_t)
    if state.entry_key != entry_key:
        state.entry_key  = entry_key
        state.entry_v    = float(np.linalg.norm(vel))
        state.braking    = state.entry_v > _TERM_BRAKE_THRESH * _TERM_V_MAX_MS

    # ── Port sanity check ─────────────────────────────────────────────
    if port_lvlh is not None and np.linalg.norm(port_lvlh) > 1e-6:
        cand_range = float(np.linalg.norm(port_lvlh - pos))
        port = port_lvlh.copy() if cand_range < _TERM_PORT_SANITY_M else np.zeros(3)
    else:
        port = np.zeros(3)

    port_range = float(np.linalg.norm(port - pos))

    # ── Entry brake ───────────────────────────────────────────────────
    if state.braking:
        accel = -vel / 1.0
        mag   = float(np.linalg.norm(accel))
        if mag > accel_max:
            accel *= accel_max / mag
        if np.linalg.norm(vel) < _TERM_BRAKE_DONE_MS * 1e-3:
            state.braking = False
        return accel

    # ── TAU gain scheduling ───────────────────────────────────────────
    if com_range < 0.30:
        tau = 10.0
    elif com_range < 0.60:
        tau = 8.0
    else:
        tau = 6.0

    # ── Target selection ──────────────────────────────────────────────
    if port_range > 0.001:
        tgt_hat   = (port - pos) / port_range
        tgt_range = port_range
    else:
        tgt_hat   = -pos / max(com_range, 1e-6)
        tgt_range = com_range

    # ── Speed law ─────────────────────────────────────────────────────
    # Normalised to TERMINAL_M=0.8m (C FSW entry point).
    # Python sim enters TERMINAL early at MAIN_TERMINAL_M=10m (CW orbit-trap
    # workaround) but V_MAX_MS cap takes effect above ~0.8m so the kernel
    # behaviour is identical.  Do NOT change 0.8 to match MAIN_TERMINAL_M.
    k_sqrt  = _TERM_V_MAX_MS / math.sqrt(0.8)
    v_des_mag = min(k_sqrt * math.sqrt(max(com_range, 0.001)), _TERM_V_MAX_MS)
    if tgt_range < _TERM_DOCK_RANGE_M:
        v_des_mag = min(v_des_mag, _TERM_V_CAPTURE_MS)

    vel_des = tgt_hat * v_des_mag
    accel   = (vel_des - vel) / tau

    mag = float(np.linalg.norm(accel))
    if mag > accel_max:
        accel *= accel_max / mag

    return accel
