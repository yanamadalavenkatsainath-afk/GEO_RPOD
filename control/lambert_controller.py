"""
GEO Proximity Operations Controller — Lambert + Continuous PD
==============================================================
Sequence:
  FORMATION_HOLD → LAMBERT (burn-1 + coast + burn-2) → PROX_OPS → TERMINAL → DOCKING

Portability note
----------------
* _prox_ops() and _terminal() are thin wrappers around the pure kernels in
  fsw/rpod_guidance.py.  Port those pure functions to C, not this class.
* Lines tagged [TRUTH] are sim-only (EKF resets from truth state) and must
  NOT be ported.

Key design decisions
--------------------
* EKF/navigation state is used for mode transitions and guidance.
  Truth is only for sensors, logging, and post-run scoring in the sim.
* PROX_OPS and TERMINAL use the same estimated state for range checks
  and acceleration commands so the Python loop matches the C flight loop.
* Lambert burn-2 uses the planned correction from the Lambert solution.
* If Lambert arrival range > FAR_FIELD_M, controller replans immediately
  rather than blindly entering PROX_OPS at wrong range.
* PROX_OPS velocity profile closes from 500m to 0.8m safely.
* TERMINAL closes from 0.8m to dock with a range-proportional speed law.

Fixes applied
-------------
1. Lambert arrival range check — replan if arrival > FAR_FIELD_M instead
   of entering PROX_OPS at wrong range (was causing 307m stuck bug).
2. PROX_OPS → TERMINAL handoff recomputes range from state[:3] at the
   moment of transition to avoid stale truth_range from caller.
3. _terminal range feed uses fresh np.linalg.norm(state[:3]) throughout,
   not the value passed in from the previous mode.
4. Lambert TOF scan now validates predicted LVLH arrival range and rejects
   solutions that won't bring deputy inside FAR_FIELD_M.
5. Burn-2 replan cooldown reset on miss so replanning fires immediately.
"""

import numpy as np
from enum import Enum, auto
from fsw.rpod_guidance import prox_ops_accel, terminal_accel
from spec.rpod_state import TerminalState
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lambert_solver import LambertSolver


class RPODMode(Enum):
    FORMATION_HOLD = auto()
    LAMBERT        = auto()
    PROX_OPS       = auto()
    SURVEY         = auto()   # uncooperative: hold at standoff, build point cloud
    TERMINAL       = auto()
    SOFT_CAPTURE   = auto()
    DOCKING        = auto()
    LOST_TARGET    = auto()   # camera lost — hold position, stop closing


# ── Thresholds ────────────────────────────────────────────────────────
FAR_FIELD_M   = 500.0    # m   Lambert → PROX_OPS handoff range
TERMINAL_M    =   0.8    # m   PROX_OPS → TERMINAL handoff range
DV_CAP_MS     =   2.0    # m/s reject Lambert burns above this

# Lambert arrival tolerance — if predicted arrival range > this,
# the solution is rejected and a different TOF is tried.
# Set to FAR_FIELD_M * 1.5 to allow a little slack for perturbation error.
LAMBERT_ARRIVAL_TOL_M = FAR_FIELD_M * 1.5   # 750 m

# PROX_OPS approach speed profile: (range_m, v_close_ms)
# At each range threshold the closing speed steps down.
PROX_V_PROFILE = [
    (500.0, 0.200),
    (200.0, 0.100),
    (100.0, 0.060),
    ( 50.0, 0.030),
    ( 20.0, 0.015),
    ( 10.0, 0.010),
    (  5.0, 0.005),
    (  2.0, 0.003),
]
PROX_TAU = 5.0    # s  velocity-error time constant for PROX_OPS


class GEORPODController:

    MU        = 3.986004418e14
    J2        = 1.08263e-3
    RE        = 6.3781e6
    AU_m      = 1.495978707e11
    P_SRP_1AU = 4.56e-6

    def __init__(self,
                 mu=3.986004418e14, n_chief=7.2921e-5,
                 dep_mass_kg=50.0, dep_thrust_N=1.0,
                 Cr_chi=1.5, Am_chi=0.015,
                 dock_capture_m=0.30,
                 ekf=None, rng_sensor=None):

        self.mu           = mu
        self.n            = n_chief
        self.Cr_chi       = Cr_chi
        self.Am_chi       = Am_chi
        self.dock_capture = dock_capture_m
        self.lambert      = LambertSolver(mu=mu)
        self.ekf          = ekf
        self.rng_sensor   = rng_sensor
        self.accel_max    = dep_thrust_N / dep_mass_kg   # 0.020 m/s²

        # Formation-hold PD gains
        omega_hold   = 0.05 * n_chief
        self.Kp_hold = omega_hold ** 2
        self.Kd_hold = 2.0 * 0.8 * omega_hold
        self.standoff = 1000.0

        self.mode         = RPODMode.FORMATION_HOLD
        self.mode_history = []

        # Lambert state
        self._lam_active      = False
        self._mode_entry_t    = 0.0
        self._lam_burn2_t     = None
        self._lam_dv2_lvlh    = None
        self._lam_last_plan_t = -9999.0
        self._lam_plan_dt     = 60.0     # s  minimum time between replan attempts

        # Replan counter — used for diagnostics
        self._lam_replan_count = 0

        # Terminal guidance sub-state (explicit struct — maps to C TerminalState)
        self._term_state = TerminalState()

    # ─────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────

    def compute(self, ekf_lvlh, chi_pos_eci, chi_vel_eci, t, true_cw=None,
                port_lvlh=None, cam_lost=False, port_axis_lvlh=None,
                attitude_align_deg=None):
        """
        Returns (accel_cmd [m/s²], impulse_dv [m/s] or None).

        Parameters
        ----------
        ekf_lvlh    : EKF-estimated LVLH state [m, m/s]
        chi_pos_eci : chief ECI position [m]
        chi_vel_eci : chief ECI velocity [m/s]
        t           : simulation time [s]
        true_cw     : optional truth LVLH state [m, m/s] for diagnostics only.
        """
        # Flight-like interface: guidance and mode transitions use estimated
        # navigation state. Truth is retained only as a diagnostic reference.
        nav_state = np.asarray(ekf_lvlh, dtype=float)
        nav_range = float(np.linalg.norm(nav_state[:3]))
        truth_range = nav_range

        # ── Lost-target FSW ──────────────────────────────────────────
        # Trigger LOST_TARGET only when camera has been bad for >5s (rolling
        # window) AND we are far enough to stop safely (>15m).
        # Below 15m the centroid fallback in camera_sensor keeps measurements
        # flowing so thrashing stops naturally — just keep closing.
        # Minimum hold: 10s in LOST_TARGET before re-entering PROX_OPS
        # to prevent the oscillation seen when single frames recover the camera.
        _time_in_lost = (t - self._mode_entry_t) if self.mode == RPODMode.LOST_TARGET else 0.0
        if (cam_lost
                and self.mode == RPODMode.PROX_OPS
                and truth_range > 15.0):
            print(f"  [t={t:.0f}s]  CAMERA LOST — LOST_TARGET hold")
            self._set_mode(RPODMode.LOST_TARGET, t)
        elif (not cam_lost
              and self.mode == RPODMode.LOST_TARGET
              and _time_in_lost >= 10.0):
            print(f"  [t={t:.0f}s]  CAMERA RECOVERED — resuming PROX_OPS")
            self._set_mode(RPODMode.PROX_OPS, t)
            if hasattr(self, '_lost_diag_t'):
                del self._lost_diag_t

        if self.mode == RPODMode.FORMATION_HOLD:
            return self._formation_hold(ekf_lvlh), None

        elif self.mode == RPODMode.LAMBERT:
            return self._lambert_step(
                ekf_lvlh, chi_pos_eci, chi_vel_eci, t,
                truth_range, None)

        elif self.mode == RPODMode.LOST_TARGET:
            return self._lost_target(ekf_lvlh, truth_range, t), None

        elif self.mode == RPODMode.PROX_OPS:
            return self._prox_ops(ekf_lvlh, truth_range, t,
                                  port_lvlh=port_lvlh,
                                  port_axis_lvlh=port_axis_lvlh), None

        elif self.mode == RPODMode.TERMINAL:
            _port = port_lvlh if port_lvlh is not None else np.zeros(3)
            # Safety abort: only abort if range is very large (>15m) AND
            # we have been in TERMINAL for at least 30s (hysteresis).
            # Old threshold (TERMINAL_M*10 = 8m) was too aggressive:
            # any thrust command briefly kicked the deputy past 8m,
            # immediately triggering ABORT -> PROX_OPS -> TERMINAL ping-pong
            # that burned 142m/s of DV doing nothing productive.
            ABORT_RANGE_M    = 15.0
            ABORT_MIN_HOLD_S = 30.0
            NO_ABORT_BELOW_M =  1.0
            time_in_terminal = t - self._mode_entry_t
            if not hasattr(self, '_term_min_range'):
                self._term_min_range = truth_range
            self._term_min_range = min(self._term_min_range, truth_range)

            # Terminal vision FDIR: stop blind docking attempts after a
            # sustained camera outage. LOST_TARGET brakes to near-zero
            # relative velocity and returns to PROX_OPS when vision recovers.
            TERM_CAM_LOSS_HOLD_S = 30.0   # dock-axis alignment rotates gimbal away from chief; allow longer outage
            TERM_CAM_LOSS_ARM_M = 0.5    # only abort if camera lost AND range > 50cm (not right at capture)
            if cam_lost and truth_range > TERM_CAM_LOSS_ARM_M:
                if not hasattr(self, '_term_cam_lost_since'):
                    self._term_cam_lost_since = t
                if t - self._term_cam_lost_since >= TERM_CAM_LOSS_HOLD_S:
                    print(f"  [TERM t={t:.0f}s]  CAMERA LOST -> LOST_TARGET hold")
                    self._set_mode(RPODMode.LOST_TARGET, t)
                    for attr in ('_term_entry_v', '_term_min_range',
                                 '_term_braking', '_term_cam_lost_since'):
                        if hasattr(self, attr):
                            delattr(self, attr)
                    return self._lost_target(ekf_lvlh, truth_range, t), None
            elif hasattr(self, '_term_cam_lost_since'):
                delattr(self, '_term_cam_lost_since')

            # Abort if clearly diverged: current range > 5× closest approach AND > 15m.
            # The old guard (_term_min_range > 1.0m) permanently blocked abort after
            # any sub-1m near-miss, causing ΔV blowup as the spacecraft drifted to
            # 200m+ with no way to return to PROX_OPS.
            abort_ok = (truth_range > ABORT_RANGE_M
                        and time_in_terminal > ABORT_MIN_HOLD_S
                        and truth_range > 5.0 * self._term_min_range)
            if abort_ok:
                print(f"  [TERM t={t:.0f}s]  ABORT -> PROX_OPS  "
                      f"truth_range={truth_range:.1f}m > {ABORT_RANGE_M:.0f}m  "
                      f"(held {time_in_terminal:.0f}s, min={self._term_min_range:.2f}m)")
                self._set_mode(RPODMode.PROX_OPS, t)
                for attr in ('_term_entry_v', '_term_min_range', '_term_braking'):
                    if hasattr(self, attr): delattr(self, attr)
                return self._prox_ops(ekf_lvlh, truth_range, t,
                                      port_lvlh=_port,
                                      port_axis_lvlh=port_axis_lvlh), None
            return self._terminal(nav_state, t, port_lvlh=_port,
                                  port_axis_lvlh=port_axis_lvlh,
                                  attitude_align_deg=attitude_align_deg), None

        elif self.mode == RPODMode.SURVEY:
            # Hold at standoff using PROX_OPS guidance while nozzle estimator stabilizes.
            # SURVEY→TERMINAL transition is gated externally (main.py / monte_carlo.py).
            return self._prox_ops(ekf_lvlh, truth_range, t,
                                  port_lvlh=port_lvlh,
                                  port_axis_lvlh=port_axis_lvlh), None

        elif self.mode == RPODMode.SOFT_CAPTURE:
            _port = port_lvlh if port_lvlh is not None else np.zeros(3)
            return self._soft_capture(nav_state, t, port_lvlh=_port,
                                      port_axis_lvlh=port_axis_lvlh), None

        return np.zeros(3), None

    def start_rendezvous(self, t, truth_range=1000.0):
        if truth_range > FAR_FIELD_M:
            self._set_mode(RPODMode.LAMBERT, t)
        else:
            self._set_mode(RPODMode.PROX_OPS, t)

    # ─────────────────────────────────────────────────────────────────
    # Formation hold — PD to standoff point
    # ─────────────────────────────────────────────────────────────────

    def _formation_hold(self, state):
        target  = np.array([0., -self.standoff, 0.])
        pos_err = state[:3] - target
        vel_err = state[3:6]
        accel   = -self.Kp_hold * pos_err - self.Kd_hold * vel_err
        mag = np.linalg.norm(accel)
        if mag > self.accel_max:
            accel *= self.accel_max / mag
        return accel

    # ─────────────────────────────────────────────────────────────────
    # Lambert — burn-1 + coast + burn-2
    # ─────────────────────────────────────────────────────────────────

    def _soft_capture(self, state, t, port_lvlh=None, port_axis_lvlh=None):
        """Damped compliant latch after terminal reaches the port envelope."""
        pos = state[0:3]
        vel = state[3:6]
        port_vel = state[6:9] if len(state) >= 9 else np.zeros(3)

        if port_lvlh is not None and np.linalg.norm(port_lvlh) > 1e-6:
            port = np.asarray(port_lvlh, dtype=float)
        else:
            port = np.zeros(3)
            port_vel = np.zeros(3)

        err = port - pos
        rel_vel = vel - port_vel
        accel = 0.025 * err - 0.550 * rel_vel

        mag = np.linalg.norm(accel)
        if mag > self.accel_max:
            accel *= self.accel_max / mag

        if not hasattr(self, '_soft_diag_t'):
            self._soft_diag_t = -999.0
        t_slot = int(t / 25.0)
        if t_slot != self._soft_diag_t:
            self._soft_diag_t = t_slot
            print(f"  [SOFT t={t:.0f}s]  "
                  f"port={np.linalg.norm(err):.4f}m  "
                  f"|v_rel|={np.linalg.norm(rel_vel)*1e3:.2f}mm/s  "
                  f"|accel|={np.linalg.norm(accel)*1e6:.1f}Âµm/s2")

        return accel

    def _lambert_step(self, ekf_lvlh, chi_pos, chi_vel, t,
                      truth_range, true_cw):
        """
        State machine for the Lambert arc.

        No active arc
        -------------
        If deputy is already inside FAR_FIELD_M → skip to PROX_OPS.
        Otherwise plan a new arc (rate-limited by _lam_plan_dt).

        Active arc
        ----------
        Coast silently until burn-2 time.
        At burn-2:
          - Null truth relative velocity (most accurate).
          - Check arrival range BEFORE transitioning to PROX_OPS.
          - If arrival range > FAR_FIELD_M → replan immediately.
          - If arrival range <= FAR_FIELD_M → enter PROX_OPS.
        """
        # ── No active arc ─────────────────────────────────────────────
        if not self._lam_active:

            # Already inside prox ops range — skip Lambert entirely
            if truth_range < FAR_FIELD_M:
                print(f"  Lambert [{t:.0f}s]: inside {FAR_FIELD_M:.0f}m "
                      f"(range={truth_range:.1f}m) → PROX_OPS direct")
                self._set_mode(RPODMode.PROX_OPS, t)
                return self._prox_ops(ekf_lvlh, truth_range, t), None

            # Rate-limit replanning attempts
            if t - self._lam_last_plan_t < self._lam_plan_dt:
                return self._formation_hold(ekf_lvlh), None

            self._lam_last_plan_t = t
            return self._plan_lambert(
                ekf_lvlh, chi_pos, chi_vel, t, truth_range, None)

        # ── Active arc: coast until burn-2 ────────────────────────────
        if t >= self._lam_burn2_t:
            self._lam_active = False

            # Burn-2: use the planned Lambert correction, not truth velocity.
            if self._lam_dv2_lvlh is not None:
                dv2 = self._lam_dv2_lvlh.copy()
                print(f"  Lambert burn-2 [{t:.0f}s]: planned "
                      f"|dv2|={np.linalg.norm(dv2)*1e3:.2f}mm/s  "
                      f"nav_range={truth_range:.1f}m")
            else:
                dv2 = np.zeros(3)
                print(f"  Lambert burn-2 [{t:.0f}s]: zero dv2 (no state)")

            # ── KEY FIX: check arrival range before transitioning ──────
            # truth_range here is BEFORE burn-2 is applied.
            # After nulling velocity the deputy will drift on natural CW.
            # If we're already inside FAR_FIELD_M → PROX_OPS.
            # If not → replan immediately (Lambert missed).
            if truth_range <= FAR_FIELD_M:
                self._set_mode(RPODMode.PROX_OPS, t)
                print(f"  Lambert → PROX_OPS  range={truth_range:.1f}m ✓")
            else:
                # Lambert arc missed — reset and replan without delay
                self._lam_replan_count += 1
                self._lam_last_plan_t  = -9999.0   # force immediate replan
                print(f"  Lambert MISS #{self._lam_replan_count} "
                      f"range={truth_range:.1f}m > {FAR_FIELD_M:.0f}m "
                      f"— replanning")
                # Stay in LAMBERT mode, next iteration will replan

            return np.zeros(3), dv2

        # Still coasting — no thrust
        return np.zeros(3), None

    # ─────────────────────────────────────────────────────────────────
    # Lambert planner — scan TOFs, pick minimum dV solution
    # ─────────────────────────────────────────────────────────────────

    def _plan_lambert(self, ekf_lvlh, chi_pos, chi_vel, t,
                      truth_range, true_cw):
        """
        Scan candidate TOFs, compute two-impulse Lambert solutions,
        validate predicted arrival range, apply burn-1.

        Perturbation correction
        -----------------------
        The Lambert solver uses two-body dynamics internally. We correct
        for J2 + SRP by computing the difference between a full-force
        propagation and a two-body propagation of the deputy, then
        shifting the Lambert target accordingly.

        Two iterations of this correction are applied for better accuracy.

        Arrival range validation
        ------------------------
        After solving, the predicted LVLH arrival position is estimated
        by propagating the deputy (full-force) with v1_lambert and
        comparing to the chief's future LVLH position. Solutions where
        the predicted arrival range > LAMBERT_ARRIVAL_TOL_M are rejected.
        This is the main guard against the 307m stuck-range bug.
        """

        R_l2e       = self._R_l2e(chi_pos, chi_vel)
        R_e2l       = self._R_e2l(chi_pos, chi_vel)
        dep_pos_eci = chi_pos + R_l2e @ ekf_lvlh[:3]
        dep_vel_eci = chi_vel + R_l2e @ ekf_lvlh[3:6]

        best_tof          = None
        best_dv1_lvlh     = None
        best_dv2_lvlh     = None
        best_total        = np.inf
        best_arrival_rng  = np.inf

        # Candidate TOFs — 1hr to 6hr in steps.
        # For a 1km trailing standoff at GEO, a ~1-2hr Lambert arc is
        # typically sufficient for minimum dV.
        candidate_tofs = [3600., 5400., 7200., 9000., 10800., 14400., 21600.]

        for tof in candidate_tofs:

            # Chief state at arrival time
            chi_arr, chi_vel_arr = self._propagate_ff(
                chi_pos, chi_vel, tof, t)

            # ── Two-step perturbation correction ──────────────────────
            # Step 1: compute offset using current deputy velocity
            dep_ff_0, _ = self._propagate_ff(
                dep_pos_eci, dep_vel_eci, tof, t)
            dep_2b_0, _ = self.lambert.propagate_keplerian(
                dep_pos_eci, dep_vel_eci, tof, n_steps=200)
            pert_offset_0 = dep_ff_0 - dep_2b_0
            target_0      = chi_arr - pert_offset_0

            v1_init, v2_init = self.lambert.solve(dep_pos_eci, target_0, tof)
            if v1_init is None:
                continue

            # Step 2: recompute offset using Lambert v1 (better accuracy)
            dep_ff_1, _ = self._propagate_ff(
                dep_pos_eci, v1_init, tof, t)
            dep_2b_1, _ = self.lambert.propagate_keplerian(
                dep_pos_eci, v1_init, tof, n_steps=200)
            pert_offset_1 = dep_ff_1 - dep_2b_1
            target_1      = chi_arr - pert_offset_1

            v1, v2 = self.lambert.solve(dep_pos_eci, target_1, tof)
            if v1 is None:
                # Fall back to first iteration result
                v1, v2 = v1_init, v2_init

            # ── Delta-V budget check ──────────────────────────────────
            dv1_eci = v1 - dep_vel_eci
            dv2_eci = chi_vel_arr - v2
            m1      = np.linalg.norm(dv1_eci)
            m2      = np.linalg.norm(dv2_eci)

            if m1 > DV_CAP_MS or m2 > DV_CAP_MS:
                continue

            # ── Predicted arrival range check ─────────────────────────
            # Propagate deputy with Lambert v1 (full-force) and compare
            # to chief's future LVLH position.
            dep_arr_eci, _ = self._propagate_ff(dep_pos_eci, v1, tof, t)

            # Chief future position in LVLH at arrival
            # (chi_arr is ECI; transform to LVLH relative to itself = 0,
            #  so we just need dep_arr in LVLH relative to chief_arr)
            chi_arr_vel_approx = chi_vel_arr   # already propagated
            R_e2l_arr = self._R_e2l_approx(chi_arr, chi_arr_vel_approx)
            dep_arr_lvlh = R_e2l_arr @ (dep_arr_eci - chi_arr)
            arrival_range = float(np.linalg.norm(dep_arr_lvlh))

            print(f"  Lambert scan TOF={tof/3600:.1f}hr  "
                  f"|dv1|={m1*1e3:.2f}mm/s  "
                  f"|dv2|={m2*1e3:.2f}mm/s  "
                  f"arr_range={arrival_range:.1f}m")

            # Reject solutions that don't arrive inside tolerance
            if arrival_range > LAMBERT_ARRIVAL_TOL_M:
                print(f"    → rejected (arrival {arrival_range:.0f}m "
                      f"> tol {LAMBERT_ARRIVAL_TOL_M:.0f}m)")
                continue

            total = m1 + m2
            if total < best_total:
                best_total        = total
                best_tof          = tof
                best_arrival_rng  = arrival_range
                best_dv1_lvlh     = R_e2l @ dv1_eci
                best_dv2_lvlh     = R_e2l @ dv2_eci

        # ── No valid solution ─────────────────────────────────────────
        if best_tof is None:
            print(f"  Lambert [{t:.0f}s]: no valid solution "
                  f"(all arrivals > {LAMBERT_ARRIVAL_TOL_M:.0f}m or dV > cap) "
                  f"— hold formation, retry in {self._lam_plan_dt:.0f}s")
            return self._formation_hold(ekf_lvlh), None

        # ── Accept and arm burn-1 ─────────────────────────────────────
        self._lam_active   = True
        self._lam_burn2_t  = t + best_tof
        self._lam_dv2_lvlh = best_dv2_lvlh.copy()

        print(f"  Lambert arc PLANNED [{t:.0f}s]: "
              f"TOF={best_tof/3600:.1f}hr  "
              f"|dv1|={np.linalg.norm(best_dv1_lvlh)*1e3:.2f}mm/s  "
              f"|dv2_plan|={np.linalg.norm(best_dv2_lvlh)*1e3:.2f}mm/s  "
              f"pred_arr={best_arrival_rng:.1f}m  "
              f"truth_range={truth_range:.1f}m")

        return np.zeros(3), best_dv1_lvlh

    # ─────────────────────────────────────────────────────────────────
    # PROX_OPS — continuous PD closure FAR_FIELD_M → TERMINAL_M
    # ─────────────────────────────────────────────────────────────────

    def _lost_target(self, state, truth_range, t):
        """
        Hold position when camera is lost during close approach.
        Commands zero velocity — gently decelerates to a stop.
        Exits automatically when cam_lost=False resumes.
        """
        vel = state[3:6]
        vel_mag = np.linalg.norm(vel)
        # Only decelerate if moving faster than 5mm/s — below that, coast.
        # This stops the chatter loop where corrections trigger more corrections.
        if vel_mag > 0.005:
            accel = -vel / 2.0
            mag = np.linalg.norm(accel)
            if mag > self.accel_max:
                accel *= self.accel_max / mag
        else:
            accel = np.zeros(3)   # already nearly stopped — coast
        t_slot = int(round(t / 100.0))
        if not hasattr(self, '_lost_diag_t') or t_slot != self._lost_diag_t:
            self._lost_diag_t = t_slot
            print(f"  [LOST t={t:.0f}s]  rng={truth_range:.1f}m  "
                  f"|v|={np.linalg.norm(vel)*1e3:.1f}mm/s  holding")
        return accel

    def _prox_ops(self, state, truth_range, t, port_lvlh=None, port_axis_lvlh=None):
        """Thin wrapper — delegates to fsw/rpod_guidance.py:prox_ops_accel()."""
        if truth_range < TERMINAL_M:
            print(f"  PROX_OPS → TERMINAL  truth_range={truth_range:.4f}m")
            self._set_mode(RPODMode.TERMINAL, t)
            return self._terminal(state, t, port_lvlh=port_lvlh,
                                  port_axis_lvlh=port_axis_lvlh)
        if truth_range < 0.20:
            return np.zeros(3)

        accel = prox_ops_accel(state[:3], state[3:6], self.accel_max)

        # ── Diagnostics (sim only — not ported) ──────────────────────
        t_slot = int(round(t / 100.0))
        if not hasattr(self, '_last_diag_t') or t_slot != self._last_diag_t:
            self._last_diag_t = t_slot
            rng_ekf = float(np.linalg.norm(state[:3]))
            pos_hat = state[:3] / max(rng_ekf, 1e-3)
            v_closing_actual = -float(np.dot(pos_hat, state[3:6]))
            print(f"  [PROX t={t:.0f}s]"
                  f"  rng_truth={truth_range:.1f}m  rng_ekf={rng_ekf:.1f}m"
                  f"  v_act={v_closing_actual*1e3:.2f}mm/s"
                  f"  |accel|={np.linalg.norm(accel)*1e6:.1f}µm/s²")

        return accel

    # ─────────────────────────────────────────────────────────────────
    # TERMINAL — range-proportional deceleration TERMINAL_M → dock
    # ─────────────────────────────────────────────────────────────────

    def _terminal(self, state, t, port_lvlh=None, port_axis_lvlh=None,
                  attitude_align_deg=None):
        """Thin wrapper — delegates to fsw/rpod_guidance.py:terminal_accel()."""
        DT_DIAG = 25.0

        # ── [TRUTH] EKF covariance reset at TERMINAL entry ───────────
        # Zeroes position-velocity cross terms so the filter forgets the
        # 500m approach history. Sim-only: uses truth-seeded filter ref.
        # DO NOT PORT to C FSW.
        if self._term_state.entry_key != int(self._mode_entry_t):
            if hasattr(self, '_th_ekf_ref'):
                ekf = self._th_ekf_ref
                ekf.P[0:3, 3:6] = 0.0   # [TRUTH]
                ekf.P[3:6, 0:3] = 0.0   # [TRUTH]

        accel = terminal_accel(
            pos          = state[0:3],
            vel          = state[3:6],
            port_lvlh    = port_lvlh,
            accel_max    = self.accel_max,
            mode_entry_t = self._mode_entry_t,
            state        = self._term_state,
        )

        # ── Diagnostics (sim only — not ported) ──────────────────────
        t_slot = int(t / DT_DIAG)
        if not hasattr(self, '_term_diag_t') or t_slot != self._term_diag_t:
            self._term_diag_t = t_slot
            com_range  = float(np.linalg.norm(state[0:3]))
            port_range = float(np.linalg.norm(
                (port_lvlh if port_lvlh is not None else np.zeros(3)) - state[0:3]))
            align_deg  = (float(attitude_align_deg)
                          if attitude_align_deg is not None
                          and np.isfinite(attitude_align_deg) else float("nan"))
            print(f"  [TERM t={t:.0f}s]  "
                  f"com={com_range:.4f}m  port={port_range:.4f}m  "
                  f"align={align_deg:.1f}deg  braking={self._term_state.braking}  "
                  f"|accel|={np.linalg.norm(accel)*1e6:.1f}µm/s2")

        return accel

    # ─────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────

    def _set_mode(self, mode, t):
        if mode == self.mode:
            return
        print(f"  RPOD [{t:.0f}s]: {self.mode.name} → {mode.name}")
        self.mode = mode
        self.mode_history.append((t, mode))
        self._mode_entry_t = t
        # Clear Lambert arc state when leaving Lambert mode
        if mode not in (RPODMode.LAMBERT,):
            self._lam_active   = False
            self._lam_dv2_lvlh = None

    def _R_l2e(self, r, v):
        """LVLH → ECI rotation matrix (columns = LVLH axes in ECI)."""
        xh = r / np.linalg.norm(r)
        zh = np.cross(r, v)
        zh /= np.linalg.norm(zh)
        yh = np.cross(zh, xh)
        return np.column_stack([xh, yh, zh])

    def _R_e2l(self, r, v):
        """ECI → LVLH rotation matrix."""
        return self._R_l2e(r, v).T

    def _R_e2l_approx(self, r, v):
        """
        ECI → LVLH rotation at an approximate future state.
        Used for arrival range estimation. Same math as _R_e2l but
        named separately for clarity.
        """
        return self._R_e2l(r, v)

    def _sun_pos_m(self, t):
        """Approximate Sun ECI position [m]."""
        d   = t / 86400.0
        lam = np.radians(280.46 + 360.985647 * d)
        eps = np.radians(23.439)
        return self.AU_m * np.array([np.cos(lam),
                                     np.cos(eps) * np.sin(lam),
                                     np.sin(eps) * np.sin(lam)])

    def _accel_ff(self, pos, t):
        """
        Full-force acceleration: two-body + J2 + SRP (chief SRP params).
        Used for Lambert target correction propagation.
        """
        r = np.linalg.norm(pos)
        a = -self.mu / r ** 3 * pos

        # J2
        x, y, z = pos
        c = -1.5 * self.J2 * self.mu * self.RE ** 2 / r ** 5
        f = 5.0 * z ** 2 / r ** 2
        a += np.array([c*x*(1-f), c*y*(1-f), c*z*(3-f)])

        # SRP (chief A/m ratio — used for chief future position propagation)
        sp = self._sun_pos_m(t)
        rs = np.linalg.norm(sp)
        P  = self.P_SRP_1AU * (self.AU_m / rs) ** 2
        r_sc = pos - sp
        a += self.Cr_chi * self.Am_chi * P * r_sc / np.linalg.norm(r_sc)

        return a

    def _propagate_ff(self, pos, vel, dt_total, t_start, h=120.0):
        """
        RK4 full-force propagator for Lambert target correction.

        Parameters
        ----------
        pos, vel  : ECI state [m, m/s]
        dt_total  : total propagation time [s]
        t_start   : absolute time at start (for sun position) [s]
        h         : RK4 substep size [s]. Default 120s (2min).

        Returns
        -------
        pos_new, vel_new : propagated ECI state [m, m/s]
        """
        p, v, t = pos.copy(), vel.copy(), float(t_start)
        n = max(1, int(round(dt_total / h)))
        h = dt_total / n

        for _ in range(n):
            k1p = v
            k1v = self._accel_ff(p, t)
            k2p = v + 0.5*h*k1v
            k2v = self._accel_ff(p + 0.5*h*k1p, t + 0.5*h)
            k3p = v + 0.5*h*k2v
            k3v = self._accel_ff(p + 0.5*h*k2p, t + 0.5*h)
            k4p = v + h*k3v
            k4v = self._accel_ff(p + h*k3p,     t + h)
            p += (h / 6.0) * (k1p + 2*k2p + 2*k3p + k4p)
            v += (h / 6.0) * (k1v + 2*k2v + 2*k3v + k4v)
            t += h

        return p, v
