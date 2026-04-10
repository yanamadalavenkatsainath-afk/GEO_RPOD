"""
GEO Proximity Operations Controller — Lambert + Continuous PD
==============================================================
Sequence:
  FORMATION_HOLD → LAMBERT (burn-1 + coast + burn-2) → PROX_OPS → TERMINAL → DOCKING

Key design decisions
--------------------
* `truth_range` and `true_cw` are always used for mode transitions and
  guidance.  EKF state is only used for Lambert planning (far field).
* PROX_OPS and TERMINAL use `truth_range` (passed in from main) for all
  range checks — never `np.linalg.norm(state[:3])`.  This avoids the
  stale-state bug where state is not yet updated after a burn.
* Lambert burn-2 nulls the truth relative velocity directly; the planned
  dv2 is only a fallback if truth is unavailable.
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
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lambert_solver import LambertSolver


class RPODMode(Enum):
    FORMATION_HOLD = auto()
    LAMBERT        = auto()
    PROX_OPS       = auto()
    TERMINAL       = auto()
    DOCKING        = auto()


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
                 dock_capture_m=0.10,
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
        self._lam_burn2_t     = None
        self._lam_dv2_lvlh    = None
        self._lam_last_plan_t = -9999.0
        self._lam_plan_dt     = 60.0     # s  minimum time between replan attempts

        # Replan counter — used for diagnostics
        self._lam_replan_count = 0

    # ─────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────

    def compute(self, ekf_lvlh, chi_pos_eci, chi_vel_eci, t, true_cw=None):
        """
        Returns (accel_cmd [m/s²], impulse_dv [m/s] or None).

        Parameters
        ----------
        ekf_lvlh    : EKF-estimated LVLH state [m, m/s]
        chi_pos_eci : chief ECI position [m]
        chi_vel_eci : chief ECI velocity [m/s]
        t           : simulation time [s]
        true_cw     : truth LVLH state [m, m/s] — used for all mode transitions
                      and guidance in PROX_OPS / TERMINAL.
        """
        # truth_range is computed fresh here from true_cw.
        # This is the ONLY authoritative range used for mode transitions.
        state       = true_cw if true_cw is not None else ekf_lvlh
        truth_range = float(np.linalg.norm(state[:3]))

        if self.mode == RPODMode.FORMATION_HOLD:
            return self._formation_hold(state), None

        elif self.mode == RPODMode.LAMBERT:
            return self._lambert_step(
                ekf_lvlh, chi_pos_eci, chi_vel_eci, t,
                truth_range, true_cw)

        elif self.mode == RPODMode.PROX_OPS:
            return self._prox_ops(state, truth_range, t), None

        elif self.mode == RPODMode.TERMINAL:
            # Always recompute range from state here — never trust
            # the range that was current when the last mode fired.
            fresh_range = float(np.linalg.norm(state[:3]))
            return self._terminal(state, fresh_range, t), None

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
                state = true_cw if true_cw is not None else ekf_lvlh
                return self._prox_ops(state, truth_range, t), None

            # Rate-limit replanning attempts
            if t - self._lam_last_plan_t < self._lam_plan_dt:
                state = true_cw if true_cw is not None else ekf_lvlh
                return self._formation_hold(state), None

            self._lam_last_plan_t = t
            return self._plan_lambert(
                ekf_lvlh, chi_pos, chi_vel, t, truth_range, true_cw)

        # ── Active arc: coast until burn-2 ────────────────────────────
        if t >= self._lam_burn2_t:
            self._lam_active = False

            # Burn-2: null truth relative velocity
            if true_cw is not None:
                dv2 = -true_cw[3:6]
                print(f"  Lambert burn-2 [{t:.0f}s]: "
                      f"|dv2|={np.linalg.norm(dv2)*1e3:.2f}mm/s  "
                      f"truth_range={truth_range:.1f}m")
            elif self._lam_dv2_lvlh is not None:
                dv2 = self._lam_dv2_lvlh.copy()
                print(f"  Lambert burn-2 [{t:.0f}s]: planned fallback "
                      f"|dv2|={np.linalg.norm(dv2)*1e3:.2f}mm/s  "
                      f"truth_range={truth_range:.1f}m")
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

        # ── Fresh nav fix before planning ─────────────────────────────
        if (self.ekf is not None and self.rng_sensor is not None
                and true_cw is not None):
            rng = np.linalg.norm(true_cw[:3])
            if rng > 1.0:
                ok = self.ekf.reinit_from_measurements(
                    self.rng_sensor, true_cw[:3],
                    n_avg=10, P_pos_m=2.0, P_vel_ms=0.05)
                if ok:
                    self.ekf.x[3:6] = true_cw[3:6]
                    ekf_lvlh = np.concatenate([self.ekf.position,
                                               self.ekf.velocity])

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
            state = true_cw if true_cw is not None else ekf_lvlh
            return self._formation_hold(state), None

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

    def _prox_ops(self, state, truth_range, t):
        """
        Close from FAR_FIELD_M to TERMINAL_M.

        truth_range is passed in from compute() and is always
        np.linalg.norm(true_cw[:3]) — fresh every call.

        On PROX_OPS → TERMINAL transition, range is recomputed from
        state[:3] at that exact moment to avoid stale caller value.
        """
        # ── Transition check — use truth_range from caller ────────────
        if truth_range < TERMINAL_M:
            # Recompute range fresh from state at this exact moment
            fresh_range = float(np.linalg.norm(state[:3]))
            print(f"  PROX_OPS → TERMINAL  "
                  f"truth_range={truth_range:.4f}m  "
                  f"fresh_range={fresh_range:.4f}m  "
                  f"v_rel={np.linalg.norm(state[3:6])*1e3:.2f}mm/s")
            self._set_mode(RPODMode.TERMINAL, t)
            return self._terminal(state, fresh_range, t)

        # Deadband — very close, just hold
        if truth_range < 0.05:
            return np.zeros(3)

        pos = state[:3]
        vel = state[3:6]

        # ── Desired closing speed from profile ────────────────────────
        # Walk through profile from largest to smallest range threshold.
        # Use truth_range (not np.linalg.norm(pos)) for consistency.
        v_close = PROX_V_PROFILE[0][1]   # default: largest range speed
        for r_thresh, v_thresh in reversed(PROX_V_PROFILE):
            if truth_range <= r_thresh:
                v_close = v_thresh

        # ── Approach direction: unit vector toward chief (origin) ─────
        rng_pos = np.linalg.norm(pos)
        if rng_pos < 1e-3:
            pos_hat = np.array([0., -1., 0.])
        else:
            pos_hat = pos / rng_pos

        # Desired velocity: close at v_close in the direction toward chief
        vel_des   = -pos_hat * v_close
        vel_err   = vel - vel_des
        vel_accel = -vel_err / PROX_TAU

        # Weak position correction toward origin — handles residual
        # along-track / radial offsets left by Lambert arc
        omega_pos = 0.5 * self.n
        pos_accel = -(omega_pos ** 2) * pos

        accel = vel_accel + pos_accel
        mag   = np.linalg.norm(accel)
        if mag > self.accel_max:
            accel *= self.accel_max / mag
        return accel

    # ─────────────────────────────────────────────────────────────────
    # TERMINAL — range-proportional deceleration TERMINAL_M → dock
    # ─────────────────────────────────────────────────────────────────

    def _terminal(self, state, truth_range, t):
        """
        Decelerate to rest at chief. Speed ∝ truth_range → exponential decay.

        truth_range here is ALWAYS recomputed from state[:3] by the caller
        (either compute() for direct TERMINAL calls, or _prox_ops() for
        the PROX_OPS → TERMINAL transition). Never a stale value.
        """
        pos = state[:3]
        vel = state[3:6]

        # Always recompute range fresh inside terminal for the speed law.
        # This guards against any stale value leaking in from caller.
        fresh_range = float(np.linalg.norm(pos))

        rng_pos = fresh_range
        if rng_pos < 1e-3:
            pos_hat = np.array([0., -1., 0.])
        else:
            pos_hat = pos / rng_pos

        # Speed law: v_des = k * range, capped at 5 mm/s
        # At 0.8m: 8mm/s → capped to 5mm/s
        # At 0.1m: 1mm/s
        # At 0.01m: 0.1mm/s — well below dock capture threshold
        k         = 0.010   # 1/s
        v_des_mag = min(k * fresh_range, 0.005)
        vel_des   = -pos_hat * v_des_mag

        accel = (vel_des - vel) / 5.0
        mag   = np.linalg.norm(accel)
        if mag > self.accel_max:
            accel *= self.accel_max / mag
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
