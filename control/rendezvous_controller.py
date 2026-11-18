"""
Relative Motion Controller — Formation Hold + Rendezvous
=========================================================
Implements two modes:

1. FORMATION_HOLD
   Maintains a target relative position using continuous PD control
   in LVLH. Corrects natural drift from CW secular terms by keeping
   δẏ = -2n·δx (drift-free condition).

2. RENDEZVOUS
   Executes a two-impulse CW rendezvous maneuver (Hohmann-like in
   relative motion space). Computes optimal delta-V burns at t=0 and
   t=T/2 to drive deputy to the chief's location (origin).

   The two-impulse transfer solves:
       [δx(T), δy(T), δz(T)] = 0
   given the current state and transfer time T.

3. STATION_KEEPING (passive safety ellipse maintenance)
   Applies small corrective impulses to maintain PSE shape when
   perturbations cause deviation beyond a tolerance.

Reference:
    Schaub & Junkins, "Analytical Mechanics of Space Systems", §14.4
    Gaias & D'Amico, JGCD 38(6), 2015.
    Woffinden & Geller, JGCD 30(6), 2007.
"""

import numpy as np
from enum import Enum, auto
from environment.cw_dynamics import CWDynamics


class RelNavMode(Enum):
    FORMATION_HOLD  = auto()
    RENDEZVOUS      = auto()
    STATION_KEEPING = auto()
    COASTING        = auto()    # no thrust, free drift


class RendezvousController:
    """
    Relative motion controller for formation hold and rendezvous.

    Parameters
    ----------
    cw          : CWDynamics instance (shares state with main propagator)
    mode        : initial RelNavMode
    target_lvlh : target relative position for formation hold [m]
    """

    def __init__(self,
                 n:           float,
                 mode:        RelNavMode = RelNavMode.FORMATION_HOLD,
                 target_lvlh: np.ndarray = None):

        self.n    = n
        self.mode = mode

        # Formation hold target [m]
        if target_lvlh is None:
            self.target = np.array([0.0, 100.0, 0.0])   # 100 m along-track
        else:
            self.target = np.array(target_lvlh)

        # PD gains for continuous formation hold
        # Tuned: position bandwidth ~0.1n, damping ratio ~0.7
        omega_c  = 0.1 * n           # control bandwidth
        zeta     = 0.7               # damping
        self.Kp  = omega_c**2
        self.Kd  = 2.0 * zeta * omega_c

        # Rendezvous state
        self._rdv_burn1_applied = False
        self._rdv_burn2_applied = False
        self._rdv_t_start       = None
        self._rdv_T             = None   # transfer time
        self._rdv_dv1           = None
        self._rdv_dv2           = None

        # Thrust magnitude cap [m/s²] (~0.5 mN thruster on 3 kg CubeSat)
        self.accel_max = 1e-4   # m/s²

        # Mode history
        self.mode_history = []

    # ─────────────────────────────────────────────────────────────────
    # Main update — call every timestep
    # ─────────────────────────────────────────────────────────────────

    def compute(self,
                state_est:  np.ndarray,
                t:          float
                ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Compute control output.

        Parameters
        ----------
        state_est : estimated relative state [δx,δy,δz,δẋ,δẏ,δż] [m, m/s]
        t         : current simulation time [s]

        Returns
        -------
        accel_cmd  : continuous thrust command [m/s²] in LVLH
        impulse_dv : instantaneous delta-V [m/s] if a burn is due, else None
        """
        if self.mode == RelNavMode.FORMATION_HOLD:
            accel = self._formation_hold(state_est)
            return accel, None

        elif self.mode == RelNavMode.RENDEZVOUS:
            accel, dv = self._rendezvous(state_est, t)
            return accel, dv

        elif self.mode == RelNavMode.STATION_KEEPING:
            accel = self._station_keeping(state_est)
            return accel, None

        else:   # COASTING
            return np.zeros(3), None

    def set_mode(self, mode: RelNavMode, t: float = 0.0,
                 target_lvlh: np.ndarray = None):
        """
        Switch controller mode.

        Parameters
        ----------
        mode         : new RelNavMode
        t            : current time (needed to plan rendezvous burns)
        target_lvlh  : new formation hold target [m]
        """
        if target_lvlh is not None:
            self.target = np.array(target_lvlh)

        if mode == RelNavMode.RENDEZVOUS:
            self._rdv_t_start       = t
            self._rdv_burn1_applied = False
            self._rdv_burn2_applied = False

        print(f"  RelNav [{t:.1f}s]: {self.mode.name} → {mode.name}")
        self.mode = mode
        self.mode_history.append((t, mode))

    # ─────────────────────────────────────────────────────────────────
    # Formation hold — continuous PD
    # ─────────────────────────────────────────────────────────────────

    def _formation_hold(self, state: np.ndarray) -> np.ndarray:
        """
        PD controller about target relative position.
        Includes feed-forward CW coupling cancellation.
        """
        dr  = state[0:3]
        dv  = state[3:6]
        err = dr - self.target

        # PD feedback
        accel = -self.Kp * err - self.Kd * dv

        # Cancel the current-state CW terms, leaving the commanded PD
        # acceleration in the two-body baseline (not only gravity at target).
        accel[0] -= 3.0*self.n**2*dr[0] + 2.0*self.n*dv[1]
        accel[1] += 2.0*self.n*dv[0]
        accel[2] += self.n**2*dr[2]

        # Cap thrust
        mag = np.linalg.norm(accel)
        if mag > self.accel_max:
            accel = accel / mag * self.accel_max

        return accel

    # ─────────────────────────────────────────────────────────────────
    # Rendezvous — two-impulse CW transfer
    # ─────────────────────────────────────────────────────────────────

    def plan_rendezvous(self,
                        state_est:    np.ndarray,
                        transfer_time: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute two-impulse CW rendezvous burns.

        Given current state x0, find Δv1 (now) and Δv2 (at t=T) such that
        the deputy arrives at the origin (chief position) at t=T.

        Uses CW state transition matrix to solve the boundary value problem:
            x(T) = Φ(T)·[x0 + B·Δv1]
            [δr(T)] = 0  → solve for Δv1
            Δv2 = -δv(T) to zero final velocity

        Parameters
        ----------
        state_est     : current estimated state
        transfer_time : desired transfer duration [s]

        Returns
        -------
        dv1, dv2 : delta-V vectors [m/s] in LVLH
        """
        n  = self.n
        T  = transfer_time
        nt = n * T
        s  = np.sin(nt)
        c  = np.cos(nt)

        # CW state transition matrix partitioned as Φ = [[Φrr, Φrv],[Φvr, Φvv]]
        Phi_rr = np.array([
            [4-3*c,    0,  0],
            [6*(s-nt), 1,  0],
            [0,        0,  c],
        ])
        Phi_rv = np.array([
            [s/n,         2*(1-c)/n,    0   ],
            [-2*(1-c)/n,  (4*s-3*nt)/n, 0   ],
            [0,           0,            s/n ],
        ])
        Phi_vr = np.array([
            [3*n*s,      0, 0   ],
            [6*n*(c-1),  0, 0   ],
            [0,          0, -n*s],
        ])
        Phi_vv = np.array([
            [c,    2*s,   0],
            [-2*s, 4*c-3, 0],
            [0,    0,     c],
        ])

        r0 = state_est[0:3]
        v0 = state_est[3:6]

        # Solve: Phi_rr·r0 + Phi_rv·(v0 + dv1) = 0
        # => dv1 = -Phi_rv^{-1}·(Phi_rr·r0 + Phi_rv·v0)
        #
        # Guard: Phi_rv is singular at nt = k*pi (sin=0).
        # Check condition number and bail out if ill-conditioned.
        cond = np.linalg.cond(Phi_rv)
        if cond > 1e8:
            # Ill-conditioned — caller must retry with a different T
            return None, None

        try:
            rhs  = Phi_rr @ r0 + Phi_rv @ v0
            dv1  = -np.linalg.solve(Phi_rv, rhs)
        except np.linalg.LinAlgError:
            return None, None

        # State at arrival
        v0_after_burn1 = v0 + dv1
        r_T = Phi_rr @ r0 + Phi_rv @ v0_after_burn1
        v_T = Phi_vr @ r0 + Phi_vv @ v0_after_burn1

        # dv2 cancels final velocity (soft-dock condition)
        dv2 = -v_T

        return dv1, dv2

    def _rendezvous(self,
                    state:  np.ndarray,
                    t:      float
                    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Execute planned rendezvous sequence."""

        if self._rdv_T is None:
            # Scan transfer time fractions to find well-conditioned solution.
            # Phi_rv is singular at nt = k*pi — avoid those.
            # Pick the transfer time with minimum total delta-V subject to
            # a hard cap on |dv1| (prevents runaway burns from stale EKF state).
            T_orb  = 2.0 * np.pi / self.n
            DV_CAP = 0.10   # m/s — any individual burn above this is rejected
            best_T, best_dv1, best_dv2, best_total = None, None, None, np.inf

            for frac in np.linspace(0.15, 0.95, 80):   # finer scan
                T_try  = frac * T_orb
                nt_try = self.n * T_try
                k_near = round(nt_try / np.pi)
                if abs(nt_try - k_near * np.pi) < 0.15:   # skip near singularity
                    continue
                result = self.plan_rendezvous(state, T_try)
                if result[0] is None:
                    continue
                dv1_try, dv2_try = result
                # Reject solutions with runaway individual burns
                if np.linalg.norm(dv1_try) > DV_CAP:
                    continue
                if np.linalg.norm(dv2_try) > DV_CAP:
                    continue
                total_try = np.linalg.norm(dv1_try) + np.linalg.norm(dv2_try)
                if total_try < best_total:
                    best_total = total_try
                    best_T, best_dv1, best_dv2 = T_try, dv1_try, dv2_try

            if best_T is None:
                print("  WARNING: No valid rendezvous transfer found — reverting to FORMATION_HOLD")
                self.mode = RelNavMode.FORMATION_HOLD
                return np.zeros(3), None

            self._rdv_T          = best_T
            self._rdv_dv1        = best_dv1
            self._rdv_dv2        = best_dv2
            print(f"  Rendezvous planned: T={self._rdv_T/60:.1f} min, "
                  f"|Dv1|={np.linalg.norm(best_dv1)*1000:.2f} mm/s, "
                  f"|Dv2|={np.linalg.norm(best_dv2)*1000:.2f} mm/s, "
                  f"SumDv={best_total*1000:.2f} mm/s")

        dt_rdv = t - self._rdv_t_start

        # Burn 1: at start of rendezvous
        if not self._rdv_burn1_applied:
            self._rdv_burn1_applied = True
            return np.zeros(3), self._rdv_dv1

        # Burn 2: at scheduled T
        if (not self._rdv_burn2_applied and
                dt_rdv >= self._rdv_T - 0.5):
            self._rdv_burn2_applied = True
            self.mode   = RelNavMode.COASTING
            self.target = np.zeros(3)
            return np.zeros(3), self._rdv_dv2

        return np.zeros(3), None

    # ─────────────────────────────────────────────────────────────────
    # Station keeping — PSE maintenance
    # ─────────────────────────────────────────────────────────────────

    def _station_keeping(self, state: np.ndarray) -> np.ndarray:
        """
        Maintain passive safety ellipse. Applies drift correction only —
        damps the secular y-drift by restoring the drift-free condition.
        """
        n  = self.n
        dr = state[0:3]
        dv = state[3:6]

        # Drift-free condition: ẏ = -2n·x
        vy_ideal = -2.0 * n * dr[0]
        vy_err   = dv[1] - vy_ideal

        # Correct y-velocity error with a small along-track acceleration
        # Only act if error exceeds 0.01 m/s to avoid thruster chatter
        accel = np.zeros(3)
        if abs(vy_err) > 0.01:
            accel[1] = -self.Kd * vy_err
            accel[1] = np.clip(accel[1], -self.accel_max, self.accel_max)

        return accel