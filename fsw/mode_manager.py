import numpy as np
from enum import Enum, auto


class Mode(Enum):
    """
    FSW operating modes in priority order.
    Transitions are one-way downward (safe→detumble→sun_acq→fine_pointing)
    except SAFE_MODE which can be entered from any state.
    """
    SAFE_MODE       = auto()   # fault state — all actuators off
    DETUMBLE        = auto()   # B-dot rate damping
    SUN_ACQUISITION = auto()   # slew to sun-pointing for TRIAD geometry
    FINE_POINTING   = auto()   # MEKF + PD control
    MOMENTUM_DUMP   = auto()   # magnetorquer desaturation, wheels biased


class ModeManager:
    """
    FSW Mode Management State Machine — 3U CubeSat.

    Manages transitions between operating modes based on sensor
    measurements and actuator states. Each mode has:
        - Entry condition  : what triggers transition INTO this mode
        - Exit condition   : what triggers transition OUT of this mode
        - Active actuators : what runs while in this mode

    Transition map:
        SAFE_MODE       → DETUMBLE        : rate < safe_rate_threshold AND no faults
        DETUMBLE        → SUN_ACQUISITION : rate < detumble_threshold
        SUN_ACQUISITION → FINE_POINTING   : TRIAD error < triad_threshold OR timeout
        FINE_POINTING   → MOMENTUM_DUMP   : any wheel |h| > dump_trigger
        MOMENTUM_DUMP   → FINE_POINTING   : all wheels |h| < dump_complete
        ANY             → SAFE_MODE       : rate > safe_rate_threshold OR fault flag

    Reference:
        Wertz, "Space Mission Engineering", §11.3 (ADCS mode management)
        Sidi, "Spacecraft Dynamics and Control", §9.4
    """

    # ── Transition thresholds ─────────────────────────────────────────
    SAFE_RATE_THRESHOLD   = np.radians(40.0)   # rad/s — above this → SAFE_MODE
    DETUMBLE_THRESHOLD    = np.radians(3.5)    # rad/s = 2.9 deg/s exit condition
    TRIAD_ERR_THRESHOLD   = 15.0               # deg — TRIAD accepted below this
    SUN_ACQ_TIMEOUT       = 600.0              # s — give up sun acq after this long

    # Momentum dump thresholds — scaled to 75% / 20% of h_max (4.0 N·m·s).
    # 50 kg jetpack, Bradford WSAT wheels (4.0 N·m·s).
    DUMP_TRIGGER          = 3.000              # N·m·s — start dump above this |h|  (75% of 4.0 N·m·s)
    DUMP_COMPLETE         = 0.800              # N·m·s — end dump below this |h|    (20% of 4.0 N·m·s)

    # Only enter MOMENTUM_DUMP when pointing is already good (deg).
    # Prevents thrashing: large pointing errors cause large wheel torques -> instant re-trigger.
    DUMP_POINTING_GUARD   = 5.0                # deg — skip dump entry if err > this

    def __init__(self):
        self.mode           = Mode.DETUMBLE
        self.prev_mode      = None
        self.mode_entry_t   = 0.0
        self.fault_flags    = set()

        self.triad_err_deg     = None
        self.pointing_err_deg  = None   # last known MEKF pointing error

        # History for telemetry/plotting
        self.mode_history   = []   # list of (t, mode) tuples

        print("  FSW ModeManager initialised — starting in DETUMBLE")

    # ─────────────────────────────────────────────────────────────────
    # Main update — call once per control cycle
    # ─────────────────────────────────────────────────────────────────

    def update(self, t, omega, wheel_h, triad_err_deg=None, fault=False,
               pointing_err_deg=None):
        """
        Evaluate transition conditions and update mode.

        Parameters
        ----------
        t                : simulation time [s]
        omega            : angular rate vector [rad/s]
        wheel_h          : reaction wheel momentum vector [N·m·s]
        triad_err_deg    : TRIAD initialisation error [deg], None if not run
        fault            : external fault flag
        pointing_err_deg : current MEKF pointing error [deg], used to gate
                           MOMENTUM_DUMP entry (avoids thrashing when off-target)

        Returns
        -------
        mode : current Mode enum value
        """
        rate  = np.linalg.norm(omega)
        h_max = np.max(np.abs(wheel_h))

        if pointing_err_deg is not None:
            self.pointing_err_deg = pointing_err_deg

        # ── Fault / safe mode — highest priority ─────────────────────
        if fault or rate > self.SAFE_RATE_THRESHOLD:
            if rate > self.SAFE_RATE_THRESHOLD:
                self.fault_flags.add("RATE_EXCEEDED")
            if fault:
                self.fault_flags.add("EXTERNAL_FAULT")
            self._transition(Mode.SAFE_MODE, t)
            return self.mode

        # ── Recover from safe mode once calm ─────────────────────────
        if self.mode == Mode.SAFE_MODE:
            if rate < self.DETUMBLE_THRESHOLD * 5 and not fault:
                self.fault_flags.clear()
                self._transition(Mode.DETUMBLE, t)
            return self.mode

        # ── DETUMBLE exit ─────────────────────────────────────────────
        if self.mode == Mode.DETUMBLE:
            if rate < self.DETUMBLE_THRESHOLD:
                self._transition(Mode.SUN_ACQUISITION, t)
            return self.mode

        # ── SUN_ACQUISITION exit ──────────────────────────────────────
        if self.mode == Mode.SUN_ACQUISITION:
            time_in_mode = t - self.mode_entry_t

            if triad_err_deg is not None:
                self.triad_err_deg = triad_err_deg

            triad_ok  = (self.triad_err_deg is not None and
                         self.triad_err_deg < self.TRIAD_ERR_THRESHOLD)
            timed_out = time_in_mode > self.SUN_ACQ_TIMEOUT

            if triad_ok or timed_out:
                if timed_out and not triad_ok:
                    print(f"  FSW: Sun acq timeout at t={t:.1f}s "
                          f"— proceeding with q_exit seed")
                self._transition(Mode.FINE_POINTING, t)
            return self.mode

        # ── FINE_POINTING ↔ MOMENTUM_DUMP ────────────────────────────
        if self.mode == Mode.FINE_POINTING:
            # Only trigger dump when pointing is already settled.
            # If the spacecraft is still converging, large attitude-control
            # torques would immediately re-saturate the wheels.
            pointing_ok = (self.pointing_err_deg is None or
                           self.pointing_err_deg < self.DUMP_POINTING_GUARD)
            if h_max > self.DUMP_TRIGGER and pointing_ok:
                self._transition(Mode.MOMENTUM_DUMP, t)
            return self.mode

        if self.mode == Mode.MOMENTUM_DUMP:
            if h_max < self.DUMP_COMPLETE:
                self._transition(Mode.FINE_POINTING, t)
            return self.mode

        return self.mode

    # ─────────────────────────────────────────────────────────────────
    # Mode queries
    # ─────────────────────────────────────────────────────────────────

    @property
    def is_detumbling(self):
        return self.mode == Mode.DETUMBLE

    @property
    def is_sun_acquiring(self):
        return self.mode == Mode.SUN_ACQUISITION

    @property
    def is_fine_pointing(self):
        return self.mode == Mode.FINE_POINTING

    @property
    def is_momentum_dumping(self):
        return self.mode == Mode.MOMENTUM_DUMP

    @property
    def is_safe(self):
        return self.mode == Mode.SAFE_MODE

    def time_in_mode(self, t):
        return t - self.mode_entry_t

    # ─────────────────────────────────────────────────────────────────
    # Private
    # ─────────────────────────────────────────────────────────────────

    def _transition(self, new_mode, t):
        if new_mode == self.mode:
            return
        print(f"  FSW [{t:7.1f}s] {self.mode.name:20s} → {new_mode.name}")
        self.prev_mode    = self.mode
        self.mode         = new_mode
        self.mode_entry_t = t
        self.mode_history.append((t, new_mode))