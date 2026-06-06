"""
sensors/dock_port_sensor.py - Close-range dock port position sensor.

Encapsulates the full measurement pipeline:
  truth port position (LVLH)
    -> range-dependent Gaussian noise
    -> PortTracker (gated alpha filter)
    -> filtered port position estimate

The visibility gate (range and RPOD mode) is the caller's responsibility;
it requires flight-mode awareness that belongs in the flight loop, not the
sensor. The sensor is intentionally independent of full-chief body-camera
feature tracking.
"""

import numpy as np
from estimation.port_tracker import PortTracker


class DockPortSensor:
    """
    Dock port position sensor with range-dependent noise and gated filtering.

    Parameters
    ----------
    alpha             : PortTracker smoothing factor
    innovation_gate_m : PortTracker innovation gate [m]
    noise_base_m      : minimum position noise 1-sigma [m]
    noise_range_frac  : noise coefficient proportional to range [1/m]
    """

    def __init__(self,
                 alpha: float = 0.40,
                 innovation_gate_m: float = 0.25,
                 noise_base_m: float = 0.01,
                 noise_range_frac: float = 0.002):
        self._tracker = PortTracker(alpha=alpha,
                                    innovation_gate_m=innovation_gate_m)
        self._noise_base_m = float(noise_base_m)
        self._noise_range_frac = float(noise_range_frac)

    def update(self,
               port_lvlh_true,
               range_m: float,
               dt: float,
               measurement_valid: bool = True,
               rng=None,
               noise_scale: float = 1.0):
        """
        Update with a new measurement opportunity.

        Parameters
        ----------
        port_lvlh_true    : (3,) true port position in LVLH [m]
        range_m           : deputy-to-chief range for noise scaling [m]
        dt                : timestep [s]
        measurement_valid : False when the independent port sensor is outside
                            its acquisition/range window
        rng               : numpy Generator for reproducible noise; None -> np.random
        noise_scale       : stress-profile multiplier (MC use only)

        Returns
        -------
        estimate : (3,) filtered port position in LVLH [m]
        valid    : True once tracker has been initialised
        """
        if measurement_valid:
            sigma = noise_scale * max(self._noise_base_m,
                                      self._noise_range_frac * range_m)
            _rng = rng if rng is not None else np.random
            noise = _rng.normal(0.0, sigma, 3)
            meas = np.asarray(port_lvlh_true, dtype=float) + noise
        else:
            meas = np.zeros(3)
        return self._tracker.update(meas, dt, measurement_valid=measurement_valid)

    @property
    def estimate(self) -> np.ndarray:
        return self._tracker.pos.copy()

    @property
    def is_valid(self) -> bool:
        return self._tracker.initialized

    def reset(self):
        self._tracker.reset()
