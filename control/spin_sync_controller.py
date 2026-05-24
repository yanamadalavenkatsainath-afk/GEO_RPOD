import numpy as np


class SpinSyncController:
    """
    Chaser attitude command for target spin synchronisation.

    The controller produces a desired body rate and an alignment quaternion
    target for a deputy docking axis. It is intentionally separate from the
    translational RPOD controller because spin-sync couples ADCS and relative
    navigation and should be enabled only in dedicated experiments.
    """

    def __init__(self, rate_blend=0.35, max_rate_rad_s=np.radians(0.5)):
        self.rate_blend = float(rate_blend)
        self.max_rate_rad_s = float(max_rate_rad_s)
        self.omega_cmd = np.zeros(3)

    def compute_rate_command(self, omega_target_lvlh, R_lvlh_to_body):
        omega_body = R_lvlh_to_body @ np.asarray(omega_target_lvlh, dtype=float)
        mag = float(np.linalg.norm(omega_body))
        if mag > self.max_rate_rad_s:
            omega_body *= self.max_rate_rad_s / mag
        self.omega_cmd = ((1.0 - self.rate_blend) * self.omega_cmd
                          + self.rate_blend * omega_body)
        return self.omega_cmd.copy()

    @staticmethod
    def sync_quality(omega_body, omega_cmd):
        denom = max(np.linalg.norm(omega_cmd), 1e-9)
        err = float(np.linalg.norm(np.asarray(omega_body) - np.asarray(omega_cmd)))
        return float(np.clip(1.0 - err / denom, 0.0, 1.0))
