import numpy as np

class BDotController:
    def __init__(self, k_bdot=1e5, m_max=0.2):
        self.k_bdot = k_bdot
        self.m_max  = m_max

    def compute(self, B_body, omega_body, B_inertial, dt):
        """
        True B-dot in body frame.
        B_dot_body = dB_body/dt = -omega x B_body  (for rotating body)
        This is the physically correct formulation.
        """
        # Compute B_dot analytically from rotation
        # dB/dt = -omega x B  in body frame
        B_dot = -np.cross(omega_body, B_body)

        m_cmd  = -self.k_bdot * B_dot
        m_cmd  = np.clip(m_cmd, -self.m_max, self.m_max)
        torque = np.cross(m_cmd, B_body)

        return m_cmd, torque