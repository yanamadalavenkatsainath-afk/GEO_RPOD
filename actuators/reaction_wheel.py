import numpy as np

class ReactionWheel:

    def __init__(self, h_max=0.05):
        self.h = np.zeros(3)
        self.h_max = h_max

    def apply_torque(self, torque_cmd, dt):
        self.h += torque_cmd * dt
        self.h = np.clip(self.h, -self.h_max, self.h_max)
        return torque_cmd
