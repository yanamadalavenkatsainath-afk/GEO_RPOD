"""Initial mission-inspired handoff qualification, not complete abort guidance."""
import numpy as np


class HandoffSupervisor:
    def __init__(self,hold_s=5.):
        if not np.isfinite(hold_s) or hold_s<=0:raise ValueError('Positive hold duration required')
        self.required=float(hold_s);self.held=0.;self.phase='FAR_RENDEZVOUS'

    def update(self,estimate,dt,*,navigation_fresh,actuation_ready,burn_active):
        state=np.asarray(estimate,dtype=float)
        if state.shape!=(6,) or not np.all(np.isfinite(state)) or not np.isfinite(dt) or dt<=0:
            raise ValueError('Finite six-state and positive dt required')
        inside=np.linalg.norm(state[:3])<20. and np.linalg.norm(state[3:])<.15
        qualified=inside and navigation_fresh and actuation_ready and not burn_active
        if not qualified:
            self.held=0.;self.phase='HOLD_INHIBITED' if inside else 'FAR_RENDEZVOUS'
            return False
        self.held+=dt;self.phase='TERMINAL' if self.held+1e-9>=self.required else 'HANDOFF_HOLD'
        return self.phase=='TERMINAL'
