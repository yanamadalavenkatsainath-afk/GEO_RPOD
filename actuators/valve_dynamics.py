"""Per-channel command delay and exact held-target first-order valve response."""
from collections import deque
import numpy as np
from utils.validation import finite_array


def channels(value,count,name):
    array=finite_array(value,name)
    if array.ndim==0:array=np.full(count,float(array))
    if array.shape!=(count,) or np.any(array<0):raise ValueError(name+' requires nonnegative channel values')
    return array


class ValveBank:
    """Integrates nonnegative force targets, returning interval impulse.

    The caller owns the contiguous actuator timeline and physical capacities.
    Response is a declared lumped lag, not a pressure/thermal valve simulation.
    Isolation immediately removes stored response and queued commands.
    """
    def __init__(self,count,*,delay_s=0.,time_constant_s=0.):
        self.delay=channels(delay_s,count,'delay_s')
        self.tau=channels(time_constant_s,count,'time_constant_s')
        self.count=count
        self.reset()

    def reset(self):
        self.output=np.zeros(self.count)
        self.target=np.zeros(self.count)
        self.last_command=np.full(self.count,np.nan)
        self.events=[deque() for _ in range(self.count)]

    def integrate(self,start_s,duration_s,targets,isolated):
        targets=channels(targets,self.count,'targets')
        impulse=np.zeros(self.count)
        for k in range(self.count):
            if isolated[k]:
                self.events[k].clear();self.output[k]=self.target[k]=0.;self.last_command[k]=np.nan
                continue
            if self.last_command[k]!=targets[k]:
                self.events[k].append((start_s+self.delay[k],targets[k]))
                self.last_command[k]=targets[k]
            current=start_s;end=start_s+duration_s
            while current<end:
                while self.events[k] and self.events[k][0][0]<=current:
                    _,self.target[k]=self.events[k].popleft()
                boundary=min(end,self.events[k][0][0]) if self.events[k] else end
                h=boundary-current
                if h>0:
                    target=self.target[k]
                    if self.tau[k]==0:
                        impulse[k]+=target*h;self.output[k]=target
                    else:
                        change=-np.expm1(-h/self.tau[k])
                        impulse[k]+=target*h+(self.output[k]-target)*self.tau[k]*change
                        self.output[k]+=(target-self.output[k])*change
                    current=boundary
        return np.maximum(0.,impulse)
