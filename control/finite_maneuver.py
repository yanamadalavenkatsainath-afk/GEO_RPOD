"""Finite command sequencing with separate achieved-impulse evidence."""
import numpy as np
from utils.validation import vector,finite_scalar


class FiniteManeuver:
    """Hold a fixed reference-frame delta-v direction until commands are issued.

    Consuming the command budget is not a successful burn. The caller records
    achieved execution separately, screens residual uncertainty, and replans.
    No truth delta-v closes this command sequencer's guidance loop.
    """
    def __init__(self,*,maximum_acceleration_m_s2):
        self.limit=finite_scalar(maximum_acceleration_m_s2,'maximum_acceleration_m_s2',strictly_positive=True)
        self.active=None;self.history=[]

    def start(self,delta_v_reference_m_s,*,time_s,phase):
        delta=vector(delta_v_reference_m_s,'delta_v_reference_m_s',size=3)
        if self.active is not None:raise RuntimeError('Cannot overwrite an unfinished finite maneuver')
        magnitude=float(np.linalg.norm(delta))
        if magnitude==0:return
        record=dict(start_s=finite_scalar(time_s,'time_s',minimum=0),phase=str(phase),
            commanded_delta_v_m_s=delta.tolist(),issued_delta_v_m_s=[0.,0.,0.],achieved_delta_v_m_s=[0.,0.,0.],
            command_sequence_complete=False,aborted=False)
        self.active=dict(direction=delta/magnitude,remaining=magnitude,record=record)
        self.history.append(record)

    def command(self,duration_s,*,permitted=True):
        dt=finite_scalar(duration_s,'duration_s',strictly_positive=True)
        if self.active is None or not permitted:return np.zeros(3)
        amount=min(self.limit*dt,self.active['remaining'])
        delta=self.active['direction']*amount
        self.active['remaining']-=amount
        record=self.active['record']
        record['issued_delta_v_m_s']=(np.asarray(record['issued_delta_v_m_s'])+delta).tolist()
        # Keep active through recording the final interval's achieved execution.
        return delta/dt

    def record_execution(self,delta_v_reference_m_s,*,end_s,sequence_interval_complete=True):
        if self.active is None:return
        record=self.active['record']
        record['achieved_delta_v_m_s']=(np.asarray(record['achieved_delta_v_m_s'])+
            vector(delta_v_reference_m_s,'delta_v_reference_m_s',size=3)).tolist()
        if self.active['remaining']<=1e-12 and sequence_interval_complete:
            record['command_sequence_complete']=True;record['end_s']=float(end_s)
            record['vector_error_m_s']=float(np.linalg.norm(np.asarray(record['achieved_delta_v_m_s'])-record['commanded_delta_v_m_s']))
            self.active=None

    def abort(self,*,time_s,reason):
        if self.active is None:return
        self.active['record'].update(aborted=True,end_s=float(time_s),abort_reason=str(reason))
        self.active=None
