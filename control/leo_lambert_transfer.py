"""GEO Lambert seed plus LEO full-force, two-finite-burn shooting.

Reuses the flight-sim universal-variable solver. All inputs are navigation
states, known chief ephemeris and declared force-model callbacks. No truth
deputy state or achieved delta-v is used to close the planning loop.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from control.lambert_solver import LambertSolver
from utils.orbital_frames import eci_to_lvlh_rotation,lvlh_frame_rate
from utils.validation import finite_scalar,vector


class FiniteLambertTransfer:
    def __init__(self,*,mu,maximum_acceleration_m_s2=1e-4,target_rtn_m=(0.,12.,0.)):
        self.mu=finite_scalar(mu,'mu',strictly_positive=True)
        self.limit=finite_scalar(maximum_acceleration_m_s2,'maximum_acceleration_m_s2',strictly_positive=True)
        self.target=vector(target_rtn_m,'target_rtn_m',size=3)
        self.solver=LambertSolver(mu=self.mu)

    @staticmethod
    def propagate(state,start,end,acceleration,extra=None,dense=False):
        def rhs(t,s):
            a=acceleration(t,s[:3],s[3:])
            if extra is not None:a=a+extra
            return np.r_[s[3:],a]
        result=solve_ivp(rhs,(start,end),state,rtol=2e-11,atol=1e-7,max_step=30.,dense_output=dense)
        if not result.success:raise RuntimeError('Lambert prediction propagation failed')
        return result

    def plan(self,*,chief_state,deputy_state,time_s,tof_s,chief_acceleration,deputy_acceleration):
        chief=vector(chief_state,'chief_state',size=6);deputy=vector(deputy_state,'deputy_state',size=6)
        start=finite_scalar(time_s,'time_s',minimum=0);tof=finite_scalar(tof_s,'tof_s',strictly_positive=True)
        end=start+tof
        chief_end=self.propagate(chief,start,end,chief_acceleration).y[:,-1]
        frame=eci_to_lvlh_rotation(chief_end[:3],chief_end[3:])
        omega=lvlh_frame_rate(chief_end[:3],chief_end[3:],chief_acceleration(end,chief_end[:3],chief_end[3:]))
        target_position=chief_end[:3]+frame.T@self.target
        target_velocity=chief_end[3:]+frame.T@np.cross(omega,self.target)
        free=self.propagate(deputy,start,end,deputy_acceleration).y[:,-1]
        two_r,two_v=self.solver.propagate_keplerian(deputy[:3],deputy[3:],tof,n_steps=max(200,int(np.ceil(tof/5))))
        # Same full-force target-offset correction used by GEO Lambert: seed
        # with the difference between perturbed and two-body deputy motion.
        seed_target=target_position-(free[:3]-two_r)
        v1,v2=self.solver.solve(deputy[:3],seed_target,tof)
        if v1 is None:raise RuntimeError('No Lambert seed for this transfer time')
        guess=np.r_[v1-deputy[3:],target_velocity-(v2+free[3:]-two_v)]
        # This bound guarantees non-overlap for every optimizer evaluation.
        bound=min(.10,self.limit*tof/(2.1*np.sqrt(3)))
        guess=np.clip(guess,-.99*bound,.99*bound)

        def predict(delta):
            first,second=delta[:3],delta[3:]
            d1=float(np.linalg.norm(first)/self.limit);d2=float(np.linalg.norm(second)/self.limit)
            state=deputy.copy()
            for a,b,command in ((start,start+d1,first/max(d1,1e-30)),
                                (start+d1,end-d2,None),(end-d2,end,second/max(d2,1e-30))):
                if b-a>1e-10:state=self.propagate(state,a,b,deputy_acceleration,command).y[:,-1]
            return state,d1,d2

        def residual(delta):
            state,_,_=predict(delta)
            return np.r_[(state[:3]-target_position)/100.,(state[3:]-target_velocity)/.1]

        solution=least_squares(residual,guess,bounds=(-bound,bound),diff_step=1e-3,
            x_scale=.03,max_nfev=30,ftol=1e-9,xtol=1e-9,gtol=1e-9)
        arrival,d1,d2=predict(solution.x)
        miss=float(np.linalg.norm(arrival[:3]-target_position));speed=float(np.linalg.norm(arrival[3:]-target_velocity))
        if miss>.5 or speed>.001:raise RuntimeError(f'Finite Lambert did not converge: {miss:.3f} m, {speed:.6f} m/s')
        return dict(planner='flight_sim_Lambert_seed_full_force_finite_shooting',start_s=start,end_s=end,
            departure_delta_v_m_s=solution.x[:3].tolist(),arrival_delta_v_m_s=solution.x[3:].tolist(),
            arrival_burn_start_s=end-d2,departure_duration_s=d1,arrival_duration_s=d2,
            target_rtn_m=self.target.tolist(),predicted_position_error_m=miss,predicted_speed_error_m_s=speed,
            maximum_acceleration_m_s2=self.limit,optimizer_evaluations=solution.nfev)
