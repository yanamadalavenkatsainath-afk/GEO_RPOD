"""Two osculating Cartesian spacecraft in one inertial frame and time base.

This truth plant is independent of the secular ROE navigation model. Forces
comprise central gravity/J2 plus explicitly injected environment callbacks.
Rigid-body deputy wrench is evaluated at integration stages. Selecting the
plant alone does not select an atmosphere, geometry or ephemeris provider.
"""
import numpy as np
from environment.common import central_gravity, j2_acceleration
from utils.orbital_frames import relative_state_from_eci, relative_state_to_eci, eci_to_lvlh_rotation
from utils.validation import vector, finite_scalar


class CartesianRelativeOrbit:
    def __init__(self, chief_position_m, chief_velocity_m_s, relative_state,
                 *, mu_m3_s2=3.986004418e14, radius_m=6378100., j2=1.08263e-3,
                 max_step_s=.5, chief_environment_acceleration=None,
                 deputy_environment_acceleration=None, deputy_environment_wrench=None):
        self.t = 0.
        self.chief_environment_acceleration=chief_environment_acceleration
        self.deputy_environment_acceleration=deputy_environment_acceleration
        self.deputy_environment_wrench=deputy_environment_wrench
        for callback in (chief_environment_acceleration,deputy_environment_acceleration,deputy_environment_wrench):
            if callback is not None and not callable(callback):
                raise ValueError('Environment providers must be callable')
        self.mu = finite_scalar(mu_m3_s2, 'mu_m3_s2', strictly_positive=True)
        self.radius = finite_scalar(radius_m, 'radius_m', strictly_positive=True)
        self.j2 = finite_scalar(j2, 'j2')
        self.max_step = finite_scalar(max_step_s, 'max_step_s', strictly_positive=True)
        self.chief_position = vector(chief_position_m, 'chief_position_m', size=3)
        self.chief_velocity = vector(chief_velocity_m_s, 'chief_velocity_m_s', size=3)
        if np.linalg.norm(self.chief_position) <= self.radius:
            raise ValueError('Chief must start outside Earth')
        self.deputy_position, self.deputy_velocity = relative_state_to_eci(
            self.chief_position, self.chief_velocity, relative_state,
            chief_accel=self.chief_acceleration(self.chief_position))
        if np.linalg.norm(self.deputy_position) <= self.radius:
            raise ValueError('Deputy must start outside Earth')
        self.t = 0.
        self.total_dv_ms = 0.
        self.body = None
        self.chief_body = None

    def attach_body(self, attitude_body_to_rtn, relative_rate_rtn, *, dry_mass_kg,
                    propellant_kg, inertia_kg_m2):
        """Initialize rigid-body truth once at the existing orbital epoch.

        The supplied rate is relative to RTN; the rigid plant needs inertial
        body rate. No orbital state is replaced when attaching the body.
        """
        from plant.coupled import CoupledSpacecraft
        from spec import Timestamp
        from scipy.spatial.transform import Rotation
        from utils.orbital_frames import lvlh_frame_rate
        if self.body is not None:
            raise ValueError('Rigid body already attached')
        frame = eci_to_lvlh_rotation(self.chief_position,self.chief_velocity)
        omega = lvlh_frame_rate(self.chief_position,self.chief_velocity,
                               self.chief_acceleration(self.chief_position))
        rotation = np.asarray(attitude_body_to_rtn)
        xyzw = Rotation.from_matrix(frame.T@rotation).as_quat()
        self.body = CoupledSpacecraft(time=Timestamp(self.t),
            position_m=self.deputy_position,velocity_m_s=self.deputy_velocity,
            quaternion=np.r_[xyzw[3],xyzw[:3]],
            omega_body_rad_s=rotation.T@(relative_rate_rtn+omega),
            dry_mass_kg=dry_mass_kg,propellant_kg=propellant_kg,
            dry_inertia_kg_m2=inertia_kg_m2,max_step_s=self.max_step)

    def body_state_rtn(self):
        from utils.quaternion import rot_matrix
        from utils.orbital_frames import lvlh_frame_rate
        if self.body is None:
            raise ValueError('No rigid body attached')
        frame = eci_to_lvlh_rotation(self.chief_position,self.chief_velocity)
        rotation = frame@rot_matrix(self.body.q)
        omega = lvlh_frame_rate(self.chief_position,self.chief_velocity,
                               self.chief_acceleration(self.chief_position))
        return rotation, rotation@self.body.omega-omega

    def attach_chief_body(self,attitude_body_to_rtn,relative_rate_rtn,*,mass_kg,inertia_kg_m2):
        """Initialize target rigid dynamics once, preserving chief orbit and clock."""
        from plant.coupled import CoupledSpacecraft
        from spec import Timestamp
        from utils.quaternion import quat_from_matrix
        from utils.orbital_frames import lvlh_frame_rate
        if self.chief_body is not None:raise ValueError('Chief body already attached')
        frame=eci_to_lvlh_rotation(self.chief_position,self.chief_velocity)
        rotation=np.asarray(attitude_body_to_rtn)
        rate=lvlh_frame_rate(self.chief_position,self.chief_velocity,self.chief_acceleration(self.chief_position))
        self.chief_body=CoupledSpacecraft(time=Timestamp(self.t),position_m=self.chief_position,
            velocity_m_s=self.chief_velocity,quaternion=quat_from_matrix(frame.T@rotation),
            omega_body_rad_s=rotation.T@(np.asarray(relative_rate_rtn)+rate),dry_mass_kg=mass_kg,
            propellant_kg=0.,dry_inertia_kg_m2=inertia_kg_m2,body_frame='body/target',max_step_s=self.max_step)

    def chief_body_state_rtn(self):
        from utils.quaternion import rot_matrix
        from utils.orbital_frames import lvlh_frame_rate
        if self.chief_body is None:raise ValueError('No target rigid body')
        frame=eci_to_lvlh_rotation(self.chief_position,self.chief_velocity)
        rotation=frame@rot_matrix(self.chief_body.q)
        rate=lvlh_frame_rate(self.chief_position,self.chief_velocity,self.chief_acceleration(self.chief_position))
        return rotation,rotation@self.chief_body.omega-rate

    def step_achieved(self, achieved,*,internal_momentum_body_nms=None,chief_actuation=None,propulsive_force_body_n=None):
        """Advance chief gravity and deputy achieved body wrench on one clock.

        Chief and deputy are uncoupled gravitating test bodies, so their RK4
        integrations are independent. Body force rotates at every deputy RK
        stage, rather than being frozen as an RTN acceleration for the step.
        """
        if self.body is None:
            raise ValueError('Attach body before applying a wrench')
        dt = achieved.duration_s
        mass_mid = self.body.mass_kg-achieved.propellant_used_kg/2
        budget_force=achieved.force_n if propulsive_force_body_n is None else vector(propulsive_force_body_n,'propulsive_force_body_n',size=3)
        start=self.t
        self.body.step(achieved,environment_acceleration=lambda time,r,v:self._environment_acceleration(
            time.seconds,r,v,self.deputy_environment_acceleration),environment_wrench=self.deputy_environment_wrench,
            internal_momentum_body_nms=internal_momentum_body_nms)
        if self.chief_body is not None:
            from spec import ActuationResult,WrenchCommand
            if chief_actuation is None:
                chief_actuation=ActuationResult(WrenchCommand('target_coast',self.chief_body.time,'body/target'),dt,
                    np.zeros(3),np.zeros(3))
            if chief_actuation.command.time!=achieved.command.time or chief_actuation.duration_s!=dt:
                raise ValueError('Coupled target/chaser intervals must match')
            from environment.surface_forces import gravity_gradient_from_tensor
            def target_wrench(stamp,r,v,q,w,m):
                radius=np.linalg.norm(r)
                tensor=self.mu*(3*np.outer(r,r)/radius**5-np.eye(3)/radius**3)
                return np.zeros(3),gravity_gradient_from_tensor(tensor,q,self.chief_body.inertia_kg_m2)
            self.chief_body.step(chief_actuation,environment_acceleration=lambda time,r,v:self._environment_acceleration(
                time.seconds,r,v,self.chief_environment_acceleration),environment_wrench=target_wrench)
            self.chief_position=self.chief_body.r.copy();self.chief_velocity=self.chief_body.v.copy()
            self.deputy_position,self.deputy_velocity=self.body.r.copy(),self.body.v.copy()
            self.t=self.body.time.seconds
            self.total_dv_ms+=float(np.linalg.norm(budget_force))*dt/mass_mid
            if min(np.linalg.norm(self.chief_position),np.linalg.norm(self.deputy_position))<=self.radius:
                raise RuntimeError('Cartesian orbit entered Earth')
            return self.to_lvlh()
        if chief_actuation is not None:raise ValueError('Target actuation requires an attached target body')
        chief = np.r_[self.chief_position,self.chief_velocity]
        count = max(1,int(np.ceil(dt/self.max_step)))
        h = dt/count
        def rhs(time,state):
            return np.r_[state[3:],self.chief_acceleration(state[:3],state[3:],time)]
        for index in range(count):
            time=start+index*h
            a=rhs(time,chief); b=rhs(time+h/2,chief+h*a/2); c=rhs(time+h/2,chief+h*b/2); d=rhs(time+h,chief+h*c)
            chief += h*(a+2*b+2*c+d)/6
        self.chief_position,self.chief_velocity=chief[:3],chief[3:]
        self.deputy_position,self.deputy_velocity=self.body.r.copy(),self.body.v.copy()
        self.t=self.body.time.seconds
        self.total_dv_ms += float(np.linalg.norm(budget_force))*dt/mass_mid
        if min(np.linalg.norm(self.chief_position),np.linalg.norm(self.deputy_position)) <= self.radius:
            raise RuntimeError('Cartesian orbit entered Earth')
        return self.to_lvlh()

    def acceleration(self, position):
        return central_gravity(position, self.mu)+j2_acceleration(position,
            mu_m3_s2=self.mu, radius_m=self.radius, j2=self.j2)

    def _environment_acceleration(self,time_s,position,velocity,provider):
        from spec import Timestamp
        acceleration=self.acceleration(position)
        if provider is not None:
            acceleration+=vector(provider(Timestamp(time_s),position,velocity),'environment_acceleration',size=3)
        return acceleration

    def chief_acceleration(self,position,velocity=None,time_s=None):
        return self._environment_acceleration(self.t if time_s is None else time_s,position,
            self.chief_velocity if velocity is None else velocity,self.chief_environment_acceleration)

    def to_lvlh(self):
        return relative_state_from_eci(self.chief_position, self.chief_velocity,
            self.deputy_position, self.deputy_velocity,
            chief_accel=self.chief_acceleration(self.chief_position))

    def set_relative_state(self, state):
        """Explicit scenario initialization/dispersion, never an estimator update."""
        if self.body is not None:
            raise ValueError('Cannot reset relative state after rigid-body attachment')
        state = vector(state, 'state', size=6)
        self.deputy_position, self.deputy_velocity = relative_state_to_eci(
            self.chief_position, self.chief_velocity, state,
            chief_accel=self.chief_acceleration(self.chief_position))

    @property
    def range_m(self):
        return float(np.linalg.norm(self.deputy_position-self.chief_position))

    def apply_impulse_rtn(self, delta_v_m_s):
        """Execute an ideal instantaneous impulse at the current physical epoch."""
        if self.body is not None:
            raise ValueError('Use achieved wrench after rigid-body attachment')
        impulse = vector(delta_v_m_s, 'delta_v_m_s', size=3)
        rotation = eci_to_lvlh_rotation(self.chief_position,self.chief_velocity)
        self.deputy_velocity += rotation.T@impulse
        self.total_dv_ms += float(np.linalg.norm(impulse))

    def step(self, dt, deputy_acceleration_rtn_m_s2):
        if self.body is not None:
            raise ValueError('Use step_achieved after rigid-body attachment')
        duration = finite_scalar(dt, 'dt', strictly_positive=True)
        control = vector(deputy_acceleration_rtn_m_s2, 'deputy_acceleration_rtn_m_s2', size=3)
        state = np.r_[self.chief_position, self.chief_velocity,
                      self.deputy_position, self.deputy_velocity]
        count = max(1, int(np.ceil(duration/self.max_step)))
        h = duration/count
        def rhs(time,x):
            rotation = eci_to_lvlh_rotation(x[:3], x[3:6])
            return np.r_[x[3:6], self.chief_acceleration(x[:3],x[3:6],time), x[9:12],
                self._environment_acceleration(time,x[6:9],x[9:12],self.deputy_environment_acceleration)+rotation.T@control]
        for index in range(count):
            time=self.t+index*h
            a = rhs(time,state);b = rhs(time+h/2,state+h*a/2);c = rhs(time+h/2,state+h*b/2);d = rhs(time+h,state+h*c)
            state += h*(a+2*b+2*c+d)/6
        self.chief_position, self.chief_velocity = state[:3], state[3:6]
        self.deputy_position, self.deputy_velocity = state[6:9], state[9:12]
        self.t += duration
        self.total_dv_ms += float(np.linalg.norm(control))*duration
        if min(np.linalg.norm(self.chief_position), np.linalg.norm(self.deputy_position)) <= self.radius:
            raise RuntimeError('Cartesian orbit entered Earth')
        return self.to_lvlh()
