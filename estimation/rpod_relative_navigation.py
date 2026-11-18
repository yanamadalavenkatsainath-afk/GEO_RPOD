"""Shared LEO RelNav phase used by main.py and monte_carlo_t.py.

Truth is selected as secular ROE or independent Cartesian two-body/J2.
Sensors observe the selected truth, both EKFs
receive the same measurements and commanded inputs, and control uses the
ROE estimate. Chief ephemeris/phase is assumed known. A noisy Cartesian
prior is explicitly provided ONCE at activation (not autonomous initial
acquisition). Sensor tracking is an idealized LVLH pointing command from
estimated line of sight, with no ADCS/gimbal dynamics.
"""
from dataclasses import dataclass
import numpy as np
from environment.roe_dynamics import ROEDynamics
from estimation.roe_ekf import ROEEKF
from estimation.cw_ekf import CWEKF
from sensors.ranging_sensor import RangingBearingSensor
from control.roe_controller import ROEController, ROEMode
from control.rendezvous_controller import RendezvousController


@dataclass(frozen=True)
class RendezvousRequirements:
    # 15m preserves the historical proximity endpoint. New speed/hold values
    # are configurable demonstration assumptions, not docking requirements.
    position_m: float = 15.0
    speed_ms: float = 0.02
    hold_s: float = 30.0

    def __post_init__(self):
        values = (self.position_m, self.speed_ms, self.hold_s)
        if not np.all(np.isfinite(values)) or min(values) <= 0:
            raise ValueError("Rendezvous thresholds must be positive")


class RendezvousAcceptance:
    """Independent sampled truth evaluator; contains no control/navigation."""
    def __init__(self, requirements=None):
        self.requirements = requirements or RendezvousRequirements()
        self.held_s, self.complete = 0.0, False

    def update(self, state, dt, cleanup_active):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        req = self.requirements
        valid = (cleanup_active and np.all(np.isfinite(state))
                 and np.linalg.norm(state[:3]) < req.position_m
                 and np.linalg.norm(state[3:]) < req.speed_ms)
        self.held_s = self.held_s + dt if valid else 0.0
        self.complete = self.held_s + 1e-9 >= req.hold_s
        return self.complete


class RelativeNavigationScenario:
    def __init__(self, dt=0.1, chief_radius_km=6781.0,
                 formation_offset=None, seed=42, j2=True,
                 requirements=None, cleanup_bandwidth=None, truth_model='roe',physical_config=None,
                 chief_inclination_deg=51.6,transfer_time_s=2700.):
        self.dt = float(dt)
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("dt must be positive")
        from utils.validation import finite_scalar
        self.transfer_time_s=finite_scalar(transfer_time_s,'transfer_time_s',strictly_positive=True)
        self.rng = np.random.default_rng(seed)
        offset = np.array([0., 100., 0.]) if formation_offset is None else np.asarray(formation_offset)
        self.truth = ROEDynamics(chief_radius_km*1e3, e_chief=0.0,
                                 i_chief=np.radians(chief_inclination_deg), j2=j2)
        self.truth.set_from_lvlh(offset, np.zeros(3))
        if truth_model not in ('roe','cartesian_j2'):
            raise ValueError('Unknown relative truth model')
        self.orbit_truth = None
        if truth_model == 'cartesian_j2':
            from plant.relative_orbit import CartesianRelativeOrbit
            self.orbit_truth = CartesianRelativeOrbit([self.truth.a,0.,0.],
                np.sqrt(self.truth.MU/self.truth.a)*np.array([0.,np.cos(self.truth.i),np.sin(self.truth.i)]),
                self.truth.to_lvlh(),j2=self.truth.J2 if j2 else 0.,max_step_s=self.dt)
        self.roe_ekf = ROEEKF(self.truth, self.dt)
        self.cw_ekf = CWEKF(self.truth.n, self.dt)
        self.roe_ctrl = ROEController(self.truth, target_roe=self.truth.roe.copy())
        self.cleanup_ctrl = RendezvousController(self.truth.n, target_lvlh=np.zeros(3))
        # Explicit scenario control settings; retain the 1e-4 m/s^2 actuator cap.
        bandwidth = 3*self.truth.n if cleanup_bandwidth is None else cleanup_bandwidth
        self.cleanup_ctrl.Kp = bandwidth**2
        self.cleanup_ctrl.Kd = 2*0.9*bandwidth
        self.sensor = RangingBearingSensor(
            sigma_range_m=0.5, sigma_range_frac=0.002,
            sigma_angle_rad=np.radians(0.1), fov_half_deg=60.,
            max_range_m=5000., rng=self.rng)
        self.acceptance = RendezvousAcceptance(requirements)
        self.active = self.triggered = self.cleanup_active = False
        self.burn_ranges, self.burns = [], []
        self.measurements = self.roe_updates = self.cw_updates = 0
        self.physical_config=physical_config
        self.physical_rcs=self.physical_attitude_chain=self.physical_wheels=None
        self.finite_transfer=None
        self.lambert_plan=None;self.lambert_plans=[];self.lambert_replanned=False
        self.physical_actuation_health=None
        if physical_config is not None:
            if self.orbit_truth is None:raise ValueError('Physical far leg requires Cartesian truth')
            from actuators import FuelTank,ThrusterArray,ThrusterLayout,ReactionWheelArray
            from fsw.leo_attitude import LeoAttitudeChain
            from control.finite_maneuver import FiniteManeuver
            from scipy.spatial.transform import Rotation
            cfg=physical_config;orbit=self.orbit_truth
            orbit.attach_body(np.eye(3),np.zeros(3),dry_mass_kg=cfg.dry_mass_kg,
                propellant_kg=cfg.propellant_kg,inertia_kg_m2=np.diag(cfg.inertia_diagonal_kg_m2))
            orbit.attach_chief_body(Rotation.from_euler('xyz',[15.,-10.,20.],degrees=True).as_matrix(),
                np.radians([.025,-.035,.10])*cfg.tumble_scale,mass_kg=cfg.target_mass_kg,inertia_kg_m2=np.diag([300.,300.,200.]))
            self.physical_rcs=ThrusterArray(ThrusterLayout.corner_pod_16(max_force_n=cfg.jet_force_n),
                FuelTank(cfg.dry_mass_kg,cfg.propellant_kg),isp_s=cfg.isp_s,minimum_impulse_ns=cfg.minimum_impulse_ns,
                allocation_method='minimum_force',valve_delay_s=.005,valve_time_constant_s=.005)
            self.physical_wheels=ReactionWheelArray(max_torque_nm=.2,max_momentum_nms=4.,rotor_inertia_kg_m2=.005)
            self.physical_attitude_chain=LeoAttitudeChain(dt=.01,seed=seed,max_coast_s=cfg.attitude_coast_limit_s,
                uncertainty_limit_deg=cfg.attitude_uncertainty_limit_deg)
            from fsw.readiness import ActuationHealthMonitor
            self.physical_actuation_health=ActuationHealthMonitor(force_residual_limit_n=.2,torque_residual_limit_nm=.2,dwell_s=5.)
            self.finite_transfer=FiniteManeuver(maximum_acceleration_m_s2=1e-4)
            # Physical acquisition must remain inside the finite authority
            # envelope; the earlier 3n cleanup saturated after an impulsive miss.
            if cleanup_bandwidth is None:
                self.cleanup_ctrl.Kp=self.truth.n**2
                self.cleanup_ctrl.Kd=2*.9*self.truth.n
            if cfg.environment_model=='earth_plates':
                from environment.earth_vehicle import InterpolatedSunPosition,box_plate_environment,CannonballEarthEnvironment
                from environment.leo_orbit import _atmo_density
                from spec import Epoch,TimeScale
                orbit.sun_position=InterpolatedSunPosition(Epoch('LEO environment',TimeScale.TAI,cfg.environment_epoch_tai))
                density=lambda time,r:_atmo_density(np.linalg.norm(r)/1000.-orbit.radius/1000.)
                orbit.chief_environment_acceleration=CannonballEarthEnvironment(mass_kg=cfg.target_mass_kg,
                    area_m2=cfg.target_effective_area_m2,sun_position=orbit.sun_position,density=density,
                    cd=cfg.drag_coefficient,cr=cfg.radiation_coefficient,radius_m=orbit.radius)
                orbit.deputy_environment_wrench=box_plate_environment(half_extents_m=cfg.bus_half_extents_m,
                    pressure_offset_m=cfg.pressure_offset_m,sun_position=orbit.sun_position,density=density,
                    inertia=lambda time,mass:orbit.body.inertia_kg_m2,cd=cfg.drag_coefficient,
                    cr=cfg.radiation_coefficient,mu_m3_s2=orbit.mu,radius_m=orbit.radius)

    def _plan_lambert_transfer(self,tof_s):
        from control.leo_lambert_transfer import FiniteLambertTransfer
        from utils.orbital_frames import relative_state_to_eci,eci_to_lvlh_rotation
        from utils.quaternion import quat_from_matrix
        from spec import Timestamp
        orbit=self.orbit_truth;cfg=self.physical_config
        chief=np.r_[orbit.chief_position,orbit.chief_velocity]  # declared known chief ephemeris
        relative=self.roe_ekf._roe_to_lvlh(self.roe_ekf.x)
        r,v=relative_state_to_eci(chief[:3],chief[3:],relative,
            chief_accel=orbit.chief_acceleration(chief[:3],chief[3:],orbit.t))
        def chief_acceleration(t,r,v):return orbit.chief_acceleration(r,v,t)
        predictor=FiniteLambertTransfer(mu=orbit.mu,maximum_acceleration_m_s2=self.finite_transfer.limit)
        chief_prediction=predictor.propagate(chief,orbit.t,orbit.t+tof_s,chief_acceleration,dense=True)
        mass=self.physical_rcs.tank.mass_kg  # known fuel/actuation ledger, not orbital truth
        def deputy_acceleration(t,r,v):
            acceleration=orbit.acceleration(r)
            if orbit.deputy_environment_wrench is not None:
                known_chief=chief_prediction.sol(t)
                frame=eci_to_lvlh_rotation(known_chief[:3],known_chief[3:])
                q=quat_from_matrix(frame.T)  # planned LVLH tracking attitude
                force,_=orbit.deputy_environment_wrench(Timestamp(t),r,v,q,np.zeros(3),mass)
                acceleration=acceleration+frame.T@force/mass
            return acceleration
        plan=predictor.plan(chief_state=chief,deputy_state=np.r_[r,v],time_s=orbit.t,tof_s=tof_s,
            chief_acceleration=chief_acceleration,deputy_acceleration=deputy_acceleration)
        plan.update(departure_issued=False,arrival_issued=False)
        self.lambert_plan=plan;self.lambert_plans.append(plan)

    def _sequence_lambert_transfer(self):
        plan=self.lambert_plan;now=self.orbit_truth.t
        if self.finite_transfer.active is not None:return
        # One navigation-driven midcourse reshoot retains the same arrival
        # epoch and the same force/acceleration limits. Rejects retain old plan.
        if (not self.lambert_replanned and plan['departure_issued']
                and now>=(plan['start_s']+plan['end_s'])/2
                and plan['end_s']-now>=300. and not plan['arrival_issued']):
            self.lambert_replanned=True
            try:
                self._plan_lambert_transfer(plan['end_s']-now)
                plan=self.lambert_plan
            except RuntimeError as error:
                plan['midcourse_replan_rejected']=str(error)
        if not plan['departure_issued']:
            delta=np.asarray(plan['departure_delta_v_m_s']);plan['departure_issued']=True;phase='FAR_LAMBERT_DEPARTURE'
        elif not plan['arrival_issued'] and now>=plan['arrival_burn_start_s']:
            delta=np.asarray(plan['arrival_delta_v_m_s']);plan['arrival_issued']=True;phase='FAR_LAMBERT_ARRIVAL'
        else:
            if plan['arrival_issued'] and now>=plan['end_s']:self.cleanup_active=True
            return
        self.burn_ranges.append(float(np.linalg.norm(self.roe_ekf.position)))
        self.burns.append((now,delta.copy()))
        self.finite_transfer.start(delta,time_s=now,phase=phase)

    def _advance_physical(self,acceleration_rtn):
        from spec import WrenchCommand
        from control.attitude_guidance import AttitudeReference,QuaternionPD
        from actuators import combine_actuation
        from utils.quaternion import rot_matrix,quat_from_matrix
        from utils.orbital_frames import eci_to_lvlh_rotation,lvlh_frame_rate
        orbit=self.orbit_truth;body=orbit.body;cfg=self.physical_config
        count=int(np.ceil(self.dt/.01));dt=self.dt/count
        controller=QuaternionPD(kp_nm_rad=[.4]*3,kd_nm_s_rad=[2.]*3,torque_limit_nm=[.2]*3)
        achieved_delta=np.zeros(3)
        for index in range(count):
            chain=self.physical_attitude_chain;chain.observe(orbit)
            frame=eci_to_lvlh_rotation(orbit.chief_position,orbit.chief_velocity)
            if chain.filter is None or not chain.last_gyro.valid:
                wheel_command=WrenchCommand('far_safe',body.time,body.body_frame)
                force=np.zeros(3)
            else:
                ref=AttitudeReference(body.time,'earth/gcrs',body.body_frame,quat_from_matrix(frame.T),
                    lvlh_frame_rate(orbit.chief_position,orbit.chief_velocity,orbit.chief_acceleration(orbit.chief_position)),np.zeros(3))
                wheel_command=controller.command(time=body.time,quaternion_estimate=chain.filter.q,
                    omega_estimate_body_rad_s=chain.last_gyro.value-chain.filter.bias,reference=ref)
                force=rot_matrix(chain.filter.q).T@frame.T@acceleration_rtn*self.physical_rcs.tank.mass_kg if chain.healthy else np.zeros(3)
            jet=self.physical_rcs.execute(WrenchCommand('far_rendezvous',body.time,body.body_frame,force_n=force),dt)
            old=self.physical_wheels.angular_momentum_body.copy();wheel=self.physical_wheels.execute(wheel_command,dt)
            mass_mid=body.mass_kg-jet.propellant_used_kg/2
            before=rot_matrix(body.q)
            result=combine_actuation((jet,wheel));self.physical_actuation_health.update(result)
            state=orbit.step_achieved(result,
                internal_momentum_body_nms=(old+self.physical_wheels.angular_momentum_body)/2)
            achieved_delta+=(before+rot_matrix(body.q))@jet.force_n*dt/(2*mass_mid)
            chain.predict(body.time)
            if not self.physical_actuation_health.ready:
                self.finite_transfer.record_execution(achieved_delta,end_s=orbit.t,sequence_interval_complete=False)
                self.finite_transfer.abort(time_s=orbit.t,reason='ACTUATOR_AUTHORITY_LOSS')
                raise RuntimeError('Far-leg actuator authority lost; partial burn retained')
        return state,achieved_delta

    def coast(self):
        """Advance truth exactly once while RelNav is inactive."""
        if self.active:
            raise RuntimeError("Use step after activation")
        state = self.truth.step(self.dt)
        if self.physical_config is not None:return self._advance_physical(np.zeros(3))[0]
        return state if self.orbit_truth is None else self.orbit_truth.step(self.dt,np.zeros(3))

    def truth_state(self):
        return self.truth.to_lvlh() if self.orbit_truth is None else self.orbit_truth.to_lvlh()

    def activate(self, position_jitter=None, velocity_jitter=None,
                 prior_noise=None):
        if self.active:
            raise RuntimeError("Scenario already active; truth reseeding is forbidden")
        state = self.truth_state()
        if position_jitter is not None:
            state[:3] += position_jitter
        if velocity_jitter is not None:
            state[3:] += velocity_jitter
        self.truth.set_from_lvlh(state[:3], state[3:])
        if self.orbit_truth is not None and (position_jitter is not None or velocity_jitter is not None):
            self.orbit_truth.set_relative_state(state)
        sigma = np.array([5., 5., 5., .02, .02, .02])
        noise = self.rng.normal(size=6)*sigma if prior_noise is None else np.asarray(prior_noise)
        prior = state + noise  # explicitly idealized prior, same for both filters
        cov = np.diag(sigma**2)
        self.cw_ekf.initialise(prior, cov)
        self.roe_ekf.initialise_cartesian(prior, self.truth.mean_anomaly, cov)
        self.active = True

    def step(self, t, allow_rendezvous=False, stop_time=None):
        if not self.active:
            raise RuntimeError("Activate RelNav before step")
        phase = self.roe_ekf.mean_anomaly
        if allow_rendezvous and not self.triggered:
            if self.physical_config is not None:
                tof=min(self.transfer_time_s,(stop_time-t-120.) if stop_time is not None else self.transfer_time_s)
                if tof<600.:raise RuntimeError('Insufficient time for bounded finite Lambert transfer')
                self._plan_lambert_transfer(tof)
            else:
                self.roe_ctrl.set_mode(ROEMode.RENDEZVOUS, self.roe_ekf.x,phase,t,stop_time)
            self.triggered = True
        impulse=None
        if self.physical_config is None:
            _,impulse=self.roe_ctrl.compute(self.roe_ekf.x,phase,t)
        elif self.lambert_plan is not None:
            self._sequence_lambert_transfer()
        if impulse is not None:
            self.burn_ranges.append(float(np.linalg.norm(self.truth_state()[:3])))
            self.burns.append((float(t), impulse.copy()))
            if self.physical_config is not None:
                from utils.orbital_frames import eci_to_lvlh_rotation
                frame=eci_to_lvlh_rotation(self.orbit_truth.chief_position,self.orbit_truth.chief_velocity)
                self.finite_transfer.start(frame.T@impulse,time_s=self.orbit_truth.t,phase='FAR_RENDEZVOUS')
            else:self.truth.apply_impulse_lvlh(impulse)
            if self.orbit_truth is not None and self.physical_config is None:
                self.orbit_truth.apply_impulse_rtn(impulse)
            if self.physical_config is None:
                self.cw_ekf.x[3:] += impulse
                self.roe_ekf.apply_impulse_lvlh(impulse)
        if self.physical_config is None and self.triggered and self.roe_ctrl._rdv_burn2_applied:
            self.cleanup_active = True
        estimated = self.roe_ekf._roe_to_lvlh(self.roe_ekf.x)
        if self.cleanup_active:
            accel, _ = self.cleanup_ctrl.compute(estimated, t)
        else:
            accel = np.zeros(3)
        if self.physical_config is not None and self.finite_transfer.active is not None:
            from utils.orbital_frames import eci_to_lvlh_rotation
            frame=eci_to_lvlh_rotation(self.orbit_truth.chief_position,self.orbit_truth.chief_velocity)
            accel=frame@self.finite_transfer.command(self.dt,permitted=self.physical_attitude_chain.healthy)
        state = self.truth.step(self.dt, accel)
        if self.physical_config is not None:
            state,achieved=self._advance_physical(accel)
            self.finite_transfer.record_execution(achieved,end_s=self.orbit_truth.t)
        elif self.orbit_truth is not None:
            state = self.orbit_truth.step(self.dt,accel)
        self.cw_ekf.predict(accel)
        self.roe_ekf.predict(accel)
        pointing = self.roe_ekf.position
        norm = np.linalg.norm(pointing)
        pointing = pointing/norm if norm > 1e-6 else np.array([0., 1., 0.])
        z, R = self.sensor.measure(state[:3], pointing)
        if z is not None:
            self.measurements += 1
            self.cw_updates += int(self.cw_ekf.update(z, R))
            self.roe_updates += int(self.roe_ekf.update(z, R))
        self.acceptance.update(state, self.dt, self.cleanup_active)
        return state

    @property
    def complete(self):
        return self.acceptance.complete

    def summary(self):
        state = self.truth_state()
        return dict(rdv_triggered=self.triggered, rdv_success=self.complete,
                    final_range=float(np.linalg.norm(state[:3])),
                    final_speed=float(np.linalg.norm(state[3:])),
                    hold_s=self.acceptance.held_s,
                    total_dv_mms=(self.truth.total_dv_ms if self.orbit_truth is None else self.orbit_truth.total_dv_ms)*1000,
                    truth_model='roe' if self.orbit_truth is None else 'cartesian_j2',
                    transfer_guidance='finite_cartesian_lambert' if self.physical_config is not None else 'impulsive_secular_roe',
                    lambert_plans=self.lambert_plans,
                    finite_burn_execution=[] if self.finite_transfer is None else self.finite_transfer.history,
                    remaining_propellant_kg=None if self.physical_rcs is None else self.physical_rcs.tank.propellant_kg,
                    burn_count=len(self.burns), measurements=self.measurements,
                    roe_updates=self.roe_updates, cw_updates=self.cw_updates)
