"""Configurable Earth-orbit non-cooperative RPOD demonstrator, separate from stand-off GNC.

All vectors/rates are in chief RTN, rotations map body to RTN, velocities
are derivatives in RTN. Synthetic visual pose is a sensor abstraction, not
an image-processing/acquisition demonstration. Target bell geometry is known
after identification. No target truth is exposed to the mission controller.
"""
from dataclasses import dataclass
from collections import deque
import numpy as np
from scipy.spatial.transform import Rotation
from environment.roe_dynamics import ROEDynamics
from utils.quaternion import rot_matrix


def limit(v, maximum):
    return v * min(1., maximum / max(np.linalg.norm(v), 1e-15))


def contact_metrics(position, velocity, attitude, rate, target, target_rate, center=None, normal=None):
    center = np.array([0.,0.,2.]) if center is None else center
    normal = np.array([0.,0.,1.]) if normal is None else normal
    axis = target@normal
    bell = target@center
    tool_offset = attitude @ np.array([0., 0., -.5])
    delta = position + tool_offset - bell
    relative_velocity = velocity + np.cross(rate, tool_offset) - np.cross(target_rate, bell)
    axial = float(delta @ axis)
    return dict(axial_m=axial, lateral_m=float(np.linalg.norm(delta-axial*axis)),
                speed_ms=float(np.linalg.norm(relative_velocity)),
                angle_deg=float(np.degrees(np.arccos(np.clip(axis@attitude[:,2],-1.,1.)))),
                spin_error_deg_s=float(np.degrees(np.linalg.norm(rate-target_rate))))


def eligible(metrics):
    return (0. <= metrics['axial_m'] < .04 and metrics['lateral_m'] < .04
            and metrics['speed_ms'] < .01 and metrics['angle_deg'] < 2.
            and metrics['spin_error_deg_s'] < .1)


class PoseTracker:
    """Spatial-frame finite-baseline rate estimate; no quaternion error mixing."""
    def __init__(self):
        # A 60 s window at the nominal 4 Hz pose cadence reduces camera
        # differentiation noise; local-rate lag under tumble remains a
        # diagnostic limitation to verify, rather than relaxing sync gates.
        self.history = deque(maxlen=241)
        self.rotation = None
        self.rate = np.zeros(3)
        self.last_seen = -np.inf

    def observe(self, rotation, t):
        self.history.append((t, rotation.copy()))
        self.rotation = rotation.copy()
        self.last_seen = t
        old_t, old_r = self.history[0]
        if t-old_t >= 5:
            # Endpoint differencing gives two noisy PnP attitudes all the
            # influence. Fit a local spatial angular-rate trend with an
            # intercept and robust residual weights across the full history.
            epochs=np.array([stamp-old_t for stamp,_ in self.history])
            angles=np.array([Rotation.from_matrix(value@old_r.T).as_rotvec()
                             for _,value in self.history])
            design=np.c_[np.ones(len(epochs)),epochs]
            weights=np.ones(len(epochs))
            for _ in range(4):
                root=np.sqrt(weights)[:,None]
                fit=np.linalg.lstsq(design*root,angles*root,rcond=None)[0]
                residual=np.linalg.norm(angles-design@fit,axis=1)
                scale=max(np.radians(.02),np.median(residual)/1.54,
                          1.4826*np.median(np.abs(residual-np.median(residual))))
                weights=np.minimum(1.,2.5*scale/np.maximum(residual,1e-12))
            self.rate=fit[1]
            self.rotation=Rotation.from_rotvec(design[-1]@fit).as_matrix()@old_r

    def estimate(self, t):
        if self.rotation is None:
            return None, self.rate.copy(), False
        age = t-self.last_seen
        rotation = Rotation.from_rotvec(self.rate*age).as_matrix() @ self.rotation
        healthy = bool(age <= 2. and len(self.history) >= 20 and np.linalg.norm(self.rate) < np.radians(1.))
        return rotation, self.rate.copy(), healthy


class BellSurvey:
    """Fit a circle to an unlabelled segmented 3D rim cloud in target body axes.

    Segmentation/ranging are explicit ideal sensor assumptions. No truth fixture
    geometry is provided to this estimator; acceptance needs stable repeated fits.
    """
    def __init__(self):
        self.center = None
        self.normal = None
        self.accepted = 0
        self.residual = np.inf
        self.last_seen = -np.inf
        self.fit_covariance=None
        self.rejected=0
        self.last_reason='NOT_OBSERVED'
        self.last_nis=None

    def observe(self, points, t, rotation_covariance=None):
        origin = points.mean(axis=0)
        _,_,basis = np.linalg.svd(points-origin,full_matrices=False)
        uv = (points-origin)@basis[:2].T
        solution = np.linalg.lstsq(np.c_[2*uv,np.ones(len(points))],np.sum(uv**2,axis=1),rcond=None)[0]
        center = origin+basis[:2].T@solution[:2]
        normal = basis[2]
        if normal@center < 0:
            normal = -normal
        radius = np.sqrt(max(0.,solution[2]+solution[:2]@solution[:2]))
        residual = float(np.sqrt(np.mean(((points-center)@normal)**2)))
        stable = self.center is None or (np.linalg.norm(center-self.center)<.025 and normal@self.normal>np.cos(np.radians(2)))
        if not (.15<radius<.4 and residual<.008):
            self.accepted=0
            self.last_reason='RIM_GEOMETRY'
            return
        if rotation_covariance is not None:
            from utils.validation import covariance
            from utils.orbital_frames import skew
            angular=covariance(rotation_covariance,3)
            jac=np.vstack((-skew(center),-skew(normal)))
            noise=jac@angular@jac.T+np.diag([.001**2]*3+[.005**2]*3)
            if self.fit_covariance is None or t-self.last_seen>=2.:
                self.center,self.normal=center.copy(),normal.copy()
                self.fit_covariance=noise.copy();self.accepted=1
            else:
                innovation=np.r_[center-self.center,normal-self.normal]
                # The fitted geometry shares a pose calibration bias across
                # samples. Its declared non-averaging floor must also enter
                # the innovation gate, rather than only the validity check.
                bias_floor=np.diag([.005**2]*3+[np.radians(.1)**2]*3)
                total=self.fit_covariance+noise+bias_floor
                nis=float(innovation@np.linalg.solve(total,innovation))
                self.last_nis=nis
                if nis>25.:
                    self.rejected+=1;self.last_reason='FIT_INNOVATION_GATE'
                    return
                gain=np.linalg.solve(total,self.fit_covariance).T
                state=np.r_[self.center,self.normal]+gain@innovation
                identity=np.eye(6)-gain
                self.fit_covariance=identity@self.fit_covariance@identity.T+gain@(noise+bias_floor)@gain.T
                self.fit_covariance=(self.fit_covariance+self.fit_covariance.T)/2
                self.center=state[:3];self.normal=state[3:]/np.linalg.norm(state[3:])
                self.accepted+=1
            self.residual=residual;self.last_seen=t;self.last_reason='COVARIANCE_FIT'
            return
        if not stable or t-self.last_seen >= 2.:
            # Start a new qualification sequence. An obsolete reference must
            # not veto every subsequent observation after a pose correction.
            # This candidate remains unusable until 20 stable fits accumulate.
            self.center = center.copy()
            self.normal = normal.copy()
            self.accepted = 0
        self.center = center if self.center is None else .1*center+.9*self.center
        self.normal = normal if self.normal is None else .1*normal+.9*self.normal
        self.normal /= np.linalg.norm(self.normal)
        self.accepted += 1
        self.residual = residual
        self.last_seen = t

    def valid(self,t):
        precise=True
        if self.fit_covariance is not None:
            # A non-averaging engineering floor represents shared pose bias.
            # Noise calibration and full joint pose/geometry covariance are
            # still outside this illustrative segmented-rim abstraction.
            position_sigma=np.sqrt(np.linalg.eigvalsh(self.fit_covariance[:3,:3]).max()+.005**2)
            normal_sigma=np.sqrt(np.linalg.eigvalsh(self.fit_covariance[3:,3:]).max()+np.radians(.1)**2)
            precise=position_sigma<.025 and normal_sigma<np.radians(2.)
        return self.accepted >= 20 and t-self.last_seen < 2. and precise


@dataclass
class RPODConfig:
    dt: float = .25
    duration: float = 2400.
    seed: int = 0
    dropout_start: float = np.inf
    dropout_end: float = np.inf
    contact_enabled: bool = True
    tumble_scale: float = 1.
    truth_model: str = 'roe'
    actuation_model: str = 'ideal'
    dry_mass_kg: float = 50.
    propellant_kg: float = 1.
    inertia_diagonal_kg_m2: tuple = (10.,10.,6.)
    jet_force_n: float = .25
    isp_s: float = 220.
    minimum_impulse_ns: float = .0001
    plume_exclusion_half_angle_deg: float = 45.
    attitude_model: str = 'synthetic'
    attitude_coast_limit_s: float = 300.
    attitude_uncertainty_limit_deg: float = .1
    environment_model: str = 'gravity_j2'
    environment_epoch_tai: str = '2025-01-01T00:00:00'
    bus_half_extents_m: tuple = (.3,.3,.4)
    pressure_offset_m: tuple = (.001,.001,.005)
    drag_coefficient: float = 2.2
    radiation_coefficient: float = 1.3
    target_mass_kg: float = 500.
    target_effective_area_m2: float = 4.
    vision_model: str = 'synthetic_reference'
    contact_model: str = 'geometric_reference'
    maximum_contact_load_n: float = 100.
    navigation_acceleration_noise_density: float = .003
    chief_radius_m: float = 6781e3
    chief_inclination_deg: float = 51.6

    def __post_init__(self):
        if isinstance(self.seed,(bool,np.bool_)) or not isinstance(self.seed,(int,np.integer)) or not 0<=self.seed<2**32:
            raise ValueError('seed must be an integer in [0, 2**32)')
        if np.isnan(self.dropout_start) or np.isnan(self.dropout_end) or self.dropout_start>self.dropout_end:
            raise ValueError('Dropout bounds must be ordered and not NaN')
        if not isinstance(self.contact_enabled,(bool,np.bool_)):
            raise ValueError('contact_enabled must be boolean')
        if self.vision_model not in ('synthetic_reference','mapped_pixels'):
            raise ValueError('Unknown target vision model')
        if self.vision_model=='mapped_pixels' and self.attitude_model!='sensor_mekf':
            raise ValueError('Mapped pixels require measured inertial observer attitude')
        if self.contact_model not in ('geometric_reference','compliant'):
            raise ValueError('Unknown contact model')
        if self.contact_model=='compliant' and self.actuation_model!='allocated_rcs':
            raise ValueError('Compliant capture requires achieved RCS and rigid bodies')
        if not np.isfinite(self.maximum_contact_load_n) or self.maximum_contact_load_n<=0:
            raise ValueError('Positive finite contact load limit required')
        if self.truth_model not in ('roe','cartesian_j2'):
            raise ValueError('Unknown LEO truth model')
        if self.actuation_model not in ('ideal','allocated_rcs'):
            raise ValueError('Unknown actuation model')
        if self.actuation_model == 'allocated_rcs' and self.truth_model != 'cartesian_j2':
            raise ValueError('Allocated RCS requires Cartesian truth')
        if self.attitude_model not in ('synthetic','sensor_mekf'):
            raise ValueError('Unknown attitude model')
        if self.attitude_model == 'sensor_mekf' and self.actuation_model != 'allocated_rcs':
            raise ValueError('Inertial attitude sensor chain requires allocated RCS and rigid body')
        if self.environment_model not in ('gravity_j2','earth_plates'):
            raise ValueError('Unknown environment model')
        if self.environment_model=='earth_plates' and self.actuation_model!='allocated_rcs':
            raise ValueError('Plate environment requires the coupled rigid-body plant')
        from utils.validation import vector,finite_scalar
        finite_scalar(self.navigation_acceleration_noise_density,'navigation_acceleration_noise_density',minimum=0)
        finite_scalar(self.chief_radius_m,'chief_radius_m',strictly_positive=True)
        finite_scalar(self.chief_inclination_deg,'chief_inclination_deg',minimum=0)
        if self.chief_radius_m<=6378100. or self.chief_inclination_deg>180:raise ValueError('Invalid Earth chief orbit')
        half=vector(self.bus_half_extents_m,'bus_half_extents_m',size=3)
        if np.any(half<=0):raise ValueError('Positive bus half extents required')
        vector(self.pressure_offset_m,'pressure_offset_m',size=3)
        finite_scalar(self.target_mass_kg,'target_mass_kg',strictly_positive=True)
        for name in ('target_effective_area_m2','drag_coefficient','radiation_coefficient'):
            finite_scalar(getattr(self,name),name,minimum=0)
        from plant.coupled import inertia_tensor
        inertia_tensor(np.diag(np.asarray(self.inertia_diagonal_kg_m2,dtype=float)))
        for name in ('dry_mass_kg','jet_force_n','isp_s','attitude_coast_limit_s','attitude_uncertainty_limit_deg'):
            if not np.isfinite(getattr(self,name)) or getattr(self,name)<=0:
                raise ValueError(name+' must be finite and positive')
        for name in ('propellant_kg','minimum_impulse_ns'):
            if not np.isfinite(getattr(self,name)) or getattr(self,name)<0:
                raise ValueError(name+' must be finite and nonnegative')
        if not np.isfinite(self.plume_exclusion_half_angle_deg) or not 0<=self.plume_exclusion_half_angle_deg<=90:
            raise ValueError('Plume exclusion half angle must be in [0,90] degrees')
        if not 0 < self.dt <= .5 or not np.isfinite(self.duration) or self.duration < self.dt:
            raise ValueError('Require 0 < dt <= 0.5 and finite duration >= dt')
        if not np.isfinite(self.tumble_scale) or self.tumble_scale <= 0:
            raise ValueError('Tumble scale must be finite and positive')


class RPODMission:
    """Same engine for single-run/MC; truth acceptance isolated from guidance."""
    def __init__(self, config=None, relative_navigation=None):
        self.config = config or RPODConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.model = ROEDynamics(self.config.chief_radius_m,i_chief=np.radians(self.config.chief_inclination_deg))
        initial = np.array([0., 0., 14.]) + self.rng.normal(0, .2, 3)
        self.model.set_from_lvlh(initial, np.zeros(3))
        # Explicit one-time noisy acquisition prior, never reseeded.
        self.x = np.r_[initial, np.zeros(3)] + self.rng.normal(size=6)*[.1,.1,.1,.002,.002,.002]
        self.P = np.diag(np.array([.1,.1,.1,.002,.002,.002])**2)
        self.handoff = None
        self.retained_range_sensor=getattr(relative_navigation,'sensor',None)
        self.last_range_sample=-np.inf;self.accepted_range_updates=0;self.rejected_range_updates=0
        self.recovery_waypoint=None;self.recovery_attitude=None
        from estimation.cw_ekf import CWEKF
        self.range_update=CWEKF(self.model.n,self.config.dt)
        attached_orbit = None
        if relative_navigation is not None:
            self.model = relative_navigation.truth
            nav = relative_navigation.roe_ekf
            mapping = self.model.mapping_matrix(nav.mean_anomaly)
            self.x = mapping@nav.x
            self.P = mapping@nav.P@mapping.T
            self.handoff = dict(truth_time_s=self.model.t,estimated_range_m=float(np.linalg.norm(self.x[:3])),
                                estimate=self.x.tolist(),phase=nav.mean_anomaly)
            attached_orbit = getattr(relative_navigation,'orbit_truth',None)
            if self.config.truth_model == 'cartesian_j2' and attached_orbit is None:
                raise ValueError('Cartesian terminal truth requires a Cartesian far-leg handoff')
            if self.config.truth_model == 'roe' and attached_orbit is not None:
                raise ValueError('Cannot discard Cartesian truth at a secular terminal handoff')
            if attached_orbit is not None:
                self.handoff.update(truth_model='cartesian_j2',truth_state=attached_orbit.to_lvlh().tolist(),
                    chief_position_m=attached_orbit.chief_position.tolist(),
                    chief_velocity_m_s=attached_orbit.chief_velocity.tolist(),
                    deputy_position_m=attached_orbit.deputy_position.tolist(),
                    deputy_velocity_m_s=attached_orbit.deputy_velocity.tolist(),
                    continuity='Same Cartesian plant object and clock retained; no position/velocity reset')
        self.target = Rotation.from_euler('xyz', [15., -10., 20.], degrees=True).as_matrix()
        self.target_rate = np.radians(np.array([.025, -.035, .10]))*self.config.tumble_scale
        self.target = Rotation.from_rotvec(self.target_rate*self.model.t).as_matrix()@self.target
        self.attitude = Rotation.from_euler('xyz', [-5., 5., 0.], degrees=True).as_matrix()
        self.rate = np.zeros(3)
        self.pose = PoseTracker()
        self.survey = BellSurvey()
        self.fixture_center = np.array([0.,0.,2.])+self.rng.normal(0,.04,3)
        self.fixture_normal = Rotation.from_euler('xy',self.rng.normal(0,2.,2),degrees=True).apply([0.,0.,1.])
        self.t = 0.
        self.mode = 'SURVEY'
        self.transitions = [{'t':0., 'mode':self.mode}]
        self.standoff = 10.
        self.hold = self.capture_hold = 0.
        self.latched = False
        self.complete = False
        self.rejected_contacts = 0
        self.pre_capture_evidence = None
        self.records = []
        self.reason = 'time_limit'
        self.orbit_truth = attached_orbit
        if self.config.truth_model == 'cartesian_j2':
            if self.orbit_truth is None:
                from plant.relative_orbit import CartesianRelativeOrbit
                self.orbit_truth = CartesianRelativeOrbit(
                    [self.model.a,0.,0.],
                    np.sqrt(self.model.MU/self.model.a)*np.array([0.,np.cos(self.model.i),np.sin(self.model.i)]),
                    self.model.to_lvlh(),max_step_s=self.config.dt)
        self.rcs = None
        self.actuation = None
        self.peak_force_residual_n = 0.
        self.peak_torque_residual_nm = 0.
        self.actuator_flag_counts = {}
        from fsw.readiness import ActuationHealthMonitor
        self.actuation_health=ActuationHealthMonitor(force_residual_limit_n=.2,torque_residual_limit_nm=.2,dwell_s=5.)
        self.attitude_chain=None
        self.attitude_ready=True
        self.mapped_camera=None
        if self.config.vision_model=='mapped_pixels':
            from sensors import create_sensor
            from utils.sampling import SensorTiming
            from actuators.gimbal import RateLimitedGimbal
            self.camera_gimbal=RateLimitedGimbal(minimum_angles_rad=[-np.pi,-np.pi/2],
                maximum_angles_rad=[np.pi,np.pi/2],maximum_rates_rad_s=[.2,.2],maximum_accelerations_rad_s2=[.1,.1])
            self.camera_gimbal.reset([0.,-np.pi/2])
            self.camera_mount=np.diag([1.,-1.,-1.])
            self.mapped_camera=create_sensor('landmark_pose',source_id='leo/target_camera',seed=self.config.seed+771,
                timing=SensorTiming(.2),camera_to_body=self.camera_mount,
                camera_matrix=[[250.,0.,512.],[0.,250.,512.],[0.,0.,1.]],sigma_px=.3)
            # Known bus features/correspondences, not a known engine-bell map.
            points=[];normals=[]
            for axis in range(3):
                for sign in (-1.,1.):
                    for u in (-.8,0.,.8):
                        for v in (-.8,0.,.8):
                            p=np.zeros(3);p[axis]=sign*(1.+.03*((u==0.)+2*(v==0.)))
                            rest=[a for a in range(3) if a!=axis];p[rest]=[u,v]
                            normal=np.zeros(3);normal[axis]=sign
                            points.append(p);normals.append(normal)
            self.target_map=np.asarray(points);self.target_normals=np.asarray(normals)
            self.accepted_pixel_poses=0;self.rejected_pixel_poses=0
        self.compliant_fixture=None
        self.contact_load_n=0.;self.contact_couple_nm=0.
        self.peak_contact_load_n=0.;self.attachment_broken=False
        from plant.finite_body import BoxBody, FiniteBodyPair
        self.bus_pair=FiniteBodyPair(BoxBody((1.,1.,1.),'target example bus'),
                                     BoxBody(self.config.bus_half_extents_m,'deputy configured bus'))
        self.bus_collision=False
        self.min_bus_sphere_clearance_m=np.inf
        if self.config.actuation_model == 'allocated_rcs':
            from actuators.execution import FuelTank, ThrusterArray
            from actuators.thruster_layout import ThrusterLayout
            inherited=relative_navigation is not None and getattr(relative_navigation,'physical_rcs',None) is not None
            self.inherited_wheels=None
            if self.orbit_truth.body is None:
                self.orbit_truth.attach_body(self.attitude,self.rate,
                    dry_mass_kg=self.config.dry_mass_kg,propellant_kg=self.config.propellant_kg,
                    inertia_kg_m2=np.diag(self.config.inertia_diagonal_kg_m2))
            elif not inherited:raise ValueError('Existing rigid body requires its inherited actuator/filter state')
            if inherited:
                self.attitude,self.rate=self.orbit_truth.body_state_rtn()
                self.rcs=relative_navigation.physical_rcs
                self.attitude_chain=relative_navigation.physical_attitude_chain
                self.inherited_wheels=relative_navigation.physical_wheels
                self.handoff['attitude_continuity']='Same body, gyro/star filter, wheels, jets/valves and tank retained'
            else:
                self.rcs=ThrusterArray(ThrusterLayout.corner_pod_16(max_force_n=self.config.jet_force_n),
                    FuelTank(self.config.dry_mass_kg,self.config.propellant_kg),isp_s=self.config.isp_s,
                    minimum_impulse_ns=self.config.minimum_impulse_ns)
            self.terminal_initial_propellant_kg=self.rcs.tank.propellant_kg
            if self.config.environment_model=='earth_plates':
                from environment.earth_vehicle import InterpolatedSunPosition,box_plate_environment,CannonballEarthEnvironment
                from environment.leo_orbit import _atmo_density
                from spec import Epoch,TimeScale
                orbit=self.orbit_truth
                orbit.sun_position=InterpolatedSunPosition(Epoch('LEO environment',TimeScale.TAI,self.config.environment_epoch_tai))
                density=lambda time,r:_atmo_density(np.linalg.norm(r)/1000.-orbit.radius/1000.)
                orbit.chief_environment_acceleration=CannonballEarthEnvironment(
                    mass_kg=self.config.target_mass_kg,area_m2=self.config.target_effective_area_m2,
                    sun_position=orbit.sun_position,density=density,cd=self.config.drag_coefficient,
                    cr=self.config.radiation_coefficient,radius_m=orbit.radius)
                orbit.deputy_environment_wrench=box_plate_environment(
                    half_extents_m=self.config.bus_half_extents_m,pressure_offset_m=self.config.pressure_offset_m,
                    sun_position=orbit.sun_position,density=density,inertia=lambda time,mass:orbit.body.inertia_kg_m2,
                    cd=self.config.drag_coefficient,cr=self.config.radiation_coefficient,
                    mu_m3_s2=orbit.mu,radius_m=orbit.radius)
            if self.config.contact_model=='compliant' and self.orbit_truth.chief_body is None:
                self.orbit_truth.attach_chief_body(self.target,self.target_rate,mass_kg=self.config.target_mass_kg,
                    inertia_kg_m2=np.diag([300.,300.,200.]))
            if self.orbit_truth.chief_body is not None:
                self.target,self.target_rate=self.orbit_truth.chief_body_state_rtn()
            if self.config.attitude_model == 'sensor_mekf' and self.attitude_chain is None:
                from fsw.leo_attitude import LeoAttitudeChain
                self.attitude_chain=LeoAttitudeChain(dt=self.config.dt,seed=self.config.seed,
                    max_coast_s=self.config.attitude_coast_limit_s,
                    uncertainty_limit_deg=self.config.attitude_uncertainty_limit_deg)

    def execute_rcs(self, accel, angular_accel, star, relative_rate):
        """Command from synthetic attitude/rate measurements; execute common jets.

        Example hardware parameters are declared in RPODConfig. The plume
        cone uses estimated line of sight; no truth is supplied to allocation.
        A cone is a keepout approximation, not a plume/impingement simulation.
        """
        from spec import WrenchCommand
        from utils.orbital_frames import lvlh_frame_rate
        orbit=self.orbit_truth
        body=orbit.body
        r,v=orbit.chief_position,orbit.chief_velocity
        a=orbit.chief_acceleration(r,v)
        frame_rate=lvlh_frame_rate(r,v,a)
        epsilon=.01
        next_r=r+epsilon*v+.5*epsilon**2*a
        next_v=v+epsilon*a
        frame_accel=(lvlh_frame_rate(next_r,next_v,orbit.chief_acceleration(next_r,next_v,orbit.t+epsilon))-frame_rate)/epsilon
        omega_body=star.T@(relative_rate+frame_rate)
        alpha_body=star.T@(angular_accel+frame_accel+np.cross(frame_rate,relative_rate))
        inertia=body.inertia_kg_m2
        rotor_momentum=(np.zeros(3) if self.inherited_wheels is None else
                        self.inherited_wheels.angular_momentum_body)
        # Euler feedforward must include inherited spinning rotors, as does
        # the plant. Otherwise omega x h appears as an unmodelled torque.
        torque=inertia@alpha_body+np.cross(omega_body,inertia@omega_body+rotor_momentum)
        command=WrenchCommand('leo_rpod',body.time,body.body_frame,
                              force_n=star.T@accel*self.rcs.tank.mass_kg,
                              torque_nm=torque if self.attitude_ready else np.zeros(3))
        excluded=self.rcs.layout.chief_mask(star.T@(-self.x[:3]),
                                            self.config.plume_exclusion_half_angle_deg)
        wheel_old=wheel_new=None
        if self.inherited_wheels is not None:
            # Far-leg wheels were inherited but previously never commanded in
            # terminal flight. An infeasible jet force could therefore scale
            # the entire jet wrench to zero, also suppressing attitude control.
            from spec import ActuationResult
            wheel_old=self.inherited_wheels.angular_momentum_body.copy()
            wheel=self.inherited_wheels.execute(WrenchCommand('terminal_wheel',body.time,body.body_frame,
                torque_nm=command.torque_nm),self.config.dt)
            wheel_new=self.inherited_wheels.angular_momentum_body.copy()
            jet_command=WrenchCommand('terminal_rcs',body.time,body.body_frame,force_n=command.force_n,
                torque_nm=command.torque_nm-wheel.torque_nm)
            jet=self.rcs.execute(jet_command,self.config.dt,disabled=excluded)
            # Preserve the original total request so residuals do not count
            # wheel-saturation compensation twice.
            self.actuation=ActuationResult(command,self.config.dt,jet.force_n,jet.torque_nm+wheel.torque_nm,
                jet.propellant_used_kg,tuple(sorted(set(jet.flags+wheel.flags))))
        else:
            self.actuation=self.rcs.execute(command,self.config.dt,disabled=excluded)
        self.actuation_health.update(self.actuation)
        self.peak_force_residual_n=max(self.peak_force_residual_n,float(np.linalg.norm(self.actuation.residual[:3])))
        self.peak_torque_residual_nm=max(self.peak_torque_residual_nm,float(np.linalg.norm(self.actuation.residual[3:])))
        for flag in self.actuation.flags:
            self.actuator_flag_counts[flag]=self.actuator_flag_counts.get(flag,0)+1
        if self.attitude_chain is None and self.compliant_fixture is None:
            return orbit.step_achieved(self.actuation,internal_momentum_body_nms=None if wheel_old is None else (wheel_old+wheel_new)/2)
        # Keep guidance/allocation at the outer rate, but physically interleave
        # high-rate inertial observations and filter propagation with the plant.
        # Jet execution still supplies interval-mean wrench, not resolved pulses.
        from dataclasses import replace
        from spec import ActuationResult
        count=int(np.ceil(self.config.dt/.01))
        interval=self.config.dt/count
        for index in range(count):
            if index and self.attitude_chain is not None:
                self.attitude_chain.sample_gyro(body)
            segment=ActuationResult(replace(self.actuation.command,time=body.time),interval,
                self.actuation.force_n,self.actuation.torque_nm,
                self.actuation.propellant_used_kg/count,self.actuation.flags)
            propulsive_force=segment.force_n.copy()
            target_actuation=None
            if self.compliant_fixture is not None and self.compliant_fixture.engaged:
                from plant.rigid_contact import RigidBodyState
                from actuators.execution import combine_actuation
                def rigid(p):
                    return RigidBodyState(p.mass_kg,p.inertia_kg_m2,p.r,p.v,p.q,p.omega,p.time,'earth/gcrs')
                loads=self.compliant_fixture.loads(rigid(body),rigid(orbit.chief_body))
                self.contact_load_n=loads.demanded_force_n;self.contact_couple_nm=loads.demanded_couple_nm
                self.peak_contact_load_n=max(self.peak_contact_load_n,self.contact_load_n)
                self.attachment_broken=loads.broken
                contact=ActuationResult(WrenchCommand('fixture',body.time,body.body_frame),interval,
                    rot_matrix(body.q).T@loads.force_first_reference_n,loads.torque_first_body_nm)
                target_actuation=ActuationResult(WrenchCommand('fixture',body.time,'body/target'),interval,
                    -rot_matrix(orbit.chief_body.q).T@loads.force_first_reference_n,loads.torque_second_body_nm)
                segment=combine_actuation((segment,contact))
            truth=orbit.step_achieved(segment,chief_actuation=target_actuation,
                internal_momentum_body_nms=None if wheel_old is None else wheel_old+(index+.5)/count*(wheel_new-wheel_old),
                propulsive_force_body_n=propulsive_force)
            if self.attitude_chain is not None:self.attitude_chain.predict(body.time)
        return truth

    def truth_state(self):
        return self.model.to_lvlh() if self.orbit_truth is None else self.orbit_truth.to_lvlh()

    def transition(self, mode):
        self.mode = mode
        self.hold = 0.
        if mode != 'CAPTURE' and mode != 'COMPLETE':
            self.capture_hold = 0.
            self.latched = False
            if self.compliant_fixture is not None:self.compliant_fixture.release()
        self.transitions.append(dict(t=self.t, mode=mode))

    def update_retained_ranging(self):
        # Retain the independent far-leg range/bearing instrument through the
        # camera handover. Truth is used by the sensor only, never as a fix.
        if (self.retained_range_sensor is not None and self.t-self.last_range_sample>=.5-1e-9
                and not self.config.dropout_start<=self.t<self.config.dropout_end):
            self.last_range_sample=self.t
            pointing=self.x[:3]/max(1e-9,np.linalg.norm(self.x[:3]))
            measurement,noise=self.retained_range_sensor.measure(self.truth_state()[:3],pointing)
            if measurement is not None:
                self.range_update.x=self.x.copy();self.range_update.P=self.P.copy()
                if self.range_update.update(measurement,noise):
                    self.x=self.range_update.x.copy();self.P=self.range_update.P.copy();self.accepted_range_updates+=1
                else:self.rejected_range_updates+=1
    def sensor_update(self):
        self.update_retained_ranging()
        if self.mapped_camera is not None:
            self._mapped_sensor_update()
            self._mapped_rim_update()
            return
        if self.config.dropout_start <= self.t < self.config.dropout_end:
            return
        # Explicit heritage synthetic reference observation.
        measurement = self.truth_state()[:3] + self.rng.normal(0, .005, 3)
        H = np.c_[np.eye(3), np.zeros((3,3))]
        R = np.eye(3)*.005**2
        K = np.linalg.solve(H@self.P@H.T+R, H@self.P).T
        self.x += K@(measurement-H@self.x)
        I_KH = np.eye(6)-K@H
        self.P = I_KH@self.P@I_KH.T+K@R@K.T
        observed = Rotation.from_rotvec(self.rng.normal(0,np.radians(.015),3)).as_matrix() @ self.target
        self.pose.observe(observed, self.t)
        if round(self.t/self.config.dt)%max(1,round(.5/self.config.dt)) == 0:
            normal = self.fixture_normal
            tangent = np.cross(normal,[0.,1.,0.]); tangent/=np.linalg.norm(tangent)
            other = np.cross(normal,tangent)
            angles = np.linspace(0,2*np.pi,48,endpoint=False)
            cloud = self.fixture_center + .24*(np.cos(angles)[:,None]*tangent+np.sin(angles)[:,None]*other)
            # Sensor points first rotate to RTN; estimated pose transforms back.
            cloud = cloud@self.target.T+self.rng.normal(0,.001,(48,3))
            self.survey.observe(cloud@observed,self.t)

    def _mapped_sensor_update(self):
        from utils.quaternion import rot_matrix
        from utils.orbital_frames import eci_to_lvlh_rotation,skew
        orbit=self.orbit_truth;body=orbit.body;target_body=orbit.chief_body
        if target_body is None:raise ValueError('Mapped target camera requires a dynamic target')
        frame=eci_to_lvlh_rotation(orbit.chief_position,orbit.chief_velocity)
        if self.attitude_chain.filter is not None:
            direction=(frame@rot_matrix(self.attitude_chain.filter.q)).T@(-self.x[:3])
            angles=[np.arctan2(direction[1],direction[0]),np.arctan2(direction[2],np.hypot(*direction[:2]))]
            gimbal=self.camera_gimbal.step(angles,self.config.dt)
            z=gimbal.boresight_body
            up=np.array([0.,1.,0.]) if abs(z[1])<.95 else np.array([1.,0.,0.])
            x=np.cross(up,z);x/=np.linalg.norm(x)
            self.camera_mount=np.column_stack((x,np.cross(z,x),z))
            self.mapped_camera.model.mount=self.camera_mount
        sample=self.mapped_camera.sample(body.time,observer_position_m=body.r,observer_quaternion=body.q,
            target_position_m=target_body.r,target_quaternion=target_body.q,
            points_body_m=self.target_map,normals_body=self.target_normals,
            enabled=bool(not self.config.dropout_start<=self.t<self.config.dropout_end))
        if not sample.valid or self.attitude_chain.filter is None:return
        frame=eci_to_lvlh_rotation(orbit.chief_position,orbit.chief_velocity)
        camera_to_rtn=frame@rot_matrix(self.attitude_chain.filter.q)@self.camera_mount
        measurement=-camera_to_rtn@sample.value[:3]
        jac=camera_to_rtn@skew(sample.value[:3])
        noise=2*(camera_to_rtn@sample.covariance[:3,:3]@camera_to_rtn.T+
            jac@self.attitude_chain.filter.P[:3,:3]@jac.T)+np.eye(3)*.002**2
        H=np.c_[np.eye(3),np.zeros((3,3))]
        residual=measurement-self.x[:3];innovation=H@self.P@H.T+noise
        nis=float(residual@np.linalg.solve(innovation,residual))
        if nis>25.:self.rejected_pixel_poses+=1;return
        gain=np.linalg.solve(innovation,H@self.P).T
        self.x+=gain@residual;ikh=np.eye(6)-gain@H
        self.P=ikh@self.P@ikh.T+gain@noise@gain.T;self.P=(self.P+self.P.T)/2
        observed=camera_to_rtn@rot_matrix(sample.value[3:])
        self.pose.observe(observed,self.t);self.accepted_pixel_poses+=1
        body_to_rtn=frame@rot_matrix(self.attitude_chain.filter.q)
        self.last_pose_angular_rtn=(camera_to_rtn@sample.covariance[3:,3:]@camera_to_rtn.T+
                         body_to_rtn@self.attitude_chain.filter.P[:3,:3]@body_to_rtn.T)

    def _mapped_rim_update(self):
        fitted_rotation,_,healthy=self.pose.estimate(self.t)
        if (not healthy or not hasattr(self,'last_pose_angular_rtn')
                or self.config.dropout_start<=self.t<self.config.dropout_end):return
        # A segmented ranging rim is a separate sensor abstraction. The
        # fixture itself is fitted from these points rather than read by FSW.
        # It runs on its own cadence, including ticks without a new PnP fix.
        if round(self.t/self.config.dt)%max(1,round(.5/self.config.dt))==0:
            normal=self.fixture_normal
            tangent=np.cross(normal,[0.,1.,0.]);tangent/=np.linalg.norm(tangent)
            other=np.cross(normal,tangent)
            angles=np.linspace(0,2*np.pi,48,endpoint=False)
            cloud=self.fixture_center+.24*(np.cos(angles)[:,None]*tangent+np.sin(angles)[:,None]*other)
            cloud=cloud@self.target.T+self.rng.normal(0,.001,(48,3))
            age=max(0.,self.t-self.pose.last_seen)
            angular_rtn=self.last_pose_angular_rtn+np.eye(3)*(np.radians(.03)*age)**2
            # Use the same filtered target frame as guidance. Raw single-frame
            # PnP rotations made a static rim appear to move between fits,
            # repeatedly expiring identification during otherwise healthy sync.
            self.survey.observe(cloud@fitted_rotation,self.t,
                                fitted_rotation.T@angular_rtn@fitted_rotation)

    def step(self):
        dt = self.config.dt
        self.sensor_update()
        target, target_rate, healthy = self.pose.estimate(self.t)
        target_pose_healthy=healthy
        if self.attitude_chain is None:
            gyro = self.rate + self.rng.normal(0,np.radians(.0005),3)
            star = Rotation.from_rotvec(self.rng.normal(0,np.radians(.005),3)).as_matrix() @ self.attitude
        else:
            star,gyro,self.attitude_ready=self.attitude_chain.observe(self.orbit_truth)
            healthy=healthy and self.attitude_ready
        if target is None:
            target, target_rate = star, np.zeros(3)
        identified = self.survey.valid(self.t)
        center = self.survey.center if self.survey.center is not None else np.array([0.,0.,2.])
        normal = self.survey.normal if self.survey.normal is not None else np.array([0.,0.,1.])
        axis = target@normal
        bell = target@center
        desired = bell+(.5+self.standoff)*axis
        if self.mode == 'SURVEY':
            direction = self.x[:3]/max(np.linalg.norm(self.x[:3]),1e-9)
            desired = 12.5*(direction+.5*axis)/np.linalg.norm(direction+.5*axis)
        if self.mode == 'SPIN_SYNC':
            desired = 12.5*axis
        desired_velocity = np.cross(target_rate,desired)
        x_axis=target[:,0]-axis*(axis@target[:,0]); x_axis/=np.linalg.norm(x_axis)
        desired_attitude=np.column_stack((x_axis,np.cross(axis,x_axis),axis))
        attitude_error = Rotation.from_matrix(desired_attitude@star.T).as_rotvec()
        angular_accel = limit(.012*attitude_error + .22*(target_rate-gyro), np.radians(.03))
        estimated_metrics = contact_metrics(self.x[:3],self.x[3:],star,gyro,target,target_rate,center,normal)
        if not (healthy and identified) and self.mode in ('SPIN_SYNC','APPROACH','CAPTURE'):
            self.transition('RETREAT')
            self.standoff = 10.
            distance=np.linalg.norm(self.x[:3])
            self.recovery_waypoint=self.x[:3]*max(20.,distance)/max(distance,1e-9)
            self.recovery_attitude=star.copy()
        if self.mode=='RETREAT':
            # A recovery destination is latched once in the rotating frame.
            # Do not chase a noisy/expired target axis or invent its velocity.
            if self.recovery_waypoint is None:
                distance=np.linalg.norm(self.x[:3])
                self.recovery_waypoint=self.x[:3]*max(20.,distance)/max(distance,1e-9)
                self.recovery_attitude=star.copy()
            desired=self.recovery_waypoint.copy()
            desired_attitude=self.recovery_attitude
            attitude_error=Rotation.from_matrix(desired_attitude@star.T).as_rotvec()
            angular_accel=limit(.012*attitude_error-.22*gyro,np.radians(.03))
        if self.mode == 'SURVEY' and self.t >= 30 and healthy and identified:
            self.transition('SPIN_SYNC')
        position_error = np.linalg.norm(self.x[:3]-desired)
        sync = (np.degrees(np.linalg.norm(attitude_error)) < 1.
                and np.degrees(np.linalg.norm(gyro-target_rate)) < .03
                and np.linalg.norm(self.x[:3]-12.5*axis) < .15)
        if self.mode == 'SPIN_SYNC':
            self.hold = self.hold+dt if sync and healthy else 0.
            if self.hold >= 10:
                self.transition('APPROACH')
        if self.mode == 'RETREAT' and healthy and identified and position_error < .2 and np.linalg.norm(self.x[3:])<.02:
            self.transition('SPIN_SYNC')
        closing_rate = 0.
        if self.mode == 'APPROACH' and healthy:
            if position_error < .15 and np.linalg.norm(attitude_error) < np.radians(2):
                previous = self.standoff
                self.standoff = max(.012, self.standoff-min(.018,.03*self.standoff)*dt)
                closing_rate = (self.standoff-previous)/dt
            if self.standoff <= .012+1e-9 and eligible(estimated_metrics):
                self.transition('CAPTURE')
        if self.mode in ('APPROACH','CAPTURE','COMPLETE'):
            desired = bell+(.5+self.standoff)*axis
        desired_velocity = np.zeros(3) if self.mode=='RETREAT' else np.cross(target_rate,desired)+closing_rate*axis
        reference_rate=np.zeros(3) if self.mode=='RETREAT' else target_rate
        # Estimate-only tracking with selected secular-ROE drift cancellation.
        tiny = .001
        F = self.model.cartesian_transition(tiny)
        drift = ((F@self.x-self.x)/tiny)[3:]
        accel = (.0036*(desired-self.x[:3]) + .12*(desired_velocity-self.x[3:])
                 + np.cross(reference_rate,np.cross(reference_rate,desired)) - drift)
        accel = limit(accel,.003)
        if not self.attitude_ready:
            accel=np.zeros(3)
        if self.rcs is None:
            self.rate += angular_accel*dt
            self.attitude = Rotation.from_rotvec(self.rate*dt).as_matrix()@self.attitude
        if self.orbit_truth is None or self.orbit_truth.chief_body is None:
            self.target = Rotation.from_rotvec(self.target_rate*dt).as_matrix()@self.target
        F = self.model.cartesian_transition(dt)
        B = self.model.mapping_matrix(self.model.mean_anomaly+self.model.phase_rate*dt)@self.model.control_matrix(dt)
        truth = self.model.step(dt,accel)
        if self.rcs is not None:
            truth = self.execute_rcs(accel,angular_accel,star,gyro)
            self.attitude,self.rate=self.orbit_truth.body_state_rtn()
            if self.orbit_truth.chief_body is not None:
                self.target,self.target_rate=self.orbit_truth.chief_body_state_rtn()
        elif self.orbit_truth is not None:
            truth = self.orbit_truth.step(dt,accel)
        self.x = F@self.x+B@accel
        q=self.config.navigation_acceleration_noise_density**2
        process=q*np.block([[np.eye(3)*dt**3/3,np.eye(3)*dt**2/2],
                             [np.eye(3)*dt**2/2,np.eye(3)*dt]])
        self.P = F@self.P@F.T + process
        self.P=(self.P+self.P.T)/2
        self.t += dt
        metrics = contact_metrics(truth[:3],truth[3:],self.attitude,self.rate,self.target,self.target_rate,self.fixture_center,self.fixture_normal)
        clearance=self.bus_pair.clearance(np.zeros(3),self.target,truth[:3],self.attitude)
        self.bus_collision=self.bus_collision or clearance['collision']
        self.min_bus_sphere_clearance_m=min(self.min_bus_sphere_clearance_m,clearance['sphere_clearance_m'])
        # No pose/state snap: geometric latch qualification stays independent.
        contact_ok = eligible(metrics) and self.config.contact_enabled and not self.bus_collision
        if self.mode == 'CAPTURE':
            if self.pre_capture_evidence is None:
                self.pre_capture_evidence = dict(t=self.t,**metrics)
            if self.config.contact_model=='compliant':
                if self.compliant_fixture is None and contact_ok:
                    from plant.compliant_capture import CompliantCapture
                    from plant.rigid_contact import RigidBodyState
                    orbit=self.orbit_truth
                    def rigid(p):
                        return RigidBodyState(p.mass_kg,p.inertia_kg_m2,p.r,p.v,p.q,p.omega,p.time,'earth/gcrs')
                    self.compliant_fixture=CompliantCapture(first_fixture_body_m=[0.,0.,-.5],
                        second_fixture_body_m=self.fixture_center,maximum_force_n=self.config.maximum_contact_load_n)
                    self.compliant_fixture.engage(rigid(orbit.body),rigid(orbit.chief_body))
                self.latched=bool(self.compliant_fixture is not None and self.compliant_fixture.engaged and not self.attachment_broken)
            else:self.latched = bool(contact_ok)
            self.capture_hold = self.capture_hold+dt if self.latched and contact_ok else 0.
            if not contact_ok:
                self.rejected_contacts += 1
            if self.capture_hold >= 5:
                self.complete = True
                self.reason = 'qualified_capture'
                self.transition('COMPLETE')
        if not self.actuation_health.ready:
            self.complete=False;self.reason='actuator_authority_loss';self.transition('ABORT')
        if self.attachment_broken:
            self.complete=False;self.reason='attachment_overload';self.transition('ABORT')
        if self.bus_collision or not np.all(np.isfinite(truth)) or np.linalg.norm(truth[:3]) > 100 or np.linalg.norm(truth[:3]) < 1.9:
            self.complete=False
            self.reason = 'bus_collision' if self.bus_collision else 'safety_abort'
            self.transition('ABORT')
        self.records.append(dict(t=self.t,mode=self.mode,range_m=float(np.linalg.norm(truth[:3])),
                                 truth_elapsed_s=self.model.t if self.orbit_truth is None else self.orbit_truth.t,
                                 chief_position_m=self.orbit_truth.chief_position.tolist() if self.orbit_truth is not None else None,
                                 deputy_position_m=self.orbit_truth.deputy_position.tolist() if self.orbit_truth is not None else None,
                                 standoff_m=self.standoff,nav_error_m=float(np.linalg.norm(truth[:3]-self.x[:3])),
                                 pose_healthy=target_pose_healthy,guidance_sensing_healthy=healthy,bell_identified=identified,
                                 bell_fit_residual_m=self.survey.residual if np.isfinite(self.survey.residual) else None,
                                 bell_fit_count=self.survey.accepted,bell_fit_rejections=self.survey.rejected,
                                 bell_fit_reason=self.survey.last_reason,
                                 bell_fit_nis=self.survey.last_nis,
                                 estimated_target_rate_rad_s=target_rate.tolist(),
                                 truth_target_rate_rad_s=self.target_rate.tolist(),
                                 estimated_sync_attitude_error_deg=float(np.degrees(np.linalg.norm(attitude_error))),
                                 estimated_sync_rate_error_deg_s=float(np.degrees(np.linalg.norm(gyro-target_rate))),
                                 estimated_sync_position_error_m=float(np.linalg.norm(self.x[:3]-12.5*axis)),
                                 sync_hold_s=self.hold,
                                 latched=self.latched,capture_hold_s=self.capture_hold,**metrics))
        self.records[-1].update(contact_force_n=self.contact_load_n,contact_couple_nm=self.contact_couple_nm,
                               attachment_broken=self.attachment_broken,bus_collision=self.bus_collision,
                               bus_sphere_clearance_m=clearance['sphere_clearance_m'])
        if self.attitude_chain is not None:
            self.records[-1].update(self.attitude_chain.diagnostics(self.orbit_truth))
            self.records[-1].update(measured_relative_rate_rad_s=gyro.tolist(),
                truth_relative_rate_rad_s=self.rate.tolist(),angular_accel_command_rad_s2=angular_accel.tolist(),
                requested_torque_body_nm=self.actuation.command.torque_nm.tolist())
        if self.actuation is not None:
            self.records[-1].update(propellant_used_kg=self.terminal_initial_propellant_kg-self.rcs.tank.propellant_kg,
                achieved_force_body_n=self.actuation.force_n.tolist(),
                achieved_torque_body_nm=self.actuation.torque_nm.tolist(),
                force_residual_n=float(np.linalg.norm(self.actuation.residual[:3])),
                torque_residual_nm=float(np.linalg.norm(self.actuation.residual[3:])),
                actuator_flags=list(self.actuation.flags))

    def run(self, progress=None):
        previous_mode = None
        next_checkpoint = 0.
        while self.t+self.config.dt <= self.config.duration+1e-9 and not self.complete and self.mode != 'ABORT':
            self.step()
            if progress is not None and (self.mode != previous_mode or self.t >= next_checkpoint):
                progress(self)
                previous_mode = self.mode
                next_checkpoint = self.t+60.
        return self.summary()

    def summary(self):
        criteria=dict(capture_hold=self.complete and self.capture_hold>=5.,
            entry_verified=self.pre_capture_evidence is not None,
            no_bus_collision=not self.bus_collision,no_attachment_overload=not self.attachment_broken,
            actuator_health=self.actuation_health.ready)
        if self.attitude_chain is not None:
            criteria['measured_attitude']=self.attitude_chain.healthy
        if self.config.vision_model=='mapped_pixels':
            criteria['pixel_acquisition']=getattr(self,'accepted_pixel_poses',0)>0
        if self.config.contact_model=='compliant':
            criteria['physical_attachment']=bool(self.compliant_fixture is not None and self.compliant_fixture.engaged)
        return dict(seed=self.config.seed,success=bool(self.complete and all(criteria.values())),criteria=criteria,
                    accepted_range_updates=self.accepted_range_updates,rejected_range_updates=self.rejected_range_updates,
                    recovery_waypoint_m=None if self.recovery_waypoint is None else self.recovery_waypoint.tolist(),
                    reason=self.reason,elapsed_s=self.t,
                    transitions=self.transitions,handoff=self.handoff,final=self.records[-1] if self.records else {},
                    pre_capture_evidence=self.pre_capture_evidence,
                    rejected_contacts=self.rejected_contacts,
                    total_dv_ms=self.model.total_dv_ms if self.orbit_truth is None else self.orbit_truth.total_dv_ms,
                    vision_model=self.config.vision_model,accepted_pixel_poses=getattr(self,'accepted_pixel_poses',0),
                    rejected_pixel_poses=getattr(self,'rejected_pixel_poses',0),
                    actuation_model=self.config.actuation_model,
                    attitude_model=self.config.attitude_model,
                    environment_model=self.config.environment_model,
                    environment_scope=('Gravity/J2 throughout' if self.config.environment_model=='gravity_j2' else
                        'Deputy plate drag/SRP/GG and chief effective-area drag/SRP; tabulated atmosphere; selected physical far path shares these providers'),
                    terminal_propellant_used_kg=None if self.rcs is None else self.terminal_initial_propellant_kg-self.rcs.tank.propellant_kg,
                    peak_force_residual_n=self.peak_force_residual_n,
                    peak_torque_residual_nm=self.peak_torque_residual_nm,
                    actuator_flag_counts=self.actuator_flag_counts,actuator_health_events=self.actuation_health.events,
                    bus_collision=self.bus_collision,
                    min_bus_sphere_clearance_m=None if not np.isfinite(self.min_bus_sphere_clearance_m) else self.min_bus_sphere_clearance_m,
                    collision_scope='Oriented example bus boxes sampled each plant step; appendages excluded',
                    model=('Circular-chief secular J2 ROE' if self.orbit_truth is None else
                        'Independent nonlinear Cartesian two-body plus J2 chief/deputy truth; secular ROE navigation prediction')
                        +(', mapped pixels/PnP with known bus features; independent segmented bell-rim ranging, ' if self.mapped_camera is not None else ', synthetic visual reference pose, ')+('ideal 3-axis thrust/torque' if self.rcs is None else
                        'achieved bounded pulsed 16-jet RCS, fuel depletion, inertial rigid-body Euler dynamics; estimate-driven chief plume cone; finite far leg when inherited physical state is supplied'),
                    peak_contact_load_n=self.peak_contact_load_n,attachment_broken=self.attachment_broken,
                    capture=('Two dynamic bodies and finite compliant fixture forces/torques; no pose/velocity reset; declared mechanism parameters'
                        if self.config.contact_model=='compliant' else 'Geometric reference qualification only; no compliant contact'))
