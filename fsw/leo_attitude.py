"""LEO wrapper's inertial attitude sensor/filter chain using common components.

Two example star-tracker heads provide opposite boresights. These are model
configurations, not a demonstrated hardware installation. The caller supplies
known chief ephemeris to convert estimates into its RTN guidance convention.
"""
import numpy as np
from sensors import create_sensor, SensorTiming
from sensors.gyro_performance import GyroPerformance
from estimation.attitude import AttitudeMEKF
from utils.orbital_frames import eci_to_lvlh_rotation, lvlh_frame_rate
from utils.quaternion import rot_matrix


class LeoAttitudeChain:
    def __init__(self,*,dt,seed,max_coast_s=300.,uncertainty_limit_deg=.1):
        # Resolve the fastest commanded slew inside the outer guidance step;
        # holding a 4 Hz gyro across acceleration biased the star innovations.
        self.dt=dt/int(np.ceil(dt/.01))
        self.performance=GyroPerformance()
        self.gyro=create_sensor('performance_gyro',source_id='leo/gyro',seed=seed,
            timing=SensorTiming(self.dt),dt=self.dt,performance=self.performance)
        self.trackers=[create_sensor('star_tracker',source_id=f'leo/star/{index}',seed=seed,
            timing=SensorTiming(.25),boresight_body=axis,acquisition_s=30.)
            for index,axis in enumerate(([1.,0.,0.],[-1.,0.,0.]))]
        self.filter=None
        self.last_star_time=-np.inf
        self.last_gyro=None
        self.accepted_stars=0
        self.rejected_stars=0
        self.unavailable_star_samples=0
        self.healthy=False
        self.max_coast_s=max_coast_s
        self.uncertainty_limit_deg=uncertainty_limit_deg
        self.uncertainty_bound_deg=None
        from environment.sun_model import SunModel
        self.sun_model=SunModel(epoch_year=2025.)
        self.sun_time=-np.inf

    def observe(self,orbit,*,sun_reference=None):
        body=orbit.body
        time=body.time
        if sun_reference is None and hasattr(orbit,'sun_position'):
            sun_reference=orbit.sun_position(time)-body.r
        if sun_reference is not None:
            from utils.quaternion import normalize
            self.sun=normalize(sun_reference)
            self.sun_time=time.seconds
        elif time.seconds-self.sun_time>=60:
            self.sun=self.sun_model.get_sun_vector(time.seconds)
            self.sun_time=time.seconds
        gyro=self.gyro.sample(time,omega_true=body.omega)
        self.last_gyro=gyro
        for tracker in self.trackers:
            sample=tracker.sample(time,q_true=body.q,sun_vec_eci=self.sun,pos_eci_m=body.r)
            if not sample.valid:
                self.unavailable_star_samples+=int(sample.reason!='NOT_DUE')
                continue
            if self.filter is None:
                # Only a valid observed attitude initializes the estimate.
                p=np.zeros((6,6));p[:3,:3]=sample.covariance
                p[3:,3:]=np.eye(3)*(np.deg2rad(self.performance.bias_repeatability_deg_hr)/3600)**2
                self.filter=AttitudeMEKF(time=time,quaternion=sample.value,bias_rad_s=np.zeros(3),
                    initial_covariance=p,gyro_noise_density=np.deg2rad(self.performance.arw_deg_sqrt_hr)/60,
                    bias_walk_density=np.deg2rad(self.performance.rrw_deg_hr_sqrt_hr)/(3600*60),
                    max_gyro_age_s=self.dt+1e-9)
                self.last_star_time=time.seconds
                self.accepted_stars+=1
            else:
                result=self.filter.update_star_tracker(sample)
                if result.accepted:
                    self.last_star_time=time.seconds
                    self.accepted_stars+=1
                else:
                    self.rejected_stars+=1
        frame=eci_to_lvlh_rotation(orbit.chief_position,orbit.chief_velocity)
        rotation=frame@(np.eye(3) if self.filter is None else rot_matrix(self.filter.q))
        frame_rate=lvlh_frame_rate(orbit.chief_position,orbit.chief_velocity,
                                  orbit.chief_acceleration(orbit.chief_position) if hasattr(orbit,'chief_acceleration') else orbit.acceleration(orbit.chief_position))
        measured_rate=np.zeros(3) if not gyro.valid else gyro.value
        bias=np.zeros(3) if self.filter is None else self.filter.bias
        relative_rate=rotation@(measured_rate-bias)-frame_rate
        age=time.seconds-self.last_star_time
        if self.filter is not None:
            # The six-state filter approximates the multi-OU flicker process.
            # Add a conservative integrated-rate envelope for those omitted
            # states instead of treating its white-noise covariance as complete.
            angle_sigma=np.sqrt(max(0.,np.max(np.linalg.eigvalsh(self.filter.P[:3,:3]))))
            flicker_sigma=self.gyro.model.flicker_sigma*np.sqrt(len(self.gyro.model.tau))
            self.uncertainty_bound_deg=float(np.degrees(3*angle_sigma+3*flicker_sigma*age))
        self.healthy=bool(self.filter is not None and gyro.valid and age<=self.max_coast_s
            and self.uncertainty_bound_deg<=self.uncertainty_limit_deg)
        return rotation,relative_rate,self.healthy

    def sample_gyro(self,body):
        """Acquire at the real intermediate plant epoch, never retroactively."""
        self.last_gyro=self.gyro.sample(body.time,omega_true=body.omega)

    def predict(self,to_time):
        if self.filter is not None:
            result=self.filter.predict(self.last_gyro,to_time)
            if not result.accepted:
                self.healthy=False

    def diagnostics(self,orbit):
        error=None
        if self.filter is not None:
            from utils.quaternion import quat_error
            error=float(np.degrees(2*np.arccos(np.clip(abs(quat_error(orbit.body.q,self.filter.q)[0]),0,1))))
        return dict(attitude_estimation_error_deg=error,attitude_healthy=self.healthy,
            attitude_uncertainty_bound_deg=self.uncertainty_bound_deg,
            gyro_bias_estimate_rad_s=None if self.filter is None else self.filter.bias.tolist(),
            star_age_s=None if not np.isfinite(self.last_star_time) else orbit.t-self.last_star_time,
            accepted_star_samples=self.accepted_stars,rejected_star_samples=self.rejected_stars,
            unavailable_star_samples=self.unavailable_star_samples)
