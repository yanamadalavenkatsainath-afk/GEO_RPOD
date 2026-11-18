"""Reusable configured Earth bus environment, independent of mission sequencing."""
import numpy as np
from spec import Timestamp,Epoch
from utils.validation import finite_scalar,vector
from environment.common import AstropySunProvider,drag_acceleration,srp_acceleration,illumination_fraction
from environment.surface_forces import PlateForceModel,EarthPlateWrench


class InterpolatedSunPosition:
    """Lazy linear position interpolation on an explicit physical epoch.

    Interval is configurable. This is a computational approximation, not an
    accuracy guarantee. Provider and caller must use the same Earth frame.
    """
    def __init__(self,epoch,*,interval_s=60.,provider=None):
        self.epoch=epoch
        self.interval=finite_scalar(interval_s,'interval_s',strictly_positive=True)
        self.provider=AstropySunProvider() if provider is None else provider
        self.cache={}

    def __call__(self,time):
        # Simulation elapsed seconds are anchored to the configured epoch.
        if time.epoch not in (Epoch(),self.epoch):
            raise ValueError('Sun position timestamp has a different epoch')
        index=int(np.floor(time.seconds/self.interval))
        for knot in (index,index+1):
            if knot not in self.cache:
                self.cache[knot]=self.provider.position_at(Timestamp(knot*self.interval,self.epoch)).value
        weight=time.seconds/self.interval-index
        result=(1-weight)*self.cache[index]+weight*self.cache[index+1]
        # Retain only adjacent knots; long runs do not grow an ephemeris table.
        self.cache={k:self.cache[k] for k in (index,index+1)}
        return result


def box_plate_environment(*,half_extents_m,pressure_offset_m,sun_position,density,inertia,
                          cd=2.2,cr=1.3,mu_m3_s2=3.986004418e14,radius_m=6378100.):
    """Six faces around the bus COM, with an explicit force-centre offset."""
    half=vector(half_extents_m,'half_extents_m',size=3)
    if np.any(half<=0):raise ValueError('Box half extents must be positive')
    offset=vector(pressure_offset_m,'pressure_offset_m',size=3)
    normals=np.vstack((np.eye(3),-np.eye(3)))
    areas=4*np.array([half[1]*half[2],half[0]*half[2],half[0]*half[1]])
    plates=PlateForceModel(areas_m2=np.tile(areas,2),normals_body=normals,
        centres_body_m=normals*half+offset)
    return EarthPlateWrench(plates=plates,sun_position=sun_position,density=density,inertia=inertia,
        cd=cd,cr=cr,mu_m3_s2=mu_m3_s2,radius_m=radius_m)


class CannonballEarthEnvironment:
    """Drag/SRP acceleration for a target with prescribed effective area.

    No attitude torque or target attitude dynamics is inferred from area.
    Central/J2 gravity is added separately by the orbital plant.
    """
    def __init__(self,*,mass_kg,area_m2,sun_position,density,cd=2.2,cr=1.3,radius_m=6378100.):
        self.mass=finite_scalar(mass_kg,'mass_kg',strictly_positive=True)
        self.area=finite_scalar(area_m2,'area_m2',minimum=0)
        self.cd=finite_scalar(cd,'cd',minimum=0);self.cr=finite_scalar(cr,'cr',minimum=0)
        self.radius=finite_scalar(radius_m,'radius_m',strictly_positive=True)
        self.sun_position=sun_position;self.density=density

    def __call__(self,time,r,v):
        sun=self.sun_position(time)
        drag=drag_acceleration(v-np.cross([0.,0.,7.292115e-5],r),density_kg_m3=self.density(time,r),
            cd=self.cd,area_m2=self.area,mass_kg=self.mass)
        srp=srp_acceleration(r,sun,cr=self.cr,area_m2=self.area,mass_kg=self.mass,
            illumination=illumination_fraction(r,sun,occulting_radius_m=self.radius))
        return drag+srp
