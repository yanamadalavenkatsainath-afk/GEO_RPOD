"""Configurable low-order plate forces, gravity-gradient torque and dipole field."""
import numpy as np

from utils.quaternion import normalize, rot_matrix
from utils.validation import covariance, finite_array, finite_scalar, vector


def gravity_gradient_torque(position_reference_m, quaternion_body_to_reference, inertia_body_kg_m2, *, mu_m3_s2):
    position = vector(position_reference_m, "position_reference_m", size=3)
    radius = np.linalg.norm(position)
    if radius == 0:
        raise ValueError("Nonzero central-body radius required")
    inertia = covariance(inertia_body_kg_m2, 3, "inertia_body_kg_m2")
    mu = finite_scalar(mu_m3_s2, "mu_m3_s2", minimum=0)
    body_radial = rot_matrix(quaternion_body_to_reference).T @ (position/radius)
    return 3*mu/radius**3*np.cross(body_radial, inertia @ body_radial)


def gravity_gradient_from_tensor(acceleration_gradient_reference_s2,quaternion_body_to_reference,inertia_body_kg_m2):
    """Rigid gravitational torque from a symmetric local acceleration gradient."""
    gradient=finite_array(acceleration_gradient_reference_s2,'acceleration_gradient_reference_s2',shape=(3,3))
    if not np.allclose(gradient,gradient.T,rtol=1e-8,atol=1e-16):raise ValueError('Conservative symmetric gravity gradient required')
    inertia=covariance(inertia_body_kg_m2,3,'inertia_body_kg_m2')
    rotation=rot_matrix(quaternion_body_to_reference);body=rotation.T@gradient@rotation
    commutator=body@inertia-inertia@body
    return np.array([commutator[1,2],commutator[2,0],commutator[0,1]])


def dipole_magnetic_field(position_m, magnetic_moment_a_m2):
    """SI vacuum dipole approximation in the same frame as position/moment.

    Not an IGRF replacement: no higher harmonics, secular variation or external
    magnetosphere currents. Select a validated provider for precision studies.
    """
    position = vector(position_m, "position_m", size=3)
    moment = vector(magnetic_moment_a_m2, "magnetic_moment_a_m2", size=3)
    distance = np.linalg.norm(position)
    if distance == 0:
        raise ValueError("Dipole field undefined at origin")
    radial = position/distance
    return 1e-7*(3*radial*(moment @ radial)-moment)/distance**3


class EarthPlateWrench:
    """Stage-evaluated body wrench for a configured Earth-orbiting bus.

    Inject Sun position and density providers with the plant's timestamp and
    reference frame. Density receives (time, position); Sun returns Earth-to-Sun
    position in metres. This assembly adds no atmosphere/ephemeris fidelity to
    its providers. Constant-Cd plate drag, scalar optical scale and spherical
    eclipse retain the limitations of the underlying common models.
    """
    def __init__(self, *, plates, sun_position, density, inertia, cd, cr,
                 mu_m3_s2=3.986004418e14, radius_m=6378100.,
                 atmosphere_omega_rad_s=(0.,0.,7.292115e-5)):
        self.plates=plates
        self.sun_position=sun_position
        self.density=density
        self.inertia=inertia
        self.cd=finite_scalar(cd,'cd',minimum=0)
        self.cr=finite_scalar(cr,'cr',minimum=0)
        self.mu=finite_scalar(mu_m3_s2,'mu_m3_s2',strictly_positive=True)
        self.radius=finite_scalar(radius_m,'radius_m',strictly_positive=True)
        self.atmosphere_omega=vector(atmosphere_omega_rad_s,'atmosphere_omega_rad_s',size=3)

    def __call__(self,time,r,v,q,omega,mass):
        from environment.common import illumination_fraction
        sun=vector(self.sun_position(time),'sun_position',size=3)
        toward=sun-r
        distance=np.linalg.norm(toward)
        if distance==0:raise ValueError('Spacecraft coincides with Sun position')
        illuminated=illumination_fraction(r,sun,occulting_radius_m=self.radius)
        relative=v-np.cross(self.atmosphere_omega,r)
        drag_force,drag_torque=self.plates.drag(relative,q,density_kg_m3=self.density(time,r),cd=self.cd)
        srp_force,srp_torque=self.plates.srp(toward,q,
            pressure_pa=4.56e-6*self.cr*(149597870700./distance)**2,illumination=illuminated)
        inertia=self.inertia(time,mass) if callable(self.inertia) else self.inertia
        gg=gravity_gradient_torque(r,q,inertia,mu_m3_s2=self.mu)
        return drag_force+srp_force,drag_torque+srp_torque+gg


class PlateForceModel:
    """Per-face drag/absorbing-SRP force and torque about supplied body COM.

    Drag is a constant-Cd free-molecular directional approximation, not a
    species-resolved Sentman model. SRP treats absorption; specular/diffuse
    optical coefficients and self-shadowing need a selected higher-fidelity model.
    """
    def __init__(self, *, areas_m2, normals_body, centres_body_m):
        self.areas = vector(areas_m2, "areas_m2")
        self.normals = finite_array(normals_body, "normals_body", shape=(len(self.areas), 3))
        self.centres = finite_array(centres_body_m, "centres_body_m", shape=(len(self.areas), 3))
        lengths = np.linalg.norm(self.normals, axis=1)
        if np.any(self.areas <= 0) or np.any(lengths == 0):
            raise ValueError("Positive areas and nonzero face normals required")
        self.normals /= lengths[:, None]

    def drag(self, relative_velocity_reference_m_s, quaternion_body_to_reference, *, density_kg_m3,
             cd, centre_of_mass_body_m=(0., 0., 0.)):
        velocity = rot_matrix(quaternion_body_to_reference).T @ vector(relative_velocity_reference_m_s, "relative_velocity_reference_m_s", size=3)
        speed = np.linalg.norm(velocity)
        density = finite_scalar(density_kg_m3, "density_kg_m3", minimum=0)
        coefficient = finite_scalar(cd, "cd", minimum=0)
        if speed == 0:
            return np.zeros(3), np.zeros(3)
        direction = velocity/speed
        exposed = self.areas*np.maximum(self.normals @ direction, 0)
        forces = -.5*density*coefficient*speed**2*exposed[:, None]*direction
        return self._sum(forces, centre_of_mass_body_m)

    def srp(self, sun_direction_reference, quaternion_body_to_reference, *, pressure_pa,
            illumination=1., centre_of_mass_body_m=(0., 0., 0.)):
        toward = rot_matrix(quaternion_body_to_reference).T @ normalize(vector(sun_direction_reference, "sun_direction_reference", size=3))
        pressure = finite_scalar(pressure_pa, "pressure_pa", minimum=0)
        fraction = finite_scalar(illumination, "illumination", minimum=0)
        if fraction > 1:
            raise ValueError("illumination must be in [0,1]")
        exposed = self.areas*np.maximum(self.normals @ toward, 0)
        forces = -fraction*pressure*exposed[:, None]*toward
        return self._sum(forces, centre_of_mass_body_m)

    def _sum(self, forces, centre):
        offset = self.centres-vector(centre, "centre_of_mass_body_m", size=3)
        return forces.sum(axis=0), np.cross(offset, forces).sum(axis=0)


class NRLMSISE00Provider:
    """Explicit physical-time and space-weather boundary for optional NRLMSISE-00.

    Geographic coordinates are geodetic degrees/altitude km as required by the
    package. No weather history or forecast is downloaded or inferred from date.
    """
    def density_at(self, time, *, altitude_m, latitude_rad, longitude_rad, f107a, f107, ap):
        from environment.common import astronomical_time
        from nrlmsise00 import msise_flat
        altitude = finite_scalar(altitude_m, "altitude_m", minimum=0)
        latitude = finite_scalar(latitude_rad, "latitude_rad")
        longitude = finite_scalar(longitude_rad, "longitude_rad")
        if abs(latitude) > np.pi/2:
            raise ValueError("Latitude must be in [-pi/2,pi/2]")
        weather = [finite_scalar(value, name, minimum=0) for value, name in ((f107a, "f107a"), (f107, "f107"), (ap, "ap"))]
        epoch = astronomical_time(time).utc.to_datetime()
        result = msise_flat(epoch, altitude/1000, np.degrees(latitude), np.degrees(longitude), *weather)
        return finite_scalar(float(result[5])*1000, "density_kg_m3", minimum=0)
