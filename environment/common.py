"""SI environment primitives and explicit ephemeris/frame providers.

The numerical core needs NumPy only. Precision Earth/Sun and SGP4 adapters
load optional astronomy packages only when explicitly constructed/called.
"""
import numpy as np

from spec import Epoch, Reference, TimeScale, Timestamp
from utils.quaternion import quat_from_rotvec, rot_matrix
from utils.validation import finite_scalar, rotation_matrix, text_label, vector


def central_gravity(position_m, mu_m3_s2):
    r = vector(position_m, "position_m", size=3)
    mu = finite_scalar(mu_m3_s2, "mu_m3_s2", minimum=0)
    radius = np.linalg.norm(r)
    if radius == 0:
        raise ValueError("Gravity requires a nonzero radius")
    return -mu*r/radius**3


def j2_acceleration(position_m, *, mu_m3_s2, radius_m, j2):
    r = vector(position_m, "position_m", size=3)
    mu = finite_scalar(mu_m3_s2, "mu_m3_s2", strictly_positive=True)
    radius = finite_scalar(radius_m, "radius_m", strictly_positive=True)
    coefficient = finite_scalar(j2, "j2")
    distance = np.linalg.norm(r)
    if distance == 0:
        raise ValueError("J2 requires a nonzero radius")
    factor = 5*r[2]**2/distance**2
    return -1.5*coefficient*mu*radius**2/distance**5*r*np.array([1-factor, 1-factor, 3-factor])


def third_body_acceleration(position_m, body_position_m, mu_m3_s2):
    r = vector(position_m, "position_m", size=3)
    body = vector(body_position_m, "body_position_m", size=3)
    mu = finite_scalar(mu_m3_s2, "mu_m3_s2", minimum=0)
    return central_gravity(r-body, mu) - central_gravity(-body, mu)


def drag_acceleration(relative_velocity_m_s, *, density_kg_m3, cd, area_m2, mass_kg):
    velocity = vector(relative_velocity_m_s, "relative_velocity_m_s", size=3)
    rho = finite_scalar(density_kg_m3, "density_kg_m3", minimum=0)
    cd = finite_scalar(cd, "cd", minimum=0)
    area = finite_scalar(area_m2, "area_m2", minimum=0)
    mass = finite_scalar(mass_kg, "mass_kg", strictly_positive=True)
    return -.5*rho*cd*area/mass*np.linalg.norm(velocity)*velocity


def illumination_fraction(position_m, sun_position_m, *, occulting_radius_m=6378137., sun_radius_m=6.957e8):
    """Uniform-brightness apparent-disk overlap; includes annular eclipse.

    Both positions are relative to the occulting body's centre in one frame.
    Uses angular-radius disks on the sky, not radiative transfer or limb darkening.
    """
    r = vector(position_m, "position_m", size=3)
    toward_sun = vector(sun_position_m, "sun_position_m", size=3)-r
    earth_radius = finite_scalar(occulting_radius_m, "occulting_radius_m", strictly_positive=True)
    solar_radius = finite_scalar(sun_radius_m, "sun_radius_m", strictly_positive=True)
    distance, sun_distance = np.linalg.norm(r), np.linalg.norm(toward_sun)
    if distance <= earth_radius or sun_distance <= solar_radius:
        raise ValueError("Observer must be outside the occulting body and Sun")
    a, b = np.arcsin(solar_radius/sun_distance), np.arcsin(earth_radius/distance)
    separation = np.arccos(np.clip((-r/distance) @ (toward_sun/sun_distance), -1, 1))
    if separation >= a+b:
        return 1.
    if separation <= abs(a-b):
        return float(max(0., 1-(b/a)**2)) if a > b else 0.
    alpha = np.arccos(np.clip((separation**2+a*a-b*b)/(2*separation*a), -1, 1))
    beta = np.arccos(np.clip((separation**2+b*b-a*a)/(2*separation*b), -1, 1))
    product = (-separation+a+b)*(separation+a-b)*(separation-a+b)*(separation+a+b)
    overlap = a*a*alpha+b*b*beta-.5*np.sqrt(max(0., product))
    return float(np.clip(1-overlap/(np.pi*a*a), 0, 1))


def srp_acceleration(position_m, sun_position_m, *, cr, area_m2, mass_kg, illumination=1., pressure_at_au_pa=4.56e-6):
    away = vector(position_m, "position_m", size=3)-vector(sun_position_m, "sun_position_m", size=3)
    distance = np.linalg.norm(away)
    if distance == 0:
        raise ValueError("Spacecraft cannot be at the Sun's centre")
    fraction = finite_scalar(illumination, "illumination", minimum=0)
    if fraction > 1:
        raise ValueError("illumination must be in [0,1]")
    cr = finite_scalar(cr, "cr", minimum=0)
    area = finite_scalar(area_m2, "area_m2", minimum=0)
    mass = finite_scalar(mass_kg, "mass_kg", strictly_positive=True)
    pressure = finite_scalar(pressure_at_au_pa, "pressure_at_au_pa", minimum=0)
    return fraction*pressure*(149597870700./distance)**2*cr*area/mass*away/distance


class FunctionEphemeris:
    """Explicit callback returning SI [position, velocity] at a named epoch.

    A reference/model provider, not a truth or uncertainty declaration. Caller
    selects which provider is used to generate truth versus navigation data.
    """
    def __init__(self, function, *, epoch, frame, source_id):
        if not callable(function) or not isinstance(epoch, Epoch):
            raise ValueError("A callable and Epoch are required")
        self.function, self.epoch = function, epoch
        self.frame = text_label(frame, "frame")
        self.source_id = text_label(source_id, "source_id")

    def state_at(self, time):
        if not isinstance(time, Timestamp) or time.epoch != self.epoch:
            raise ValueError("Ephemeris epoch mismatch")
        return Reference(self.source_id, "ephemeris_pvt", time, self.frame,
                         ("m",)*3+("m/s",)*3, self.function(time))


def astronomical_time(time):
    """Convert explicit physical epoch via Astropy; never guess simulation UTC."""
    if not isinstance(time, Timestamp) or time.epoch.scale == TimeScale.SIMULATION:
        raise ValueError("A physical TAI, TT, TDB or GPS epoch is required")
    from astropy.time import Time, TimeDelta
    if time.epoch.scale == TimeScale.GPS:
        # Epoch.origin is an ISO calendar labelled GPS, not numeric GPS seconds.
        origin = Time(time.epoch.origin, scale="tai") + TimeDelta(19., format="sec")
    else:
        origin = Time(time.epoch.origin, scale=time.epoch.scale.value.lower())
    return origin+TimeDelta(time.seconds, format="sec")


class AstropySunProvider:
    """Geocentric GCRS Sun position, using Astropy's built-in solar ephemeris."""
    def position_at(self, time):
        from astropy.coordinates import get_sun
        position = get_sun(astronomical_time(time)).cartesian.xyz.to_value("m")
        return Reference("astropy_sun", "sun_position", time, "earth/gcrs", ("m",)*3, position)


class SGP4Provider:
    """Explicit TLE propagation in TEME; never relabelled as GCRS.

    No silent fallback and no zero vectors on propagation error. The wrapper
    must select a real TEME conversion before using a GCRS-only component.
    """
    def __init__(self, line1, line2):
        from sgp4.api import Satrec
        self.satellite = Satrec.twoline2rv(line1, line2)

    def state_at(self, time):
        utc = astronomical_time(time).utc
        error, position, velocity = self.satellite.sgp4(utc.jd1, utc.jd2)
        if error:
            raise ValueError(f"SGP4 propagation failed with code {error}")
        return Reference("sgp4", "ephemeris_pvt", time, "earth/teme", ("m",)*3+("m/s",)*3,
                         np.r_[position, velocity]*1000)


class UniformRotation:
    """Declared constant-rate rotating frame, not a precision Earth transform.

    axis and angular rate are expressed in the reference frame. Translation
    of origins is not included. Returned velocity is the rotating derivative.
    """
    def __init__(self, *, time, rotating_to_reference, angular_velocity_reference_rad_s):
        if not isinstance(time, Timestamp):
            raise ValueError("time must be a Timestamp")
        self.time = time
        self.initial = rotation_matrix(rotating_to_reference)
        self.omega = vector(angular_velocity_reference_rad_s, "angular_velocity_reference_rad_s", size=3)

    def state_jacobian(self, time):
        from utils.orbital_frames import skew
        dt = time.seconds_since(self.time)
        rotation = rot_matrix(quat_from_rotvec(self.omega*dt)) @ self.initial
        return np.block([[rotation.T, np.zeros((3, 3))],
                         [-rotation.T @ skew(self.omega), rotation.T]])

    def transform_state(self, state, time):
        return self.state_jacobian(time) @ vector(state, "state", size=6)
