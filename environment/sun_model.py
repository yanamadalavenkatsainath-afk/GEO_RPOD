import numpy as np
from astropy.time import Time
from astropy.coordinates import get_sun
import astropy.units as u
from astropy.coordinates.builtin_frames.utils import get_jd12
from astropy.utils import iers

# Disable auto-download to prevent stale cache issues
iers.conf.auto_download = False
iers.conf.auto_max_age = None
class SunModel:
    """
    Sun position model using astropy JPL ephemeris.
    Returns unit sun vector in ECI (GCRS) frame.
    """

    def __init__(self, epoch_year=2025.0):
        self.epoch = Time(epoch_year, format='decimalyear')

    def get_sun_vector(self, t_seconds=0.0):
        t      = self.epoch + t_seconds * u.second
        sun    = get_sun(t)
        r      = sun.gcrs.cartesian.xyz.value
        return r / np.linalg.norm(r)