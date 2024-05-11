import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sci
import time
import astropy as astro
import astropy.units as u
import astropy.cosmology as cosmo

import sys
sys.path.append('..')
import global_constants.constants as constants

const = constants.constantsClass()
class MoonSystem:
    def __init__(self, coord, initial_moon_orbital_inclination, *args, **kwargs):
        self.moon_mass = const.MOON_MASS
        self.base_coord = coord
        self.moon_sidereal_velocity = const.MOON_SIDEREAL_VELOCITY
        self.moon_orbital_inclination = initial_moon_orbital_inclination
        