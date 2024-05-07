import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sci
import time
import astropy as astro
import astropy.units as u
import astropy.cosmology as cosmo
import astropy.coordinates as coord
import sys
sys.path.append('..')
import global_constants.customUnits as customUnits
import global_constants.constants as constants
from graphs import myGraphs as graph
from mymath.myMath import math

#IMPORT CONSTANTS AND CUSTOM UNITS
const = constants.constantsClass()
cu = customUnits.customUnitsClass()
mymath = math()
## CONSTANTS
mass = const.SATELLITE_MASS
mradius = const.MOON_RADIUS
mmass = const.MOON_MASS
G = const.GRAVITATIONAL_CONSTANT

r = 0.0
accel = np.array([0.0,0.0,0.0])*cu.mps2

def gravitational_acceleration( position):
        global r, accel
        r = mymath.magnitude(position)
        print("r units: ", r.si)

        accel = np.array([0.0, 0.0, 0.0])*cu.mps2
        print("accel units: ", accel.si)
        
        unit_vector = mymath.scalar_unit_vector(position)
        print("unit_vector units: ", unit_vector.si)

        if r > mradius:
            accel = ((G * mmass)/(r**2))*unit_vector
            print("accel units: ", accel.si)
            
        
        return accel
    
gr = gravitational_acceleration(np.asarray([mradius/u.m + 300000, 0, 0])*u.m)

print(gr)