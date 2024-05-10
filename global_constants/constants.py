from astropy import units as u
import numpy as np

class constantsClass():
        def __init__(self) -> None:
                
                # SATELLITE CONSTANTS
                self.SATELLITE_MASS = 4e6 # u.kg

                # MOON CONSTANTSa
                self.MOON_RADIUS = 1737.4 * 1000 # u.m
                self.MOON_MASS = 7.34767309e22  # u.kg
                self.MOON_SIDEREAL_VELOCITY = (2.0*np.pi)/(655.720*60*60) #rad/s

                # PHYSICAL CONSTANTS
                self.GRAVITATIONAL_CONSTANT = 6.6743e-11 # m3 / kg s2


c = constantsClass()
print(c.MOON_SIDEREAL_VELOCITY)