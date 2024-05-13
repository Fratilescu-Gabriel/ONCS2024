from astropy import units as u
import numpy as np

class constantsClass():
        def __init__(self) -> None:
                
                # SATELLITE CONSTANTS
                self.SATELLITE_MASS = 4e6 # u.kg

                # MOON CONSTANTS
                self.MOON_RADIUS = 1737.4 * 1000 # u.m
                self.MOON_MASS = 7.34767309e22  # u.kg
                self.MOON_SIDEREAL_VELOCITY = 2.66169897516368202422328245057129981309e-6 #rad/s
                self.MOON_ROTATION_AXIS_ANGLE = 6.13
                self.MOON_ANGLE_WITH_ECLIPTIC = (5.15 * np.pi) / 180 #rad
                self.MOON_ANGULAR_VELOCITY = 2*np.pi/(27.3*24*60*60) #rad/s
                
                self.SUN_RADIUS = 696340e3 #m

                # PHYSICAL CONSTANTS
                self.GRAVITATIONAL_CONSTANT = 6.6743e-11 # m3 / kg s2
                self.DISTANCE_EARTH_SUN = 151.15e9 #m
                self.DISTANCE_MOON_EARTH = 384400e3 #m
