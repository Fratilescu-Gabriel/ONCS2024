from astropy import units as u
from . import customUnits

class constantsClass():
        def __init__(self) -> None:
                
                cu = customUnits.customUnitsClass()
                # SATELLITE CONSTANTS
                self.SATELLITE_MASS = 4e6 # u.kg

                # MOON CONSTANTSa
                self.MOON_RADIUS = 1737.4 * 1000 # u.m
                self.MOON_MASS = 7.34767309e22  # u.kg

                # PHYSICAL CONSTANTS
                self.GRAVITATIONAL_CONSTANT = 6.6743e-11 # m3 / kg s2
