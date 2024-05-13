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
import mymath.myMath as math

const = constants.constantsClass()
mymath = math.math()

class MoonSystem:
    def __init__(self, collector_base_coord, earth_position_angle, *args, **kwargs):
        self.moon_mass = const.MOON_MASS
        self.base_coord = collector_base_coord
        self.moon_earth_distance = const.DISTANCE_MOON_EARTH
        self.earth_sun_distance = const.DISTANCE_EARTH_SUN
        self.initial_earth_position_angle = earth_position_angle
        self.moon_angular_velocity = const.MOON_ANGULAR_VELOCITY
        self.own_rotation_velocity = const.MOON_SIDEREAL_VELOCITY
        self.moon_radius = const.MOON_RADIUS  # Radius of the moon
        self.own_rotation_axis_angle = const.MOON_ROTATION_AXIS_ANGLE
        self.moon_angle_with_ecliptic = const.MOON_ANGLE_WITH_ECLIPTIC
        self.own_rotation_axis = mymath.to_unit_vector(np.array([0.0, np.sin(self.own_rotation_axis_angle), np.cos(self.own_rotation_axis_angle)]))
        self.earth_angles = None
        self.sun_radius = const.SUN_RADIUS

    def stationary_point_trajectory(self, t):
        # Calculate the angle rotated by the moon during the simulation
        angle_rotated = np.zeros(len(t))
        angle_rotated[0] = 0
        for i in range(len(angle_rotated)):
            if i > 0:
                angle_rotated[i] = self.own_rotation_velocity * t[i] - self.own_rotation_velocity * t[i-1]
        
        x = np.array([self.base_coord[0]])
        y = np.array([self.base_coord[1]])
        z = np.array([self.base_coord[2]])
        
        for index, time in enumerate(t):
            vector = np.array([x[index],y[index],z[index]])
            rotated_position = mymath.rotate_quat(mymath.angle_axis_quat(angle_rotated[index], self.own_rotation_axis), vector)
            
            x = np.append(x, rotated_position[0])
            y = np.append(y, rotated_position[1])
            z = np.append(z, rotated_position[2])

        return (x, y, z)
    
    def moon_surface(self, resolution = 100):
        '''Returns grid value ready for plot of the moon surface'''
        theta, phi = np.mgrid[0:2*np.pi:100j, 0:np.pi:100j]
        
        x = self.moon_radius * np.sin(phi) * np.cos(theta)
        y = self.moon_radius * np.sin(phi) * np.sin(theta)
        z = self.moon_radius * np.cos(phi)
        
        return (x, y, z)
    
    def stationary_point_trajectory_curve(self, gamma, phi_wl):
        theta = np.mgrid[0:2*np.pi:100j]

        rm = np.sqrt(self.moon_radius**2 - gamma**2)
        
        x = rm*np.cos(theta)
        y = rm*np.cos(phi_wl)*np.sin(theta)+gamma*np.sin(phi_wl)
        z = gamma * np.cos(phi_wl)-rm*np.sin(phi_wl)*np.sin(theta)
        
        return (x,y,z)
    
    def earth_trajectory(self, t):
        """
        This function calculates the trajectory of Earth at a given time `t`.
        
        :param t: The parameter `t` represents the time at which you want to calculate the position of
        Earth along its trajectory. This function likely calculates the position of Earth at a given
        time `t` along its orbit around the Sun
        """
        angle_rotated = np.zeros(len(t))
        angle_rotated[0] = 0
        for i in range(len(angle_rotated)):
            if i > 0:
                angle_rotated[i] = self.moon_angular_velocity * t[i]# - self.moon_angular_velocity * t[i-1]
        
        self.earth_angles = angle_rotated
        
        x = np.empty(0)
        y = np.empty(0)
        z = np.empty(0)
        
        for index, time in enumerate(t):
            x = np.append(x, [-self.moon_earth_distance * np.sin(self.initial_earth_position_angle + angle_rotated[index])])
            y = np.append(y, [self.moon_earth_distance * np.cos(self.moon_angle_with_ecliptic) * np.cos(self.initial_earth_position_angle + angle_rotated[index])])
            z = np.append(z, [self.moon_earth_distance * np.sin(self.moon_angle_with_ecliptic) * np.cos(self.initial_earth_position_angle + angle_rotated[index])])
            
        return (x, y, z)
    
    def sun_trajectory(self,t):
        angle_rotated = np.zeros(len(t))
        angle_rotated[0] = 0
        for i in range(len(angle_rotated)):
            if i > 0:
                angle_rotated[i] = self.moon_angular_velocity * t[i]# - self.moon_angular_velocity * t[i-1]
        
        x = np.empty(0)
        y = np.empty(0)
        z = np.empty(0)
        
        for index, time in enumerate(t):
            x = np.append(x, [-self.moon_earth_distance *np.cos(self.moon_angle_with_ecliptic) * np.sin(self.initial_earth_position_angle + angle_rotated[index])])
            y = np.append(y, [-self.moon_earth_distance * np.cos(self.moon_angle_with_ecliptic) * np.cos(self.initial_earth_position_angle + angle_rotated[index]) -self.earth_sun_distance])
            z = np.append(z, [-self.moon_earth_distance * np.sin(self.moon_angle_with_ecliptic)])
            
        return (x, y, z)