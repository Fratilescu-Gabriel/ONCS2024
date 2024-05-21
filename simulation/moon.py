from global_constants.constants import *
import numpy as np
import mymath.myMath as math

mymath = math.math()

class PlanetsSystem:
    def __init__(self, in_coll_coord, in_EPA, *args, **kwargs):
        self.coll_coord = in_coll_coord
        self.in_EPA = in_EPA
        self.moon_rotation_axis_angle = MOON_ROTATION_AXIS_ANGLE
        self.moon_angle_with_ecliptic = MOON_ANGLE_WITH_ECLIPTIC
        self.moon_rotation_axis = mymath.to_unit_vector(np.array([0.0, np.sin(-self.moon_rotation_axis_angle), np.cos(-self.moon_rotation_axis_angle)]))
        self.earth_angle_list = None

    def coll_trajectory(self, t):
        # Calculate the angle rotated by the moon during the simulation
        angle_rotated = np.zeros(len(t))
        angle_rotated[0] = 0
        for i in range(len(angle_rotated)):
            if i > 0:
                angle_rotated[i] = MOON_AV * t[i] - MOON_AV * t[i-1]
        
        x = np.empty(len(t))
        y = np.empty(len(t))
        z = np.empty(len(t))
        
        x[0] = self.coll_coord[0]
        y[0] = self.coll_coord[1]
        z[0] = self.coll_coord[2]
        
        for index, time in enumerate(t):
            if index > 0:
                vector = np.array([x[index-1],y[index-1],z[index-1]])
                rotated_position = mymath.rotate_quat(mymath.angle_axis_quat(angle_rotated[index], self.moon_rotation_axis), vector)
                
                x[index] = rotated_position[0]
                y[index] = rotated_position[1]
                z[index] = rotated_position[2]

        return (x, y, z)
    
    def moon_surface(self, resolution = 100):
        '''Returns grid value ready for plot of the moon surface'''
        theta, phi = np.mgrid[0:2*np.pi:75j, 0:np.pi:75j]
        
        x = MOON_RADIUS * np.sin(phi) * np.cos(theta)
        y = MOON_RADIUS * np.sin(phi) * np.sin(theta)
        z = MOON_RADIUS * np.cos(phi)
        
        return (x, y, z)
    
    def coll_trajectory_curve(self, gamma, phi_wl):
        theta = np.mgrid[0:2*np.pi:100j]

        rm = np.sqrt(MOON_RADIUS**2 - gamma**2)
        
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
                angle_rotated[i] = angle_rotated[i-1] + MOON_AV * (t[i] - t[i-1]) # rad
        
        self.earth_angle_list = angle_rotated
        
        x = np.empty(len(t))
        y = np.empty(len(t))
        z = np.empty(len(t))
        
        for index, time in enumerate(t):
            x[index] = -DISTANCE_MOON_EARTH* np.sin(self.in_EPA + angle_rotated[index])
            y[index] = DISTANCE_MOON_EARTH * np.cos(self.moon_angle_with_ecliptic) * np.cos(self.in_EPA + angle_rotated[index])
            z[index] = DISTANCE_MOON_EARTH * np.sin(self.moon_angle_with_ecliptic) * np.cos(self.in_EPA + angle_rotated[index])
            
        return (x, y, z)
    
    def sun_trajectory(self,t):
        angle_rotated = np.zeros(len(t))
        angle_rotated[0] = 0
        for i in range(len(angle_rotated)):
            if i > 0:
                angle_rotated[i] = MOON_AV * t[i]# - MOON_AV * t[i-1]
        
        x = np.empty(len(t))
        y = np.empty(len(t))
        z = np.empty(len(t))
        
        for index, time in enumerate(t):
            x[index] = -DISTANCE_MOON_EARTH *np.cos(self.moon_angle_with_ecliptic) * np.sin(self.in_EPA + angle_rotated[index])
            y[index] = -DISTANCE_MOON_EARTH * np.cos(self.moon_angle_with_ecliptic) * np.cos(self.in_EPA + angle_rotated[index]) - DISTANCE_EARTH_SUN
            z[index] = -DISTANCE_MOON_EARTH * np.sin(self.moon_angle_with_ecliptic)
            
        return (x, y, z)