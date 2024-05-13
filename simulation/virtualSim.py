import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy.integrate as sci
import pandas as pd
from timeit import default_timer as timer


import sys
sys.path.append(r"C:\Users\Fratilescu Gabriel\Documents\OCS\MySoftware\v1.0-new\ONCS2024")
import simulation.satellite as satellite
import simulation.moon as moon
import global_constants.constants as constants
from graphs import myGraphs as graph
from mymath.myMath import math
from myfile_management.myFileManagement import my_to_csv, get_last_filename

start_time = timer()
#DEFINE OBJECTS FOR MY MODULES
const = constants.constantsClass()
mymath = math()

## CONSTANTS
mass = const.SATELLITE_MASS
mradius = const.MOON_RADIUS
mmass = const.MOON_MASS
G = const.GRAVITATIONAL_CONSTANT
DC_TO_RF = 0.78
RF_TO_DC = 0.72
RF_COLLECTION = 0.93
# initial_acceleration = np.array([550, 5000, 0], dtype=np.longdouble)
class Environment():
    def __init__(self, collector_coordinates, earth_initial_angle, solar_pannel_energy_gain_per_second, DC_to_RF_conversion_rate_per_second) -> None:
        self.satellite = satellite.SatelliteClass(mass, np.zeros(3, dtype = np.longdouble),np.zeros(3, dtype=np.longdouble),np.zeros(3, dtype=np.longdouble))
        self.moon_system = moon.MoonSystem(collector_coordinates, earth_initial_angle)
        self.satellite_trajectory = None
        self.satellite_stored_energy = 0
        self.collector_received_energy = 0
        self.solar_pannel_energy_gain_per_second = solar_pannel_energy_gain_per_second
        self.DC_to_RF_conversion_rate_per_second = DC_to_RF_conversion_rate_per_second
        
        self.plot_collector_received_energy = None
        self.plot_sattelite_stored_energy = None
        self.satellite_in_light = None

    def calculate_trajectories(self, t):
        self.earth_trajectory = np.transpose(self.moon_system.earth_trajectory(t))
        self.sun_trajectory = np.transpose(self.moon_system.sun_trajectory(t))
        self.stationary_surface_point_trajectory = np.transpose(self.moon_system.stationary_point_trajectory(t))
        self.moon_surface = self.moon_system.moon_surface()
        
    
    def set_initial_conditions(self, coord, velocity, acceleration):
        self.satellite.set_coordinates(coord)
        self.satellite.set_velocity(velocity)
        self.satellite.set_acceleration(acceleration)

    def motion_law_ode(self, state, t):
        #State vector
        # global initial_acceleration
        position = np.array([state[0], state[1], state[2]], dtype=np.longdouble)
        velocity = np.array([state[3], state[4], state[5]], dtype=np.longdouble)

        #Compute total forces
        total_force = np.zeros(3)

        gravitational_force = -self.gravitational_acceleration(position) * self.satellite.mass

        total_force += gravitational_force

        #Acceleration based on Newton's second law
        acceleration = total_force/self.satellite.mass

        #Output state after derivation
        derivated_state = np.array([velocity, acceleration])
        derivated_state = derivated_state.reshape(-1)

        return derivated_state
    
    def gravitational_acceleration(self, position):
        #Distance between objects
        r = mymath.magnitude(position)
        
        #Acceleration 0 if it's "underground"
        accel = np.array([0.0, 0.0, 0.0], dtype=np.longdouble)
        
        #Unit vector in the direction of the position vector
        unit_vector = mymath.to_unit_vector(position)

        if r > mradius:
            #Gravitational acceleration formula
            accel = ((G * mmass)/(r**2))*unit_vector
            
        return accel
    
    def calculate_apoapsis(self, stateout):
        positions = stateout[:, :3]
        positions = np.transpose(positions)
        return np.max(mymath.magnitude(positions))

    def calculate_periapsis(self, stateout):
        positions = stateout[:, :3]
        positions = np.transpose(positions)
        return np.min(mymath.magnitude(positions))
    
    def is_satellite_in_light(self, time_index):
        cone_tip_parameter = -self.moon_system.moon_radius/(self.moon_system.sun_radius - self.moon_system.moon_radius)
        angle = np.sum(self.moon_system.earth_angles[:time_index])
        
        sun_distance_x = -(self.moon_system.moon_radius * np.cos(self.moon_system.moon_angle_with_ecliptic) * np.sin(angle))
        sun_distance_y = -self.moon_system.moon_radius * np.cos(self.moon_system.moon_angle_with_ecliptic) * np.cos(angle) + self.moon_system.earth_sun_distance
        sun_distance_z =  -self.moon_system.moon_radius * np.sin(self.moon_system.moon_angle_with_ecliptic)
        
        cone_tip = np.array([sun_distance_x, sun_distance_y, sun_distance_z])*cone_tip_parameter
        
        direction = self.sun_trajectory[time_index] - cone_tip
        
        height = mymath.magnitude(direction)
        
        direction = mymath.to_unit_vector(direction)
        
        cone_dist = np.dot(self.satellite_trajectory[time_index] - cone_tip, direction)
        
        cone_radius = (cone_dist / height) * self.moon_system.sun_radius

        orth_distance = mymath.magnitude((self.satellite_trajectory[time_index] - cone_tip) - cone_dist * direction)

        is_point_inside_cone = (orth_distance < np.abs(cone_radius))
        
        return (not is_point_inside_cone or (is_point_inside_cone and self.satellite_trajectory[time_index][1] < 0))
    
    def get_collector_satellite_angle(self, time_index):
        normal_to_moon_surface = self.stationary_surface_point_trajectory[time_index]/mymath.magnitude(self.stationary_surface_point_trajectory[time_index])
        
        collector_satellite_vector = (self.satellite_trajectory[time_index] - self.stationary_surface_point_trajectory[time_index])/mymath.magnitude(self.satellite_trajectory[time_index] - self.stationary_surface_point_trajectory[time_index])
        
        angle = np.arccos(np.dot(normal_to_moon_surface, collector_satellite_vector))
        
        return (angle/np.pi)*180
    
    def performance(self, t):
        self.plot_collector_received_energy = np.zeros(len(t))
        self.plot_sattelite_stored_energy = np.zeros(len(t))
        self.satellite_in_light = np.zeros(len(t))
        
        for index, time in enumerate(t):
            in_light = self.is_satellite_in_light(index)
            if in_light and index > 0 :
                self.satellite_stored_energy = self.satellite_stored_energy + (t[index] - t[index-1]) * self.solar_pannel_energy_gain_per_second
                self.plot_sattelite_stored_energy[index] = self.satellite_stored_energy
            
            if self.get_collector_satellite_angle(index) < 10 and index > 0 and index < len(t)-1:
                energy = (t[index] - t[index-1])*self.DC_to_RF_conversion_rate_per_second
                if energy <= self.satellite_stored_energy:
                    energy_transmited = energy*DC_TO_RF
                    # print(energy_transmited)
                else:
                    energy_transmited = self.satellite_stored_energy*DC_TO_RF
                    
                
                    
                self.satellite_stored_energy = self.satellite_stored_energy - energy_transmited
                self.collector_received_energy = self.collector_received_energy + energy_transmited * RF_TO_DC * RF_COLLECTION
            
            self.satellite_in_light[index] = 1 if in_light else -1
            self.plot_sattelite_stored_energy[index] = self.satellite_stored_energy
            self.plot_collector_received_energy[index] = self.collector_received_energy


#DEFAULT TIME VALUES
DAYx28 = 28*24*60*60 #s
DAYx27_32 = 27.32*24*60*60


#INITIAL CONDITIONS

TIME = DAYx27_32
DATA_POINTS = 50000
CHUNCKING_DATA_POINTS = 50000

collector_initial_position = np.array([mradius, 0, 0])
earth_initial_angle = 0
solar_pannel_energy_gain_per_second = 25
DC_to_RF_conversion_rate_per_second = 1000

satellite_initial_position = np.array([mradius+400000, 0, 0 ], dtype=np.longdouble)

r0 = mymath.magnitude(satellite_initial_position)
satellite_initial_velocity = np.array([0, np.sqrt(G*mmass/r0), 100], dtype=np.longdouble)
satellite_initial_acceleration = np.array([0.0, 0.0, 0.0], dtype=np.longdouble)



#MECHANISM

system = Environment(collector_initial_position, earth_initial_angle, solar_pannel_energy_gain_per_second, DC_to_RF_conversion_rate_per_second)
system.satellite.set_coordinates(satellite_initial_position)
system.satellite.set_velocity(satellite_initial_velocity)
system.satellite.set_acceleration(satellite_initial_acceleration)

initial_state = system.satellite.get_state()
initial_state = np.delete(initial_state, [6, 7, 8])

t = np.linspace(0, TIME, DATA_POINTS)

stateout = sci.odeint(system.motion_law_ode, initial_state, t)

system.calculate_trajectories(t)

xout = stateout[:,0]
yout = stateout[:,1]
zout = stateout[:,2]

system.satellite_trajectory = np.transpose(np.array([xout, yout, zout]))

vxout = stateout[:,3]
vyout = stateout[:,4]
vzout = stateout[:,5]
vout = np.sqrt(np.square(vxout)+np.square(vyout)+np.square(vzout))
axout = np.gradient(vxout, t)
ayout = np.gradient(vyout, t)
azout = np.gradient(vzout, t)
aout = np.sqrt(axout**2+ayout**2+azout**2)
r = np.sqrt(xout**2 + yout**2+zout**2) - mradius

orbital_period = system.satellite.calculate_orbital_period(system.calculate_apoapsis(stateout), system.calculate_periapsis(stateout))
print("Orbital period with kepler", orbital_period)

system.performance(t)




#3D PLOTTER

plt3d = graph.Plotter3D()

plt3d.new_figure(xlabel= 'X', ylabel='Y', zlabel='Z')


x, y, z = system.moon_surface
plt3d.add_surface(x, y, z, alpha=1)

bx = system.stationary_surface_point_trajectory[:, 0]
by = system.stationary_surface_point_trajectory[:, 1]
bz = system.stationary_surface_point_trajectory[:, 2]

plt3d.add_curve(bx, by, bz)
plt3d.add_curve(xout, yout, zout, color = 'g')

in_shadow = 0
in_light = 0
for test_index in range(len(t)):
    test = system.is_satellite_in_light(test_index)
    if test:
        in_light = in_light + 1
    else:
        in_shadow = in_shadow + 1

print("Percentage in shadow")
print(100* in_shadow/(in_shadow+in_light))

angle = np.zeros(len(t))
for test_index in range(len(t)):
    test = system.get_collector_satellite_angle(test_index)
    angle[test_index] = test   

plt3d.add_scatter_plot(system.satellite_trajectory[-1][0], system.satellite_trajectory[-1][1], system.satellite_trajectory[-1][2], color = 'b', marker= '*')

print("End to beginning distance:",mymath.magnitude(np.array([xout[0], yout[0], zout[0]])-np.array([xout[-1], yout[-1], zout[-1]])))

plt3d.set_axes_equal()

#2D PLOTTER


fig, ax = plt.subplots(2,2)
fig.canvas.manager.set_window_title("My graphs")
ax[0][0].plot(xout, yout, 'r', label = 'Orbit')
ax[0][0].plot(xout[0],yout[0],'g*', label = 'Start')
ax[0][0].plot(xout[len(xout)-1], yout[len(yout)-1], 'b*', label = 'End')
ax[0][0].plot(bx, by, 'g', label='Collector')
theta = np.linspace(0,2*np.pi,100000, dtype=np.longdouble)
xplanet = mradius*np.sin(theta)
yplanet = mradius*np.cos(theta)
ax[0][0].plot(xplanet,yplanet,'b-',label='Planet')
ax[0][0].set_xlabel('x - m')
ax[0][0].set_ylabel('y - m')
ax[0][0].grid(linestyle='--', color='gray', alpha=0.7)
ax[0][0].legend(loc = 'upper left')

# fig, ax = plt.subplots(1,1)
# fig.canvas.manager.set_window_title("Velocity")
ax[0][1].plot(t, vxout, 'b', label = 'Velocity x')
ax[0][1].plot(t, vyout,'g', label = 'Velocity y')
ax[0][1].plot(t, vzout, 'y', label = "Velocity z")
ax[0][1].plot(t, vout,'r',label='Velocity')
ax[0][1].set_xlabel('t - s')
ax[0][1].set_ylabel('velocity - m/s')
ax[0][1].grid(linestyle='--', color='gray', alpha=0.7)
ax[0][1].legend(loc = 'upper left')

# fig, ax = plt.subplots(1,1)
# fig.canvas.manager.set_window_title("Acceleration")
ax[1][0].plot(t, axout, 'b', label = 'Acceleration x')
ax[1][0].plot(t, ayout,'g', label = 'Acceleration y')
ax[1][0].plot(t, azout, 'y', label = 'Acceleration z')
ax[1][0].plot(t, aout,'r',label='Acceleration')
ax[1][0].set_xlabel('t - s')
ax[1][0].set_ylabel('acceleration - m/s2')
ax[1][0].grid(linestyle='--', color='gray', alpha=0.7)
ax[1][0].legend(loc = 'upper left')

# fig, ax = plt.subplots(1,1)
# fig.canvas.manager.set_window_title("Altitude")
ax[1][1].plot(t, r, 'r', label = 'Altitude x')
ax[1][1].set_xlabel('t - s')
ax[1][1].set_ylabel('Altitude - m')
ax[1][1].grid(linestyle='--', color='gray', alpha=0.7)
ax[1][1].legend(loc = 'upper left')

f, a = plt.subplots(1,3)
# angle = (angle/np.pi)*180
a[0].plot(t, angle, 'r', label = 'Angle')
a[0].set_xlabel('t - s')
a[0].set_ylabel('Angle - degrees')
a[0].grid(linestyle='--', color='gray', alpha=0.7)
a[0].legend(loc = 'upper left')

a[1].plot(t, system.plot_sattelite_stored_energy, 'r', label = 'Satellite')
a[1].plot(t, system.plot_collector_received_energy, 'b', label = 'Collector')
a[1].set_xlabel('t - s')
a[1].set_ylabel('Energy')
a[1].grid(linestyle='--', color='gray', alpha=0.7)
a[1].legend(loc = 'upper left')

a[2].plot(t, system.satellite_in_light, 'r', label = 'In light')
a[2].set_xlabel('t - s')
a[2].set_ylabel('light')
a[2].grid(linestyle='--', color='gray', alpha=0.7)
a[2].legend(loc = 'upper left')

end_time = timer()

print("TIME TO RUN: ", end_time-start_time)

plt.show()