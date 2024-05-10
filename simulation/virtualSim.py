import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import scipy.integrate as sci
import astropy.units as u
import astropy.cosmology as cosmo
import astropy.coordinates as coord
import pandas as pd

import simulation.satellite as satellite
import simulation.moon as moon
import sys
sys.path.append('..')
import global_constants.customUnits as customUnits
import global_constants.constants as constants
from graphs import myGraphs as graph
from mymath.myMath import math
from myfile_management.myFileManagement import my_to_csv, get_last_filename

#DEFINE OBJECTS FOR MY MODULES
const = constants.constantsClass()
cu = customUnits.customUnitsClass()
mymath = math()

## CONSTANTS
mass = const.SATELLITE_MASS
mradius = 6357000.0
mmass = 5.972e24
G = const.GRAVITATIONAL_CONSTANT
# initial_acceleration = np.array([550, 5000, 0], dtype=np.longdouble)
class Environment():
    def __init__(self) -> None:
        self.satellite = satellite.SatelliteClass(mass, np.zeros(3, dtype = np.longdouble),np.zeros(3, dtype=np.longdouble),np.zeros(3, dtype=np.longdouble))
        
    
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


system = Environment()

system.satellite.set_coordinates(np.array([mradius+300000, 0, 0 ], dtype=np.longdouble))
r0 = mymath.magnitude(system.satellite.coord)

system.satellite.set_velocity(np.array([0, np.sqrt(G*mmass/r0)*1.15, 0.0], dtype=np.longdouble))
system.satellite.set_acceleration(np.array([0.0, 0.0, 0.0], dtype=np.longdouble))

initial_state = system.satellite.get_state()
initial_state = np.delete(initial_state, [6, 7, 8])
print(initial_state)
t = np.linspace(0,10000, 10000, dtype=np.longdouble)

stateout = sci.odeint(system.motion_law_ode, initial_state, t)
xout = stateout[:,0]
yout = stateout[:,1]
zout = stateout[:,2]
vxout = stateout[:,3]
vyout = stateout[:,4]
vout = np.sqrt(vxout**2+vyout**2)
axout = np.gradient(t, vxout)
ayout = np.gradient(t, vyout)
aout = np.sqrt(axout**2+ayout**2)
r = np.sqrt(xout**2 + yout**2)

fig, ax = plt.subplots(2,2)
fig.canvas.manager.set_window_title("My graphs")
ax[0][0].plot(xout, yout, 'r', label = 'Orbit')
ax[0][0].plot(xout[0],yout[0],'g*', label = 'Start')
ax[0][0].plot(xout[len(xout)-1], yout[len(yout)-1], 'b*', label = 'End')
theta = np.linspace(0,2*np.pi,100, dtype=np.longdouble)
xplanet = mradius*np.sin(theta)
yplanet = mradius*np.cos(theta)
ax[0][0].plot(xplanet,yplanet,'b-',label='Planet')
ax[0][0].set_xlabel('x - m')
ax[0][0].set_ylabel('y - m')
ax[0][0].grid()
ax[0][0].legend(loc = 'upper left')

# fig, ax = plt.subplots(1,1)
# fig.canvas.manager.set_window_title("Velocity")
ax[0][1].plot(t, vxout, 'b', label = 'Velocity x')
ax[0][1].plot(t, vyout,'g', label = 'Velocity y')
ax[0][1].plot(t, vout,'r',label='Velocity')
ax[0][1].set_xlabel('t - s')
ax[0][1].set_ylabel('velocity - m/s')
ax[0][1].grid()
ax[0][1].legend(loc = 'upper left')

# fig, ax = plt.subplots(1,1)
# fig.canvas.manager.set_window_title("Acceleration")
ax[1][0].plot(t, axout, 'b', label = 'Acceleration x')
ax[1][0].plot(t, ayout,'g', label = 'Acceleration y')
ax[1][0].plot(t, aout,'r',label='Acceleration')
ax[1][0].set_xlabel('t - s')
ax[1][0].set_ylabel('acceleration - m/s2')
ax[1][0].grid()
ax[1][0].legend(loc = 'upper left')

# fig, ax = plt.subplots(1,1)
# fig.canvas.manager.set_window_title("Altitude")
ax[1][1].plot(t, r, 'r', label = 'Altitude x')
ax[1][1].set_xlabel('t - s')
ax[1][1].set_ylabel('Altitude - m')
ax[1][1].grid()
ax[1][1].legend(loc = 'upper left')



plt.show()

# plt.figure()
# plt.plot(xout,yout,'r-',label='Orbit')
# plt.plot(xout[0],yout[0],'g*')
# theta = np.linspace(0,2*np.pi,100)
# xplanet = mradius*np.sin(theta)
# yplanet = mradius*np.cos(theta)
# plt.plot(xplanet,yplanet,'b-',label='Planet')
# plt.grid()
# plt.legend()

# plt.show()
