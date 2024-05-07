import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy.integrate as sci
#import astropy.units as u
import astropy.cosmology as cosmo
import astropy.coordinates as coord

from satellite import Satellite
import sys
sys.path.append('..')
import global_constants.customUnits as customUnits
import global_constants.constants as constants
from graphs import myGraphs as graph
from mymath.myMath import math

#DEFINE OBJECTS FOR MY MODULES
const = constants.constantsClass()
cu = customUnits.customUnitsClass()
mymath = math()


## CONSTANTS
mass = const.SATELLITE_MASS
mradius = 6357000.0
mmass = 5.972e24
G = const.GRAVITATIONAL_CONSTANT

class Environment():
    def __init__(self) -> None:
        self.satellite = Satellite(mass, np.zeros(3),np.zeros(3),np.zeros(3))
        
    
    def set_initial_conditions(self, coord, velocity, acceleration):
        self.satellite.set_coordinates(coord)
        self.satellite.set_velocity(velocity)
        self.satellite.set_acceleration(acceleration)

    def motion_law_ode(self, state, t):
        #State vector
        position = np.asarray([state[0], state[1], state[2]])
        velocity = np.asarray([state[3], state[4], state[5]])

        #Compute total forces
        total_force = np.zeros(3)

        gravitational_force = -self.gravitational_acceleration(position) * self.satellite.mass

        total_force += gravitational_force

        #Acceleration based on Newton's second law
        acceleration = total_force/self.satellite.mass

        #Output state after derivation
        derivated_state = np.asarray([velocity, acceleration])
        derivated_state = derivated_state.reshape(-1)

        return derivated_state
    
    def gravitational_acceleration(self, position):
        #Distance between objects
        r = mymath.magnitude(position)
        #print("r units: ", r)
        
        #Acceleration 0 if it's "underground"
        accel = np.array([0.0, 0.0, 0.0])
        #print("accel units: ", accel)
        
        #Unit vector in the direction of the position vector
        unit_vector = mymath.to_unit_vector(position)
        #print("unit_vector units: ", unit_vector)

        if r > mradius:
            #Gravitational acceleration formula
            accel = ((G * mmass)/(r**2))*unit_vector
            #print("accel units: ", accel)
            
        return accel


system = Environment()

system.satellite.set_coordinates(np.asarray([mradius+300000, 0, 0 ]))
r0 = mymath.magnitude(system.satellite.coord)

system.satellite.set_velocity(np.asarray([100.0, np.sqrt(G*mmass/r0)*1.226, 0.0]))
system.satellite.set_acceleration(np.asarray([0.0, 0.0, 0.0]))

initial_state = system.satellite.get_state()
initial_state = np.delete(initial_state, [6, 7, 8])
print(initial_state)
t = np.linspace(0,500000, 10000)

stateout = sci.odeint(system.motion_law_ode, initial_state, t)
xout = stateout[:,0]
yout = stateout[:,1]
plt.figure()
plt.plot(xout,yout,'r-',label='Orbit')
plt.plot(xout[0],yout[0],'g*')
theta = np.linspace(0,2*np.pi,100)
xplanet = mradius*np.sin(theta)
yplanet = mradius*np.cos(theta)
plt.plot(xplanet,yplanet,'b-',label='Planet')
plt.grid()
plt.legend()

plt.show()
