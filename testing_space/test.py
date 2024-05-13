import numpy as np
import scipy.integrate as sci
import sys
sys.path.append(r"C:\Users\Fratilescu Gabriel\Documents\OCS\MySoftware\v1.0-new\ONCS2024")
from mymath.myMath import math
mymath = math()

class Environment():
    def __init__(self, satellite_mass, moon_mass, gravitational_constant, moon_radius) -> None:
        self.satellite_mass = satellite_mass
        self.moon_mass = moon_mass
        self.G = gravitational_constant
        self.moon_radius = moon_radius

    def motion_law_ode(self, state, t):
        position = state[:3]
        velocity = state[3:]

        total_force = self.gravitational_force(position)

        acceleration = total_force / self.satellite_mass

        return np.concatenate((velocity, acceleration))

    def gravitational_force(self, position):
        r = np.linalg.norm(position)
        unit_vector = position / r
        if r > self.moon_radius:
            return -(self.G * self.moon_mass / r**2) * unit_vector
        else:
            return np.zeros_like(position)

    def calculate_orbital_period_from_simulation(self, stateout, t, atol=50, max_iterations=10, ignored_points=500000):
        initial_position = stateout[0, :3]  # Initial position of the satellite
        crossing_indices = np.where(np.diff(np.sign(mymath.magnitude(stateout[:, :3].T) - mymath.magnitude(initial_position))))[0]

        orbital_periods = []  # Store orbital periods

        # Iterate over crossing indices
        for i in range(0, len(crossing_indices) - 1, 2):
            start_index = crossing_indices[i]
            end_index = crossing_indices[i + 1]
            
            if start_index > ignored_points:
                # Calculate orbital period based on time difference between crossings
                orbital_period = t[end_index] - t[start_index]
                orbital_periods.append(orbital_period)

            # Check if maximum number of iterations reached
            if len(orbital_periods) >= max_iterations:
                break

        # Return mean orbital period or NaN if no complete orbits found
        if orbital_periods:
            return np.mean(orbital_periods)
        else:
            return np.nan

# Example usage
satellite_mass = 1000  # Example satellite mass
moon_mass = 7.34767309e22  # Moon mass in kg
G = 6.67430e-11  # Gravitational constant
moon_radius = 1737.1e3  # Moon radius in meters

system = Environment(satellite_mass, moon_mass, G, moon_radius)

# Set initial conditions
initial_state = np.array([moon_radius + 300000, 0, 0, 0, np.sqrt(G * moon_mass / (moon_radius + 300000)), 0])
t = np.linspace(0, 48734, 100000)
stateout = sci.odeint(system.motion_law_ode, initial_state, t)
print(stateout)

# Calculate orbital period from simulation
orbital_period = system.calculate_orbital_period_from_simulation(stateout, t, atol=50, max_iterations=100, ignored_points=0)
print("Orbital period from simulation:", orbital_period)
