from global_constants.constants import *



class SatelliteClass():
    def __init__(self, mass, coord:np.ndarray, velocity:np.ndarray, acceleration:np.ndarray):
        self.mass = mass
        self.coord = coord
        self.velocity = velocity
        self.acceleration = acceleration
        self.state = np.array([self.coord, self.velocity, self.acceleration], dtype=np.longdouble)
        self.state = self.state.reshape(-1)

    def get_state(self):
        return self.state
    
    def update_state(self):
        for index in range(0,3):
            if self.coord[index] != self.state[index]:
                self.state[index] = self.coord[index]
        
        for index in range(0,3):
            if self.velocity[index] != self.state[index + 3]:
                self.state[index + 3] = self.velocity[index]
        
        for index in range(0,3):
            if self.acceleration[index] != self.state[index+6]:
                self.state[index + 6] = self.acceleration[index]
        
    def set_coordinates(self, new_coord):
        self.coord = new_coord
        self.update_state()
        
    def set_velocity(self, new_velocity):
        self.velocity = new_velocity
        self.update_state()
    
    def set_acceleration(self, new_acceleration):
        self.acceleration = new_acceleration
        self.update_state()

    def calculate_orbital_period(self, apoapsis, periapsis):
        # Calculate semi-major axis
        semi_major_axis = (apoapsis + periapsis) / 2

        # Calculate orbital period using Kepler's third law
        orbital_period = 2 * np.pi * np.sqrt(semi_major_axis**3 / (GRAVITATIONAL_CONSTANT * MOON_MASS))

        return orbital_period

