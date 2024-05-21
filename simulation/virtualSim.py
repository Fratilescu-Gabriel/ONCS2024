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
from global_constants.constants import *
from graphs import myGraphs as graph
from mymath.myMath import math
from myfile_management.myFileManagement import my_to_csv, get_last_filename, my_chunking, check_file_exists

start_time = timer()
#DEFINE OBJECTS FOR MY MODULES
mymath = math()

## CONSTANTS
G = GRAVITATIONAL_CONSTANT
init_acc = np.array([0, 0, 0])
Altitude_Test = (False, 0)

# initial_acceleration = np.array([550, 5000, 0], dtype=np.longdouble)
class Environment():
    def __init__(self, collector_coordinates, earth_initial_angle, solar_pannel_power, DC_to_RF_power, in_SSE, in_CRE) -> None:
        self.sat = satellite.SatelliteClass(SATELLITE_MASS, np.zeros(3, dtype = np.longdouble),np.zeros(3, dtype=np.longdouble),np.zeros(3, dtype=np.longdouble))
        self.planets_system = moon.PlanetsSystem(collector_coordinates, earth_initial_angle)
        self.sat_tr = None
        self.SSE = 0
        self.CRE = 0
        self.solar_pannel_power = solar_pannel_power
        self.DC_to_RF_power = DC_to_RF_power
        self.total_sat_energy = self.SSE
        self.sat_init_acc = init_acc
        
        self.plot_CRE = 0
        self.plot_SSE = 0
        self.in_CRE = in_CRE
        self.in_SSE = in_SSE
        self.plot_SP = 0
        self.plot_CP = 0
        self.sat_in_light = None
        self.sat_collector_angle = None

    def calculate_trajectories(self, t):
        self.earth_tr = np.transpose(self.planets_system.earth_trajectory(t))
        self.sun_tr = np.transpose(self.planets_system.sun_trajectory(t))
        self.collector_tr = np.transpose(self.planets_system.coll_trajectory(t))
        self.moon_surface = self.planets_system.moon_surface()
        
    
    def set_initial_conditions(self, coord, velocity, acceleration):
        self.sat.set_coordinates(coord)
        self.sat.set_velocity(velocity)
        self.sat.set_acceleration(acceleration)

    def motion_law_ode(self, state, t):
        #State vector
        # global initial_acceleration
        position = np.array([state[0], state[1], state[2]], dtype=np.longdouble)
        velocity = np.array([state[3], state[4], state[5]], dtype=np.longdouble)

        #Compute total forces
        total_force = np.zeros(3)

        gravitational_force = -self.gravitational_acceleration(position) * self.sat.mass

        total_force += gravitational_force

        #Acceleration based on Newton's second law
        acceleration = total_force/self.sat.mass


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

        if r > MOON_RADIUS:
            #Gravitational acceleration formula
            accel = ((G * MOON_MASS)/(r**2))*unit_vector
            
        return accel
    
    def calculate_apoapsis(self, stateout):
        positions = stateout[:3]
        positions = np.transpose(positions)
        return np.max(np.array([mymath.magnitude(position) for position in positions]))

    def calculate_periapsis(self, stateout):
        positions = stateout[:3]
        positions = np.transpose(positions)
        return np.min(np.array([mymath.magnitude(position) for position in positions]))
    
    def is_sat_in_light(self, time_index):
        cone_tip_parameter = -MOON_RADIUS/(SUN_RADIUS - MOON_RADIUS)
        angle = np.sum(self.planets_system.earth_angle_list[time_index])
        
        sun_distance_x = -(MOON_RADIUS * np.cos(MOON_ANGLE_WITH_ECLIPTIC) * np.sin(angle))
        sun_distance_y = -MOON_RADIUS * np.cos(MOON_ANGLE_WITH_ECLIPTIC) * np.cos(angle) + DISTANCE_EARTH_SUN
        sun_distance_z =  -MOON_RADIUS * np.sin(MOON_ANGLE_WITH_ECLIPTIC)
        
        cone_tip = np.array([sun_distance_x, sun_distance_y, sun_distance_z])*cone_tip_parameter
        
        direction = self.sun_tr[time_index] - cone_tip
        
        height = mymath.magnitude(direction)
        
        direction = mymath.to_unit_vector(direction)
        
        cone_dist = np.dot(self.sat_tr[time_index] - cone_tip, direction)
        
        cone_radius = (cone_dist / height) * SUN_RADIUS

        orth_distance = mymath.magnitude((self.sat_tr[time_index] - cone_tip) - cone_dist * direction)

        is_point_inside_cone = (orth_distance < np.abs(cone_radius))
    
        return (not is_point_inside_cone or (is_point_inside_cone and self.sat_tr[time_index][1] < 0))
    
    def get_collector_sat_angle(self, time_index):
        normal_to_moon_surface = mymath.to_unit_vector(self.collector_tr[time_index])
        sat_unit_vector = mymath.to_unit_vector(self.sat_tr[time_index])
        collector_sat_vector = mymath.to_unit_vector(normal_to_moon_surface - sat_unit_vector) 
        
        angle = np.arccos(np.dot(normal_to_moon_surface, collector_sat_vector))
        
        return (angle/np.pi)*180
    
    def performance(self, t):
        self.plot_CRE = np.zeros(len(t))
        self.plot_SSE = np.zeros(len(t))
        self.plot_CP = np.zeros(len(t))
        self.plot_SP = np.zeros(len(t))
        self.sat_in_light = np.zeros(len(t))
        self.sat_collector_angle = np.zeros(len(t))

        
        for index in range(len(t)):
            in_light = self.is_sat_in_light(index)
            # print(in_light)
            dt = t[index] - t[index-1]
            if in_light and index > 0 :
                current_energy = dt * self.solar_pannel_power #kJ = kW*s
                self.SSE = self.SSE +  current_energy #kJ
                self.total_sat_energy = self.total_sat_energy + current_energy #kJ
                
            self.sat_collector_angle[index] = self.get_collector_sat_angle(index)
            
            if self.sat_collector_angle[index] < 4.5 and index > 0:
                # print(self.sat_collector_angle[index])
                energy = dt * self.DC_to_RF_power
                if energy <= self.SSE:
                    energy_transmited = energy * DC_TO_RF * RF_TO_DC * RF_COLLECTION
                    # print(energy_transmited)
                else:
                    energy_transmited = self.SSE * DC_TO_RF * RF_TO_DC * RF_COLLECTION
                
                self.SSE = self.SSE - energy_transmited
                self.CRE = self.CRE + energy_transmited
            
            self.sat_in_light[index] = 1 if in_light else -1
            self.plot_SSE[index] = self.SSE
            self.plot_CRE[index] = self.CRE
            
            if index > 0:
                self.plot_SP[index] = (self.plot_SSE[index] - self.plot_SSE[index-1])/dt #kW
                self.plot_CP[index] = (self.plot_CRE[index] - self.plot_CRE[index-1])/dt #kW
            if index == 0:
                self.plot_SP[index] = (self.plot_SSE[index] - self.in_SSE)/dt #kW
                self.plot_CP[index] = (self.plot_CRE[index] - self.in_CRE)/dt #kW



#DEFAULT TIME VALUES
DAYx28 = 28*24*60*60 #s
DAYx29_53 = 29.530589*24*60*60


#INITIAL CONDITIONS

TIME = DAYx29_53
DATA_POINTS = 1000000
CHUNK_VOLUME= 100000


a = 10
plot_earth = False
transmission_case = 1

solar_panel_surface = 0 #m2
solar_panel_power_per_surface = 0 #W/m2
solar_pannel_power = (solar_panel_surface * solar_panel_power_per_surface)/1000 #kW
DC_to_RF_power = 0 #kW
mean_total_power_cp = 0
mean_total_power_sp = 0
mean_power_in_transmission = 0

if transmission_case == 1: #Simple liniar loss
    solar_panel_surface = 10000 #m2
    solar_panel_power_per_surface = 257 #W/m2
    solar_pannel_power = (solar_panel_surface * solar_panel_power_per_surface)/1000 #kW
    DC_to_RF = 200 #kW
    

collector_initial_position = np.array([MOON_RADIUS, 0, 0])
earth_initial_angle = 0



sat_initial_position = np.array([MOON_RADIUS+400000, 0, 0 ], dtype=np.longdouble)

r0 = mymath.magnitude(sat_initial_position)
sat_initial_velocity = np.array([0, np.sqrt(G*MOON_MASS/r0), 200], dtype=np.longdouble)
sat_initial_acceleration = np.array([0.0, 0.0, 0.0], dtype=np.longdouble)
first_time = True
in_SSE = 0
in_CRE = 0

energy_in = 0
energy_out = 0


apoapsis = 0
periapsis = 9999999999999999999999

def chunk_processing(n, time_init, plot_chunk, write_data, new_file):
    #MECHANISM
    global mean_total_power_cp, mean_total_power_sp, energy_in, energy_out, apoapsis, periapsis, collector_initial_position, earth_initial_angle, solar_pannel_power, DC_to_RF_power,sat_initial_position, sat_initial_velocity, sat_initial_acceleration, first_time, in_SSE, in_CRE
    system = Environment(collector_initial_position, earth_initial_angle, solar_pannel_power, DC_to_RF_power, in_SSE, in_CRE)
    print("EARTH INIT ANGLE", earth_initial_angle)
    system.sat.set_coordinates(sat_initial_position)
    system.sat.set_velocity(sat_initial_velocity)
    system.sat.set_acceleration(sat_initial_acceleration)
    system.SSE = in_SSE
    system.CRE = in_CRE
    system.total_sat_energy = energy_in

    initial_state = system.sat.get_state()
    initial_state = np.delete(initial_state, [6, 7, 8])

    time_per_chunk = TIME / (DATA_POINTS/CHUNK_VOLUME)
    t = np.linspace(time_init, time_init + time_per_chunk, CHUNK_VOLUME)

    stateout = sci.odeint(system.motion_law_ode, initial_state, t)

    system.calculate_trajectories(t)

    xout = stateout[:,0]
    yout = stateout[:,1]
    zout = stateout[:,2]

    system.sat_tr = np.transpose(np.array([xout, yout, zout]))

    vxout = stateout[:,3]
    vyout = stateout[:,4]
    vzout = stateout[:,5]
    vout = np.sqrt(np.square(vxout)+np.square(vyout)+np.square(vzout))
    axout = np.gradient(vxout, t)
    ayout = np.gradient(vyout, t)
    azout = np.gradient(vzout, t)
    aout = np.sqrt(axout**2+ayout**2+azout**2)
    r = np.sqrt(xout**2 + yout**2+zout**2) - MOON_RADIUS
    
    Altitude_Test = [(True, alt) if alt > 500e3 else False for alt in r]

    if apoapsis < system.calculate_apoapsis(stateout):
        apoapsis = system.calculate_apoapsis(stateout)
        
    if periapsis > system.calculate_periapsis(stateout):
        periapsis = system.calculate_periapsis(stateout)
    orbital_period = system.sat.calculate_orbital_period(apoapsis, periapsis)
    print("Orbital period with kepler", orbital_period)

    system.performance(t)
    
    energy_in = system.total_sat_energy
    energy_out = system.CRE
    
    in_shadow = 0
    in_light = 0
    for test_index in range(len(t)):
        test = system.is_sat_in_light(test_index)
        if test:
            in_light = in_light + 1
        else:
            in_shadow = in_shadow + 1

    print("Percentage in shadow")
    print(100* in_shadow/(in_shadow+in_light))
    
    # if plot_chunk:
    #     x, y, z = system.moon_surface
    #     plt3d.add_surface(x, y, z, alpha=1)
    #     update_plot(system, stateout, t, fig, ax, plt3d, f, a)
    
    if write_data:
        my_to_csv(
            pd.DataFrame([t, r, xout, yout, zout, vxout, vyout, vzout, vout, axout, ayout, azout, aout, system.collector_tr[:,0],
                    system.collector_tr[:,1], system.collector_tr[:,2], system.sun_tr[:,0],
                    system.sun_tr[:,1], system.sun_tr[:,2], system.earth_tr[:,0], system.earth_tr[:,1], system.earth_tr[:,2], 
                    system.sat_in_light, system.sat_collector_angle, system.plot_CRE, system.plot_SSE, system.plot_CP, system.plot_SP]),
            column_names= ['time', 'altitude', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'v', 'ax', 'ay', 'az', 'a', 'cx', 'cy', 'cz', 
                        'sx', 'sy', 'sz', 'ex', 'ey', 'ez','SIL', 'SCA', 'CRE', 'SSE', 'CP', 'SP'],
            new_file= new_file if first_time else False,
            file_append= True if not first_time else False
        )
        
    collector_initial_position = system.collector_tr[-1]
    earth_initial_angle = earth_initial_angle + system.planets_system.earth_angle_list[-1]
    solar_pannel_power = 25
    DC_to_RF_power = 1000

    sat_initial_position = system.sat_tr[-1]

    r0 = mymath.magnitude(sat_initial_position)
    sat_initial_velocity = np.array([vxout[-1], vyout[-1], vzout[-1]], dtype=np.longdouble)
    sat_initial_acceleration = np.array([axout[-1], ayout[-1], azout[-1]], dtype=np.longdouble)
    in_SSE = system.plot_SSE[-1]
    in_CRE = system.plot_CRE[-1]
    first_time = False
    
    mean_total_power_cp = ((n-1) * CHUNK_VOLUME * mean_total_power_cp + np.sum(system.plot_CP))/(n*CHUNK_VOLUME)
    mean_total_power_sp = ((n-1) * CHUNK_VOLUME * mean_total_power_sp + np.sum(system.plot_SP))/(n*CHUNK_VOLUME)
    print('MEAN SATELLITE POWER', mean_total_power_sp)
    print('MEAN OUTPUT POWER', mean_total_power_cp)
    print("CP SUM", np.sum(system.plot_CP))

def plot_data(file_name = get_last_filename()):
    global plot_earth, mean_total_power_sp, mean_total_power_cp
    first = True
    print(file_name)
    first_chunk = True
    plt3d = graph.Plotter3D()
    fig1, ax1 = plt.subplots(2,2)
    fig2, ax2 = plt.subplots(1,4)
    end_x = 0
    end_y = 0
    end_z = 0
    ee = None
    es = None
    for chunk in my_chunking(file_name, yield_header= False):
        
        t = chunk[0]
        r = chunk[1]
        x = chunk[2]
        y = chunk[3]
        z = chunk[4]
        vx = chunk[5]
        vy = chunk[6]
        vz = chunk[7]
        v = chunk[8]
        ax = chunk[9]
        ay = chunk[10]
        az = chunk[11]
        a = chunk[12]
        cx = chunk[13]
        cy = chunk[14]
        cz = chunk[15]
        sx = chunk[16]
        sy = chunk[17]
        sz = chunk[18]
        ex = chunk[19]
        ey = chunk[20]
        ez = chunk[21]
        sil = chunk[22]
        sca = chunk[23]
        cre = chunk[24]
        sse = chunk[25]
        cp = chunk[26]
        sp = chunk[27]
        
        end_x = x[-1]
        end_y = y[-1]
        end_z = z[-1]
        
        if first:
            es = np.array([ex[0], ey[0], ez[0]])
        
        first = False
        ee = np.array([ex[-1], ey[-1], ez[-1]])
        
        if first_chunk:
            first_chunk = not first_chunk
            
            plt3d.new_figure("", 'X', 'Y', 'Z')
            
            env = Environment(collector_initial_position, earth_initial_angle, solar_pannel_power, DC_to_RF_power, in_SSE, in_CRE)
            env.calculate_trajectories(np.asarray([0]))
            moon_surface = env.moon_surface
            del(env)
            
            plt3d.add_surface(moon_surface[0], moon_surface[1], moon_surface[2], alpha=0.4)
            
            plt3d.add_curve(x, y, z, color='g', label='sat tr')
            plt3d.add_scatter_plot(x[0], y[0], z[0], label='Start', color='b', marker='*')
            
            plt3d.add_curve(cx, cy, cz, color='r', label='Collector tr')
            
            if plot_earth:
                plt3d.add_curve(ex, ey, ez, color='k', label='Earth')
            
            plt3d.show_legend()
            
            ax1[0][0].plot(x, y, color='r', label = 'Orbit')
            ax1[0][0].plot(x[0],y[0],color='g', label = 'Start', marker = '*')
            theta = np.linspace(0,2*np.pi,100000, dtype=np.longdouble)
            xplanet = MOON_RADIUS*np.sin(theta)
            yplanet = MOON_RADIUS*np.cos(theta)
            ax1[0][0].plot(xplanet,yplanet,color='b',label='Planet')
            ax1[0][0].set_xlabel('x - m')
            ax1[0][0].set_ylabel('y - m')
            ax1[0][0].grid(linestyle='--', color='gray', alpha=0.7)
            ax1[0][0].legend(loc = 'upper left')

            ax1[0][1].plot(t, vx,color= 'b', label = 'Velocity x')
            ax1[0][1].plot(t, vy,color='g', label = 'Velocity y')
            ax1[0][1].plot(t, vz,color= 'y', label = "Velocity z")
            ax1[0][1].plot(t, v,color='r',label='Velocity')
            ax1[0][1].set_xlabel('t - s')
            ax1[0][1].set_ylabel('velocity - m/s')
            ax1[0][1].grid(linestyle='--', color='gray', alpha=0.7)
            ax1[0][1].legend(loc = 'upper left')

            ax1[1][0].plot(t, ax, color='b', label = 'Acceleration x')
            ax1[1][0].plot(t, ay,color='g', label = 'Acceleration y')
            ax1[1][0].plot(t, az,color= 'y', label = 'Acceleration z')
            ax1[1][0].plot(t, a,color='r',label='Acceleration')
            ax1[1][0].set_xlabel('t - s')
            ax1[1][0].set_ylabel('acceleration - m/s2')
            ax1[1][0].grid(linestyle='--', color='gray', alpha=0.7)
            ax1[1][0].legend(loc = 'upper left')

            ax1[1][1].plot(t, r, color='r', label = 'Altitude x')
            ax1[1][1].set_xlabel('t - s')
            ax1[1][1].set_ylabel('Altitude - m')
            ax1[1][1].grid(linestyle='--', color='gray', alpha=0.7)
            ax1[1][1].legend(loc = 'upper left')
            
            ax2[0].plot(t, sca, color= 'r', label = 'SC Angle')
            ax2[0].set_xlabel('t - s')
            ax2[0].set_ylabel('Angle - degrees')
            ax2[0].grid(linestyle='--', color='gray', alpha=0.7)
            ax2[0].legend(loc = 'upper left')

            sse = sse/3600 #kJ -> kWh
            cre = cre/3600 # kJ -> kWh
            ax2[1].plot(t, sse, color='r', label = 'Sat Energy')
            ax2[1].plot(t, cre, color='b', label = 'Coll Energy')
            ax2[1].set_xlabel('t - s')
            ax2[1].set_ylabel('Energy - Kw*h')
            ax2[1].grid(linestyle='--', color='gray', alpha=0.7)
            ax2[1].legend(loc = 'upper left')

            ax2[2].plot(t, sil, color='r', label = 'In light')
            ax2[2].set_xlabel('t - s')
            ax2[2].set_ylabel('light')
            ax2[2].grid(linestyle='--', color='gray', alpha=0.9)
            ax2[2].legend(loc = 'upper left')
            
            ax2[3].plot(t, sp, color='r', label = 'Sat Collection Power')
            ax2[3].plot(t, cp, color='b', label = 'Coll Output Power')
            ax2[3].set_xlabel('t - s')
            ax2[3].set_ylabel('Power - Kw')
            ax2[3].grid(linestyle='--', color='gray', alpha=0.7)
            ax2[3].legend(loc = 'upper left')
        else:
            plt3d.add_curve(x, y, z, color='g')
            plt3d.add_curve(cx, cy, cz, color='r')
            
            if plot_earth:
                plt3d.add_curve(ex, ey, ez, color='k')
            
            ax1[0][0].plot(x, y, color='r')
            
            
            ax1[0][1].plot(t, vx, color='b')
            ax1[0][1].plot(t, vy,color='g')
            ax1[0][1].plot(t, vz,color= 'y')
            ax1[0][1].plot(t, v,color='r')
            
            ax1[1][0].plot(t, ax, color='b')
            ax1[1][0].plot(t, ay,color='g')
            ax1[1][0].plot(t, az,color= 'y')
            ax1[1][0].plot(t, a,color='r')
            
            ax1[1][1].plot(t, r, color='r')
            
            ax2[0].plot(t, sca, color='r')
            
            
            sse = sse/3600 #kJ -> kWh
            cre = cre/3600 # kJ -> kWh
            ax2[1].plot(t, sse, color='r')
            ax2[1].plot(t, cre, color='b')
            
            ax2[2].plot(t, sil,color='r')
            
            ax2[3].plot(t, sp, color='r')
            ax2[3].plot(t, cp, color='b')
    
    # chunk = my_chunking(file_name, yield_header=False)
    ax1[0][0].plot(end_x, end_y, label = 'End', color = 'b', marker = '*')
    if plot_earth:
        plt3d.add_scatter_plot(end_x, end_y, end_z, label='Start', color='b', marker='*')
        # plt3d.add_scatter_plot(ex[-1], ey[-1], ez[-1], label='Start', color='b', marker='*')
        plt3d.add_scatter_plot(es[0], es[1], es[2], label='Start', color='g', marker='*')
    print(mymath.magnitude(ee-es))
    print('MEAN SATELLITE POWER', mean_total_power_sp)
    print('MEAN OUTPUT POWER', mean_total_power_cp)
    
    plot_show()



def plot_show():
    end = timer()
    print("TIME TAKEN", end-start_time)
    plt.show()

def start_simulation():
    print(DATA_POINTS/CHUNK_VOLUME)
    for chunk in range(int(DATA_POINTS/CHUNK_VOLUME)):
        print("TIMEINIT:",chunk*(TIME/(DATA_POINTS/CHUNK_VOLUME)))
        chunk_processing(chunk+1, chunk*(TIME/(DATA_POINTS/CHUNK_VOLUME)), True, True, True)

simulate = False
plot = True



if a == 1:
    simulate = False
    plot = True
else:
    simulate = True
    plot = False

file_to_plot = ''

if simulate:
    start_simulation()
    end = timer()
    print("TIME TAKEN", end-start_time)
    
    
if plot:
    if file_to_plot == '':
        plot_data()
    elif check_file_exists(file_to_plot):
        plot_data(file_name=file_to_plot)
    else:
        raise KeyError('File not found')


print('EFFICIENCY', 100 * energy_out/energy_in)
if Altitude_Test[0]:
    print("ALTITUDE MAX OVER: ", Altitude_Test[1])