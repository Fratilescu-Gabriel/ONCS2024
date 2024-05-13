import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sci
import time
import astropy as astro
import astropy.units as u
import astropy.cosmology as cosmo
import astropy.coordinates as coord
from simulation import virtualSim
from global_constants import constants
const = constants.constantsClass()

mradius = const.MOON_RADIUS

TIME = 150000
DATA_POINTS = 500000
CHUNCKING_DATA_POINTS = 50000

collector_initial_position = np.array([mradius, 0, 0])
earth_initial_angle = 0
solar_pannel_energy_gain_per_second = 25
DC_to_RF_conversion_rate_per_second = 1000

satellite_initial_position = np.array([mradius+400000, 0, 0 ], dtype=np.longdouble)
satellite_initial_velocity = np.array([0, np.sqrt(G*mmass/r0), 100], dtype=np.longdouble)
satellite_initial_acceleration = np.array([0.0, 0.0, 0.0], dtype=np.longdouble)

# system = Environment(collector_initial_position, earth_initial_angle, solar_pannel_energy_gain_per_second, DC_to_RF_conversion_rate_per_second)

# system.satellite.set_coordinates(np.array([mradius+400000, 0, 0 ], dtype=np.longdouble))
# r0 = mymath.magnitude(system.satellite.coord)

# system.satellite.set_velocity(satellite_initial_velocity)
# system.satellite.set_acceleration(np.array([0.0, 0.0, 0.0], dtype=np.longdouble))

# initial_state = system.satellite.get_state()
# initial_state = np.delete(initial_state, [6, 7, 8])

# t = np.linspace(0,TIME, DATA_POINTS, dtype=np.longdouble)

# stateout = sci.odeint(system.motion_law_ode, initial_state, t)

# system.calculate_trajectories(t)

# xout = stateout[:,0]
# yout = stateout[:,1]
# zout = stateout[:,2]

# system.satellite_trajectory = np.transpose(np.array([xout, yout, zout]))


# vxout = stateout[:,3]
# vyout = stateout[:,4]
# vzout = stateout[:,5]
# vout = np.sqrt(np.square(vxout)+np.square(vyout)+np.square(vzout))
# axout = np.gradient(vxout, t)
# ayout = np.gradient(vyout, t)
# azout = np.gradient(vzout, t)
# aout = np.sqrt(axout**2+ayout**2+azout**2)
# r = np.sqrt(xout**2 + yout**2+zout**2) - mradius

# orbital_period = system.satellite.calculate_orbital_period(system.calculate_apoapsis(stateout), system.calculate_periapsis(stateout))
# print("Orbital period with kepler", orbital_period)

# def plot_all():
#     pl3d = graph.Plotter3D()
#     # m = moon.MoonSystem(np.array([mradius*np.cos(np.pi/5), 0, mradius*np.sin(np.pi/5)]), np.pi)
#     pl3d.new_figure(xlabel= 'X', ylabel='Y', zlabel='Z')

#     x, y, z = system.moon_surface
#     pl3d.add_surface(x, y, z, alpha=1)


#     # bx, by, bz = m.stationary_point_trajectory_curve(900000, 6.688)
#     # pl3d.add_curve(bx, by, bz)

#     bx = system.stationary_surface_point_trajectory[:, 0]
#     by = system.stationary_surface_point_trajectory[:, 1]
#     bz = system.stationary_surface_point_trajectory[:, 2]

#     pl3d.add_curve(bx+10000, by+10000, bz)
#     pl3d.add_curve(xout, yout, zout, color = 'g')


#     e = system.sun_trajectory

#     pl3d.set_axes_equal()

#     in_shadow = 0
#     in_light = 0
#     for test_index in range(len(t)):
#         test = system.is_satellite_in_light(test_index)
#         if test:
#             in_light = in_light + 1
#         else:
#             in_shadow = in_shadow + 1

#     print("Percentage in shadow")
#     print(100* in_shadow/(in_shadow+in_light))

#     angle = np.zeros(len(t))
#     for test_index in range(len(t)):
#         test = system.get_collector_satellite_angle(test_index)
#         angle[test_index] = test   

#     pl3d.add_scatter_plot(system.satellite_trajectory[-1][0], system.satellite_trajectory[-1][1], system.satellite_trajectory[-1][2], color = 'b', marker= '*')

#     print("End to beginning distance:",mymath.magnitude(np.array([xout[0], yout[0], zout[0]])-np.array([xout[-1], yout[-1], zout[-1]])))

#     system.performance(t)
#     fig, ax = plt.subplots(2,2)
#     fig.canvas.manager.set_window_title("My graphs")
#     ax[0][0].plot(xout, yout, 'r', label = 'Orbit')
#     ax[0][0].plot(xout[0],yout[0],'g*', label = 'Start')
#     ax[0][0].plot(xout[len(xout)-1], yout[len(yout)-1], 'b*', label = 'End')
#     ax[0][0].plot(bx, by, 'g', label='Collector')
#     theta = np.linspace(0,2*np.pi,100000, dtype=np.longdouble)
#     xplanet = mradius*np.sin(theta)
#     yplanet = mradius*np.cos(theta)
#     ax[0][0].plot(xplanet,yplanet,'b-',label='Planet')
#     ax[0][0].set_xlabel('x - m')
#     ax[0][0].set_ylabel('y - m')
#     ax[0][0].grid(linestyle='--', color='gray', alpha=0.7)
#     ax[0][0].legend(loc = 'upper left')

#     # fig, ax = plt.subplots(1,1)
#     # fig.canvas.manager.set_window_title("Velocity")
#     ax[0][1].plot(t, vxout, 'b', label = 'Velocity x')
#     ax[0][1].plot(t, vyout,'g', label = 'Velocity y')
#     ax[0][1].plot(t, vzout, 'y', label = "Velocity z")
#     ax[0][1].plot(t, vout,'r',label='Velocity')
#     ax[0][1].set_xlabel('t - s')
#     ax[0][1].set_ylabel('velocity - m/s')
#     ax[0][1].grid(linestyle='--', color='gray', alpha=0.7)
#     ax[0][1].legend(loc = 'upper left')

#     # fig, ax = plt.subplots(1,1)
#     # fig.canvas.manager.set_window_title("Acceleration")
#     ax[1][0].plot(t, axout, 'b', label = 'Acceleration x')
#     ax[1][0].plot(t, ayout,'g', label = 'Acceleration y')
#     ax[1][0].plot(t, azout, 'y', label = 'Acceleration z')
#     ax[1][0].plot(t, aout,'r',label='Acceleration')
#     ax[1][0].set_xlabel('t - s')
#     ax[1][0].set_ylabel('acceleration - m/s2')
#     ax[1][0].grid(linestyle='--', color='gray', alpha=0.7)
#     ax[1][0].legend(loc = 'upper left')

#     # fig, ax = plt.subplots(1,1)
#     # fig.canvas.manager.set_window_title("Altitude")
#     ax[1][1].plot(t, r, 'r', label = 'Altitude x')
#     ax[1][1].set_xlabel('t - s')
#     ax[1][1].set_ylabel('Altitude - m')
#     ax[1][1].grid(linestyle='--', color='gray', alpha=0.7)
#     ax[1][1].legend(loc = 'upper left')

#     f, a = plt.subplots(1,3)
#     # angle = (angle/np.pi)*180
#     a[0].plot(t, angle, 'r', label = 'Angle')
#     a[0].set_xlabel('t - s')
#     a[0].set_ylabel('Angle - degrees')
#     a[0].grid(linestyle='--', color='gray', alpha=0.7)
#     a[0].legend(loc = 'upper left')

#     a[1].plot(t, system.plot_sattelite_stored_energy, 'r', label = 'Satellite')
#     a[1].plot(t, system.plot_collector_received_energy, 'b', label = 'Collector')
#     a[1].set_xlabel('t - s')
#     a[1].set_ylabel('Energy')
#     a[1].grid(linestyle='--', color='gray', alpha=0.7)
#     a[1].legend(loc = 'upper left')

#     a[2].plot(t, system.satellite_in_light, 'r', label = 'In light')
#     a[2].set_xlabel('t - s')
#     a[2].set_ylabel('light')
#     a[2].grid(linestyle='--', color='gray', alpha=0.7)
#     a[2].legend(loc = 'upper left')

#     end_time = timer()

#     print("TIME TO RUN: ", end_time-start_time)

#     plt.show()

# plot_all()

def plot_all(system):
    plt3d = graph.Plotter3D()
    pl3d.set_axes_equal()

    pl3d.new_figure(xlabel= 'X', ylabel='Y', zlabel='Z')

    x, y, z = system.moon_surface
    pl3d.add_surface(x, y, z, alpha=1)

    bx = system.stationary_surface_point_trajectory[:, 0]
    by = system.stationary_surface_point_trajectory[:, 1]
    bz = system.stationary_surface_point_trajectory[:, 2]

    pl3d.add_curve(bx, by, bz)
    pl3d.add_curve(xout, yout, zout, color = 'g')
    
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

    pl3d.add_scatter_plot(system.satellite_trajectory[-1][0], system.satellite_trajectory[-1][1], system.satellite_trajectory[-1][2], color = 'b', marker= '*')

    print("End to beginning distance:",mymath.magnitude(np.array([xout[0], yout[0], zout[0]])-np.array([xout[-1], yout[-1], zout[-1]])))
    
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


def chunk_processing(t):
    system = Environment(collector_initial_position, earth_initial_angle, solar_pannel_energy_gain_per_second, DC_to_RF_conversion_rate_per_second)
    system.satellite.set_coordinates(satellite_initial_position)
    system.satellite.set_velocity(satellite_initial_velocity)
    system.satellite.set_acceleration(satellite_initial_acceleration)
    
    initial_state = system.satellite.get_state()
    initial_state = np.delete(initial_state, [6, 7, 8])
    
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



for chunk in range(int(DATA_POINTS/CHUNCKING_DATA_POINTS)):
    time_chunk = (TIME/DATA_POINTS)*CHUNCKING_DATA_POINTS
    t = np.linspace(chunk*time_chunk, (chunk+1)*time_chunk, DATA_POINTS)
    chunk_processing(t)