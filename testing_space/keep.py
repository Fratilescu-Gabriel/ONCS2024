# def update_plot(system: Environment, stateout, t, fig, ax, plt3d, f, a):
#     #3D PLOTTER

#     xout = stateout[:,0]
#     yout = stateout[:,1]
#     zout = stateout[:,2]
#     vxout = stateout[:,3]
#     vyout = stateout[:,4]
#     vzout = stateout[:,5]
#     vout = np.sqrt(np.square(vxout)+np.square(vyout)+np.square(vzout))
#     axout = np.gradient(vxout, t)
#     ayout = np.gradient(vyout, t)
#     azout = np.gradient(vzout, t)
#     aout = np.sqrt(axout**2+ayout**2+azout**2)
#     r = np.sqrt(xout**2 + yout**2+zout**2) - mradius


#     bx = system.collector_trajectory[:, 0]
#     by = system.collector_trajectory[:, 1]
#     bz = system.collector_trajectory[:, 2]

#     plt3d.add_curve(bx, by, bz)
#     plt3d.add_curve(system.satellite_trajectory[0], system.satellite_trajectory[1], system.satellite_trajectory[2], color = 'g')


#     angle = np.zeros(len(t))
#     for test_index in range(len(t)):
#         test = system.get_collector_satellite_angle(test_index)
#         angle[test_index] = test   

#     plt3d.add_scatter_plot(system.satellite_trajectory[-1][0], system.satellite_trajectory[-1][1], system.satellite_trajectory[-1][2], color = 'b', marker= '*')

#     print("End to beginning distance:",mymath.magnitude(np.array([xout[0], yout[0], zout[0]])-np.array([xout[-1], yout[-1], zout[-1]])))

#     plt3d.set_axes_equal()

#     #2D PLOTTER

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