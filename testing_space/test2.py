import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sci
import time
import astropy as astro
import astropy.units as u
import astropy.cosmology as cosmo
import numba as nb

import sys
sys.path.append('..')
import global_constants.constants as constants
from mymath import myMath

const = constants.constantsClass()
mymath = myMath.math()

unit = np.array([0.0, 0.0, 1.0])

rotated_position = mymath.rotate_quat(mymath.angle_axis_quat((5.15*np.pi)/180, np.array([0.0,1.0,0.0])), unit)

print(rotated_position[0], rotated_position[1], rotated_position[2])
print(mymath.magnitude(rotated_position))