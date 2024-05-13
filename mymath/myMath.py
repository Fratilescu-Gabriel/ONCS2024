import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sci
import time
import astropy as astro
import astropy.units as u
import astropy.cosmology as cosmo
import astropy.coordinates as coord
from scipy.interpolate import griddata


class math():
    def magnitude(self, vector):
        if isinstance(vector, np.ndarray):
            if len(vector.shape) == 1 and vector.shape[0] == 3:
                # For one-dimensional array (single vector)
                mag = np.sqrt(np.sum(np.square(vector)))
                return mag
            elif len(vector.shape) == 2 and vector.shape[0] == 3 and vector.shape[1] >= 1:
                # For two-dimensional array (multiple vectors)
                mags = np.sqrt(np.sum(np.square(vector), axis=0))
                return mags
            else:
                raise Exception("Array must be either a one-dimensional or a two-dimensional array with 3 rows")
        else:
            raise Exception("Variable must be a numpy array")
    
    def to_unit_vector(self, vector):
        if isinstance(vector, np.ndarray):
            if len(vector.shape) == 1 and vector.shape[0] == 3:
                # For one-dimensional array (single vector)
                mag = self.magnitude(vector)
                unit_vector = vector / mag
                return unit_vector
            elif len(vector.shape) == 2 and vector.shape[0] == 3 and vector.shape[1] >= 1:
                # For two-dimensional array (multiple vectors)
                mags = np.sqrt(np.sum(np.square(vector), axis=0))
                unit_vectors = vector / mags
                return unit_vectors
            else:
                raise Exception("Array must be either a one-dimensional or a two-dimensional array with 3 rows")
        else:
            raise Exception("Variable must be a numpy array")
    
    def interpolate_to_regular_grid(self, x, y, z, resolution=1000):
        """Interpolate irregularly spaced data onto a regular grid."""
        # Define the regular grid
        xi = np.linspace(min(x), max(x), resolution)
        yi = np.linspace(min(y), max(y), resolution)

        # Create meshgrid for the regular grid
        xi, yi = np.meshgrid(xi, yi)

        # Interpolate the data onto the regular grid
        zi = griddata((x, y), z, (xi, yi), method='linear')
        

        return xi, yi, zi
    
    def angle_axis_quat(self, theta, axis):
        """
        Given an angle and an axis, it returns a quaternion.
        """
        axis = np.array(axis) / np.linalg.norm(axis)
        return np.append([np.cos(theta/2)],np.sin(theta/2) * axis)

    def mult_quat(self, q1, q2):
        """
        Quaternion multiplication.
        """
        q3 = np.copy(q1)
        q3[0] = q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3]
        q3[1] = q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2]
        q3[2] = q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1]
        q3[3] = q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0]
        return q3

    def rotate_quat(self, quat, vect):
        """
        Rotate a vector with the rotation defined by a quaternion.
        """
        # Transfrom vect into an quaternion 
        vect = np.append([0],vect)
        # Normalize it
        norm_vect = np.linalg.norm(vect)
        vect = vect/norm_vect
        # Computes the conjugate of quat
        quat_ = np.append(quat[0],-quat[1:])
        # The result is given by: quat * vect * quat_
        res = self.mult_quat(quat, self.mult_quat(vect,quat_)) * norm_vect
        return res[1:]

