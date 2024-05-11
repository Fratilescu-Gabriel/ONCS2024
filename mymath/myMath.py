import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sci
import time
import astropy as astro
import astropy.units as u
import astropy.cosmology as cosmo
import astropy.coordinates as coord

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
        

