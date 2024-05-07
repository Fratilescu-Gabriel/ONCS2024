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
                mag = np.sqrt(np.sum(np.square(vector)))
                return mag
            else:
                raise Exception("Array must be a vectore")
        else:
            raise Exception("Variable must be a numpy array")
    
    def to_unit_vector(self, vector):
        if isinstance(vector, np.ndarray):
            if len(vector.shape) == 1 and vector.shape[0] == 3:
                mag = self.magnitude(vector)
                vector /= mag
                return vector
            else:
                raise Exception("Array must be a vector")
        else:
            raise Exception("Variable must be a numpy array")
        

