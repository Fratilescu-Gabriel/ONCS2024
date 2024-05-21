import numpy as np 
import gmpy2 as gm
from timeit import default_timer as timer 
gm.get_context().precision = 20

s = timer()

a = gm.mpfr(2)
a = np.sin(a)

e = timer()

print(e - s)
print(a)