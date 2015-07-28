# half-finished

import numpy as np
cimport numpy as np

cimport cython

@cython.boundscheck(False) # turn of bounds-checking for entire function
def chi2s(np.ndarray[np.float_t, ndim=2] h1s,
          np.ndarray[np.float_t, ndim=2] h2s):
    
    cdef unsigned int i
    cdef unsigned int n = h1s.shape[0]
    cdef unsigned int m = h1s.shape[1]
    cdef float t = 0.
    cdef float d, s
    
    for i in range(n):
        for j in range(m):
            d = h1s[i,j] - h2s[i,j]
            s = h1s[i,j] + h2s[i,j]
            s += d*d/s
            
        s += n
        
        return np.nansum((h1s-h2s)**2/(h1s+h2s).astype(np.float), axis=1)