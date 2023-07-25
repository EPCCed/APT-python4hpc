# 
# Jacobi routine for CFD calculation
#
import numpy as np
from cython.parallel import prange
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def jacobi(int niter, double[:,:] psi):

    cdef int m = psi.shape[0] - 2
    cdef int n = psi.shape[1] - 2
    cdef int iter, i, j

    tmp = np.zeros((m+2, n+2))
    cdef double[:,:] tmp_view = tmp
    
    for iter in range(niter):
        # Parallelise grid updates over OpenMP threads
        for i in prange(1, m+1, nogil=True):
            for j in range(1, n+1):
                tmp_view[i,j] = 0.25 * (psi[i+1,j]+psi[i-1,j]+psi[i,j+1]+psi[i,j-1])
	
        # Update psi after each iter over the whole grid
        for i in prange(1, m+1, nogil=True):
            for j in range(1, n+1):
                psi[i,j] = tmp_view[i,j]



