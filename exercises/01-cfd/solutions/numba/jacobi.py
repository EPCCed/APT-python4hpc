# 
# Jacobi routine for CFD calculation
#
import numpy as np
from numba import jit

@jit(nopython=True)
def jacobi(niter, psi):

    (m, n) = psi.shape
    m = m - 2
    n = n - 2

    tmp = np.zeros((m+2, n+2))
    for iter in range(niter):
        # Use index notation and offsets to compute the stream function
        tmp[1:m+1,1:n+1] = 0.25 * (psi[2:m+2,1:n+1]+psi[0:m,1:n+1]+psi[1:m+1,2:n+2]+psi[1:m+1,0:n])

        # Update psi
        psi[1:m+1,1:n+1] = np.copy(tmp[1:m+1,1:n+1])


