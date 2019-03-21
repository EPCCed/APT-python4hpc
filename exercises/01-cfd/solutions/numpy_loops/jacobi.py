# 
# Jacobi routine for CFD calculation
#
import numpy as np

def jacobi(niter, psi):

    (m, n) = psi.shape
    m = m - 2
    n = n - 2

    tmp = np.zeros((m+2, n+2))
    for iter in range(niter):
        for i in range(1,m+1):
            for j in range(1,n+1):
                tmp[i,j] = 0.25 * (psi[i+1,j]+psi[i-1,j]+psi[i,j+1]+psi[i,j-1])

        # Update psi
        np.copyto(psi[1:m+1,1:n+1], tmp[1:m+1,1:n+1])
