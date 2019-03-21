# 
# Jacobi routine for CFD calculation
#
import numpy as np
from scipy import ndimage
def jacobi(niter, psi):

    (m, n) = psi.shape
    m = m - 2
    n = n - 2
    tmp = np.zeros((m+2, n+2))
    stencil = np.array([[0, 1, 0],[1, 0, 1], [0, 1, 0]])
    for iter in range(niter):
        tmp = ndimage.convolve(psi, stencil) * 0.25
        # tmp = tmp * 0.25

        # Update psi
        np.copyto(psi[1:m+1,1:n+1], tmp[1:m+1,1:n+1])


