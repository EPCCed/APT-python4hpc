# 
# Jacobi function for CFD calculation including vorticity
#
# Basic Python version using lists
#

import sys

def boundaryzet(zet, psi):

    # Get the inner dimensions
    m = len(psi) - 2
    n = len(psi[0]) -2

    for i in range (1, m+1):

        zet[i][0  ] = 2.0*(psi[i][1] - psi[i][0  ])
        zet[i][n+1] = 2.0*(psi[i][n] - psi[i][n+1])

    for j in range (1, n+1):

        zet[0  ][j] = 2.0*(psi[1][j] - psi[0  ][j])
        zet[m+1][j] = 2.0*(psi[m][j] - psi[m+1][j])

def jacobivort(niter, psi, re):

    # Get the inner dimensions
    m = len(psi) - 2
    n = len(psi[0]) -2

    # Define required local arrays and zero them
    psitmp = [[0 for col in range(n+2)] for row in range(m+2)]
    zet    = [[0 for col in range(n+2)] for row in range(m+2)]
    zettmp = [[0 for col in range(n+2)] for row in range(m+2)]

    # Iterate for number of iterations
    for iter in range(1,niter+1):

        # Set boundary conditions on zeta, which depend on psi
        boundaryzet(zet, psi)

        # Loop over the elements updating the stream function
        for i in range(1,m+1):
            for j in range(1,n+1):
                psitmp[i][j] = 0.25 * (psi[i+1][j]+psi[i-1][j] + \
                                       psi[i][j+1]+psi[i][j-1] - zet[i][j])

        # Loop over the elements updating the vorticity function
        for i in range(1,m+1):
            for j in range(1,n+1):
                zettmp[i][j] =   0.25*(zet[i+1][j] + zet[i-1][j] +     \
                                       zet[i][j+1] + zet[i][j-1]   ) - \
                              re/16.0*((psi[i][j+1]-psi[i][j-1]) *     \
                                       (zet[i+1][j]-zet[i-1][j]) -     \
                                       (psi[i+1][j]-psi[i-1][j]) *     \
                                       (zet[i][j+1]-zet[i][j-1])   )

        # Copy back
        for i in range(1,m+1):
            for j in range(1,n+1):
                psi[i][j] = psitmp[i][j]
                zet[i][j] = zettmp[i][j]

        # Debug output
        if iter%1000 == 0:
            sys.stdout.write("completed iteration {0}\n".format(iter))
