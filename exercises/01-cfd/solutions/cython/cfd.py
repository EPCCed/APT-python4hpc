#!/usr/bin/env python3
#
# CFD Calculation
# ===============
#
# Simulation of flow in a 2D box using the Jacobi algorithm.
#
# Python version - uses numpy and loops
#
# Note this version produces additional graphical output as
# "flow.png" which includes colourmap and stream lines.
#
# EPCC, 2014
#

import sys
import time

# Import numpy
import numpy as np

# Import the local "util.py" methods
import util

# Import the external jacobi function from "jacobi.py"
from jacobi import jacobi

# Import the external matplotlib-based plotting function from "matplot_flow.py"
from matplot_flow import plot_flow

def main(argv):

    # Test we have the correct number of arguments
    if len(argv) < 2:
        sys.stderr.write("Usage: cfd.py <scalefactor> <iterations> \n")
        sys.exit(1)
    
    # Get the system parameters from the arguments
    scalefactor = int(argv[0])
    niter = int(argv[1])
    
    sys.stdout.write("\n2D CFD Simulation\n")
    sys.stdout.write("=================\n")
    sys.stdout.write("Scale factor = {0}\n".format(scalefactor))
    sys.stdout.write("Iterations   = {0}\n".format(niter))

    # Time the initialisation
    tstart = time.time()

    # Set the minimum size parameters
    mbase = 32
    nbase = 32
    bbase = 10
    hbase = 15
    wbase =  5

    # Set the dimensions of the array
    m = mbase*scalefactor
    n = nbase*scalefactor
    
    # Set the parameters for boundary conditions
    b = bbase*scalefactor 
    h = hbase*scalefactor
    w = wbase*scalefactor

    # Define the psi array of dimension [m+2][n+2] and set it to zero
    psi = np.zeros((m+2, n+2))
    
    # Set the boundary conditions on bottom edge
    for i in range(b+1, b+w):
        psi[i][0] = float(i-b)
    for i in range(b+w, m+1):
        psi[i][0] = float(w)

    # Set the boundary conditions on right edge
    for j in range(1, h+1):
        psi[m+1][j] = float(w)
    for j in range(h+1, h+w):
        psi[m+1][j] = float(w-j+h)

    # Write the simulation details
    tend = time.time()
    sys.stdout.write("\nInitialisation took {0:.5f}s\n".format(tend-tstart))
    sys.stdout.write("\nGrid size = {0} x {1}\n".format(m, n))

    # Call the Jacobi iterative loop (and calculate timings)
    sys.stdout.write("\nStarting main Jacobi loop...\n")
    tstart = time.time()
    jacobi(niter, psi)
    tend = time.time()
    sys.stdout.write("\n...finished\n")
    sys.stdout.write("\nCalculation took {0:.5f}s\n\n".format(tend-tstart))
    
    # Write the output files 
    util.write_data(m, n, scalefactor, psi, "velocity.dat", "colourmap.dat")

    # Write the output visualisation
    plot_flow(psi, "flow.png")

    # Finish nicely
    sys.exit(0)


# Function to create tidy way to have main method
if __name__ == "__main__":
        main(sys.argv[1:])

