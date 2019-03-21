#!/usr/bin/env python
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
        print "Usage: cfd.py <scalefactor> <iterations>"
        sys.exit(1)
    
    # Get the systen parameters from the arguments
    scalefactor = int(sys.argv[1])
    niter = int(sys.argv[2])
    
    sys.stdout.write("\n2D CFD Simulation\n")
    sys.stdout.write("=================\n")
    sys.stdout.write("Scale Factor = {0}\n".format(scalefactor))
    sys.stdout.write("  Iterations = {0}\n".format(niter))

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
        psi[m+1][j] = float(w) # psi=(m+1,j)
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
    sys.stdout.write("...finished\n")
    sys.stdout.write("\nCalculation took {0:.5f}s\n\n".format(tend-tstart))

    # Write the output files 
    util.write_data(m, n, scalefactor, psi, "velocity.dat", "colourmap.dat")

    # Write the output visualisation
    plot_flow(psi, "flow.png")

    # Finish nicely
    sys.exit(0)

# Create a plot of the data using matplotlib
def plot_data(psi, outfile):

    # Get the inner dimensions
    (m, n) = psi.shape
    m = m - 2
    n = n - 2

    # Define and zero the velocity arryas
    umod = np.zeros((m, n))
    ux = np.zeros((m, n))
    uy = np.zeros((m, n))

    for i in range(1, m+1):
        for j in range(1, n+1):
            # Compute velocities and magnitude squared
            ux[i-1,j-1] =  (psi[i,j+1] - psi[i,j-1])/2.0
            uy[i-1,j-1] = -(psi[i+1,j] - psi[i-1,j])/2.0
            mvs = ux[i-1,j-1]**2 + uy[i-1,j-1]**2
            # Scale the magnitude
            umod[i-1,j-1] = mvs**0.5
 
    # Plot a heatmap overlayed with velocity streams
    import matplotlib

    # Plot to image file without need for X server
    # This needs to occur before "import pyplot"
    matplotlib.rcParams['font.size'] = 8
    matplotlib.use("Agg")

    # Import the required functions
    from matplotlib import pyplot as plt
    from matplotlib import cm

    fig = plt.figure()

    # Regular grids
    x = np.linspace(0, m-1, m)
    y = np.linspace(0, n-1, n)

    # We want the image to appear in natural (x,y) coordinates,
    # rather than a "matrix". This actually means we must
    # transpose x and y *and* use the "origin" argument in
    # imshow() to turn the whole thing upside down.

    ux = np.transpose(ux)
    uy = np.transpose(uy)
    umod = np.transpose(umod)

    # Line widths are scaled by modulus of velocity
    lw = 3 * umod/umod.max()

    # Create the stream lines denoting the velocities
    plt.streamplot(x, y, ux, uy, color='k', density=1.5, linewidth=lw)

    # Create the heatmap denoting the modulus of the velocities

    plt.imshow(umod, origin = 'lower', interpolation='nearest', cmap=cm.jet)

    # Save the figure to the output PNG file
    fig.savefig(outfile)

# Function to create tidy way to have main method
if __name__ == "__main__":
        main(sys.argv[1:])

