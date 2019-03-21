#!/usr/bin/env python
#
# Plot the output from Jacobi CFD calculation
#

# Import the required functions
import numpy as np
import sys
import math

def main(argv):

    # Test we have the correct number of arguments
    if len(argv) < 3:
        sys.stderr.write("Usage: {!s} ".format(argv[0]))
        sys,stderr.write("<velocity.dat> <colourmap.dat> <output.png>")
        sys.exit(1)

    # Input and output files
    velfile = argv[0]
    colfile = argv[1]
    outfile = argv[2]

    # Open the input file
    velinput = open(velfile, "r")
    colinput = open(colfile, "r")

    # Read the dimensions of the simulation
    line = colinput.readline()
    line = line.rstrip()
    tokens = line.split()
    m = int(tokens[0])
    n = int(tokens[1])

    line = velinput.readline()
    line = line.rstrip()
    tokens = line.split()
    mbase = int(tokens[0])
    nbase = int(tokens[1])
    
    # Define and zero the numpy arrays
    colour = np.zeros((m, n))

    # Loop over the fine grid reading the data in the colormap array
    for i in range(0, m):
        for j in range(0, n):
            
            line = colinput.readline()
            line = line.rstrip()
            tokens = line.split()
            i1 = int(tokens[0])
            j1 = int(tokens[1])

            hue = float(tokens[2])
            # Convert from grid coordinates to matrix indices for plotting
            colour[m-1-j1,i1] = hue

    colinput.close()

    xvel = np.zeros((mbase, nbase))
    yvel = np.zeros((mbase, nbase))

    # Loop over the coarse grid reading the data in the velocity array
    for i in range(0, mbase):
        for j in range(0, nbase):
            
            line = velinput.readline()
            line = line.rstrip()
            tokens = line.split()
            i1 = int(tokens[0])
            j1 = int(tokens[1])

            xv = float(tokens[2])
            yv = float(tokens[3])

            # Normalise, taking care of zero values
            modvsq = xv*xv + yv*yv
            if modvsq > 0.0:
                norm = math.sqrt(modvsq)
            else:
                norm = 1.0

            # Convert from grid coordinates to matrix indices for plotting
            # Use coarse  grid indices: i1, j1 refer to fine grid
            xvel[mbase-1-j,i] = xv/norm
            yvel[mbase-1-j,i] = yv/norm

    velinput.close()


    # Plot a heatmap overlayed with velocity streams
    import matplotlib

    # Plot to image file without need for X server
    matplotlib.rcParams['font.size'] = 8
    matplotlib.use("Agg")

    # Import the required functions
    from matplotlib import pyplot as plt
    from matplotlib import cm

    fig = plt.figure()

    # Regular grid - map coarse grid to larger fine grid
    x = np.linspace(0, m-1, mbase)
    y = np.linspace(0, n-1, nbase)

    # Plot arrows showing velocity at each point
    plt.quiver(x, y, xvel, yvel, scale_units="xy", scale=(mbase*1.1)/m)

    # Create the heatmap denoting the modulus of the velocities
    plt.imshow(colour, interpolation='nearest', cmap=cm.jet)

    # Save the figure to the output PNG file
    fig.savefig(outfile)

if __name__ == "__main__":
        main(sys.argv[1:])
