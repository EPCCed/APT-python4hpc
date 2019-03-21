#
# Create files of input data (for plotting) intended for the
# accompanying "plot_flow.py" script. 
#
# Note that the colour map is written at 'full' resolution,
# while the velocity field is written at a resolution which
# depends on the scale factor.

import sys

def write_data(m, n, scale, psi, velfile, colfile):

    # Open the specified files
    velout = open(velfile, "w")
    velout.write("{0} {1}\n".format(int(m/scale), int(n/scale)))
    colout = open(colfile, "w")
    colout.write("{0} {1}\n".format(m, n))

    # Loop over stream function array (excluding boundaries)
    for i in range(1, m+1):
        for j in range(1, n+1):

            # Compute velocities and magnitude
            ux =  (psi[i][j+1] - psi[i][j-1])/2.0
            uy = -(psi[i+1][j] - psi[i-1][j])/2.0
            umod = (ux**2 + uy**2)**0.5

            # We are actually going to output a colour, in which
            # case it is useful to shift values towards a lighter
            # blue (for clarity) via the following kludge...

            hue = umod**0.6
            colout.write("{0:5d} {1:5d} {2:10.5f}\n".format(i-1, j-1, hue))


    # Velocities at (potentially) reduced resolution for vector plot

    for i in range(1, m+1, scale):
        for j in range(1, n+1, scale):

            # Compute velocities
            ux =  (psi[i][j+1] - psi[i][j-1])/2.0
            uy = -(psi[i+1][j] - psi[i-1][j])/2.0

            velout.write("{0:5d} {1:5d} {2:10.5f} {3:10.5f}\n"
                         .format(int((i-1)/scale), int((j-1)/scale), ux, uy))


    velout.close()
    colout.close()
