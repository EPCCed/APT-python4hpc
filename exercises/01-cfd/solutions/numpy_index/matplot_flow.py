# Create a plot of the data using matplotlib
def plot_flow(psi, outfile):

    import numpy as np

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
