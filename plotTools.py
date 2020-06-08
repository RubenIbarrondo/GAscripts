##
## Title: Plot Tools
## Author: Rubén Ibarrondo
## Description:
##      These are some tools for plotting
##      the results obtained in PrisionersDilema

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def horizontalBarChart(values,
                       names = None,
                       error = None,
                       xlabel = None,
                       title = None):

    if names:
        if len(values) != len(names):
            raise Exception("Values and names must have the same dimensions.")
    if error:
        if len(values) != len(error):
            raise Exception("Values and error must have the same dimensions.")

    plt.rcdefaults()
    fig, ax = plt.subplots()

    y_pos = np.arange(len(values))

    ax.barh(y_pos, values, xerr=error, align='center')
    ax.set_yticks(y_pos)
    if names:
        ax.set_yticklabels(names)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    
    plt.show()

def surface3D(X, Y, Z):
    ''' Example:
    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(X, Y)
    Z = np.sin(np.sqrt(X**2 + Y**2))
    '''

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.viridis)

    plt.show()
    
