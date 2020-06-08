##
## Title: Points and sufaces in mplib
## Author: Rubén Ibarrondo
## Description:
##      Surface + points.
##  Discovering the limitations
##  of matplotlib.

import numpy as np
from bitstring import BitArray

from SurfaceMax import *

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.path as path

from mayavi import mlab
import  moviepy.editor as mpy

# ============================================
#
#               Figure generator
#
# ============================================

x = np.arange(-16, 16, 0.2)
y = np.arange(-16, 16, 0.2)
X, Y = np.meshgrid(x,y)
Z = func2D(X,Y)

repeat = input("Press 1 if you want to reestart the process, press 0 to load previouws data. (Default 0):  ")
if repeat == '1':
    fitness, population, fmax, fmean, points = FindSurfaceMax(generationNumber=1250, pc=0)

    f = open("max_fxy_data.data", 'w')
    print(repr(fitness), file=f)
    print(repr(population), file=f)
    print(repr(fmax), file=f)
    print(repr(fmean), file=f)
    print(repr(points), file=f)
    f.close()
    
else:
    f = open("data/pc0_max_fxy_data.data",'r')
    f = open("data/MAIN_EG_max_fxy_data.data", 'r')
    fitness = eval(f.readline())
    population = eval(f.readline())
    fmax = eval(f.readline())
    fmean = eval(f.readline())
    points = eval(f.readline())
    f.close()

points = np.array(points)

## The process is only shown until
## generation 190 (0-190 or 1-191)
## -- BEGIN OF LENGTH RESETTING --
fmax = fmax[:191]
fmean = fmean[:191]
points = points[:191]

xp = points[:,:,0]
yp = points[:,:,1]
zp = func2D(xp, yp)

## -- END OF LENGTH RESETTING --


''' PLOT THE FITNESS DISTRIBUTION'''
####
#### zp[M] -> histogram
M = 190
plt.grid(axis='y', linestyle='dashed')
plt.rc('axes', axisbelow=True)
plt.hist(zp[M], range=(5.98, 17.80), bins=20)
plt.ylim((0,20))
plt.ylabel("#")
plt.xlabel("$f_i$")
plt.yticks(ticks=[i for i in range(0,22,2)], labels=[i for i in range(0,22,2)])
plt.show()


'''PLOT THE EVOLUTION OF THE FITNESS DISTRIBUTION'''
####
fig = plt.figure()

def update_hist(num):
    plt.cla()
    plt.grid(axis='y', linestyle='dashed')
    plt.ylim((0,20))
    plt.ylabel("#")
    plt.xlabel("$f_i$")
    plt.yticks(ticks=[i for i in range(0,22,2)], labels=[i for i in range(0,22,2)])

    plt.hist(zp[num], range=(5.98, 17.80), bins=20, color='tab:blue')

hist = plt.hist(zp[0], range=(5.98, 17.80), bins=20, color='tab:blue')

anim = animation.FuncAnimation(fig, update_hist, frames=191, interval=50 )
#anim.save("3_evol_fitnesHist.gif")
plt.show()


'''PLOT THE MAX AND MEAN FITNES'''
####
#### fmax[:M] and fmean[:M]-> histogram
M = 191
plt.grid(axis='y', linestyle='dashed')
plt.rc('axes', axisbelow=True)

plt.plot(fmax[:M], color='tab:blue')
plt.plot(fmean[:M], color='tab:orange')

plt.ylim((5.98,17.80))
plt.ylabel("$f_i$")
plt.xlabel("generation")
#plt.yticks(ticks=[i for i in range(0,22,2)], labels=[i for i in range(0,22,2)])
plt.show()


'''PLOT THE EVOLUTION MAX AND MEAN FITNES'''
####
##fig = plt.figure()
##
##def update_f(num):
##    plt.grid(axis='y', linestyle='dashed')
##    plt.rc('axes', axisbelow=True)
##
##    plt.plot(fmax[:num], color='tab:blue')
##    plt.plot(fmean[:num], color='tab:orange')
##
##    plt.ylim((5.98,17.80))
##    plt.ylabel("$f_i$")
##    plt.xlabel("generation")
##
##anim = animation.FuncAnimation(fig, update_f, frames=191, interval=50 )
###anim.save("3_evol_fitnesMaxMean.gif")
##plt.show()

'''PLOT A GIVEN STAGE'''
##
##  N = 0   # INITIAL
##  N = 190 # FINAL
N = 190
mlab.mesh(X,Y,Z)
sp = mlab.points3d(xp[N],yp[N],zp[N], np.ones(zp[0].shape), scale_factor=1)
mlab.show()

'''ANIMATE THE FIGURE WITH MAYAVI'''
##
##@mlab.animate(delay=700, support_movie=True)
##def anim():
##    for i in range(len(xp)):
##        #s.mlab_source.scalars = np.sin((x*x+y*y+i*np.pi)/10)
##
##        sp.mlab_source.y = yp[i]
##        sp.mlab_source.x = xp[i]
##        sp.mlab_source.z = zp[i]
##        yield
##
##anim()
##mlab.show()


'''ANIMATE THE FIGURE WITH MOVIEPY, WRITE AN ANIMATED GIF'''
##
##fig_myv = mlab.figure(size=(700,700), bgcolor=(1,1,1))
##
##duration = 13 # in seconds
##
##def make_frame(t):
##    mlab.clf() # clear the figure (to reset the colors)
##    mlab.mesh(X,Y,Z, figure=fig_myv)
##    i = int( t/duration * 260)
##    if i >= len(xp):
##        i = len(xp)-1
##    mlab.points3d(xp[i],yp[i],zp[i], np.ones(zp[0].shape), scale_factor=1)
##    f = mlab.gcf() # And this. WORKS. 
##    f.scene._lift() # I found this in the net. WORKS
##    s= mlab.screenshot(fig_myv)#antialiased=True) #There is a problem (BEFORE) here, can add transparency
##    return s
##
##animation = mpy.VideoClip(make_frame, duration=duration)
##animation.write_gif("3_surface_GA_proces.gif", fps=20, fuzz=0) ## Stopped in frame 191
