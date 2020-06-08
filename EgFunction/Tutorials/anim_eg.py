import numpy as np
from mayavi import mlab

#### Eg1: Modificating data in the anim function
#### which has to be a generator (yield)
##
##x, y = np.mgrid[-6:6:.2,-6:6:.2]
###s = mlab.surf(x, y, np.sin((x*x+y*y)/10))
##xp = (.5-np.random.random(20))*12
##yp = (.5-np.random.random(20))*12
##zp = np.sin((xp*xp+yp*yp)/10)
##
##sp = mlab.points3d(xp,yp,zp)
##@mlab.animate
##def anim1():
##    for i in range(100):
##        #s.mlab_source.scalars = np.sin((x*x+y*y+i*np.pi)/10)
##
##        sp.mlab_source.y = yp*(.9)**i
##        sp.mlab_source.x = xp*(.9)**i
##        sp.mlab_source.z = np.sin((xp*xp+yp*yp+i*np.pi)/10)
##        yield
##
##anim1()
##mlab.show()

## Eg2: The animation is computed before and
## the anim function gets the data from indexed arrays
x, y = np.mgrid[-6:6:.2,-6:6:.2]
s = mlab.surf(x, y, np.sin((x*x+y*y)/10))
xp = (.5-np.random.random((100,20)))*12
yp = (.5-np.random.random((100,20)))*12
zp = np.sin((xp*xp+yp*yp)/10)

sp = mlab.points3d(xp[0],yp[0],zp[0], np.ones(zp[0].shape), scale_factor=1)
@mlab.animate
def anim2():
    for i in range(100):
        #s.mlab_source.scalars = np.sin((x*x+y*y+i*np.pi)/10)

        sp.mlab_source.y = yp[i]
        sp.mlab_source.x = xp[i]
        sp.mlab_source.z = zp[i]
        yield

anim2()
mlab.show()
