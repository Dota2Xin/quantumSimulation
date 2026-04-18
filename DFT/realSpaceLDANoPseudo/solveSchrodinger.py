import numpy as np
import scipy

def setupGrid(nx,ny,nz, bounds):
    x=np.linspace(bounds[0][0], bounds[0][1], nx)
    y=np.linspace(bounds[1][0], bounds[1][1], ny)
    z=np.linspace(bounds[2][0], bounds[2][1], nz)

    return x,y,z

def getOperators(x,y,z, density):
    return 0

def getKinetic(x,y,z):
    nx=len(x)
    ny=len(y)
    nz=len(z)
    dx=(x[-1]-x[0])/nx
    dy=(y[-1]-y[0])/ny
    dz=(z[-1]-z[0])/nz
    units=1 #replace with hbar^2/2m later

    toeplitzBaseX=np.zeros(nx)
    toeplitzBaseX[1]=1.0
    laplaceX=(units/(dx**2.0))*(-2.0*np.eye(nx)+ scipy.linalg.toeplitz(toeplitzBaseX))

    toeplitzBaseY = np.zeros(ny)
    toeplitzBaseY[1] = 1.0
    laplaceY=(units/(dy**2.0))*(-2.0*np.eye(ny)+ scipy.linalg.toeplitz(toeplitzBaseY))

    toeplitzBaseZ = np.zeros(nz)
    toeplitzBaseZ[1] = 1.0
    laplaceZ=(units/(dz**2.0))*(-2.0*np.eye(nz)+ scipy.linalg.toeplitz(toeplitzBaseZ))

    # x otimes y otimes z for consistency
    laplaceFull=np.kron(laplaceX,np.kron(np.eye(ny), np.eye(nz)))+np.kron(np.eye(nx), np.kron(laplaceY, np.eye(nz)))+np.kron(np.eye(nx), np.kron(np.eye(ny), laplaceZ))
    return laplaceFull

def getExternal(x,y,z,density, atomicPositions):
    sum=0
    #vectorize in a bit
    for i in range(len(x)):
        for j in range(len(y)):
            for k in range(len(z)):
                sum+=1

    return 1