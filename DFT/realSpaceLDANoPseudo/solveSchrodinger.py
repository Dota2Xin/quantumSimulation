import numpy as np
import scipy
from numba import njit

def setupGrid(nx,ny,nz, bounds):
    x=np.linspace(bounds[0][0], bounds[0][1], nx)
    y=np.linspace(bounds[1][0], bounds[1][1], ny)
    z=np.linspace(bounds[2][0], bounds[2][1], nz)
    xGrid, yGrid, zGrid=np.meshgrid(x,y,z)

    return x,y,z

def solveEquation(x,y,z, density, atomicPositions, fftCoulomb, n):
    kinetic, external, hartree, exchangeCorrelation=getOperators(x,y,z,density, atomicPositions, fftCoulomb)

    hamiltonian=kinetic+external+hartree+exchangeCorrelation
    eigenvectors=np.zeros((n, len(hamiltonian[0])))
    eigenvalues=np.zeros(n)
    tau=0.1
    maxIter=20
    for i in range(n):
        initialVec=np.random.normal(0,0.1, len(hamiltonian[0]))
        initialVec=initialVec/np.linalg.norm(initialVec)
        eigenvectors[i]=imaginaryTimeEvolve(hamiltonian, tau, initialVec, maxIter, eigenvectors[0:i])
        eigenvalues[i]=np.mean((hamiltonian@eigenvectors[i])/eigenvectors[i])
    return eigenvectors, eigenvalues

#make currVec orthogonal to every vector in orthogonal (assuming the orthogonal vectors are normalized
def grahamSchmidt(initialVec, orthogonals):
    currVec=initialVec
    for i in range(len(orthogonals)):
        currVec=currVec-np.dot(currVec, orthogonals[i])*orthogonals[i]
        currVec=currVec/np.linalg.norm(currVec)
    return currVec

def imaginaryTimeEvolve(hamiltonian,tau, initialVec, maxIter, orthogonals):
    count=0
    currVec=initialVec
    dV=1*len(initialVec)
    while(dV/len(initialVec)> 1e-2 and count<maxIter):
        temp=currVec
        currVec=(1-tau*hamiltonian)@currVec
        currVec=grahamSchmidt(currVec, orthogonals)
        dV=np.linalg.norm(currVec-temp)
        count=count+1
    return currVec/np.linalg.norm(currVec)

def getOperators(x,y,z, density, atomicPositions, fftCoulomb):
    kinetic=getKinetic(x,y,z)
    external=getExternalGrid(x,y,z, atomicPositions)
    hartree=getHartree(fftCoulomb, density)
    exchangeCorrelation=getExchangeCorrelation(density)

    return kinetic, external, hartree, exchangeCorrelation

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

@njit
def getExternalPoint(x,y,z,atomicPositions, units):
    potential=0

    for i in range(len(atomicPositions)):
        dist=np.sqrt((atomicPositions[0]-x)**2.0+(atomicPositions[1]-y)**2.0+(atomicPositions[2]-z)**2.0)
        potential+=(-units)*(1/dist)

    return potential

@njit
def getExternalGrid(x,y,z,atomicPositions):
    units=1

    potential=np.zeros(int(len(x)**3))
    #vectorize in a bit
    for i in range(len(x)):
        for j in range(len(y)):
            for k in range(len(z)):
                index=i*int(len(x)**2)+j*int(len(x))+k
                potential[index]=getExternalPoint(x[i], y[k], z[k], atomicPositions, units)

    return potential

#could do a fast way with things precomputed
def getFourierGrid(x,y,z):
    dkx=2*np.pi/(x[-1]-x[0])
    dky=2*np.pi/(y[-1]-y[0])
    dkz=2*np.pi/(y[-1]-y[0])

def getFourierGridSlow(xGrid,yGrid,zGrid):
    inverseDist = 1 / (np.sqrt(xGrid ** 2.0 + yGrid ** 2.0 + zGrid ** 2.0))
    fftCoulomb = np.fft.fftn(inverseDist)
    return fftCoulomb

#for the hartree potential do a FFT on n(r) and on 1/|r| and product them in a convolution then FFT back.
def getHartree(fftCoulomb, density):
    units=1

    fftDensity=np.fft.fftn(density)
    fftHartree=fftCoulomb*fftDensity

    hartree=units*np.real(np.fft.ifftn(fftHartree))
    return np.reshape(hartree, len(hartree)**3.0)

def ec1(rs):
    return -0.0480 + 0.0311 * np.log(rs) - 0.0166 * rs + 0.0020 * rs * np.log(rs)

#-rs/3*(d(ec1)/d(rs))
def dec1(rs):
    return -(0.0311/3.0) +(0.0166/3.0) * rs - (0.0020/3.0)*(rs * np.log(rs)+rs)

def ec2(rs):
    return -0.14213/(1+1.0529*np.sqrt(rs)+0.3334*rs)

#-rs/3*(d(ec2)/d(rs))
def dec2(rs):
    return -(0.14213/3.0)*rs*(0.5*1.0529/np.sqrt(rs)+0.3334)/((1+1.0529*np.sqrt(rs)+0.3334*rs)**2.0)

#SHOULD VERIFY UNITS ON THIS FORMULA WHEN USING
def getExchangeCorrelation(density):
    rs=(3/(4*np.pi*density))**(1.0/3)
    exchange=-(3/4)*(6*density/np.pi)**(1.0/3.0)
    correlationMain=np.piecewise(rs, [rs<1, rs>=1], [ec1, ec2])
    correlationDerivative=np.piecewise(rs, [rs<1, rs>=1], [dec1, dec2])

    return exchange+correlationMain+correlationDerivative