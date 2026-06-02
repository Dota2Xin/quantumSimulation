import numpy as np
import scipy
from numba import njit
from numba.core.types import np_float16

from blockDavidson import *
import math
import numpy as np
from scipy.signal import fftconvolve

def calcHartree(density, qGrid,cellVol):
    qNorm=np.linalg.norm(qGrid, axis=-1)
    result=4*np.pi*np.divide(density, (qNorm**2.0), out=np.zeros_like(density), where=(qNorm!=0))

    return result

def getPrimalStructureFactor(qGrid, atomicPositions, atomicNumbers, cellVol):
    expArg=np.einsum('ijml,kl->ijmk',qGrid, atomicPositions, dtype=np.complex64, casting='unsafe')
    structureFactor=(atomicNumbers*np.exp(1j*expArg)).sum(axis=-1)

    return structureFactor/cellVol

def getExternalPotentialLocal(integrals, qGrid,atomicPositions, atomicNumbers, rC, cellVol):
    structureFactor=getPrimalStructureFactor(qGrid, atomicPositions, atomicNumbers, cellVol)
    qNorm=np.linalg.norm(qGrid, axis=-1)
    structureFactor=structureFactor*np.exp(-0.5*(rC*qNorm)**2.0)*integrals
    result=-(4*np.pi)*np.divide(structureFactor, (qNorm**2.0), out=np.zeros_like(structureFactor), where=(qNorm!=0))
    return result

#r_s<1
def corr1(x):
    base=-0.0480+0.0311*np.log(x)-0.0116*x+0.0020*x*np.log(x)
    derivative= (-1/3.0)*(0.0311-0.0096*x+0.0020*x*np.log(x))
    return base+derivative

#r_s>1
def corr2(x):
    base=-0.14213/(1+1.0529*np.sqrt(x)+0.3334*x)
    derivative=(-1/3.0)*(0.334*x+0.52645*np.sqrt(x))/((1+1.0529*np.sqrt(x)+0.3334*x)**2.0)
    return base+derivative

def functionalLDA(realDensity):
    #exchange=(-3/4)*((3*realDensity/np.pi)**(1.0/3.0))
    #have to include derivative which gives 1/3+1=4/3 to cancel the -3/4
    exchange=-((3*realDensity/np.pi)**(1.0/3.0))
    rs = (3.0 / (4.0 * np.pi * (realDensity + 1e-12))) ** (1.0 / 3.0)
    correlation=np.piecewise(rs, [rs<1, rs>=1], [corr1, corr2])
    return exchange+correlation

#assumes it has a functional in real-space
def getExchangeCorrelation(densityTemp, coreCorrection,cellVol):
    density=densityTemp+coreCorrection
    NGrid=np.prod(density.shape)
    tempDensity=np.fft.ifftshift(density)
    realDensity=np.fft.ifftn(tempDensity)*NGrid

    exchangeCorrelationReal=functionalLDA(realDensity)
    exchangeCorrelation=np.fft.fftn(exchangeCorrelationReal)/NGrid
    exchangeCorrelation=np.fft.fftshift(exchangeCorrelation)
    return exchangeCorrelation
#qGridBig,density, atomicPositions, atomicNumbers, rC, cellVol
def getPotential(potentialArgs):
    qGridBig=potentialArgs['qGridBig']
    density=potentialArgs['density']
    atomicPositions=potentialArgs['atomicPositions']
    atomicNumbers=potentialArgs['atomicNumbers']
    rC=potentialArgs['rC']
    cellVol=potentialArgs['cellVOl']
    coreCorrection=potentialArgs['coreCorrection']
    localIntegrals=potentialArgs['localIntegrals']
    return getExchangeCorrelation(density,coreCorrection, cellVol)+getExternalPotentialLocal(localIntegrals, qGridBig, atomicPositions, atomicNumbers, rC, cellVol)+calcHartree(density, qGridBig, cellVol)

def actHamiltonianGrid(state, V,qGridSmall, k):
    #outState=np.copy(state)
    outState=fftconvolve(V, state[::-1, ::-1, ::-1], mode='valid')
    #for i in range(len(outState)):
    #    for j in range(len(outState[0])):
    #        for k in range(len(outState[0][0])):
    #            outState[i][j][k]=np.sum(V[i:i+len(state)][j:j+len(state[0])][k:k+len(state[0][0])]*state)
    outState+=0.5*np.linalg.norm((qGridSmall+k), axis=-1)*state
    return outState

def padZeros(n1Arr, n2Arr, n3Arr, smallGrid):
    n1=len(n1Arr)
    d1=int((n1-1)/2)
    n2=len(n2Arr)
    d2 =int((n2 - 1)/2)
    n3=len(n3Arr)
    d3=int((n3-1)/2)

    big = (2 * n1 - 1, 2 * n2 - 1, 2 * n3 - 1)
    bigGrid = np.zeros(big, dtype=np.complex64)
    bigGrid[d1:3*d1+1, d2:3*d2+1, d3:3*d3+1]=smallGrid

    return bigGrid

def actHamiltonianVec(state, hamiltonianArgs):
    V=hamiltonianArgs['V']
    qGridSmall=hamiltonianArgs['qGridSmall']
    k=hamiltonianArgs['k']
    n1=hamiltonianArgs['n1']
    n2=hamiltonianArgs['n2']
    n3=hamiltonianArgs['n3']
    gridShape = qGridSmall.shape[:-1]
    stateWork=state.reshape(gridShape)
    # 2. Potential Energy: G-space -> Real-space -> G-space
    # (Using the same logic as your getExchangeCorrelation)
    stateBig=padZeros(n1,n2,n3, stateWork)
    NGrid = np.prod(np.shape(stateBig))
    #PAD OUT WORKING STATE OUT USING THE PADZEROS FROM HANDLESCF
    stateReal = np.fft.ifftn(np.fft.ifftshift(stateBig))*NGrid

    # Multiply by potential in real space (This is equivalent to G-space convolution)
    #NEED TO FIX THIS BY MAKING THE V WE INPUT REAL
    outReal = V * stateReal

    # Transform back to G-space
    outState = np.fft.fftshift(np.fft.fftn(outReal)) / NGrid
    d1 = int((len(n1) - 1) / 2)
    d2 = int((len(n2) - 1) / 2)
    d3 = int((len(n3) - 1) / 2)
    outState=outState[d1:3*d1+1, d2:3*d2+1, d3:3*d3+1]
    #CUT THE OUTSTATE DOWN TO REMOVE THE PART THAT WAS PADDED WITH ZEROS
    #for i in range(len(outState)):
    #    for j in range(len(outState[0])):
    #        for k in range(len(outState[0][0])):
    #            outState[i][j][k]=np.sum(V[i:i+len(state)][j:j+len(state[0])][k:k+len(state[0][0])]*state)
    outState+=0.5*(np.linalg.norm((qGridSmall+k), axis=-1)**2.0)*stateWork
    return np.reshape(outState, -1)


#solves the Schrodinger equation in our SCF loop given an initial density
#creates all the operators and then calls block-davidson to diagonalize everything.
def solveSchrodinger(inputDict):
    density=inputDict['density']
    smallGrid=inputDict['smallGrid']
    n1=inputDict['n1']
    n2=inputDict['n2']
    n3=inputDict['n3']
    bigGrid=inputDict['bigGrid']
    atomicPositions=inputDict['atomicPositions']
    atomicNumbers=inputDict['atomicNumbers']
    rC=inputDict['rC']
    cellVol=inputDict['cellVol']
    k=inputDict['k']
    l=inputDict['nBand']
    coreCorrection=inputDict['coreCorrection']
    localIntegrals=inputDict['localIntegrals']

    potentialArgs={}
    potentialArgs['qGridBig']=bigGrid
    potentialArgs['density']=density
    potentialArgs['atomicPositions']=atomicPositions
    potentialArgs['atomicNumbers']=atomicNumbers
    potentialArgs['rC']=rC
    potentialArgs['cellVOl']= cellVol
    potentialArgs['coreCorrection']=coreCorrection
    potentialArgs['localIntegrals']=localIntegrals

    VPotential=getPotential(potentialArgs)

    RealV=np.fft.ifftn(np.fft.ifftshift(VPotential)).real*np.prod(np.shape(VPotential))

    hamiltonianArgs={}
    hamiltonianArgs['V']=RealV
    hamiltonianArgs['VFourier']=VPotential
    hamiltonianArgs['qGridSmall']=smallGrid
    hamiltonianArgs['k']=k
    hamiltonianArgs['n1']=n1
    hamiltonianArgs['n2']=n2
    hamiltonianArgs['n3'] = n3

    stateSize=len(smallGrid)*len(smallGrid[0])*len(smallGrid[0][0])
    energies, eigenfuncs=blockDavidson(l, 9*l, stateSize, smallGrid, actHamiltonianVec, hamiltonianArgs)

    return energies, eigenfuncs