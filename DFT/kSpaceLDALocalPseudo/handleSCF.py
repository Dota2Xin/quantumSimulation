import numpy as np
import math
from solveSchrodinger import *

def makeSmallGrid(ecutwfc, reciprocalVecs):
    n1=math.ceil(np.sqrt(ecutwfc)/np.linalg.norm(reciprocalVecs[:,0]))+1
    n2=math.ceil(np.sqrt(ecutwfc)/np.linalg.norm(reciprocalVecs[:,1]))+1
    n3=math.ceil(np.sqrt(ecutwfc)/np.linalg.norm(reciprocalVecs[:,2]))+1

    n1Arr=np.linspace(-n1, n1, 2*n1+1)
    n2Arr=np.linspace(-n2, n2, 2*n2+1)
    n3Arr=np.linspace(-n3, n3, 2*n3+1)

    v1 = n1Arr[:, None, None, None] * reciprocalVecs[:, 0]
    v2 = n2Arr[None, :, None, None] * reciprocalVecs[:, 1]
    v3 = n3Arr[None, None, :, None] * reciprocalVecs[:, 2]

    qGridSmall = v1 + v2 + v3

    return qGridSmall, n1Arr, n2Arr, n3Arr

def makeBigGrid(ecutwfc, reciprocalVecs):
    n1 = 2*math.ceil(np.sqrt(ecutwfc) / np.linalg.norm(reciprocalVecs[:, 0])) + 1
    n2 = 2*math.ceil(np.sqrt(ecutwfc) / np.linalg.norm(reciprocalVecs[:, 1])) + 1
    n3 = 2*math.ceil(np.sqrt(ecutwfc) / np.linalg.norm(reciprocalVecs[:, 2])) + 1

    n1Arr = np.linspace(-n1, n1, 2 * n1 + 1)
    n2Arr = np.linspace(-n2, n2, 2 * n2 + 1)
    n3Arr = np.linspace(-n3, n3, 2 * n3 + 1)

    v1 = n1Arr[:, None, None, None] * reciprocalVecs[:, 0]
    v2 = n2Arr[None, :, None, None] * reciprocalVecs[:, 1]
    v3 = n3Arr[None, None, :, None] * reciprocalVecs[:, 2]

    qGridBig = v1 + v2 + v3

    return qGridBig, n1Arr, n2Arr, n3Arr

def padZeros(n1Arr, n2Arr, n3Arr, smallGrid):
    n1=len(n1Arr)
    n2=len(n2Arr)
    n3=len(n3Arr)

    big = (4 * n1 + 1, 4 * n2 + 1, 4 * n3 + 1)
    bigGrid = np.zeros(big, dtype=np.complex64)
    bigGrid[n1:3*n1+1, n2:3*n2+1, n3:3*n3+1]=smallGrid

    return bigGrid

#assume k is given in basis of G vecs?
#for bigger grids need to work with grid you'll get from doing the FFT twice (forward and back)

def calcDensity(psi,n1,n2,n3):
    bigPsi=padZeros(n1,n2,n3, psi)
    bigPsi=np.fft.ifftshift(bigPsi)

    realPsi=np.fft.ifftn(bigPsi)*np.prod(bigPsi.shape)
    realDensity=np.abs(realPsi)**2.0

    density=np.fft.fftn(realDensity)/np.prod(realDensity.shape)
    density=np.fft.fftshift(density)

    return density

def getStartingDensity(bigGrid):
    return np.linalg.norm(bigGrid, axis=-1)

def getOccupations(*args):
    return [0]

def mainSCFLoop(initialConditions):
    ecutwfc=initialConditions['ecutwfc']
    atomicPositions=initialConditions['atomicPositions']
    atomicNumbers=initialConditions['atomicNumbers']
    atomicMasses=initialConditions['atomicMasses']

    nBand=initialConditions['nBand']
    bzSetting=initialConditions['bzSetting']

    if bzSetting=="Grid":
        bzGrid=initialConditions['bzGrid']

    latticeVecs=initialConditions['latticeVecs']
    reciprocalVecs=np.linalg.solve(latticeVecs.T, 2*np.pi*np.eye(3))
    cellVol=np.linalg.det(latticeVecs)

    rC=initialConditions['rC']

    smallGrid, n1,n2,n3=makeSmallGrid(ecutwfc, reciprocalVecs)
    bigGrid=makeBigGrid(ecutwfc, reciprocalVecs)

    occupations=getOccupations(atomicNumbers, nBand)

    #add BZ loop later
    density=getStartingDensity(bigGrid)
    relDiff = 1e5
    tol = 1e-1

    #dict to make things more readable
    solveSchrodingerInputDict={}
    solveSchrodingerInputDict['density']=density
    solveSchrodingerInputDict['smallGrid']=smallGrid
    solveSchrodingerInputDict['bigGrid']=bigGrid
    solveSchrodingerInputDict['atomicPositions']=atomicPositions
    solveSchrodingerInputDict['atomicNumbers']=atomicNumbers
    solveSchrodingerInputDict['rC']=rC
    solveSchrodingerInputDict['cellVol']=cellVol
    solveSchrodingerInputDict['k']=0
    solveSchrodingerInputDict['nBand']=nBand
    #fill out with rest of args as needed

    while relDiff > tol:
        wavefuncs, energies= solveSchrodinger(solveSchrodingerInputDict)
        oldDensity=np.copy(density)
        density=np.zeros_like(oldDensity)
        for i in range(len(occupations)):
            density+=calcDensity(wavefuncs[i], n1,n2,n3)

        relDiff=np.linalg.norm(density-oldDensity)/np.linalg.norm(density)
        relDiff=tol/2
    return density