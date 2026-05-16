import numpy as np
import math
from solveSchrodinger import *
from calculateEnergies import *

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
    n1 = math.ceil(np.sqrt(ecutwfc) / np.linalg.norm(reciprocalVecs[:, 0])) + 1
    n2 = math.ceil(np.sqrt(ecutwfc) / np.linalg.norm(reciprocalVecs[:, 1])) + 1
    n3 = math.ceil(np.sqrt(ecutwfc) / np.linalg.norm(reciprocalVecs[:, 2])) + 1

    n1Arr = np.linspace(-2*n1, 2*n1, 4 * n1 + 1)
    n2Arr = np.linspace(-2*n2, 2*n2, 4 * n2 + 1)
    n3Arr = np.linspace(-2*n3, 2*n3, 4 * n3 + 1)

    v1 = n1Arr[:, None, None, None] * reciprocalVecs[:, 0]
    v2 = n2Arr[None, :, None, None] * reciprocalVecs[:, 1]
    v3 = n3Arr[None, None, :, None] * reciprocalVecs[:, 2]

    qGridBig = v1 + v2 + v3

    return qGridBig, n1Arr, n2Arr, n3Arr

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

#assume k is given in basis of G vecs?
#for bigger grids need to work with grid you'll get from doing the FFT twice (forward and back)

def calcDensity(psi,n1,n2,n3, cellVol):
    bigPsi=padZeros(n1,n2,n3, psi)
    bigPsi=np.fft.ifftshift(bigPsi)
    nGrid=np.prod(bigPsi.shape)

    realPsi=(np.fft.ifftn(bigPsi)*nGrid/np.sqrt(cellVol))
    realDensity=np.abs(realPsi)**2.0

    density=np.fft.fftn(realDensity)/nGrid
    density=np.fft.fftshift(density)

    return density

def getMeanDensity(density):
    s = density.shape
    return density[s[0]//2, s[1]//2, s[2]//2]

def getStartingDensity(atomicNumbers, cellVol, bigGrid, n1, n2, n3):
    total1=len(n1)
    total2=len(n2)
    total3=len(n3)

    half1=total1//2
    half2=total2//2
    half3=total3//2

    density=np.zeros((len(bigGrid), len(bigGrid[0]), len(bigGrid[0][0])), dtype=np.complex64)

    density[half1][half2][half3]=np.sum(atomicNumbers)/cellVol
    return density

def getOccupations(nBand):
    return nBand

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
    tolErr=initialConditions['tol']

    smallGrid, n1,n2,n3=makeSmallGrid(ecutwfc, reciprocalVecs)
    bigGrid, n1Big, n2Big, n3Big=makeBigGrid(ecutwfc, reciprocalVecs)

    occupations=getOccupations(nBand)

    #add BZ loop later
    density=getStartingDensity(atomicNumbers,cellVol, bigGrid, n1Big, n2Big, n3Big)
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
    solveSchrodingerInputDict['n1']=n1
    solveSchrodingerInputDict['n2']=n2
    solveSchrodingerInputDict['n3']=n3
    #fill out with rest of args as needed

    energyInputDict={}
    energyInputDict['density']=density
    energyInputDict['atomicPositions']=atomicPositions
    energyInputDict['atomicNumbers']=atomicNumbers
    energyInputDict['qGridBig']=bigGrid
    energyInputDict['qGridSmall']=smallGrid
    energyInputDict['latticeVecs']=latticeVecs
    energyInputDict['rC']=rC
    energyInputDict['tol']=tolErr
    energyInputDict['cellVol']=cellVol
    energyInputDict['kGrid']=[[0]]
    prevEnergy=100

    alpha=0.3

    while relDiff > tol:
        energies, wavefuncs= solveSchrodinger(solveSchrodingerInputDict)
        oldDensity=np.copy(density)
        density=np.zeros_like(oldDensity)
        for i in range(occupations):
            density+=2*calcDensity(wavefuncs[i], n1,n2,n3, cellVol)
        density=(1-alpha)*oldDensity+alpha*density
        #print("CURR SOMETHING OR OTHER:")
        #print(f"Mean Density Data: {getMeanDensity(density)}")
        #print(f"Mean Density True:{np.sum(atomicNumbers)/cellVol}")
        solveSchrodingerInputDict['density'] = density
        energyInputDict['density']=density
        energyInputDict['wavefunctions']=wavefuncs

        currEnergy=calculateEnergy(energyInputDict)
        relDiff=np.abs(currEnergy-prevEnergy)/np.abs(currEnergy)
        prevEnergy=currEnergy


        #print(currEnergy)
        #print(relDiff)
        #relDiff=np.linalg.norm(density-oldDensity)/np.linalg.norm(density)

    return density, currEnergy