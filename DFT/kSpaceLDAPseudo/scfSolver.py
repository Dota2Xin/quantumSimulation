import numpy as np
import math
from handlePseudopotential import *
from solveSchrodinger import *
from calculateEnergies import *

#didn't think I'd need this one lol
def makeTable():
    elementDict={"H":1, "He":2, "Li":3, "Be":4, "B":5, "C":6, "N":7, "O":8,  "F":9, "Ne":10, "Na":11, "Mg":12,
                 "Al":13, "Si":14, "P":15, "S":16, "Cl":17, "Ar":18, "K": 19, "Ca":20, "Sc":21, "Ti":22, "V":23,
                 "Cr":24, "Mn":25, "Fe":26, "Co":27, "Ni":28, "Cu":29, "Zn":30, "Ga":31, "Ge":32, "As":33, "Se":34, "Br":35,
                 "Kr":36, "Rb":37, "Sr":38, "Y":39, "Zr":40, "Nb":41, "Mo":42, "Tc":43, "Ru":44, "Rh":45, "Pd":46, "Ag":47,
                 "Cd": 48, "In": 49, "Sn":50, "Sb":51, "Te":52, "I":53, "Xe":54, "Cs":55, "Ba":56, "La":57, "Ce":58, "Pr":59,
                 "Nd":60, "Pm":61, "Sm":62, "Eu":63, "Gd":64, "Tb":65, "Dy":66, "Ho":67, "Er":68, "Tm":69, "Yb":70,
                 "Lu":71, "Hf":72, "Ta":73, "W":74, "Re":75, "Os":76, "Ir":77, "Pt":78, "Au":79, "Hg":80, "Tl":81,
                 "Pb":82, "Bi":83, "Po":84, "At":85, "Rn":86, "Fr":87, "Ra":88, "Ac":89, "Th":90, "Pa":91, "U":92,
                 "Np":93, "Pu":94, "Am":95}
    return elementDict

def swapTable(table):
    newTable={}
    for key in table:
        newTable[table[key]]=key
    return newTable

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

#calculates all the integrals we need to do for our pseudo-potential at the start of the simulation.
def getIntegrals(integralArgs):
    qGridBig=integralArgs['qGridBig']
    qGridSmall=integralArgs['qGridSmall']
    r=integralArgs['r']
    rab=integralArgs['rab']
    rC=integralArgs['rC']
    Z=integralArgs['Z']
    localPseudo=integralArgs['localPseudo']
    coreDensity=integralArgs['coreDensity']
    projectors=integralArgs['projectors']
    angularMomenta=integralArgs['angularMomenta']

    qNormBig=np.linalg.norm(qGridBig, axis=-1)
    qNormSmall=np.linalg.norm(qGridSmall, axis=-1)

    qMaxBig=np.max(qNormBig)
    qMaxSmall=np.max(qNormSmall)

    bigQ=np.linspace(0, qMaxBig, 100)
    smallQ=np.linspace(0,qMaxSmall,100)

    # interpolation of integrals in local pseudo-potential
    localIntegralCalc=localIntegral(r,rab, localPseudo, gaussianLocal, bigQ, rC, Z)
    localInterp=splineInterpolate(bigQ, localIntegralCalc)
    localPart=fillInterpolation(localInterp, qNormBig)

    #interpolation of integrals in core correction density
    coreIntegralCalc=coreDensityIntegral(coreDensity,r,rab,bigQ)
    coreCorrectionInterp=splineInterpolate(bigQ, coreIntegralCalc)
    coreCorrection=fillInterpolation(coreCorrectionInterp, qNormBig)

    #interpolation of integrals in our projector
    projectorResult=[]
    for i in range(len(projectors)):
        l=angularMomenta[i]
        projector=projectors[i]
        currIntegral=projectorIntegral(r,rab, projector, smallQ, l)
        projectorInterp=splineInterpolate(smallQ, currIntegral)
        projectorResult.append(fillInterpolation(projectorInterp, qNormSmall))
    return localPart, coreCorrection, np.asarray(projectorResult)

def makeBeta(projectorIntegrals, qGridSmall, angularMomenta, atomicNumbersEff):
    #we want to make the projectors big efficiently
    #following this we want to scale every term by the appropriate spherical harmonic factor
    #and factor of our imaginary number.
    #do we do the m-scaling earlier?
    #if we can do the index wrapping easily then unrolling is certainly the way to go here.
    #otherwise the final arrays will be inhomogenous and the cache will be annihilated.
    return 0

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


    #pseudopotential handling
    elementToNumber=makeTable()
    numberToElement=swapTable(elementToNumber)

    localIntegrals=[]
    coreIntegrals=[]
    projectorIntegrals=[]

    for i in range(len(atomicNumbers)):
        num=atomicNumbers[i]
        element=numberToElement[num]
        root, metadata, radialGrid, rab = getPseudo(element)
        Z=getZ(root)
        coreDensity = getCoreDensity(root)
        localPseudo = getLocalPart(root)
        projectors, angularMomenta, cutoffs, D=getProjectors(root)

        integralArgs={}
        integralArgs['qGridBig']=bigGrid
        integralArgs['qGridSmall']=smallGrid
        integralArgs['rab']=rab
        integralArgs['r']=radialGrid
        #THIS IS WRONG NEED TO ACCOUNT FOR FACT THAT WE ONLY NEED VALENCE
        integralArgs['Z']=Z
        integralArgs['localPseudo']=localPseudo
        integralArgs['coreDensity']=coreDensity
        integralArgs['projectors']=projectors
        integralArgs['angularMomenta']=angularMomenta

        localIntegral, coreIntegral, projectorIntegral=getIntegrals(integralArgs)

        localIntegrals.append(localIntegral)
        coreIntegrals.append(coreIntegral)
        projectorIntegrals.append(projectorIntegral)

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
    solveSchrodingerInputDict['localIntegrals']=localIntegrals[0]
    solveSchrodingerInputDict['coreCorrection']=coreIntegrals[0]
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