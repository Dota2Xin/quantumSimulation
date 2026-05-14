from blockDavidson import *
import math
import numpy as np
from scipy.signal import fftconvolve

def calculateEnergy(density,wavefunctions, atomicPositions, atomicNumbers, rC):
    return 0


def makeBigGridT(Rc, latticeVecs):
    n1 = math.ceil(Rc/ np.linalg.norm(latticeVecs[:, 0])) + 1
    n2 = math.ceil(Rc/ np.linalg.norm(latticeVecs[:, 1])) + 1
    n3 = math.ceil(Rc/ np.linalg.norm(latticeVecs[:, 2])) + 1

    n1Arr = np.linspace(-2*n1, 2*n1, 4 * n1 + 1)
    n2Arr = np.linspace(-2*n2, 2*n2, 4 * n2 + 1)
    n3Arr = np.linspace(-2*n3, 2*n3, 4 * n3 + 1)

    v1 = n1Arr[:, None, None, None] * latticeVecs[:, 0]
    v2 = n2Arr[None, :, None, None] * latticeVecs[:, 1]
    v3 = n3Arr[None, None, :, None] * latticeVecs[:, 2]

    qGridBig = v1 + v2 + v3

    return qGridBig, n1Arr, n2Arr, n3Arr
#gets the T grid and eta we'll use based on our target error
def getEtaTGrid(qGridBig, targetErr, latticeVecs):
    gMax=np.linalg.norm(qGridBig[-1][-1][-1])
    eta=np.sqrt(-(gMax**2.0)/(4*np.log(targetErr)))

    Rc=np.sqrt(np.log(targetErr)/(eta**2.0))
    tGridBig=makeBigGridT(Rc, latticeVecs)

    return eta, tGridBig

def calcEwald(qGridBig, atomicPositions,atomicNumbers, eta, cellVol, tGridBig):
    pre1=np.sum(atomicNumbers**2.0)
    shift=-0.5*pre1*(2*eta/np.sqrt(np.pi)+np.pi/(cellVol*(eta**2.0)))
    gPart=ewaldG(qGridBig, atomicPositions, atomicNumbers, eta, cellVol)
    tPart=ewaldTranslation(atomicNumbers, atomicPositions, eta, tGridBig)
    return tPart+gPart+shift

def ewaldTranslation(atomicNumbers, atomicPositions, eta, tGridBig):
    atomicDiffs = np.broadcast_to(atomicPositions, (len(atomicPositions), len(atomicPositions), 3))
    atomicDiffs = atomicDiffs - np.transpose(atomicDiffs, axes=[1, 0, 2])
    atomicDiffs = np.reshape(atomicDiffs, (len(atomicDiffs) ** 2, 3))

    return 0

def ewaldG(qGridBig, atomicPositions,atomicNumbers, eta, cellVol):
    atomicDiffs=np.broadcast_to(atomicPositions, (len(atomicPositions), len(atomicPositions), 3))
    atomicDiffs=atomicDiffs-np.transpose(atomicDiffs, axes=[1,0,2])
    atomicDiffs=np.reshape(atomicDiffs, (len(atomicDiffs)**2, 3))

    cosArg = np.einsum('ijml,kl->ijmk', qGridBig, atomicDiffs, dtype=np.complex64, casting='unsafe')
    numberProduct=np.outer(atomicNumbers, atomicNumbers)
    numberProduct=np.reshape(numberProduct, -1)
    structureFactor = (np.cos(cosArg)*numberProduct).sum(axis=-1)

    qNorm = np.linalg.norm(qGridBig, axis=-1)

    main = structureFactor * np.exp(-(qNorm**2.0)/(4*(eta**2.0)))
    main = np.divide(main, (qNorm ** 2.0), out=np.zeros_like(qNorm), where=(qNorm != 0))
    return 2*np.pi*np.sum(main)/cellVol

def hartreeEnergy(qGridBig, density):
    main=np.abs(density)*np.abs(density)
    qNorm=np.linalg.norm(qGridBig, axis=-1)
    main=np.divide(main, (qNorm**2.0), out=np.zeros_like(density), where=(qNorm!=0))

    return 2*np.pi*np.sum(main)

def kineticEnergy(wavefuncs, qGridSmall, k):
    T=0
    for i in range(len(wavefuncs)):
        state=wavefuncs[i]
        T+=np.sum(np.abs(state)*0.5 * (np.linalg.norm((qGridSmall + k), axis=-1) ** 2.0) * np.abs(state))
    return T

def externalEnergy(density, qGridBig, structureFactor, rC):
    main = np.abs(density)*structureFactor
    qNorm = np.linalg.norm(qGridBig, axis=-1)
    main=main*np.exp(-0.5*((rC*qNorm)**2.0))
    main = np.divide(main, (qNorm ** 2.0), out=np.zeros_like(density), where=(qNorm != 0))

    return -4*np.pi*main

#r_s<1
def corr1(x):
    base=-0.0480+0.0311*np.log(x)-0.0116*x+0.0020*x*np.log(x)
    return base

#r_s>1
def corr2(x):
    base=-0.14213/(1+1.0529*np.sqrt(x)+0.3334*x)
    return base

def functionalLDA(realDensity):
    exchange=(-3/4)*((3*realDensity/np.pi)**(1.0/3.0))
    rs = (3.0 / (4.0 * np.pi * (realDensity + 1e-12))) ** (1.0 / 3.0)
    correlation=np.piecewise(rs, [rs<1, rs>=1], [corr1, corr2])
    return exchange+correlation

#assumes it has a functional in real-space
def getExchangeCorrelation(density, cellVol):
    NGrid=np.prod(density.shape)
    tempDensity=np.fft.ifftshift(density)
    realDensity=np.fft.ifftn(tempDensity)*NGrid

    exchangeCorrelationReal=functionalLDA(realDensity)
    exchangeCorrelation=np.fft.fftn(exchangeCorrelationReal)/NGrid
    exchangeCorrelation=np.fft.fftshift(exchangeCorrelation)
    return exchangeCorrelation

def exchangeCorrelationEnergy(density, cellVol):
    exchangeCorrelationDensity=getExchangeCorrelation(density, cellVol)
    return np.sum(exchangeCorrelationDensity*density)