from blockDavidson import *
import math
import numpy as np
from scipy.signal import fftconvolve

def calculateEnergy(density,wavefunctions, atomicPositions, atomicNumbers, rC):
    return 0

def calcEwald():
    return 0

def ewaldTranslation():
    return 0

def ewaldG(qGridBig, atomicPositions, eta, cellVol):
    #make grid over pairs of atomic positions and then flatten to get position arrays
    #can then just repeat structure factor caculation with the flattened grid replacing atomicPositions
    #in the structure factor from solveSchrodinger.
    return 0

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