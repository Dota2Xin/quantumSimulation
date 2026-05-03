
import math
import numpy as np
from scipy.signal import fftconvolve

def calcHartree(density, qGrid):
    qNorm=np.linalg.norm(qGrid, axis=-1)
    result=4*np.pi*np.divide(density, (qNorm**2.0), out=np.zeros_like(density), where=(qGrid!=0))

    return result

def getStructureFactor(qGrid, atomicPositions, atomicNumbers):
    expArg=np.einsum('ijl,kl->ijk',qGrid, atomicPositions, dtype=np.complex64)
    structureFactor=(atomicNumbers*np.exp(1j*expArg)).sum(axis=-1)

    return structureFactor

def getExternalPotential(qGrid,atomicPositions, atomicNumbers, rC, cellVol):
    structureFactor=getStructureFactor(qGrid, atomicPositions, atomicNumbers)
    qNorm=np.linalg.norm(qGrid, axis=-1)
    structureFactor=structureFactor*np.exp(-0.5*(rC*qNorm)**2.0)
    result=(4*np.pi/cellVol)*np.divide(structureFactor, (qNorm**2.0), out=np.zeros_like(structureFactor), where=(qGrid!=0))

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
    exchange=(4/3)*(-3/4)*((6*realDensity/np.pi)**(1.0/3.0))
    correlation=np.piecewise(realDensity, [realDensity<1, realDensity>=1], [corr1, corr2])
    return exchange+correlation

#assumes it has a functional in real-space
def getExchangeCorrelation(density):
    tempDensity=np.fft.ifftshift(density)
    realDensity=np.fft.ifft(tempDensity)*np.prod(tempDensity.shape)

    exchangeCorrelationReal=functionalLDA(realDensity)

    exchangeCorrelation=np.fft.fftn(exchangeCorrelationReal)/np.prod(exchangeCorrelationReal.shape)
    exchangeCorrelation=np.fft.fftshift(exchangeCorrelation)

    return exchangeCorrelation

def getPotential(qGridBig,density, atomicPositions, atomicNumbers, rC, cellVol):
    return getExchangeCorrelation(density)+getExternalPotential(qGridBig, atomicPositions, atomicNumbers, rC, cellVol)+calcHartree(density, qGridBig)

def actHamiltonian(state, V,qGridSmall, k):
    outState=np.copy(state)
    outState=fftconvolve(V, state[::-1, ::-1, ::-1], mode='valid')
    #for i in range(len(outState)):
    #    for j in range(len(outState[0])):
    #        for k in range(len(outState[0][0])):
    #            outState[i][j][k]=np.sum(V[i:i+len(state)][j:j+len(state[0])][k:k+len(state[0][0])]*state)
    outState+=0.5*np.linalg.norm((qGridSmall+k), axis=-1)
    return outState