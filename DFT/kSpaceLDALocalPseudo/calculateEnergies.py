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

def ewaldG():
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

def exchangeCorrelationEnergy():
    return 0