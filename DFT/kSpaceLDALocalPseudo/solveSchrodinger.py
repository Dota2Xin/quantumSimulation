
import math
import numpy as np

def makeGrids(ecutwfc, reciprocalVecs):
    n1=math.ceil(np.sqrt(ecutwfc)/np.linalg.norm(reciprocalVecs[:,0]))+1
    n2=math.ceil(np.sqrt(ecutwfc)/np.linalg.norm(reciprocalVecs[:,1]))+1
    n3=math.ceil(np.sqrt(ecutwfc)/np.linalg.norm(reciprocalVecs[:,2]))+1

    n1Arr=np.linspace(-n1, n1, 2*n1+1)
    n2Arr=np.linspace(-n2, n2, 2*n2+1)
    n3Arr=np.linspace(-n3, n3, 2*n3+1)

    v1 = n1Arr[:, None, None, None] * reciprocalVecs[:, 0]
    v2 = n2Arr[None, :, None, None] * reciprocalVecs[:, 1]
    v3 = n3Arr[None, None, :, None] * reciprocalVecs[:, 2]

    qGrid = v1 + v2 + v3

    return qGrid, n1Arr, n2Arr, n3Arr

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

def calcHartree(density, qGrid):
    qNorm=np.linalg.norm(qGrid, axis=-1)
    result=4*np.pi*np.divide(density, (qNorm**2.0), out=np.zeros_like(density), where=(qGrid!=0))

    return result

def getStructureFactor(qGrid, atomicPositions, atomicNumbers):
    structureFactor=np.zeros((len(qGrid), len(qGrid[0]), len(qGrid[0][0])), dtype=np.complex64)
    return 0