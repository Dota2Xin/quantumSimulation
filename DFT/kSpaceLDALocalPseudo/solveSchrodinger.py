
import math
import numpy as np

def makeGridSmall(ecutwfc, reciprocalVecs):
    n1=math.ceil(np.sqrt(ecutwfc)/np.linalg.norm(reciprocalVecs[:,0]))+1
    n2=math.ceil(np.sqrt(ecutwfc)/np.linalg.norm(reciprocalVecs[:,0]))+1
    n3=math.ceil(np.sqrt(ecutwfc)/np.linalg.norm(reciprocalVecs[:,0]))+1

    n1Arr=np.linspace(-n1, n1, 2*n1+1)
    n2Arr=np.linspace(-n2, n2, 2*n2+1)
    n3Arr=np.linspace(-n3, n3, 2*n3+1)

    grid=np.einsum('i,j,k->ijk', n1Arr, n2Arr, n3Arr)
    return grid, n1Arr, n2Arr, n3Arr

def makeGridBig(n1Arr, n2Arr, n3Arr, smallGrid):
    n1=len(n1Arr)
    n2=len(n2Arr)
    n3=len(n3Arr)

    big = (4 * n1 + 1, 4 * n2 + 1, 4 * n3 + 1)
    bigGrid = np.zeros(big)
    bigGrid[n1:3 * n1 + 1][n2:3 * n2 + 1][n3:3 * n3 + 1] = smallGrid

    return bigGrid


#assume k is given in basis of G vecs?
#for bigger grids need to work with grid you'll get from doing the FFT twice (forward and back)

def calcDensity(psi, n1,n2,n3):
    big=(4*n1+1, 4*n2+1, 4*n3+1)
    bigPsi=np.zeros(big, dtype=complex)
    bigPsi[n1:3*n1+1][n2:3*n2+1][n3:3*n3+1]=psi

    shiftBig=np.fft.ifftshift(bigPsi)

    return 0