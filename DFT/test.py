from blockDavidsonTestWorking import *
import numpy as np
import math

def main():
    a=np.asarray([[[0,1,2],[3,4,5],[6,7,8]],[[0,1,2],[3,4,5],[6,7,8]],[[0,1,2],[3,4,5],[6,7,8]]])
    print(a)
    print(np.reshape(a, (len(a)**2,3)))
    #l=5
    #m=20
    #N=100
    #H=genPSD(N,5)
    #S=genPSD(N,3)
    #H,S=genTestMatrix(N)
    #daveVal, daveEig=blockDavidsonTest(H,S,l,m)
    #realVal, realEig=getExact(H,S)

    #print(daveVal)
    #print(realVal)

    return 0
def genTestMatrix(N):
    # Strong diagonal + small random noise
    H = np.diag(np.arange(1, N + 1) * 10.0)
    noise = np.random.rand(N, N)
    H += (noise + noise.T)
    S = np.eye(N) + 0.01 * np.random.rand(N, N)
    S = (S + S.T) # Ensure symmetric
    return H, S
def genPSD(N, k):
    A = np.random.rand(N, N)
    A = 0.5 * (A + A.T) + np.diag(np.arange(N) * k) # Strong diagonal
    return A
if __name__=="__main__":
    main()