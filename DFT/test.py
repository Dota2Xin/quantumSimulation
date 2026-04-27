from solveSchrodinger import *

def main():
    l=5
    m=20
    N=100
    H=genPSD(N,5)
    S=genPSD(N,3)
    daveVal, daveEig=blockDavidsonTest(H,S,l,m)
    realVal, realEig=getExact(H,S)

    print(daveVal)
    print(realVal)

    return 0

def genPSD(N, k):
    Q,_=np.linalg.qr(k*np.random.rand(N,N)+0.1)
    diag=k*np.random.rand(N)+0.1
    A=Q@np.diag(diag)@np.transpose(Q)
    return A

if __name__=="__main__":
    main()