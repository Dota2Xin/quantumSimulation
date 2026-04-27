from solveSchrodinger import *

def main():
    l=5
    m=20
    H=np.eye(100)
    S=np.eye(100)
    daveVal, daveEig=blockDavidsonTest(H,S,l,m)
    realVal, realEig=getExact(H,S)

    print(daveVal)
    print(realVal)

    return 0

if __name__=="__main__":
    main()