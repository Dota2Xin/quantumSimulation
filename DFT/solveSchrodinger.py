import numpy as np

def actHamiltonian(args,vec):
    return 0

def actOverlap(args, vec):
    return 0

def blockDavidson(args):

    return 0

#a test of the general algorithm on small matrices that we know already
def blockDavidsonTest(H,S, l,m):
    epsilon=0.5
    V=np.eye(len(H))[0:l]
    V=V.transpose()
    ritz, eigval, res= blockDavidsonIterTest(H,S,V)

    if np.linalg.norm(res)<=1e-1:
        sortedEig = np.argsort(eigval)
        return eigval[sortedEig], ritz[:,sortedEig]

    C = np.linalg.inv(np.eye(len(H)) - epsilon*np.eye(len(H)) * H)
    T = C @ res
    V=sOrthoTest(V,S,ritz, T, eigval,m,l)

    while np.linalg.norm(res)>=1e-1:
        print("HI")
        ritz, eigval, res = blockDavidsonIterTest(H, S, V)
        C = np.linalg.inv(np.eye(len(H)) - epsilon*np.eye(len(H)) * H)
        T = C @ res
        V = sOrthoTest(V,S,ritz, T, eigval, m, l)

    sortedEig=np.argsort(eigval)
    return eigval[sortedEig], ritz[:,sortedEig]

def blockDavidsonIterTest(H,S,V):
    W = H @ V
    Hk = np.transpose(V) @ W
    U = S @ V
    Sk = np.transpose(V) @ U

    L = np.linalg.cholesky(Sk)
    Hk2 = (np.linalg.inv(L)) @ Hk @ (np.linalg.inv(np.transpose(L)))

    eigval, eigvec = np.linalg.eig(Hk2)

    realeig = (np.linalg.inv(np.transpose(L))) @ eigvec

    ritz = V @ realeig
    res = eigval * (S @ ritz) - H @ ritz
    return ritz,eigval, res

def sOrthoTest(V,S,ritz, T, eigval,m,l):
    GS=S@T
    GS=np.transpose(V)@GS
    A=(np.transpose(V)@S@V)
    GS=GS/A[:, np.newaxis]
    GS=V@GS
    TNew=T-GS
    norms=np.transpose(TNew)@S@TNew
    norms=np.sqrt(np.max(norms, 0))

    tol=1e-8
    keep=norms>tol
    TFilt=TNew[:, keep]
    norms=norms[keep]
    TFilt=TFilt/norms

    if (len(V[0]) > m - l):
        sortedEig = np.argsort(eigval)
        lowEigs = ritz[:, sortedEig]
        VNew = np.concatenate((lowEigs, TFilt), axis=1)
    else:
        VNew = np.concatenate((V, TFilt), axis=1)

    # now we have to S-orthonormalize V
    B = np.transpose(VNew) @ S @ VNew
    Q = np.linalg.cholesky(B)
    VNew=VNew@(np.linalg.inv(np.transpose(Q)))

    return VNew


def getExact(H,S):
    L = np.linalg.cholesky(S)
    H2 = (np.linalg.inv(L)) @ H @ (np.linalg.inv(np.transpose(L)))

    eigval, eigvec = np.linalg.eig(H2)
    realeig=(np.linalg.inv(np.transpose(L))) @ eigvec

    return eigval, realeig