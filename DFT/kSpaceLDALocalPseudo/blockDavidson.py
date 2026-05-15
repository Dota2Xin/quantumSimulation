import numpy as np
from scipy.linalg import qr, inv

#ASSUME ACTHAMILTONIAN ACTS ON OUR RESHAPED PSI
def reshapeState(state):
    return np.reshape(state, -1)

#version where we assume S=I, and we have an actHamiltonian that can act on a vector
def blockDavidson(l,m,stateSize,qGridSmall, actHamiltonian, hamiltonianArgs):
    #have to deal with fact vectors are 3D grid, einsum?
    V = np.random.rand(stateSize, l)
    B = V.conj().T@V
    Q = np.linalg.cholesky(B)
    V = V @ np.linalg.inv(Q.T)
    ritz, eigval, res = blockDavidsonIter(V, actHamiltonian, hamiltonianArgs)

    # Bug 1 fix: only check l target residuals
    if np.linalg.norm(res[:, :l]) <= 1e-1:
        return eigval[:l], ritz[:, :l]

    HDiag=getDiagonal(qGridSmall, hamiltonianArgs)
    SDiag=np.ones_like(HDiag)
    denom =HDiag[:, np.newaxis] - SDiag[:, np.newaxis] * eigval[:l]
    tol = 1e-5
    denom = np.where(np.abs(denom) < tol, tol * np.sign(denom + 1e-16), denom)
    T = res[:, :l] / denom
    V = sOrtho(V, ritz, T, m, l)
    #print("Hello")
    temp=np.linalg.norm(res[:, :l])
    ratio=1e5
    #INSTABILITY HERE IT MAY BE APT TO SWITCH TO SOME PERCENTAGE CHANGE LOGIC AS WELL
    #AS OPPOSED TO STRICT NROM WITH SOMETHING LIKE A DIFF
    while ratio >= 0.001:  # Bug 1 fix
        #print(ratio)
        ritz, eigval, res = blockDavidsonIter(V, actHamiltonian, hamiltonianArgs)

        HDiag = getDiagonal(qGridSmall, hamiltonianArgs)
        SDiag=np.ones_like(HDiag)

        denom = HDiag[:, np.newaxis] - SDiag[:, np.newaxis] * eigval[:l]
        tol = 1e-5
        denom = np.where(np.abs(denom) < tol, tol * np.sign(denom + 1e-16), denom)
        T = res[:, :l] / denom  # Bug 2 fix
        V = sOrtho(V, ritz, T, m, l)
        ratio=np.abs(np.linalg.norm(res[:, :l])-temp)/np.linalg.norm(res[:, :l])
        temp=np.linalg.norm(res[:, :l])


    return eigval[:l], ritz[:, :l]

def getDiagonal(qGridSmall, hamiltonianArgs):
    k=hamiltonianArgs['k']
    VPotential=hamiltonianArgs['VFourier']
    HDiag=(0.5 * np.linalg.norm((qGridSmall + k), axis=-1)**2.0).astype(complex)
    half1=len(VPotential)//2
    half2=len(VPotential[0])//2
    half3=len(VPotential[0][0])//2
    #add the 0 argument point of potential.
    HDiag+=VPotential[half1][half2][half3]
    HDiag=reshapeState(HDiag)

    return HDiag

#use fact that we act on vectors to create matmul between Hamiltonian and matrix of column vectors
#we just have to note that vectors are 3D grid now and work with that
def hamiltonianMatmul(A, actHamiltonian, hamiltonianArgs):
    B=np.copy(A).astype(complex)
    for i in range(len(A[0])):
        B[:,i ]=actHamiltonian(A[:, i], hamiltonianArgs)
    return B

#do block davidson iteration but with our hamiltonian action and understanding
def blockDavidsonIter(V, actHamiltonian, hamiltonianArgs):
    W =hamiltonianMatmul(V, actHamiltonian, hamiltonianArgs)
    Hk = V.conj().T @ W

    eigval, eigvec = np.linalg.eigh(Hk)

    ritz = V @ eigvec
    res = ritz@np.diag(eigval) - hamiltonianMatmul(ritz,actHamiltonian, hamiltonianArgs)
    return ritz, eigval, res

def grahamSchmidt(V, T, tol):
    TNew = T.copy()
    keep = []
    for i in range(T.shape[1]):
        currT = TNew[:, i].copy()

        # Two passes for numerical stability (re-orthogonalization)
        for _ in range(2):
            for j in range(V.shape[1]):
                currV = V[:, j]
                currT -= currV * (currV.conj().T @ currT)
            for j in keep:        # only previously accepted columns
                oldT = TNew[:, j]
                currT -= oldT * (oldT.conj().T @currT)

        norm = np.sqrt(currT.conj().T @ currT)
        if norm >= tol:
            keep.append(i)
            TNew[:, i] = currT / norm

    return TNew, keep

def sOrtho(V, ritz, T, m, l):
    tol=1e-8
    TNew,keep=grahamSchmidt(V, T, tol)
    TFilt=TNew[:,keep]

    if (V.shape[1] + TFilt.shape[1] > m):
        # Subspace too big: keep the 'l' best Ritz vectors
        # Note: ritz should already be sorted by your calling function
        VNew = np.concatenate((ritz[:, :l], TFilt), axis=1)
    else:
        # Subspace has room: just add the new directions
        VNew = np.concatenate((V, TFilt), axis=1)

    return VNew

def getExact(H):
    eigval, eigvec=np.linalg.eigh(H)
    return eigval, eigvec