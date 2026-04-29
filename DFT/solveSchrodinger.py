import numpy as np
from scipy.linalg import qr, inv

def actHamiltonian(args,vec):
    return 0

def actOverlap(args, vec):
    return 0

def blockDavidson(args):

    return 0

#a test of the general algorithm on small matrices that we know already
def blockDavidsonTest(H,S, l,m):
    epsilon=0
    V = np.random.rand(len(H), l)
    # But we MUST S-normalize it before the first iteration
    B = V.T @ S @ V
    Q = np.linalg.cholesky(B)
    V = V @ np.linalg.inv(Q.T)
    ritz, eigval, res= blockDavidsonIterTest(H,S,V)

    if np.linalg.norm(res)<=1e-1:
        sortedEig = np.argsort(eigval)
        return eigval[sortedEig], ritz[:,sortedEig]

    #C = np.linalg.inv(np.eye(len(H)) - epsilon*np.eye(len(H)) * H)
    h_diag = np.diag(H)
    s_diag = np.diag(S)
    denom = h_diag[:, np.newaxis] - s_diag[:, np.newaxis] * eigval
    tol = 1e-5
    denom = np.where(np.abs(denom) < tol, tol * np.sign(denom + 1e-16), denom)
    T = res/denom
    V=sOrthoTest(V,S,ritz, T, eigval,m,l)

    while np.linalg.norm(res)>=1e-1:
        print("HI")
        print(np.linalg.norm(res))
        ritz, eigval, res = blockDavidsonIterTest(H, S, V)
        #C = np.linalg.inv(np.eye(len(H)) - epsilon*np.eye(len(H)) * H)
        #T = C @ res
        h_diag = np.diag(H)
        s_diag = np.diag(S)
        denom = h_diag[:, np.newaxis] - s_diag[:, np.newaxis] * eigval
        tol = 1e-5
        denom = np.where(np.abs(denom) < tol, tol * np.sign(denom + 1e-16), denom)
        T = res / denom
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

    idx = np.argsort(eigval)
    eigval = eigval[idx]
    eigvec = eigvec[:, idx]

    realeig = (np.linalg.inv(np.transpose(L))) @ eigvec

    ritz = V @ realeig
    res = (S @ ritz) @ np.diag(eigval) - H @ ritz
    return ritz,eigval, res

def grahamSchmidt(V,S,T,tol):
    TNew=T.copy()
    keep=[]
    for i in range(len(T[0])):
        currT=TNew[:, i]
        for j in range(len(V[0])):
            currV=V[:,j]
            dij=currV.T@S@currT
            currT-=currV*dij

        for j in range(i):
            oldT=TNew[:,j]
            dij=oldT.T@S@currT
            currT-=oldT*dij

        norm=np.sqrt(currT.T@S@currT)
        if norm>=tol:
            keep.append(i)
            currT = currT / norm
            TNew[:, i] = currT
        else:
            TNew[:,i]=0
    return TNew, keep

'''
def sOrthoTest(V,S,ritz, T, eigval,m,l):
    tol=1e-8
    TNew, keep =grahamSchmidt(V,S,T, tol)

    TFilt=TNew[:, keep]

    if (len(V[0]) > m - l):
        sortedEig = np.argsort(eigval)[0:l]
        lowEigs = ritz[:, sortedEig]
        VNew = np.concatenate((lowEigs, TFilt), axis=1)
    else:
        VNew = np.concatenate((V, TFilt), axis=1)

    # now we have to S-orthonormalize V
    #B = np.transpose(VNew) @ S @ VNew
    #B+=np.eye(len(B))*(1e-13)
    #Q = np.linalg.cholesky(B)
    #VNew=VNew@(np.linalg.inv(np.transpose(Q)))
    h = VNew.T @ (S @ VNew)
    t_orthogonal = VNew - VNew @ h

    # 2. Normalize with respect to S-norm
    # norm = sqrt(t_orth^T S t_orth)
    norm = np.sqrt(t_orthogonal.T @ S @ t_orthogonal)

    return VNew/norm
'''


def sOrthoTest(V, S, ritz, T, eigval, m, l):
    tol = 1e-8
    # 1. Standardize T using your GS function to remove noise
    TNew, keep = grahamSchmidt(V, S, T, tol)
    TFilt = TNew[:, keep]

    # 2. Manage the Subspace (Thick Restart)
    if (V.shape[1] + TFilt.shape[1] > m):
        # Subspace too big: keep the 'l' best Ritz vectors
        # Note: ritz should already be sorted by your calling function
        VNew = np.concatenate((ritz[:, :l], TFilt), axis=1)
    else:
        # Subspace has room: just add the new directions
        VNew = np.concatenate((V, TFilt), axis=1)

    # 3. Robust S-Orthonormalization via QR
    # We want V_final such that V_final.T @ S @ V_final = I
    # Step A: Get the Cholesky of S (S = L @ L.T)
    L = np.linalg.cholesky(S)

    # Step B: Transform V into the space where S is the identity
    A = L.T @ VNew

    # Step C: Perform QR on the transformed vectors
    # A = Q @ R, where Q.T @ Q = I
    Q_orth, R_upper = qr(A, mode='economic')

    # Step D: Back-transform to the original space
    # V_final = VNew @ inv(R)
    V_final = VNew @ inv(R_upper)

    return V_final

def getExact(H,S):
    L = np.linalg.cholesky(S)
    H2 = (np.linalg.inv(L)) @ H @ (np.linalg.inv(np.transpose(L)))

    eigval, eigvec = np.linalg.eig(H2)
    realeig=(np.linalg.inv(np.transpose(L))) @ eigvec

    return eigval, realeig