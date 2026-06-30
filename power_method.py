import numpy as np
import matplotlib.pyplot as plt

ROUND = 3
VAL_RANGE = 100

#Sample of dimension N
#Number of sample L
#Number of dimension of the subspace P

N = 3
L = 5
P = 2

#Verifications of global dimension parameters
assert N < L
assert P < N

np.random.seed(42)



def generate_data_matrix(N, L):
    X = np.random.random((N, L))
    X -= np.mean(X, axis=1, keepdims=True)
    #Verification rows sums to 0
    assert np.allclose(np.sum(X, axis=1), np.zeros(N), atol=1e-12)
    return X

def covariance_matrix(X):
    S = (1/L)*np.dot(X, X.T)
    #Verification shape and symetry
    assert np.shape(S) == (N, N)
    assert np.all(S == S.T)
    return S

def spectral_decomposition(S):
    eigval, Q = np.linalg.eigh(S) 

    reversed_indices = np.argsort(eigval)[::-1]
    eigval = eigval[reversed_indices]

    Lambda = np.diag(eigval)
    Q = Q[:, reversed_indices]

    assert np.allclose(S, Q @ Lambda @ Q.T, atol=1e-12)
    return Lambda, Q


X = generate_data_matrix(N, L)
S = covariance_matrix(X)
Lambda, Q = spectral_decomposition(S)
