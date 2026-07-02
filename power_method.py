import numpy as np
import matplotlib.pyplot as plt

ROUND = 3
VAL_RANGE = 100

#Sample of dimension N
#Number of sample L
#Number of dimension of the subspace P
#Number of iteration T per principal component

N = 3
L = 5
P = 2
T = 100

#Verifications of global dimension parameters
assert N < L
assert P < N

#np.random.seed(42)





def generate_data_matrix():
    """
    Generates randomly an instance of data X for the problem. The rows of the matrix sums to 1.
    """
    X = np.random.random((N, L))
    X -= np.mean(X, axis=1, keepdims=True)

    #Verification rows sums to 0
    assert np.allclose(np.sum(X, axis=1), np.zeros(N), atol=1e-12)
    return X

def covariance_matrix(X):
    """
    Create the covarance matrix S associated to the data matrix X.
    """
    S = (1/L)*np.dot(X, X.T)

    #Verification shape and symetry
    assert np.shape(S) == (N, N)
    assert np.all(S == S.T)
    return S

def spectral_decomposition(S):
    """
    Finds the spectral decomposition of the covariance matrix S = Q @ Lambda @ Q.T.
    """
    eigval, Q = np.linalg.eigh(S) 

    reversed_indices = np.argsort(eigval)[::-1]
    eigval = eigval[reversed_indices]

    Lambda = np.diag(eigval)
    Q = Q[:, reversed_indices]

    #Verification of the spectral decomposition
    assert np.allclose(S, Q @ Lambda @ Q.T, atol=1e-12)
    return Lambda, Q



def normalised_vector():
    """
    Generate a random vector of dimension N on the standard normal distribution.
    """
    u = np.random.randn(N, 1)
    u /= np.linalg.norm(u)
    return u

def power_method(X):
    """
    Apply the power method.
    """
    data = dict()
    Sp = covariance_matrix(X)
    _, Q = spectral_decomposition(Sp)

    for p in range(1, P+1):
        up_0 = normalised_vector()
        data[p] = [up_0]

        for t in range(T):
            Sp_up_t = np.dot(Sp, data[p][t])
            up_t1 = Sp_up_t/np.linalg.norm(Sp_up_t)
            data[p].append(up_t1)
        
        Sp = Sp - Sp @ data[p][T] @ data[p][T].T
    
    U = np.hstack([data[p][T] for p in data])

    #Vérification that the vectors u correspond to the columns of Q such as S = Q @ Lambda @ Q.T
    for p in range(P):
        assert np.isclose(abs(U[:, p] @ Q[:, p]), 1, atol=1e-12)

    return data, U

def update_rule(X, k1, k2, eps):
    data = dict()
    S = covariance_matrix(X)

    for i in range(1, P+1):
        pass





def get_u_t(data, t):
    u_t = [data[p][t] for p in range(1, P+1)]
    return u_t

def plot(data):
    plt.title(label=f'Evolution of x_i(t)')
    plt.xlabel(xlabel='t')
    plt.ylabel(ylabel=f'x_i(t)')

    plt.show()






X = generate_data_matrix()
S = covariance_matrix(X)
Lambda, Q = spectral_decomposition(S)

data, U = power_method(X)

print(U)
print()
print(Q[:, :P])