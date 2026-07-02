import numpy as np
import matplotlib.pyplot as plt

VAL_RANGE = 100

#Sample of dimension N
#Number of sample L
#Number of dimension of the subspace P
#Number of iteration T per principal component

N = 30
L = 100
P = 20
T = 100

#Verifications of global dimension parameters
assert N < L
assert P < N

np.random.seed(42)





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



def power_method(X):
    """
    Apply the power method.
    """

    def _normalised_vector():
        """
        Generate a random vector of dimension N on the standard normal distribution.
        """

        u = np.random.randn(N, 1)
        u /= np.linalg.norm(u)
        return u


    data = dict()
    Sp = covariance_matrix(X)

    for p in range(1, P+1):
        up_0 = _normalised_vector()
        data[p] = [up_0]

        for t in range(T):
            Sp_up_t = np.dot(Sp, data[p][t])
            up_t1 = Sp_up_t/np.linalg.norm(Sp_up_t)
            data[p].append(up_t1)
        
        Sp = Sp - Sp @ data[p][T] @ data[p][T].T
    
    U = get_U_t(data, T)
    return data, U



def update_rule(X, k1, k2, eps):
    data = dict()
    S = covariance_matrix(X)

    for i in range(1, P+1):
        pass




#getter functions
def get_U_t(data, t):
    U_t = np.hstack([data[p][t] for p in range(1, P+1)])
    return U_t



#display functions
def plot(data, Q, i=None):
    """
    i = None : plotting the total distance between U(T) and Q
    i = 0 : plotting the distances between ui(T) and qi in a sigle plot
    i = -1 : plotting the distances between ui(T) and qi separately
    1 <= i <= P : plotting the evolution of the distance of the i-th principal component
    """

    def _get_norm_unsigned_ui_t_qi(data, i, t, Q):
        U_t = get_U_t(data, t)
        return abs( abs(U_t[:, i-1] @ Q[:, i-1]) - 1 )
    
    def _get_norm_unsigned_ui_t_qi_range(data, i, t0, t1, Q):
        return [_get_norm_unsigned_ui_t_qi(data, i, t, Q) for t in range(t0, t1+1)]

    def _get_norm_unsigned_U_t_QP(data, t, Q):
        return sum([_get_norm_unsigned_ui_t_qi(data, i, t, Q) for i in range(1, P+1)])
    
    def _get_norm_unsigned_U_t_QP_range(data, t0, t1, Q):
        return [_get_norm_unsigned_U_t_QP(data, t, Q) for t in range(t0, t1+1)]


    plt.title(label=f"Evolution of the distance between U = (u1 ... uP) and Q' = (q1 ... qP)")
    plt.xlabel(xlabel="t")
    plt.ylabel(ylabel=f"Distance between U and Q")

    if i == None:
        plt.plot(np.arange(T+1), _get_norm_unsigned_U_t_QP_range(data, 0, T, Q), label="U, Q'")
    elif i == 0:
        cmap = plt.get_cmap('plasma')
        for j in range(1, P+1):
            plt.plot(np.arange(T+1), _get_norm_unsigned_ui_t_qi_range(data, j, 0, T, Q), color=cmap(j/(P)), label=f"u_{j+1}(t), q_{j+1}")
    elif i == -1:
        for a in range(P):
            plot(data, Q, a)
        return
    elif 1 <= i <= P:
        plt.plot(np.arange(T+1), _get_norm_unsigned_ui_t_qi_range(data, i, 0, T, Q), label=f"u_{i}(t), q_{i}")
    else:
        raise KeyError

    plt.axhline(y=0, color='red', linestyle='--', linewidth=0.75, label=f'0')

    plt.legend(fontsize='small', bbox_to_anchor=(1.05, 1), ncol=max(1, P//20+1))
    plt.show()





X = generate_data_matrix()
S = covariance_matrix(X)
Lambda, Q = spectral_decomposition(S)

data, U = power_method(X)

print(U)
print()
print(Q[:, :P])

plot(data, Q)
plot(data, Q, 0)