import numpy as np
import matplotlib.pyplot as plt



#np.random.seed(42)
VAL_RANGE = 100

#Sample of dimension N
#Number of sample L
#Number of dimension of the subspace P
#Number of iteration T per principal component
N = 10
L = 15
P = 6
T = 50

#Verifications of global dimension parameters
assert N < L, "global parameters aren't set correctly."
assert P < N, "global parameters aren't set correctly."



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



def generate_initial_vectors():
    """
    Generate P normalised vectors of dimension N.
    """

    def _normalised_vector():
        """
        Generate a random vector of dimension N on the standard normal distribution.
        """
        u = np.random.randn(N, 1)
        u /= np.linalg.norm(u)
        return u
    
    
    return [_normalised_vector() for _ in range(P)]



def power_method(X, initial_vectors):
    """
    Apply the power method.
    """

    data = {p: [initial_vectors[p-1]] for p in range(1, P+1)}
    Sp = covariance_matrix(X)

    for p in range(1, P+1):

        for t in range(T):

            Sp_up_t = np.dot(Sp, data[p][-1])
            up_t1 = Sp_up_t/np.linalg.norm(Sp_up_t)
            data[p].append(up_t1)
            
            for p0 in range(1, P+1):
                if p0 != p:
                    data[p0].append(data[p0][-1])
        
        Sp = Sp - Sp @ data[p][-1] @ data[p][-1].T
    
    U = get_U_t(data, len(data[1])-1)
    return data, U



def update_rule(X, initial_vectors, k1, k2, eps):
    """
    Apply the update rule proposed in the paper.
    """

    data = {p: [initial_vectors[p-1]] for p in range(1, P+1)}
    S = covariance_matrix(X)
    
    for t in range(T):
        for i in range(1, P+1):
            ui_t1 = (-k1 * sum([data[j][t] @ data[j][t].T @ data[i][t] for j in range(1, i+1)]))
            ui_t1 = ui_t1 + k2 * (S @ data[i][t])
            ui_t1 = ui_t1 * eps + data[i][t]

            ui_t1 /= np.linalg.norm(ui_t1)
            data[i].append(ui_t1)
    
    U = get_U_t(data, T)
    return data, U



#getter functions
def get_U_t(data, t):
    """
    Get U(t) = (u1(t) ... uP(t)).
    """
    U_t = np.hstack([data[p][t] for p in range(1, P+1)])
    return U_t



#display functions
def plot(data, Q, i=None, *, pm=False):
    """
    i = None : plotting the total distance between U(T) and Q
    i = 0 : plotting the distances between ui(T) and qi in a single plot
    i = -1 : plotting the distances between ui(T) and qi separately
    1 <= i <= P : plotting the evolution of the distance of the i-th principal component
    """

    def _get_norm_unsigned_ui_t_qi(data, i, t, Q):
        """
        Get the distance between ui(t) and qi, accurate to the sign.
        """
        U_t = get_U_t(data, t)
        return abs( abs(U_t[:, i-1] @ Q[:, i-1]) - 1 )
    
    def _get_norm_unsigned_ui_t_qi_range(data, i, Q, t0=0, t1=len(data[1])-1):
        """
        Get the distance between ui(t) and qi, from time t0 to time t1, accurate to the sign.
        """
        return [_get_norm_unsigned_ui_t_qi(data, i, t, Q) for t in range(t0, t1+1)]

    def _get_norm_unsigned_U_t_QP(data, t, Q):
        """
        Get the distance between U(t) = (u1(t) ... uP(t)) and Q' = (q1 ... qP), accurate to the sign.
        """
        return sum([_get_norm_unsigned_ui_t_qi(data, p, t, Q) for p in range(1, P+1)])
    
    def _get_norm_unsigned_U_t_QP_range(data, Q, t0=0, t1=len(data[1])-1):
        """
        Get the distance between U(t) = (u1(t) ... uP(t)) and Q' = (q1 ... qP), from time t0 to time t1, accurate to the sign.
        """
        return [_get_norm_unsigned_U_t_QP(data, t, Q) for t in range(t0, t1+1)]


    plt.title(label=f"Evolution of the distance between U = (u1 ... uP) and Q' = (q1 ... qP)")
    plt.xlabel(xlabel="t")
    plt.ylabel(ylabel=f"Distance between U and Q")

    if i == None:
        plt.plot(np.arange(len(data[1])), _get_norm_unsigned_U_t_QP_range(data, Q), label="U, Q'")
    elif i == 0:
        cmap = plt.get_cmap('plasma')
        for j in range(1, P+1):
            if pm:
                #Verification that data is computed with the classical power method
                assert len(data[1]) == T*P+1, "data isn't computed with the power_method function. Please set pm=False."
                plt.plot(np.arange(T*(j-1), T*j+1), _get_norm_unsigned_ui_t_qi_range(data, j, Q, T*(j-1), T*j), color=cmap(j/(P)), label=f"u{j}(t), q{j}")
            else:
                plt.plot(np.arange(len(data[1])), _get_norm_unsigned_ui_t_qi_range(data, j, Q), color=cmap(j/(P)), label=f"u{j}(t), q{j}")

    elif i == -1:
        for a in range(1, P+1):
            plot(data, Q, a)
        return
    elif 1 <= i <= P:
        plt.plot(np.arange(len(data[1])), _get_norm_unsigned_ui_t_qi_range(data, i, Q), label=f"u{i}(t), q{i}")
    else:
        raise KeyError("i does not correspond to any principal component's id.")

    plt.axhline(y=0, color='red', linestyle='--', linewidth=0.75, label=f'0')

    plt.legend(fontsize='small', bbox_to_anchor=(1.05, 1), ncol=max(1, P//20+1))
    plt.show()



X = generate_data_matrix()
initial_vectors = generate_initial_vectors()

S = covariance_matrix(X)
Lambda, Q = spectral_decomposition(S)



data, U = power_method(X, initial_vectors)
plot(data, Q, pm=True)
plot(data, Q, 0, pm=True)



data1, U1 = update_rule(X, initial_vectors, 0.2, 0.4, 0.1)
plot(data1, Q)
plot(data1, Q, 0)
