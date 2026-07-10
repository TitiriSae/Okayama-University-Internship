import numpy as np
import matplotlib.pyplot as plt


SEED = 42

#Sample of dimension N_DIM
#Number of sample L_DIM
#Number of dimension of the subspace P_DIM
#Number of iteration T per principal component
N_DIM = 10
L_DIM = 15
P_DIM = 6
T_PM = 50

K1 = 0.2
K2 = 0.4
EPS = 0.1

#Verifications of global dimension parameters
assert N_DIM < L_DIM, "global parameters aren't set correctly."
assert P_DIM < N_DIM, "global parameters aren't set correctly."



def generate_instance_PM(L):
    """
    Generates randomly an instance of data X and the P normalized initial vectors of dimension N for the problem. The rows of the matrix sums to 1.

    return:
        X: list[list[float]]
        initial_vectors: list[list[list[float]]]
    """

    X = np.random.random((N_DIM, L))
    X -= np.mean(X, axis=1, keepdims=True)

    #Verification rows sums to 0
    assert np.allclose(np.sum(X, axis=1), np.zeros(N_DIM), atol=1e-12)

    def _normalised_vector():
        """
        Generate a random vector of dimension N on the standard normal distribution.

        return:
            u: list[list[float]]
        """

        u = np.random.randn(N_DIM, 1)
        u /= np.linalg.norm(u)
        return u
    
    initial_vectors = [_normalised_vector() for _ in range(P_DIM)]
    return X, initial_vectors



def covariance_matrix(X, L):
    """
    Create the covarance matrix S associated to the data matrix X.

    return 
        S: list[list[float]]
    """

    S = (1/L)*np.dot(X, X.T)

    #Verification shape and symetry
    assert np.shape(S) == (N_DIM, N_DIM)
    assert np.all(S == S.T)
    return S



def spectral_decomposition(S):
    """
    Finds the spectral decomposition of the covariance matrix S = Q @ Lambda @ Q.T.

    return:
        Lambda: list[list[float]]
        Q: list[list[float]]
    """

    eigval, Q = np.linalg.eigh(S) 

    reversed_indices = np.argsort(eigval)[::-1]
    eigval = eigval[reversed_indices]

    Lambda = np.diag(eigval)
    Q = Q[:, reversed_indices]

    #Verification of the spectral decomposition
    assert np.allclose(S, Q @ Lambda @ Q.T, atol=1e-12)
    return Lambda, Q



def power_method(X, initial_vectors, L):
    """
    Apply the power method.

    return:
        data: dict[int, list[list[list[float]]]
        U: list[list[float]]
    """

    data = {p: [initial_vectors[p-1]] for p in range(1, P_DIM+1)}
    Sp = covariance_matrix(X, L)

    for p in range(1, P_DIM+1):

        for t in range(T_PM):

            Sp_up_t = np.dot(Sp, data[p][-1])
            up_t1 = Sp_up_t/np.linalg.norm(Sp_up_t)
            data[p].append(up_t1)
            
            for p0 in range(1, P_DIM+1):
                if p0 != p:
                    data[p0].append(data[p0][-1])
        
        Sp = Sp - Sp @ data[p][-1] @ data[p][-1].T
    
    U = get_U_t(data, len(data[1])-1)
    return data, U



def update_rule(X, initial_vectors, L, k1, k2, eps):
    """
    Apply the update rule proposed in the paper.

    return:
        data: dict[int, list[list[list[float]]]
        U: list[list[float]]
    """

    data = {p: [initial_vectors[p-1]] for p in range(1, P_DIM+1)}
    S = covariance_matrix(X, L)
    
    for t in range(T_PM):
        for i in range(1, P_DIM+1):
            ui_t1 = (-k1 * sum([data[j][t] @ data[j][t].T @ data[i][t] for j in range(1, i+1)]))
            ui_t1 = ui_t1 + k2 * (S @ data[i][t])
            ui_t1 = ui_t1 * eps + data[i][t]

            ui_t1 /= np.linalg.norm(ui_t1)
            data[i].append(ui_t1)
    
    U = get_U_t(data, T_PM)
    return data, U



#Setter functions
def set_N_DIM(n_dim):
    """
    Setter for the global variable N_DIM.

    return:
        n_dim: int
    """
    global N_DIM
    N_DIM = n_dim
    return n_dim

def set_P_DIM(p_dim):
    """
    Setter for the global variable P_DIM.

    return:
        p_dim: int
    """
    global P_DIM
    P_DIM = p_dim
    return p_dim

def set_T_PM(t_pm):
    """
    Setter for the global variable T_PM.

    return:
        t_pm: int
    """
    global T_PM
    T_PM = t_pm
    return t_pm


def set_K1(k1):
    """
    Setter for the global variable K1.

    return:
        k1: float
    """
    global K1
    K1 = k1
    return k1

def set_K2(k2):
    """
    Setter for the global variable K2.

    return:
        k2: float
    """
    global K2
    K2 = k2
    return k2

def set_EPS(eps):
    """
    Setter for the global variable EPS.

    return:
        eps: float
    """
    global EPS
    EPS = eps
    return eps



#Getter functions
def get_U_t(data, t):
    """
    Get U(t) = (u1(t) ... uP(t)).

    return:
        U_t: list[list[float]]
    """
    U_t = np.hstack([data[p][t] for p in range(1, P_DIM+1)])
    return U_t



#Display functions
def plot(data, Q, i=None, *, pm=False):
    """
    Plot the data, especially the distance (unsigned) beteween the estimated principal components and their optimal values.
    The plot is different depending on the parameter i:
    i = None : plotting the total distance between U(T) and Q.
    i = 0 : plotting the distances between ui(T) and qi in a single plot.
        Parameter pm should be set as True if the data is computed with the power_method function to obtain a different plot.
    i = -1 : plotting the distances between ui(T) and qi separately.
    1 <= i <= P : plotting the evolution of the distance of the i-th principal component.

    return:
        None
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
        return sum([_get_norm_unsigned_ui_t_qi(data, p, t, Q) for p in range(1, P_DIM+1)])
    
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
        for j in range(1, P_DIM+1):
            if pm:
                #Verification that data is computed with the classical power method
                assert len(data[1]) == T_PM*P_DIM+1, "data isn't computed with the power_method function. Please set pm=False."
                plt.plot(np.arange(T_PM*(j-1), T_PM*j+1), _get_norm_unsigned_ui_t_qi_range(data, j, Q, T_PM*(j-1), T_PM*j), color=cmap(j/(P_DIM)), label=f"u{j}(t), q{j}")
            else:
                plt.plot(np.arange(len(data[1])), _get_norm_unsigned_ui_t_qi_range(data, j, Q), color=cmap(j/(P_DIM)), label=f"u{j}(t), q{j}")

    elif i == -1:
        for a in range(1, P_DIM+1):
            plot(data, Q, a)
        return
    elif 1 <= i <= P_DIM:
        plt.plot(np.arange(len(data[1])), _get_norm_unsigned_ui_t_qi_range(data, i, Q), label=f"u{i}(t), q{i}")
    else:
        raise KeyError("i does not correspond to any principal component's id.")

    plt.axhline(y=0, color='red', linestyle='--', linewidth=0.75, label=f'0')

    plt.legend(fontsize='small', bbox_to_anchor=(1.05, 1), ncol=max(1, P_DIM//20+1))
    plt.show()





if __name__ == "__main__":
    np.random.seed(SEED)

    X, initial_vectors = generate_instance_PM(L_DIM)
    S = covariance_matrix(X, L_DIM)
    Lambda, Q = spectral_decomposition(S)

    data, U = power_method(X, initial_vectors, L_DIM)
    plot(data, Q, pm=True)
    plot(data, Q, 0, pm=True)

    data1, U1 = update_rule(X, initial_vectors, L_DIM, K1, K2, EPS)
    plot(data1, Q)
    plot(data1, Q, 0)
