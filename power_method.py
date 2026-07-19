import numpy as np
import matplotlib.pyplot as plt





def generate_data_matrix(global_var, l):
    """
    Generates randomly an instance of data X for the problem. The rows of the matrix sums to 1.

    return:
        X: list[list[float]]
    """
    N_DIM = global_var["N_DIM"]
    assert N_DIM < l, "global parameters aren't set correctly."

    X = np.random.random((N_DIM, l))
    X -= np.mean(X, axis=1, keepdims=True)

    #Verification rows sums to 0
    assert np.allclose(np.sum(X, axis=1), np.zeros(N_DIM), atol=1e-12)

    return X



def generate_initial_vectors(global_var):
    """
    Generates randomly the P normalized initial vectors of dimension N for the problem.

    return:
        initial_vectors: list[list[list[float]]]
    """
    N_DIM = global_var["N_DIM"]
    P_DIM = global_var["P_DIM"]
    assert P_DIM < N_DIM, "global parameters aren't set correctly."

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
    return initial_vectors



def covariance_matrix(global_var, X, l):
    """
    Create the covarance matrix S associated to the data matrix X.

    return 
        S: list[list[float]]
    """
    N_DIM = global_var["N_DIM"]

    S = (1/l)*np.dot(X, X.T)

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



def power_method(global_var, X, initial_vectors, l):
    """
    Apply the power method.

    return:
        data: dict[int, list[list[list[float]]]
            data -> (i) i-th principal component: history of values of the i-th principal component
        U: list[list[float]]
    """
    P_DIM = global_var["P_DIM"]
    T_PM = global_var["T_PM"]

    data = {p: [initial_vectors[p-1]] for p in range(1, P_DIM+1)}
    Sp = covariance_matrix(global_var, X, l)

    for p in range(1, P_DIM+1):

        for _ in range(T_PM):

            Sp_up_t = np.dot(Sp, data[p][-1])
            up_t1 = Sp_up_t/np.linalg.norm(Sp_up_t)
            data[p].append(up_t1)
            
            for p0 in range(1, P_DIM+1):
                if p0 != p:
                    data[p0].append(data[p0][-1])
        
        Sp = Sp - Sp @ data[p][-1] @ data[p][-1].T
    
    U = get_U_t(global_var, data, len(data[1])-1)
    return data, U



def update_rule(global_var, X, initial_vectors, l):
    """
    Apply the update rule proposed in the paper.

    return:
        data: dict[int, list[list[list[float]]]
        U: list[list[float]]
    """
    P_DIM = global_var["P_DIM"]
    T_PM = global_var["T_PM"]
    K1 = global_var["K1"]
    K2 = global_var["K2"]
    EPS = global_var["EPS"]

    data = {p: [initial_vectors[p-1]] for p in range(1, P_DIM+1)}
    S = covariance_matrix(global_var, X, l)
    
    for t in range(T_PM):
        for i in range(1, P_DIM+1):
            ui_t1 = (-K1 * sum([data[j][t] @ data[j][t].T @ data[i][t] for j in range(1, i+1)]))
            ui_t1 = ui_t1 + K2 * (S @ data[i][t])
            ui_t1 = ui_t1 * EPS + data[i][t]

            ui_t1 /= np.linalg.norm(ui_t1)
            data[i].append(ui_t1)
        
        """
        if all( [np.linalg.norm(data[j][t+1] - data[j][t]) < CONVERGENCE_VAL for j in range(1, P_DIM+1)] ):
            set_T_PM(t+1)
            break
        """
    
    U = get_U_t(global_var, data, T_PM)
    return data, U



#Getter functions
def get_U_t(global_var, data, t):
    """
    Get U(t) = (u1(t) ... uP(t)).

    return:
        U_t: list[list[float]]
    """
    P_DIM = global_var["P_DIM"]

    U_t = np.hstack([data[p][t] for p in range(1, P_DIM+1)])
    return U_t



#Display functions
def plot(global_var, data, Q, i=None, *, pm=False):
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
    P_DIM = global_var["P_DIM"]
    T_PM = global_var["T_PM"]

    def _get_norm_unsigned_ui_t_qi(data, i, t, Q):
        """
        Get the distance between ui(t) and qi, accurate to the sign.
        """
        U_t = get_U_t(global_var, data, t)
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
                plt.plot(np.arange(len(data[1])), _get_norm_unsigned_ui_t_qi_range(data, j, Q), color=cmap(j/P_DIM), label=f"u{j}(t), q{j}")

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

    global_var = dict()

    global_var["SEED"] = 42
    np.random.seed(global_var["SEED"])

    #Sample of dimension N_DIM
    #Number of sample L_DIM
    #Number of dimension of the subspace P_DIM
    #Number of iteration T_PM per principal component
    global_var["N_DIM"] = 10
    global_var["L_DIM"] = 15
    global_var["P_DIM"] = 6
    global_var["T_PM"] = 5000

    global_var["K1"] = 0.2
    global_var["K2"] = 0.4
    global_var["EPS"] = 0.1

    #Verifications of global dimension parameters
    assert global_var["N_DIM"] < global_var["L_DIM"], "global parameters aren't set correctly."
    assert global_var["P_DIM"] < global_var["N_DIM"], "global parameters aren't set correctly."

    global_var["CONVERGENCE_VAL"] = 1e-10



    X = generate_data_matrix(global_var, global_var["L_DIM"])
    initial_vectors = generate_initial_vectors(global_var)
    S = covariance_matrix(global_var, X, global_var["L_DIM"])
    Lambda, Q = spectral_decomposition(S)
    
    data, U = power_method(global_var, X, initial_vectors, global_var["L_DIM"])
    plot(global_var, data, Q, pm=True)
    plot(global_var, data, Q, 0, pm=True)
    
    data1, U1 = update_rule(global_var, X, initial_vectors, global_var["L_DIM"])
    plot(global_var, data1, Q)
    plot(global_var, data1, Q, 0)
