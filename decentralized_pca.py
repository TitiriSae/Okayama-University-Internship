from distributed_averaging import generate_graph, init_graph, init_initial_values, init_local_degree_weight, distributed_linear_iteration, show_graph
from power_method import generate_data_matrix, generate_initial_vectors, covariance_matrix, spectral_decomposition

import numpy as np
import matplotlib.pyplot as plt





def generate_L_DIM_LIST(nb_agent, n_dim, *, l_max=None, l_dim_list=None):
    """
    Generate L_DIM_LIST.

    return:
        l_dim_list: list[int]
    """

    if l_dim_list != None:
        return l_dim_list
    
    if l_max == None:
        l_max = n_dim*2
    assert l_max > n_dim

    l_dim_list = [np.random.randint(n_dim+1, l_max+1) for _ in range(nb_agent)]
    return l_dim_list



def init_decentralized_PCA(global_var):
    """
    Initialize an instance of the decentralized algorithm for PCA.

    return:
        adjacency_matrix: list[list[int]]
        pos: dict[int, list[float]]
        data: dict[int, dict[str, Any]]
            data -> (m) agent m -> ('n') neighbors of agent m: list[int]
                                -> ('d') degree of agent m: int
        W: list[list[float]]
        X_m_init_vect_list: list[tuple[ list[list[float]], list[list[list[float]]] ]]
    """
    NB_AGENT = global_var["NB_AGENT"]
    L_DIM_LIST = global_var["L_DIM_LIST"]

    #Initialization of the agent network
    adjacency_matrix, pos = generate_graph(global_var)

    #Updating the data dictionary to add information on the neighbours, degree for each agent
    data = init_graph(global_var, adjacency_matrix)

    #Initialization of the weights matrix
    W = init_local_degree_weight(global_var, data)

    #Initialization of the data and initial vectors for the PCA instances for each agent
    X_m_init_vect_list = [(generate_data_matrix(global_var, L_DIM_LIST[m]), generate_initial_vectors(global_var)) for m in range(NB_AGENT)]

    return adjacency_matrix, pos, data, W, X_m_init_vect_list



def decentralized_PCA(global_var, data, W, X_m_init_vect_list):
    """
    Apply the decentralized algorithm for PCA.

    The entries of the data dictonary is updated.
    data: dict[Any, Any]
    data -> (m) agent m -> ('n') neighbors of agent m: list[int]
                        -> ('d') degree of agent m: int
                        -> ('X') local data matrix X of agent m: list[list[float]]
                        -> ('U') history of the matrices U_m(t) = (u1_m(t) ... uP_m(t)) computed at each time t: list[list[list[list[float]]]]
                        -> ('Y') history of the matrices Y_m(t) = (y1_m(t) ... yP_m(t)) computed at each time t: list[list[list[list[float]]]]
                        -> ('Z') history of the matrices Z_m(t) = (z1_m(t) ... zP_m(t)) computed at each time t: list[list[list[list[float]]]]
         -> ('X') global data: list[list[float]]
         -> ('Q') optimal matrix of eigenvectors: list[list[float]]
         -> ('W') weight matrix: list[list[float]]

    return:
        None
    """
    NB_AGENT = global_var["NB_AGENT"]
    T_PM = global_var["T_PM"]
    L_DIM_LIST = global_var["L_DIM_LIST"]

    #Updating data dictionary to start iterations
    for m in range(1, NB_AGENT+1):

        #Updating the data dictionary to add information on the local data matrix and its eigenvectors matrix for each agent
        data[m]["X"] = X_m_init_vect_list[m-1][0]

        #Preparing the key value "U", "Y" and "Z" of the data for each agent to store matrices U, Y and Z at each time
        data[m]["U"] = [X_m_init_vect_list[m-1][1]]
        data[m]["Y"] = []
        data[m]["Z"] = []

    #Beginning of iterations loop
    for t in range(T_PM):

        #Preparing to compute matrices Y(t), Z(t) and U(t) for this iteration
        for m in range(1, NB_AGENT+1):
            data[m]["Y"].append([])
            data[m]["Z"].append([])
            data[m]["U"].append([])

        #STEPS 
        compute_Y_t(global_var, data, W, t)
        compute_Z_t(global_var, data, W, t)
        update_rule_t(global_var, data, t)

    #Adding global data, optimal matrix of eigenvectors, and the weight matrix to the data
    X = np.hstack([X_m_init_vect_list[m][0] for m in range(NB_AGENT)])
    _, Q = spectral_decomposition(covariance_matrix(global_var, X, sum(L_DIM_LIST)))
    data["X"] = X
    data["Q"] = np.hsplit(Q, Q.shape[1])
    data["W"] = W



def compute_Y_t(global_var, data, W, t):
    """
    #STEP 1: AVERAGE CONSENSUS ON U_m(t) = (u1_m(t) ... uP_m(t)) TO FIND MATRIX Y_m(t) = (y1_m(t) ... yP_m(t)) FOR EACH AGENT m.

    return:
        None
    """
    NB_AGENT = global_var["NB_AGENT"]
    P_DIM = global_var["P_DIM"]
    
    for p in range(P_DIM):

        #Updating the data dictionary to add entry on values' history for the p-th principal component for each agent
        up_m_t_list = [data[m]["U"][t][p] for m in range(1, NB_AGENT+1)]
        init_initial_values(global_var, data, up_m_t_list)

        #Distributed averaging consensus on the vector up_m(t) to find yp_m(t)
        global_var["T_DA"] = global_var["T_CONSENSUS_Y"]
        distributed_linear_iteration(global_var, data, W)

        for m in range(1, NB_AGENT+1):

            #Storing the last estimated p-th principal component as the vector yp_m(t) for each agent
            data[m]["Y"][t].append(data[m]["x"][-1])

            #Resetting the values' history entry to prepare for next the principal component's computing for each agent
            del data[m]["x"]



def compute_Z_t(global_var, data, W, t):
    """
    #STEP 2: AVERAGE CONSENSUS ON THE SUM TERMS ON THE LAST TERM OF THE UPDATE RULE TO FIND MATRIX Z_m(t) = (z1_m(t) ... zP_m(t)) FOR EACH AGENT m.

    return:
        None
    """
    NB_AGENT = global_var["NB_AGENT"]
    P_DIM = global_var["P_DIM"]

    for p in range(P_DIM):

        #Computing the p-th term in the last sum term in the update rule for each agent
        last_term_list = [data[m]["X"] @ data[m]["X"].T @ data[m]["Y"][t][p] for m in range(1, NB_AGENT+1)]

        #Updating the data dictionary to add entry on values' history for the vector zp_m(t) for each agent
        init_initial_values(global_var, data, last_term_list)

        #Distributed averaging consensus on the yp_m(t) vector to find zp_m(t)
        global_var["T_DA"] = global_var["T_CONSENSUS_Z"]
        distributed_linear_iteration(global_var, data, W)

        for m in range(1, NB_AGENT+1):

            #Storing the last estimated p-th vector of Z_m(t) for each agent
            data[m]["Z"][t].append(data[m]["x"][-1])

            #Resetting the values' history entry to prepare for the computing of the next vector of Z_m(t) for each agent
            del data[m]["x"]



def update_rule_t(global_var, data, t):
    """
    #STEP 3: COMPUTING A SINGLE ITERATION OF THE UPDATE RULE to find U_m(t) = (u1_m(t) ... uP_m(t)) FOR EACH AGENT m.

    return:
        None
    """
    NB_AGENT = global_var["NB_AGENT"]
    P_DIM = global_var["P_DIM"]
    L_DIM_LIST = global_var["L_DIM_LIST"]
    K1 = global_var["K1"]
    K2 = global_var["K2"]
    EPS = global_var["EPS"]

    for m in range(1, NB_AGENT+1):
        for p in range(P_DIM):
            
            up_m_t = -K1 * sum(data[m]["Y"][t][j] @ data[m]["Y"][t][j].T @ data[m]["Y"][t][p] for j in range(p))
            up_m_t = up_m_t + K2 * (NB_AGENT/L_DIM_LIST[m-1]) * data[m]["Z"][t][p]
            up_m_t = data[m]["U"][t][p] + EPS * up_m_t

            up_m_t /= np.linalg.norm(up_m_t)
            data[m]["U"][t+1].append(up_m_t)



def plot(global_var, data, m=None, p=None, *, show=True, label=None, color=None):
    """
    Plot function to visualize the distance (accurate to the sign) of the principal components’ estimations to the optimal principal components.

    This function has two optional parameters to specify what to plot. The first optional parameter m (default value = None) has 3 different behaviors:
        - m = None: the plot displays global information (a single curve that is the sum each agent’s curve)
        - m = 0: Tthe plot displays local information for each agent (one curve associated to one agent)
        - 1 <= m <= M: the plot only displays local information associated to the agent m (the curve associated to agent m).
    
    The second optional parameter p (default value = None) has 3 differents behaviors:
        - p = None: the plot displays information on the total distance of all P principal components (a single curve that is the sum of each principal component’s curve)
        - p = 0: the plot displays information on the distances of all P principal components (one curve associated to one principal component)
        - 1 <= p <= P: the plot only display information associated to the pth principal component (the curve associated to the pth principal component).

    return:
        None
    """
    NB_AGENT = global_var["NB_AGENT"]
    P_DIM = global_var["P_DIM"]
    T_PM = len(data[1]["U"])-1


    def _get_norm_unsigned_up_m_t_qp(data, p, m, t):
        """
        Get the distance between up_m(t) and qp, accurate to the sign.
        """
        return abs( abs((data["Q"][p-1].T @ data[m]["U"][t][p-1]).item()) -1)

    def _get_norm_unsigned_U_m_t_QP(data, m, t):
        """
        Get the distance between U_m(t) and Q', accurate to the sign.
        """
        return sum([_get_norm_unsigned_up_m_t_qp(data, p, m, t) for p in range(1, P_DIM+1)])

    def _get_norm_unsigned_U_m_t_QP_range(data, m, t0=0, t1=T_PM):
        """
        Get the distance between U_m(t) and Q', from time t0 to time t1, accurate to the sign.
        """
        return [_get_norm_unsigned_U_m_t_QP(data, m, t) for t in range(t0, t1+1)]

    def _get_norm_unsigned_U_t_QP(data, t):
        """
        Get the distance between U(t) and Q', accurate to the sign.
        """
        return sum([_get_norm_unsigned_U_m_t_QP(data, m, t) for m in range(1, NB_AGENT+1)])
    
    def _get_norm_unsigned_U_t_QP_range(data, t0=0, t1=T_PM):
        """
        Get the distance between U(t) and Q', from time t0 to time t1, accurate to the sign.
        """
        return [_get_norm_unsigned_U_t_QP(data, t) for t in range(t0, t1+1)]

    def _get_norm_unsigned_up_t_qp(data, p, t):
        """
        Get the distance between up(t) and qp, accurate to the sign.
        """
        return sum([_get_norm_unsigned_up_m_t_qp(data, p, m, t) for m in range(1, NB_AGENT+1)])

    def _get_norm_unsigned_up_t_qp_range(data, p, t0=0, t1=T_PM):
        """
        Get the distance between up(t) and qp, from time t0 to time t1, accurate to the sign.
        """
        return [_get_norm_unsigned_up_t_qp(data, p, t) for t in range(t0, t1+1)]
    
    def _get_norm_unsigned_up_m_t_qp_range(data, p, m, t0=0, t1=T_PM):
        """
        Get the distance between up_m(t) and qp, from time t0 to time t1, accurate to the sign.
        """
        return [_get_norm_unsigned_up_m_t_qp(data, p, m, t) for t in range(t0, t1+1)]



    cmap = plt.get_cmap('plasma')
    m0, p0 = 1, 1

    if m == None:

        if p == None:
            if label==None:
                label=f"U_(t), Q'"
            plt.plot(np.arange(T_PM+1), _get_norm_unsigned_U_t_QP_range(data), label=label, color=color)

        elif p == 0:
            for p0 in range(1, P_DIM+1):
                plt.plot(np.arange(T_PM+1), _get_norm_unsigned_up_t_qp_range(data, p0), color=cmap(p0/P_DIM), label=f"u{p0}_(t), q{p0}")

        elif 1 <= p <= P_DIM:
            plt.plot(np.arange(T_PM+1), _get_norm_unsigned_up_t_qp_range(data, p), label=f"u{p}_(t), q{p}")

        else:
            raise KeyError("argument p is invalid.")

    elif m == 0:
    
        for m0 in range(1, NB_AGENT+1):
            if p == None:
                plt.plot(np.arange(T_PM+1), _get_norm_unsigned_U_m_t_QP_range(data, m0), color=cmap(m0/NB_AGENT), label=f"U_{m0}(t), Q'")
            
            elif p == 0:
                for p0 in range(1, P_DIM+1):
                    plt.plot(np.arange(T_PM+1), _get_norm_unsigned_up_m_t_qp_range(data, p0, m0), color=cmap((p0+m0+(m0-1)*(P_DIM-1))/(P_DIM*NB_AGENT)), label=f"u{p0}_{m0}_(t), q{p0}")
            
            elif 1 <= p <= P_DIM:
                plt.plot(np.arange(T_PM+1), _get_norm_unsigned_up_m_t_qp_range(data, p, m0), color=cmap(m0/NB_AGENT), label=f"u{p}_{m0}_(t), q{p}")
            
            else:
                raise KeyError("argument p is invalid.")

    elif 1 <= m <= NB_AGENT:

        if p == None:
            plt.plot(np.arange(T_PM+1), _get_norm_unsigned_U_m_t_QP_range(data, m), label=f"U_{m}(t), Q'")
        
        elif p == 0:
            for p0 in range(1, P_DIM+1):
                plt.plot(np.arange(T_PM+1), _get_norm_unsigned_up_m_t_qp_range(data, p0, m), color=cmap(p0/P_DIM), label=f"u{p0}_{m}_(t), q{p0}")
        
        elif 1 <= p <= P_DIM:
            plt.plot(np.arange(T_PM+1), _get_norm_unsigned_up_m_t_qp_range(data, p, m), label=f"u{p}_{m}_(t), q{p}")
        
        else:
            raise KeyError("argument p is invalid.")
    
    else:
        raise KeyError("argument m is invalid.")

    if show:
        plt.title(label=f"Evolution of the distance between U = (u1 ... uP) and Q' = (q1 ... qP)")
        plt.xlabel(xlabel="t")
        plt.ylabel(ylabel=f"Distance between U and Q")
        
        plt.axhline(y=0, color='red', linestyle='--', linewidth=0.75, label=f'0')
        plt.legend(fontsize='small', bbox_to_anchor=(1.05, 1), ncol=max(1, (m0*p0)//20+1))
        plt.show()



def check_accuracy(global_var, data, precision_list):
    """
    Check the accuracy of the estimated principal component.

    return:
        None
    """
    for precision in precision_list:
        print(f"\nAccuracy to {precision}:")
        for m in range(1, global_var["NB_AGENT"]+1):
            print(f"U_{m}, Q':", np.allclose(np.abs(data["Q"][:global_var["P_DIM"]]), np.abs(data[m]["U"][-1]), atol=precision))


if __name__ == "__main__":

    global_var = dict()

    global_var["SEED"] = 15
    np.random.seed(global_var["SEED"])

    #Number of agents NB_AGENT
    #Number of edges NB_EDGE
    #Number of iteration for the distributed linear iteration for Y and Z T_CONSENSUS_Y T_CONSENSUS_Z
    global_var["NB_AGENT"] = 10
    global_var["NB_EDGE"] = 30
    global_var["T_CONSENSUS_Y"] = 1
    global_var["T_CONSENSUS_Z"] = 20

    #Dimension of a data N_DIM
    #Number of principal component wanted P_DIM
    global_var["N_DIM"] = 5
    global_var["P_DIM"] = 3

    #Number of data sample for each agent L_DIM_LIST
    #Number of iteration for the update rule T_PM
    global_var["L_DIM_LIST"] = generate_L_DIM_LIST(global_var["NB_AGENT"], global_var["N_DIM"])
    global_var["T_PM"] = 1000

    #Parameter of the update rule K1
    #Parameter of the update rule K2
    #Parameter of the update rule EPS
    global_var["K1"] = 0.2
    global_var["K2"] = 0.4
    global_var["EPS"] = 0.1



    adjacency_matrix, pos, data, W, X_m_init_vect_list = init_decentralized_PCA(global_var)
    show_graph(adjacency_matrix, pos)
    decentralized_PCA(global_var, data, W, X_m_init_vect_list)
    
    check_accuracy(global_var, data, [10**-i for i in range(1, 6)])    
    plot(global_var, data)