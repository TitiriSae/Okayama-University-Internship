from distributed_averaging import generate_instance_DA, initialize_instance, initialize_initial_values, initialize_local_degree_weight, distributed_linear_iteration, show_graph
from distributed_averaging import set_NB_AGENT, set_NB_EDGE, set_T_DA
from power_method import generate_instance_PM, covariance_matrix, spectral_decomposition
from power_method import set_N_DIM, set_P_DIM, set_T_PM, set_K1, set_K2, set_EPS

import numpy as np
import matplotlib.pyplot as plt



SEED = 42

#Number of agents NB_AGENT
#Number of edges NB_EDGE
#Number of iteration for the distributed linear iteration T_DA
NB_AGENT = set_NB_AGENT(6)
NB_EDGE = set_NB_EDGE(7)
T_DA = set_T_DA(250)

assert NB_AGENT-1 <= NB_EDGE <= (NB_AGENT*(NB_AGENT-1))/2

#Dimension of a data N_DIM
#Number of principal component wanted P_DIM
#Number of data sample for each agent L_DIM_LIST
#Number of iteration for the update rule T_PM
N_DIM = set_N_DIM(5)
P_DIM = set_P_DIM(3)
L_DIM_LIST = [6, 6, 7, 7, 8, 8]
T_PM = set_T_PM(300)

for l_dim in L_DIM_LIST:
    assert N_DIM < l_dim, "global parameters aren't set correctly."
assert P_DIM < N_DIM, "global parameters aren't set correctly."
assert len(L_DIM_LIST) == NB_AGENT, "global parameters aren't set correctly."

#Parameter of the update rule K1
#Parameter of the update rule K2
#Parameter of the update rule EPS
K1 = set_K1(0.2)
K2 = set_K2(0.4)
EPS = set_EPS(0.1)



#np.random.seed(SEED)





def init_decentralized_PCA():
    """
    Initialize an instance of the decentralized algorithm for PCA.

    return:
        adjacency_matrix: list[list[int]]
        data: dict[int, dict[str, Any]]
            data -> (m) agent m -> ('n') neighbors of agent m: list[int]
                                -> ('d') degree of agent m: int
        W: list[list[float]]
        X_m_init_vect_list: list[tuple[ list[list[float]], list[list[list[float]]] ]]
    """

    #Initialization of the agent network
    adjacency_matrix, _, _= generate_instance_DA()
    #show_graph(adjacency_matrix)

    #Updating the data dictionary to add information on the neighbours, degree for each agent
    data = initialize_instance(adjacency_matrix)

    #Initialization of the weights matrix
    W = initialize_local_degree_weight(data)

    #Initialization of the data and initial vectors for the PCA instances for each agent
    X_m_init_vect_list = [generate_instance_PM(L_DIM_LIST[m]) for m in range(NB_AGENT)]

    return adjacency_matrix, data, W, X_m_init_vect_list



def decentralized_PCA(data, W, X_m_init_vect_list):
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

    return:
        None
    """

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
        compute_Y_t(data, W, t)
        compute_Z_t(data, W, t)
        update_rule_t(data, t)

    #Adding global data and optimal matrix of eigenvectors to the data
    X = np.hstack([X_m_init_vect_list[m][0] for m in range(NB_AGENT)])
    _, Q = spectral_decomposition(covariance_matrix(X, sum(L_DIM_LIST)))
    data["X"] = X
    data["Q"] = np.hsplit(Q, Q.shape[1])



def compute_Y_t(data, W, t):
    """
    #STEP 1: AVERAGE CONSENSUS ON U_m(t) = (u1_m(t) ... uP_m(t)) TO FIND MATRIX Y_m(t) = (y1_m(t) ... yP_m(t)) FOR EACH AGENT m

    return:
        None
    """

    for p in range(P_DIM):

        #Updating the data dictionary to add entry on values' history for the p-th principal component for each agent
        up_m_t_list = [data[m]["U"][t][p] for m in range(1, NB_AGENT+1)]
        initialize_initial_values(data, up_m_t_list)

        #Distributed averaging consensus on the vector up_m(t) to find yp_m(t)
        distributed_linear_iteration(data, W)

        for m in range(1, NB_AGENT+1):

            #Storing the last estimated p-th principal component as the vector yp_m(t) for each agent
            data[m]["Y"][t].append(data[m]["x"][-1])

            #Resetting the values' history entry to prepare for next the principal component's computing for each agent
            del data[m]["x"]



def compute_Z_t(data, W, t):
    """
    #STEP 2: AVERAGE CONSENSUS ON THE SUM TERMS ON THE LAST TERM OF THE UPDATE RULE TO FIND MATRIX Z_m(t) = (z1_m(t) ... zP_m(t)) FOR EACH AGENT m

    return:
        None
    """
    
    for p in range(P_DIM):

        #Computing the p-th term in the last sum term in the update rule for each agent
        last_term_list = [data[m]["X"] @ data[m]["X"].T @ data[m]["Y"][t][p] for m in range(1, NB_AGENT+1)]

        #Updating the data dictionary to add entry on values' history for the vector zp_m(t) for each agent
        initialize_initial_values(data, last_term_list)

        #Distributed averaging consensus on the yp_m(t) vector to find zp_m(t)
        distributed_linear_iteration(data, W)

        for m in range(1, NB_AGENT+1):

            #Storing the last estimated p-th vector of Z_m(t) for each agent
            data[m]["Z"][t].append(data[m]["x"][-1])

            #Resetting the values' history entry to prepare for the computing of the next vector of Z_m(t) for each agent
            del data[m]["x"]



def update_rule_t(data, t):
    """
    #STEP 3: COMPUTING A SIGLE ITERATION OF THE UPDATE RULE to find U_m(t) = (u1_m(t) ... uP_m(t)) FOR EACH AGENT m

    return:
        None
    """

    for m in range(1, NB_AGENT+1):
        for p in range(P_DIM):
            
            up_m_t = -K1 * sum(data[m]["Y"][t][j] @ data[m]["Y"][t][j].T @ data[m]["Y"][t][p] for j in range(p))
            up_m_t = up_m_t + K2 * (NB_AGENT/L_DIM_LIST[m-1]) * data[m]["Z"][t][p]
            up_m_t = data[m]["U"][t][p] + EPS * up_m_t

            up_m_t /= np.linalg.norm(up_m_t)
            data[m]["U"][t+1].append(up_m_t)




def plot(data, m=None, p=None):

    def _get_norm_unsigned_up_m_t_qp(data, p, m, t):
        """
        Get the distance between up_m(t) and qp, accurate to the sign.
        """
        return 1- abs((data["Q"][p].T @ data[m]['U'][t][p]).item())

    def _get_norm_unsigned_U_m_t_QP(data, m, t):
        """
        Get the distance between U_m(t) and Q', accurate to the sign.
        """
        return sum([_get_norm_unsigned_up_m_t_qp(data, p, m, t) for p in range(P_DIM)])

    def _get_norm_unsigned_U_m_t_QP_range(data, m, t0=0, t1=len(data[1]["U"])-1):
        """
        Get the distance between U_m(t) and Q', from time t0 to time t1, accurate to the sign.
        """
        return [_get_norm_unsigned_U_m_t_QP(data, m, t) for t in range(t0, t1+1)]

    def _get_norm_unsigned_U_t_QP(data, t):
        """
        Get the distance between U(t) and Q', accurate to the sign.
        """
        return sum([_get_norm_unsigned_U_m_t_QP(data, m, t) for m in range(1, NB_AGENT+1)])
    
    def _get_norm_unsigned_U_t_QP_range(data, t0=0, t1=len(data[1]["U"])-1):
        """
        Get the distance between U(t) and Q', from time t0 to time t1, accurate to the sign.
        """
        return [_get_norm_unsigned_U_t_QP(data, t) for t in range(t0, t1+1)]
    


    def _get_norm_unsigned_up_t_qp(data, p, t):
        """
        Get the distance between up(t) and qp, accurate to the sign.
        """
        return sum([_get_norm_unsigned_up_m_t_qp(data, p, m, t) for m in range(1, NB_AGENT+1)])

    def _get_norm_unsigned_up_t_qp_range(data, p, t0=0, t1=len(data[1]["U"])-1):
        """
        Get the distance between up(t) and qp, from time t0 to time t1, accurate to the sign.
        """
        return [_get_norm_unsigned_up_t_qp(data, p, t) for t in range(t0, t1+1)]
    
    def _get_norm_unsigned_up_m_t_qp_range(data, p, m, t0=0, t1=len(data[1]["U"])-1):
        """
        Get the distance between up_m(t) and qp, from time t0 to time t1, accurate to the sign.
        """
        return [_get_norm_unsigned_up_m_t_qp(data, p, m, t) for t in range(t0, t1+1)]

    plt.title(label=f"Evolution of the distance between U = (u1 ... uP) and Q' = (q1 ... qP)")
    plt.xlabel(xlabel="t")
    plt.ylabel(ylabel=f"Distance between U and Q")

    cmap = plt.get_cmap('plasma')

    if m == None:

        if p == None:
            plt.plot(np.arange(T_PM+1), _get_norm_unsigned_U_t_QP_range(data), label=f"U_(t), Q'")
        elif p == 0:
            for p0 in range(P_DIM):
                plt.plot(np.arange(T_PM+1), _get_norm_unsigned_up_t_qp_range(data, p0), color=cmap(p0/P_DIM), label=f"u{p0}_(t), q{p0}")
        elif 1 <= p <= P_DIM:
            pass

    elif m == 0:
        for m0 in range(1, NB_AGENT+1):

            if p == None:
                plt.plot(np.arange(T_PM+1), _get_norm_unsigned_U_m_t_QP_range(data, m0), color=cmap(m0/NB_AGENT), label=f"U_{m0}(t), Q'")
            elif p == 0:
                for p0 in range(P_DIM):
                    plt.plot(np.arange(T_PM+1), _get_norm_unsigned_up_m_t_qp_range(data, p0, m0), color=cmap((p0+m0+(m0-1)*(P_DIM-1))/(P_DIM*NB_AGENT)), label=f"u{p0}_{m0}_(t), q{p0}")

    elif 1 <= m <= NB_AGENT:

        if p == None:
            plt.plot(np.arange(T_PM+1), _get_norm_unsigned_U_m_t_QP_range(data, m), label=f"U_{m}(t), Q'")
        elif p == 0:
            for p0 in range(P_DIM):
                plt.plot(np.arange(T_PM+1), _get_norm_unsigned_up_m_t_qp_range(data, p0, m), color=cmap((p0+m+(m-1)*(P_DIM-1))/(P_DIM*NB_AGENT)), label=f"u{p0}_{m}_(t), q{p0}")
    
    
    else:
        raise KeyError("argument m is invalid.")

    plt.axhline(y=0, color='red', linestyle='--', linewidth=0.75, label=f'0')

    plt.legend(fontsize='small', bbox_to_anchor=(1.05, 1), ncol=max(1, P_DIM//20+1))
    plt.show()




adjacency_matrix, data, W, X_m_init_vect_list = init_decentralized_PCA()
show_graph(adjacency_matrix)
decentralized_PCA(data, W, X_m_init_vect_list)

print(data["Q"][:P_DIM])
print()
print(data[1]["U"][-1])

for m in range(1, NB_AGENT+1):
    print(np.allclose(data["Q"][:P_DIM], data[m]["U"][-1], atol=1e-2))

plot(data, 0)