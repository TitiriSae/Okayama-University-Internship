from distributed_averaging import generate_instance_DA, initialize_instance, initialize_initial_values, initialize_local_degree_weight, distributed_linear_iteration, show_graph, plot as plot_DA
from distributed_averaging import set_NB_AGENT, set_NB_EDGE, set_T_DA
from power_method import generate_instance_PM, covariance_matrix, spectral_decomposition, power_method, update_rule, plot as plot_PM
from power_method import set_N_DIM, set_P_DIM, set_T_PM, set_K1, set_K2, set_EPS

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx



SEED = 42

#Number of agents NB_AGENT
#Number of edges NB_EDGE
#Number of iteration for the distributed linear iteration T_DA
NB_AGENT = set_NB_AGENT(6)
NB_EDGE = set_NB_EDGE(7)
T_DA = set_T_DA(200)

assert NB_AGENT-1 <= NB_EDGE <= (NB_AGENT*(NB_AGENT-1))/2

#Dimension of a data N_DIM
#Number of principal component wanted P_DIM
#Number of data sample for each agent L_DIM_LIST
#Number of iteration for the update rule T_PM
N_DIM = set_N_DIM(5)
P_DIM = set_P_DIM(3)
L_DIM_LIST = [6, 6, 7, 7, 8, 8]
T_PM = set_T_PM(20)

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


np.random.seed(SEED)







#Initialization of the agent network
adjacency_matrix, _, _= generate_instance_DA()
show_graph(adjacency_matrix)

#Initialization of the data and initial vectors for the PCA instances for each agent
X_m_init_vect_list = [generate_instance_PM(L_DIM_LIST[m]) for m in range(NB_AGENT)]

#Updating the data dictionary to add information on the neighbours, degree for each agent
data = initialize_instance(adjacency_matrix)

#Initialization of the weights matrix
W = initialize_local_degree_weight(data)

for m in range(1, NB_AGENT+1):

    #Updating the data dictionary to add information on the local data matrix and its eigenvectors matrix for each agent
    data[m]["X"] = X_m_init_vect_list[m-1][0]
    _, Q = spectral_decomposition(covariance_matrix(data[m]["X"], L_DIM_LIST[m-1]))
    data[m]["Q"] = Q

    #Preparing the key value "U", "Y" and "Z" of the data for each agent to store matrices U, Y and Z at each time
    data[m]["U"] = [X_m_init_vect_list[m-1][1]]
    data[m]["Y"] = []
    data[m]["Z"] = []

    





#BEGINNING OF ITERATION t=0
t=0

#Preparing to compute matrices Y(t) and Z(t) for this iteration
for m in range(1, NB_AGENT+1):
    data[m]["Y"].append([])
    data[m]["Z"].append([])




#STEP 1: COMPUTING AND AVERAGE CONSENSUS OF THE MATRIX Y
for p in range(P_DIM):

    #Updating the data dictionary to add entry on values' history for the p-th principal component for each agent
    up_m_t = [data[m]["U"][t][p] for m in range(1, NB_AGENT+1)]
    initialize_initial_values(data, up_m_t)

    #Distributed averaging consensus on the vector up_m(t) to find yp_m(t)
    distributed_linear_iteration(data, W)

    for m in range(1, NB_AGENT+1):

        #Storing the last estimated p-th principal component as the vector yp_m(t) for each agent
        data[m]["Y"][t].append(data[m]["x"][-1])

        #Resetting the values' history entry to prepare for next the principal component's computing for each agent
        del data[m]["x"]





#STEP 2: COMPUTING AND AVERAGE CONSENSUS OF THE MATRIX Z
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





#STEP 3: COMPUTING A SIGLE ITERATION OF THE UPDATE RULE
    for m in range(P_DIM):
        for p in range(1, NB_AGENT+1):
            pass



"""
    data[m]["PC"], _ = power_method(data[m]["X"], initial_vectors_m, L_DIM_LIST[m-1])
    plot_PM(data[m]["PC"], Q)
    plot_PM(data[m]["PC"], Q, 0, pm=True)
"""


