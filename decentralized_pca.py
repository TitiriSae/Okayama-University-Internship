from distributed_averaging import generate_instance_DA, initialize_instance, initialize_local_degree_weight, distributed_linear_iteration, show_graph, plot as plot_DA
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
T_DA = set_T_DA(50)

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
assert len(L_DIM_LIST) == NB_AGENT

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

data = dict()
#Initialization of the data and initial vectors for the PCA instances for each agent
X_m_init_vect_list = [generate_instance_PM(L_DIM_LIST[m]) for m in range(NB_AGENT)]
X_m_list = [init_m[0] for init_m in X_m_init_vect_list]
init_vect_list_by_agent = [init_m[1] for init_m in X_m_init_vect_list]

init_vect_list_by_pc = [[init_vect_list_by_agent[m][p] for m in range(NB_AGENT)] for p in range(P_DIM)]


#Updating the data dictionary to add information on the neighbours, degree and values' history for each agent
#The values' history data[m]["x"] concerns the first principal component init_vect_list_by_pc[0]
data = initialize_instance(adjacency_matrix, init_vect_list_by_pc[0])

#Updating the data dictionary to add information on the local data matrix and its eigenvectors matrix for each agent
for m in range(1, NB_AGENT+1):
    data[m]["X"] = X_m_init_vect_list[m-1][0]
    _, Q = spectral_decomposition(covariance_matrix(data[m]["X"], L_DIM_LIST[m-1]))
    data[m]["Q"] = Q

#Initialization of the weights matrix
W = initialize_local_degree_weight(data)

#Distributed averaging on the initial vector of the first prinsipal component
distributed_linear_iteration(data, W)
for m in range(1, NB_AGENT+1):
    data[m]["y"] = [data[m]["x"][-1]]
    data[m]["x"] = []


"""
    data[m]["PC"], _ = power_method(data[m]["X"], initial_vectors_m, L_DIM_LIST[m-1])
    plot_PM(data[m]["PC"], Q)
    plot_PM(data[m]["PC"], Q, 0, pm=True)
"""


