from distributed_averaging import generate_instance_DA, initialize_instance, initialize_local_degree_weight, distributed_linear_iteration, show_graph, plot as plot_DA
from distributed_averaging import set_NB_AGENT, set_NB_EDGE
from power_method import generate_instance_PM, covariance_matrix, spectral_decomposition, power_method, update_rule, plot as plot_PM
from power_method import set_N_DIM, set_P_DIM, set_T_PM, set_K1, set_K2, set_EPS

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx



SEED = 42

#Number of agents NB_AGENT
#Number of edges NB_EDGE
NB_AGENT = set_NB_AGENT(6)
NB_EDGE = set_NB_EDGE(7)

N_DIM = set_N_DIM(5)
P_DIM = set_P_DIM(3)
L_DIM_LIST = [6, 6, 7, 7, 8, 8]
T_PM = set_T_PM(20)

K1 = set_K1(0.2)
K2 = set_K2(0.4)
EPS = set_EPS(0.1)

np.random.seed(SEED)


adjacency_matrix, _, _= generate_instance_DA()
show_graph(adjacency_matrix)

data = dict()

for a in range(1, NB_AGENT+1):
    data[a] = dict()
    X, initial_vectors = generate_instance_PM(L_DIM_LIST[a-1])
    _, Q = spectral_decomposition(covariance_matrix(X, L_DIM_LIST[a-1]))

    data[a]["X"] = X
    data[a]["Q"] = Q
    data[a]["PC"], _ = power_method(data[a]["X"], initial_vectors, L_DIM_LIST[a-1])
    plot_PM(data[a]["PC"], Q)
    plot_PM(data[a]["PC"], Q, 0, pm=True)

