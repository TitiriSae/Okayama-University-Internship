from decentralized_pca import generate_L_DIM_LIST, init_decentralized_PCA, decentralized_PCA, plot, check_accuracy, show_graph

import numpy as np



if __name__ == "__main__":

    global_var = dict()

    global_var["SEED"] = 15
    np.random.seed(global_var["SEED"])



    global_var["NB_AGENT"] = 10
    global_var["NB_EDGE"] = 30

    global_var["N_DIM"] = 5
    global_var["P_DIM"] = 3
    global_var["L_DIM_LIST"] = generate_L_DIM_LIST(global_var["NB_AGENT"], global_var["N_DIM"])
    
    global_var["T_PM"] = 10000
    global_var["T_CONSENSUS_Y"] = 1
    global_var["T_CONSENSUS_Z"] = 15

    global_var["K1"] = 0.2
    global_var["K2"] = 0.4
    global_var["EPS"] = 0.1



    adjacency_matrix, pos, data, W, X_m_init_vect_list = init_decentralized_PCA(global_var)
    show_graph(adjacency_matrix, pos)
    decentralized_PCA(global_var, data, W, X_m_init_vect_list)
    
    check_accuracy(global_var, data, [10**-i for i in range(1, 6)])
    plot(global_var, data)