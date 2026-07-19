from decentralized_pca import generate_L_DIM_LIST, init_decentralized_PCA, decentralized_PCA, plot, check_accuracy, show_graph

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


if __name__ == "__main__":

    global_var = dict()

    global_var["SEED"] = 15
    np.random.seed(global_var["SEED"])



    global_var["NB_AGENT"] = 10
    global_var["NB_EDGE"] = 30

    global_var["N_DIM"] = 5
    global_var["P_DIM"] = 3
    global_var["L_DIM_LIST"] = generate_L_DIM_LIST(global_var["NB_AGENT"], global_var["N_DIM"])
    
    global_var["T_PM"] = 1000
    global_var["T_CONSENSUS_Y"] = 1
    global_var["T_CONSENSUS_Z"] = 15

    global_var["K1"] = 0.2
    global_var["K2"] = 0.4
    global_var["EPS"] = 0.1



    adjacency_matrix, pos, data, W, X_m_init_vect_list = init_decentralized_PCA(global_var)
    show_graph(adjacency_matrix, pos)

    cmap = plt.get_cmap('plasma')
    start, end, step = 0, 20, 2
    for i in range(start, end, step):
        
        data_copy = deepcopy(data)
        global_var["T_CONSENSUS_Y"] = i
        decentralized_PCA(global_var, data_copy, W, X_m_init_vect_list)
        #check_accuracy(global_var, data, [10**-i for i in range(1, 6)])
        plot(global_var, data_copy, None, None, show=False, label=f"T_Y = {global_var["T_CONSENSUS_Y"]}", color=cmap(i/(end-start)))

    plt.title(label=f"Evolution of the distance between U = (u1 ... uP) and Q' = (q1 ... qP)")
    plt.xlabel(xlabel="t")
    plt.ylabel(ylabel=f"Distance between U and Q")
    
    plt.axhline(y=0, color='red', linestyle='--', linewidth=0.75, label=f'0')
    plt.legend(fontsize='small', bbox_to_anchor=(1.05, 1), ncol=max(1, ((end-start)/step)//20+1))
    plt.show()