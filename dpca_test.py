from decentralized_pca import generate_L_DIM_LIST, init_decentralized_PCA, decentralized_PCA, plot, check_accuracy, show_graph
from decentralized_pca import spectral_decomposition, covariance_matrix

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from copy import deepcopy



def check_parameter_effect(global_var, data, W, X_m_init_vect_list):
    """
    Plot the total distance between U and Q' while changing the value of a parameter.

    return:
        save: list[dict[Any, Any]]
    """
    parameter = global_var["parameter"]
    val = global_var["val"]
    check_acc = global_var["check_acc"]

    cmap = plt.get_cmap('plasma')
    save = []

    color = list(np.linspace(0, 1, len(val)))
    for i in range(len(val)):

        data_copy = deepcopy(data)
        global_var[parameter] = val[i]
        decentralized_PCA(global_var, data_copy, W, X_m_init_vect_list)

        print(f"\n\n{parameter} = {global_var[parameter]}")
        check_accuracy(global_var, data_copy, [10**-i for i in range(1, check_acc+1)])
        plot(global_var, data_copy, None, None, show=False, label=f"{parameter} = {global_var[parameter]}", color=cmap(color[i]))
        save.append(data_copy)

    plt.title(label=f"Evolution of the distance between U = (u1 ... uP) and Q' = (q1 ... qP)")
    plt.xlabel(xlabel="t")
    plt.ylabel(ylabel=f"Distance between U and Q")

    plt.ylim(0)
    plt.legend(fontsize='small', bbox_to_anchor=(1.05, 1), ncol=len(val)//20+1)
    plt.show()

    return save



def replot(global_var, save, ylim=None):
    """
    Replot every saved data dictionnary without computing again.

    return:
        None
    """
    parameter = global_var["parameter"]
    val = global_var["val"]

    cmap = plt.get_cmap('plasma')
    color = np.linspace(0, 1, len(val))
    for i in range(len(val)):
        global_var[parameter] = val[i]
        plot(global_var, save[i], None, None, show=False, label=f"{parameter} = {global_var[parameter]}", color=cmap(color[i]))

    plt.title(label=f"Evolution of the distance between U = (u1 ... uP) and Q' = (q1 ... qP)")
    plt.xlabel(xlabel="t")
    plt.ylabel(ylabel=f"Distance between U and Q")

    plt.ylim(0, ylim)
    plt.legend(fontsize='small', bbox_to_anchor=(1.05, 1), ncol=len(val)//20+1)
    plt.show()

if __name__ == "__main__":

    global_var = dict()

    global_var["SEED"] = 15
    np.random.seed(global_var["SEED"])



    global_var["NB_AGENT"] = 8
    global_var["NB_EDGE"] = 25

    #global_var["NB_AGENT"], global_var["NB_EDGE"] = 8, 7
    star_g = np.array([
        [0, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
    ]) 

    #global_var["NB_AGENT"], global_var["NB_EDGE"] = 8, 7
    path_g = np.array([
        [0, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0],
    ])

    #global_var["NB_AGENT"], global_var["NB_EDGE"] = 8, 8
    circle_g = np.array([
        [0, 1, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 1, 0],
    ])

    #global_var["NB_AGENT"], global_var["NB_EDGE"] = 8, 16
    regular_g = np.array([
        [0, 1, 1, 0, 0, 0, 1, 1],
        [1, 0, 1, 1, 0, 0, 0, 1],
        [1, 1, 0, 1, 1, 0, 0, 0],
        [0, 1, 1, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 1, 1, 0],
        [0, 0, 0, 1, 1, 0, 1, 1],
        [1, 0, 0, 0, 1, 1, 0, 1],
        [1, 1, 0, 0, 0, 1, 1, 0],
    ])



    global_var["N_DIM"] = 5
    global_var["P_DIM"] = 3
    global_var["L_DIM_LIST"] = generate_L_DIM_LIST(global_var)
    
    global_var["T_PM"] = 10000
    global_var["T_Y"] = 1000
    global_var["T_Z"] = 1000

    global_var["K1"] = 0.2
    global_var["K2"] = 0.4
    global_var["EPS"] = 0.1

    global_var["CONSENSUS_EPS"] = 1e-8
    global_var["CONVERGENCE_EPS"] = 1e-10


    global_var["NB_AGENT"], global_var["NB_EDGE"] = 7, 7
    G = nx.cycle_graph(global_var["NB_AGENT"])
    adjacency_matrix = nx.to_numpy_array(G, dtype=int)

    adjacency_matrix, pos, data, W, X_m_init_vect_list = init_decentralized_PCA(global_var, adjacency_matrix)
    show_graph(adjacency_matrix, pos)

    global_var["parameter"] = "CONVERGENCE_EPS"
    global_var["val"] = [10**-i for i in range(1, 13)]
    global_var["check_acc"] = 15

    data["TY"], data["TZ"] = [], []
    
    save = check_parameter_effect(global_var, data, W, X_m_init_vect_list)
    #for i in range(11):replot(global_var, save, 10**-i)

    for i in range(len(global_var["val"])):
        print(f"{global_var["parameter"]} = {global_var["val"][i]}")
        check_accuracy(global_var, save[i], [10**-i for i in range(1, 15+1)])

    for i in range(len(global_var["val"])):
        print(global_var["val"][i], np.mean(save[i]["TY"]), np.mean(save[i]["TZ"]), len(save[i][1]["U"])-1)