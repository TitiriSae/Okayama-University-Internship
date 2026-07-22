from decentralized_pca import generate_L_DIM_LIST, init_decentralized_PCA, decentralized_PCA, plot, check_accuracy, show_graph

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy



def check_parameter_effect(global_var, data, W, X_m_init_vect_list):
    """
    Plot the total distance between U and Q' while changing the value of a parameter.

    return:
        None
    """
    parameter = global_var["parameter"]
    start, end, step = global_var["range"]
    check_acc = global_var["check_acc"]

    cmap = plt.get_cmap('plasma')
    save = []
    for i in range(start, end+1, step):

        data_copy = deepcopy(data)
        global_var[parameter] = i
        decentralized_PCA(global_var, data_copy, W, X_m_init_vect_list)

        print(f"\n\n{parameter} = {global_var[parameter]}")
        if check_acc != None:
            check_accuracy(global_var, data_copy, [10**-i for i in range(1, check_acc+1)])
        plot(global_var, data_copy, None, None, show=False, label=f"{parameter} = {global_var[parameter]}", color=cmap(((i-start)/step)/((end-start)/step+1)))
        save.append(data_copy)

    plt.title(label=f"Evolution of the distance between U = (u1 ... uP) and Q' = (q1 ... qP)")
    plt.xlabel(xlabel="t")
    plt.ylabel(ylabel=f"Distance between U and Q")

    plt.ylim(0)
    plt.legend(fontsize='small', bbox_to_anchor=(1.05, 1), ncol=max(1, ((end-start)/step)//20+1))
    plt.show()

    return save

def replot(global_var, save, ylim=None):
    parameter = global_var["parameter"]
    start, end, step = global_var["range"]

    cmap = plt.get_cmap('plasma')
    li = range(len(range(start, end+1, step)))
    for i in li:
        global_var[parameter] = i
        plot(global_var, save[i], None, None, show=False, label=f"{parameter} = {global_var[parameter]}", color=cmap(((i-start)/step)/((end-start)/step+1)))

    plt.title(label=f"Evolution of the distance between U = (u1 ... uP) and Q' = (q1 ... qP)")
    plt.xlabel(xlabel="t")
    plt.ylabel(ylabel=f"Distance between U and Q")

    plt.ylim(0, ylim)
    plt.legend(fontsize='small', bbox_to_anchor=(1.05, 1), ncol=max(1, ((end-start)/step)//20+1))
    plt.show()

if __name__ == "__main__":

    global_var = dict()

    global_var["SEED"] = 15
    np.random.seed(global_var["SEED"])



    global_var["NB_AGENT"] = 8
    global_var["NB_EDGE"] = 25

    global_var["N_DIM"] = 5
    global_var["P_DIM"] = 3
    global_var["L_DIM_LIST"] = generate_L_DIM_LIST(global_var["NB_AGENT"], global_var["N_DIM"])
    
    global_var["T_PM"] = 14000
    global_var["T_CONS_Y"] = 1
    global_var["T_CONS_Z"] = 10

    global_var["K1"] = 0.2
    global_var["K2"] = 0.4
    global_var["EPS"] = 0.1



    adjacency_matrix, pos, data, W, X_m_init_vect_list = init_decentralized_PCA(global_var)
    show_graph(adjacency_matrix, pos)



    global_var["parameter"] = "T_CONS_Y"
    global_var["range"] = (0, 15, 1)
    global_var["check_acc"] = 8

    save = check_parameter_effect(global_var, data, W, X_m_init_vect_list)
    for i in range(11):
        replot(global_var, save, 10**-i)

