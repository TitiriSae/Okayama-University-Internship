import numpy as np
import matplotlib.pyplot as plt
import networkx as nx





def generate_graph(global_var):
    """
    Generates randomly a symetrical adjacency matrix (with zero diagonal).

    return: 
        adjacency_matrix: list[list[int]]
        pos: dict[int, list[float]]
    """
    NB_AGENT = global_var["NB_AGENT"]
    NB_EDGE = global_var["NB_EDGE"]

    assert NB_AGENT-1 <= NB_EDGE <= (NB_AGENT*(NB_AGENT-1))/2, "Impossible graph."

    #Create positions of nodes
    random_positions = np.random.random((NB_AGENT, 2))
    pos = {i+1: random_positions[i] for i in range(len(random_positions))}

    #Compute the distance between each nodes
    edges = []
    for i in range(1, NB_AGENT+1):
        for j in range(i+1, NB_AGENT+1):
            d = np.linalg.norm(pos[i] - pos[j])
            edges.append((d, i, j))

    #Sort the edges by distance
    edges.sort(key=lambda x: x[0])

    #Create the complete graph
    graph_complete = nx.Graph()
    graph_complete.add_nodes_from(range(1, NB_AGENT+1))
    for d, i, j in edges:
        graph_complete.add_edge(i, j, weight=d)

    #Find minimum size spanning tree
    graph = nx.minimum_spanning_tree(graph_complete)

    #Add edges till the graph has M edges
    nb_edges = graph.number_of_edges()
    for d, i, j in edges:
        if nb_edges == NB_EDGE:
            break
        if not graph.has_edge(i, j):
            graph.add_edge(i, j, weight=d)
            nb_edges += 1

    #Verifications on the graph
    assert nx.is_connected(graph)
    assert graph.number_of_nodes() == NB_AGENT
    assert graph.number_of_edges() == NB_EDGE

    #Convert graph to adjacency matrix
    adjacency_matrix = np.where(nx.to_numpy_array(graph), 1, 0)

    return adjacency_matrix, pos

def generate_initial_values(global_var):
    """
    Generate random values between 0 and VAL_RANGE_DA for each node.

    return:
        initial_values: list[float]
    """
    NB_AGENT = global_var["NB_AGENT"]

    #Random initial values
    initial_values = global_var["VAL_RANGE"]*np.random.rand(NB_AGENT)
    return initial_values

def init_graph(global_var, adjacency_matrix):
    """
    Return instance's data defined with by the graph and the initial values of each nodes in dict format.

    return: 
        data: dict[int, dict[str, Any]] 
            data -> (i) node i -> ('n') neighbors of node i: list[int]
                               -> ('d') degree of node i: int
    """
    NB_AGENT = global_var["NB_AGENT"]

    #Verification of lists length
    assert np.shape(adjacency_matrix)[0] == np.shape(adjacency_matrix)[1]

    data = dict()

    for i in range(1, NB_AGENT+1):
        data[i] = dict()
        #Transform the adjacency matrix into an adjacency list
        data[i]["n"] = [j for j in range(1, NB_AGENT+1) if adjacency_matrix[i-1][j-1]!=0]
        #Degree of the node
        data[i]["d"] = len(data[i]["n"])
    
    return data



def init_initial_values(global_var, data, initial_values):
    """
    Add an entry "x" to the dictionary to store the values' history during the averaging consensus.
    data: dict[int, dict[str, Any]] 
        data -> (i) node i -> ('n') neighbors of node i: list[int]
                           -> ('d') degree of node i: int
                           -> ('x') history of values of node i: list[float]
    return:
        None
    """
    NB_AGENT = global_var["NB_AGENT"]

    #Verification of lists length
    assert len(data) == len(initial_values)

    for i in range(1, NB_AGENT+1):
        #Storage of a history of the values taken
        data[i]["x"] = [initial_values[i-1]]



def init_local_degree_weight(global_var, data):
    """
    The weight matrix W is defined as :
    Wij = 0 if {i, j} not in E and i != j                                           (sparsity constraint of W)
    Wij = 1/(max{deg_i, deg_j})                                                     (local degree weight)
    Wii is chosen such that the matrix sums to 1 across rows and columns.           (theorem 1)

    return:
        W: list[list[float]]
    """
    NB_AGENT = global_var["NB_AGENT"]

    #Ignoring Wii for now
    W = [
        [0 if (j not in data[i]["n"] and j!=i) or j==i else 1/(max(data[i]["d"], data[j]["d"])) for j in range(1, NB_AGENT+1)] 
        for i in range(1, NB_AGENT+1)
        ]
    
    #Set Wii
    for i in range(NB_AGENT):
        #W is symmetrical
        W[i][i] = 1 - np.sum(W[i])

    return W



def distributed_linear_iteration(global_var, data, W):
    """
    Apply the distributed linear iterations for t iterations on the "x" entry of the data.

    return: 
        None
    """
    NB_AGENT = global_var["NB_AGENT"]
    T_DA = global_var["T_DA"]


    #Convergence conditions verification
    assert np.allclose(np.sum(W, axis=0), np.ones(NB_AGENT), atol=1e-12)
    assert np.allclose(np.sum(W, axis=1), np.ones(NB_AGENT), atol=1e-12)
    assert  np.max(np.abs(
                np.linalg.eigvals(
                    W - (1/NB_AGENT)*np.ones((NB_AGENT, NB_AGENT))
                )
            )) < 1

    for t in range(1, T_DA+1):

        for i in range(1, NB_AGENT+1):
            #Iteration
            x_i_t = W[i-1][i-1]*data[i]["x"][t-1]
            for j in data[i]["n"]:
                x_i_t += W[i-1][j-1]*data[j]["x"][t-1]
            #Store value in the history
            data[i]["x"].append(x_i_t)
        """
        if all( [abs(data[j]["x"][t] - data[i]["x"][t-1]) < CONSENSUS_VAL for j in range(1, NB_AGENT+1)] ):
            set_T_DA(t)
            return
        """



#Getter functions
def get_consensus_val(data):
    """
    Returns the consenseus value wanted for an instance of the problem.

    return: 
        float
    """

    return np.mean([data[node]['x'][0] for node in data])

def get_x_t(data, t):
    """
    Returns the vector x at time t.

    return:
        x_t: list[float]
    """

    x_t = [data[node]["x"][t] for node in data]
    return x_t



#Display functions
def show_graph(adjacency_matrix, positions=None):
    """
    Display the graph associated to the instance of the problem.

    return: 
        None
    """

    graph_0 = nx.from_numpy_array(adjacency_matrix)
    graph_1 = nx.relabel_nodes(graph_0, {i: i+1 for i in graph_0.nodes()})
    nx.draw(graph_1, positions, with_labels=True, node_size=100, font_size=8)
    plt.show()

def plot(global_var, data, i=None):
    """
    Plot the data, especially the evolution of the values taken by the nodes.
    The plot is different depending on the parameter i:
    - if i isn't set, the evolution of every agents will be shown in one single plot.
    - if i is set as 0, the evolution of every agents will be shown but separately.
    - if i is set as the ID of an agent, the evolution of this agent only will be shown.

    return:
        None
    """
    NB_AGENT = global_var["NB_AGENT"]

    def _get_x_i_range(data, node, t0=0, t1=len(data[1]["x"])-1):
        """
        Returns the all the values taken by a node from time t0 to time t1, xi(t0) to xi(t1).

        return:
            x_i_range: list[float]
        """
        x_i_range = [data[node]["x"][t] for t in range(t0, t1+1)]
        return x_i_range

    def _plot_x_i_t(data, i, color=None):
        """
        Plot the evolution of the value x_i over time t.

        return:
            None
        """
        plt.plot(np.arange(len(data[1]["x"])), _get_x_i_range(data, i), color=color ,label=f'x_{i}(t)')

    def _plot_x_t(data):
        """
        Plot the evolution of the vector x over time t.
        
        return:
            None
        """
        cmap = plt.get_cmap('plasma')
        for i in range(1, NB_AGENT+1):
            _plot_x_i_t(data, i, cmap(i/(NB_AGENT)))


    plt.title(label=f'Evolution of x_i(t)')
    plt.xlabel(xlabel='t')
    plt.ylabel(ylabel=f'x_i(t)')

    if i==None:
        _plot_x_t(data)
    elif i==0:
        for a in range(1, NB_AGENT+1):
            plot(data, a)
        return
    elif 1<= i <= NB_AGENT:
        _plot_x_i_t(data, i)
    else:
        raise KeyError
        
    plt.axhline(y=get_consensus_val(data), color='red', linestyle='--', linewidth=0.75, label=f'consensus_value = {get_consensus_val(data):.3f}')

    plt.legend(fontsize='small', bbox_to_anchor=(1.05, 1), ncol=max(1, NB_AGENT//20+1))
    plt.show()





if __name__ == "__main__":

    global_var = dict()

    global_var["SEED"] = 42
    np.random.seed(global_var["SEED"])

    global_var["VAL_RANGE"] = 100

    #Number of nodes NB_AGENT
    #Number of edges NB_EDGE
    #Number of iteration T_DA
    global_var["NB_AGENT"] = 50
    global_var["NB_EDGE"] = 200
    global_var["T_DA"] = 50

    global_var["CONSENSUS_VAL"] = 1e-10




    #Example of a adjacency matrix of an undirected graph
    """
    N = 8
    M = 9
    T = 30

    n1 = np.array([0, 1, 1, 0, 1, 0, 1, 0])
    n2 = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    n3 = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    n4 = np.array([0, 0, 0, 0, 0, 1, 1, 1])
    n5 = np.array([1, 0, 0, 0, 0, 1, 0, 1])
    n6 = np.array([0, 0, 0, 1, 1, 0, 0, 0])
    n7 = np.array([1, 0, 0, 1, 0, 0, 0, 0])
    n8 = np.array([0, 0, 0, 1, 1, 0, 0, 0])

    adj = np.array([n1, n2, n3, n4, n5, n6, n7, n8])
    x_0 = np.array([4, 3, 2, 1, 6, 5, 4, 7])
    show_graph(adj)
    """

    #Random generation
    adj, pos = generate_graph(global_var)
    data = init_graph(global_var, adj)
    show_graph(adj, pos)

    x_0 = generate_initial_values(global_var)
    init_initial_values(global_var, data, x_0)

    #Algorithm 
    W = init_local_degree_weight(global_var, data)
    distributed_linear_iteration(global_var, data, W)

    plot(global_var, data)