import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

ROUND = 3
VAL_RANGE = 100

N = 100
M = 400
T = 15



def generate_instance(N, M):
    """
    Generates randomly a symetrical adjacency matrix (with zero diagonal) and the initial value of the nodes.
    """
    assert N-1 <= M <= (N*(N-1))/2

    #create positions of nodes
    pos = np.random.random((N, 2))

    #compute the distance between each nodes
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(pos[i] - pos[j])
            edges.append((d, i, j))

    #sort the edges by distance
    edges.sort(key=lambda x: x[0])

    #create the complete graph
    graph_complete = nx.Graph()
    graph_complete.add_nodes_from(range(N))
    for d, i, j in edges:
        graph_complete.add_edge(i, j, weight=d)

    #find minimum size spanning tree
    graph = nx.minimum_spanning_tree(graph_complete)

    #add edges till the graph has M edges
    nb_edges = graph.number_of_edges()
    for d, i, j in edges:
        if nb_edges == M:
            break
        if not graph.has_edge(i, j):
            graph.add_edge(i, j, weight=d)
            nb_edges += 1

    #verifications on the graph
    assert nx.is_connected(graph)
    assert graph.number_of_nodes() == N
    assert graph.number_of_edges() == M

    #convert graph to adjacency matrix
    adjacency_matrix = np.where(nx.to_numpy_array(graph), 1, 0)

    #random initial values
    initial_values = VAL_RANGE*np.random.rand(N)

    return adjacency_matrix, pos, initial_values


def initialize_instance(adjacency_matrix, initial_values):
    """
    Return instance's data defined with by the graph and the initial values of each nodes in dict format.
    data -> node i -> neighbors n
                   -> degree d
                   -> history of values x
    """
    #verification of lists length
    assert np.shape(adjacency_matrix)[0] == np.shape(adjacency_matrix)[1]
    assert np.shape(adjacency_matrix)[0] == len(initial_values)

    N = len(adjacency_matrix)
    data = dict()

    for i in range(1, N+1):
        data[i] = dict()
        #transform the adjacency matrix into an adjacency list
        data[i]["n"] = [j for j in range(1, N+1) if adjacency_matrix[i-1][j-1]!=0]
        #degree of the node
        data[i]["d"] = len(data[i]["n"])
        #storage of a history of the values taken
        data[i]["x"] = [initial_values[i-1]]
    
    return data



def initialize_local_degree_weight(data):
    """
    The weight matrix W is defined as :
    Wij = 0 if {i, j} not in E and i != j                                           (sparsity constraint of W)
    Wij = 1/(max{deg_i, deg_j})                                                     (local degree weight)
    Wii is chosen such that the matrix sums to 1 across rows and columns.           (theorem 1)
    """

    N = len(data)

    #Ignoring Wii for now
    W = [
        [0 if (j not in data[i]["n"] and j!=i) or j==i else 1/(max(data[i]["d"], data[j]["d"])) for j in range(1, N+1)] 
        for i in range(1, N+1)
        ]
    
    #Set Wii
    for i in range(N):
        #W is symmetrical
        W[i][i] = 1 - np.sum(W[i])

    return W



def algo(data, W, time):
    """
    Apply the distributed linear iterations for t iterations.
    """

    N = len(data)

    #convergence conditions verification
    assert np.allclose(np.sum(W, axis=0), np.ones(N), atol=1e-12)
    assert np.allclose(np.sum(W, axis=1), np.ones(N), atol=1e-12)
    assert  np.max(np.abs(
                np.linalg.eigvals(
                    W - (1/N)*np.ones((N, N))
                )
            )) < 1

    for t in range(1, time+1):

        for i in range(1, N+1):
            #iteration
            x_i_t = W[i-1][i-1]*data[i]["x"][t-1]
            for j in data[i]["n"]:
                x_i_t += W[i-1][j-1]*data[j]["x"][t-1]
            #store value in the history
            data[i]["x"].append(x_i_t)



#get
def get_consensus_val(data):
    return np.mean([data[node]['x'][0] for node in data])

def get_x_t(data, t):
    x_t = [data[node]["x"][t] for node in data]
    return x_t

def get_x_i_range(data, node, t0, t1):
    x_i_range = [data[node]["x"][t] for t in range(t0, t1+1)]
    return x_i_range


#show
def show_graph(adjacency_matrix, positions):
    graph = nx.from_numpy_array(adjacency_matrix)
    nx.draw(graph, positions, with_labels=True, labels={i: i+1 for i in graph.nodes}, node_size=100, font_size=8)
    plt.show()

def plot(data, i=None):
    """
    Plot the data, especially the evolution of the values taken by the nodes.
    The plot is different depending on the parameter i:
    - if i isn't set, the evolution of every agents will be shown in one single plot.
    - if i is set as 0, the evolution of every agents will be shown but separately.
    - if i is set as the ID of an agent, the evolution of this agent only will be shown.
    """

    def _plot_x_i_t(data, i, color=None):
        """
        Plot the evolution of the value x_i over time t.
        """
        plt.plot(np.arange(T+1), get_x_i_range(data, i, 0, T), color=color ,label=f'x_{i}(t)')

    def _plot_x_t(data):
        """
        Plot the evolution of the vector x over time t.
        """
        cmap = plt.get_cmap('plasma')
        for i in range(1, N+1):
            _plot_x_i_t(data, i, cmap(i/(N)))


    plt.title(label=f'Evolution of x_i(t)')
    plt.xlabel(xlabel='t')
    plt.ylabel(ylabel=f'x_i(t)')

    if i==None:
        _plot_x_t(data)
    elif i==0:
        for a in range(1, N+1):
            plot(data, a)
        return
    else:
        _plot_x_i_t(data, i)
        
    plt.axhline(y=get_consensus_val(data), color='red', linestyle='--', linewidth=0.75, label=f'consensus_value = {get_consensus_val(data):.{ROUND}f}')

    plt.legend(fontsize='small', bbox_to_anchor=(1.05, 1), ncol=max(1, N//20+1))
    plt.show()



"""
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
"""



adj, pos, x_0 = generate_instance(N, M)

data = initialize_instance(adj, x_0)

print(f"x(0) = {np.round(get_x_t(data, 0), ROUND)}")
print(f"consensus : {get_consensus_val(data):.{ROUND}f}")

W = initialize_local_degree_weight(data)

algo(data, W, T)

print(f"x(t) = {get_x_t(data, T)}")

show_graph(adj, pos)
plot(data)



