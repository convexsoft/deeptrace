
# This version is only suitable for infection networks whose node numbers are in range(len(number of nodes infection networks))

#%matplotlib inline
import copy
import random

from pylab import *
import random as rnd
import networkx as nx
import math
from contact_tracing_involve import CalGraphDS
from cal_max_min_ds import CalMaxMinDS
from cal_BFS_rand import CalBFSRandDS
# from __future__ import division

rcParams['figure.figsize'] = 6, 6  # that's default image size for this interactive session


def draw_graph(G:nx.Graph, labels=None, graph_layout='shell',
               node_size=600, node_color='blue', node_alpha=0.3,
               node_text_size=12,
               edge_color='blue', edge_alpha=0.3, edge_tickness=1,
               edge_text_pos=0.3,
               text_font='sans-serif'):
    """
    Based on: https://www.udacity.com/wiki/creating-network-graphs-with-python
    We describe a graph as a list enumerating all edges.
    Ex: graph = [(1,2), (2,3)] represents a graph with 2 edges - (node1 - node2) and (node2 - node3)
    """

    # create networkx graph


    # these are different layouts for the network you may try
    # shell seems to work best
    if graph_layout == 'spring':
        graph_pos = nx.spring_layout(G)
    elif graph_layout == 'spectral':
        graph_pos = nx.spectral_layout(G)
    elif graph_layout == 'random':
        graph_pos = nx.random_layout(G)
    else:
        graph_pos = nx.shell_layout(G)

    # draw graph
    nx.draw_networkx_nodes(G, graph_pos, node_size=node_size,
                           alpha=node_alpha, node_color=node_color)
    nx.draw_networkx_edges(G, graph_pos, width=edge_tickness,
                           alpha=edge_alpha, edge_color=edge_color)
    nx.draw_networkx_labels(G, graph_pos, font_size=node_text_size,
                            font_family=text_font)
    # show graph
    plt.show()


def build_adjacency(filename, min_degree, num_nodes):
    adjacency = [[] for i in range(num_nodes)]

    # open the datafile
    f = open(filename, 'rb')

    edges = f.readlines()

    # add all the edges
    for edge in edges:
        edge = edge.split()
        source = int(edge[0]) - 1
        destination = int(edge[1]) - 1
        if (destination < num_nodes):
            adjacency[source].append(destination)
            adjacency[destination].append(source)

    # zero out the people with fewer than min_degree friends
    while True:
        loopflag = True
        for i in range(len(adjacency)):
            if len(adjacency[i]) < min_degree and len(adjacency[i]) > 0:
                loopflag = False
                for node in adjacency[i]:
                    adjacency[node].remove(i)
                adjacency[i] = []
        if loopflag:
            break

    return adjacency


def build_adjacency_from_G(G:nx.graph, min_degree=1):
    num_nodes = G.number_of_nodes()
    adjacency = [[] for i in range(num_nodes)]

    edges = G.edges()
    print("edges:",edges)

    # add all the edges
    for edge in edges:
        # edge = edge.split()
        source = int(edge[0])
        destination = int(edge[1])
        if (destination < num_nodes):
            adjacency[source].append(destination)
            adjacency[destination].append(source)

    # zero out the people with fewer than min_degree friends
    while True:
        loopflag = True
        for i in range(len(adjacency)):
            if len(adjacency[i]) < min_degree and len(adjacency[i]) > 0:
                loopflag = False
                for node in adjacency[i]:
                    adjacency[node].remove(i)
                adjacency[i] = []
        if loopflag:
            break
    print("adjacency:",adjacency)

    return adjacency

def adjacency_to_graph(adjacency):
    graph = []
    for node in range(len(adjacency)):
        if adjacency[node]:
            for neighbors in range(len(adjacency[node])):
                graph.append((node, adjacency[node][neighbors]))
    G = nx.Graph()

    # add edges
    for edge in graph:
        G.add_edge(edge[0], edge[1])
    return G


def generate_source(adjacency):
    num_nodes = len(adjacency)
    while True:
        source = rnd.randint(0, num_nodes - 1)
        if len(adjacency[source]) > 0:
            break
    return source


# 输入未感染网络的adjacency， 其中节点的标号无意义
def si_model_rumor_spreading(source, adjacency, N):
    infctn_pattern = [-1] * N
    who_infected = [[] for i in range(N)]
    num_of_orig_node = len(adjacency)

    # adding the source node to the list of infected nodes
    infctn_pattern[0] = source
    susceptible_nodes = adjacency[source]
    susceptible_indices = [0] * len(susceptible_nodes)

    for i in range(1, N):
        # print("susceptible_nodes:", susceptible_nodes)
        # print("susceptible_indices:", susceptible_indices)
        # infect the first node
        infctd_node_idx = rnd.randrange(0, len(susceptible_nodes), 1)
        infctn_pattern[i] = susceptible_nodes[infctd_node_idx]
        who_infected[i] = [susceptible_indices[infctd_node_idx]]
        # print("[1]--who_infected:",i,who_infected)

        who_infected[susceptible_indices[infctd_node_idx]].append(i)
        # print("[2]--who_infected:",i,who_infected)

        # updating susceptible_nodes and susceptible_indices
        susceptible_indices = [susceptible_indices[j] for j in range(len(susceptible_nodes)) if susceptible_nodes[j]
                               != susceptible_nodes[infctd_node_idx]]
        # print("===susceptible_indices:", susceptible_indices)

        susceptible_nodes = [susceptible_nodes[j] for j in range(len(susceptible_nodes)) if susceptible_nodes[j]
                             != susceptible_nodes[infctd_node_idx]]
        infctd_nodes = set(infctn_pattern[:i + 1])
        new_susceptible_nodes = set(adjacency[infctn_pattern[i]])
        new_susceptible_nodes = list(new_susceptible_nodes.difference(infctd_nodes))
        susceptible_nodes = susceptible_nodes + new_susceptible_nodes
        susceptible_indices = susceptible_indices + [i] * len(new_susceptible_nodes)

    # Construct the "tree" with who_infected (node label as infection order) plus the susceptible_nodes (node labels in orignal graph)

    susceptible_nodes = list(set(susceptible_nodes))
    who_infected_plus_susceptible =  [[] for i in range(len(susceptible_nodes+who_infected))]

    num_of_who_infected = len(who_infected)

    for susceptible_i, susceptible_n in  enumerate(susceptible_nodes):
        for n in range(num_of_who_infected):
            if susceptible_n in adjacency[infctn_pattern[n]]:
                who_infected_plus_susceptible[n].append(num_of_who_infected+susceptible_i)
                who_infected_plus_susceptible[num_of_who_infected+susceptible_i].append(n)
                break

    for n in range(len(who_infected)):
        who_infected_plus_susceptible[n] = who_infected[n]+who_infected_plus_susceptible[n]
    print("susceptible_nodes:",susceptible_nodes)

    return who_infected, infctn_pattern, who_infected_plus_susceptible


def rumor_centrality_up(up_messages, who_infected, parent_node, current_node):
    if current_node == parent_node:
        for child_node in who_infected[current_node]:
            up_messages = rumor_centrality_up(up_messages, who_infected, current_node, child_node)
    elif len(who_infected[current_node]) == 1:
        up_messages[parent_node][0] += 1
        up_messages[parent_node][1] = up_messages[parent_node][1] * up_messages[current_node][1]
    # leave
    else:
        for child_node in who_infected[current_node]:
            if child_node != parent_node:
                up_messages = rumor_centrality_up(up_messages, who_infected, current_node, child_node)
                up_messages[current_node][1] = up_messages[current_node][1] * up_messages[child_node][1]
        up_messages[parent_node][0] += up_messages[current_node][0]
        up_messages[current_node][1] = up_messages[current_node][0] * up_messages[current_node][1]
    return up_messages


def rumor_centrality_down(down_messages, up_messages, who_infected, parent_node, current_node):
    if current_node == parent_node:
        down_messages[current_node] = math.log(math.factorial(len(who_infected))) - math.log((len(who_infected)))
        for child_node in who_infected[current_node]:
            down_messages[current_node] = down_messages[current_node] - math.log((up_messages[child_node][1]))
        for child_node in who_infected[current_node]:
            down_messages = rumor_centrality_down(down_messages, up_messages, who_infected, current_node, child_node)
    else:
        down_messages[current_node] = (down_messages[parent_node] + math.log(up_messages[current_node][0])) - (
            math.log((len(who_infected) - up_messages[current_node][0])))
        for child_node in who_infected[current_node]:
            if child_node != parent_node:
                down_messages = rumor_centrality_down(down_messages, up_messages, who_infected, current_node,
                                                      child_node)
    return down_messages


# 输入感染网络的adjacency， 其中节点的标号是感染的顺序
def rumor_centrality(who_infected):
    root_node = 0
    rumor_center = -1
    up_messages = []
    for i in range(len(who_infected)):
        up_messages.append([1, 1])
    down_messages = [1] * len(who_infected)
    up_message = rumor_centrality_up(up_messages, who_infected, root_node, root_node)
    down_message = rumor_centrality_down(down_messages, up_message, who_infected, root_node, root_node)
    # print("cccup_messages:",up_messages)
    # print("down_message:", down_message)

    # center = max(down_message)
    # for i in range(len(down_messages)):
    #     if down_messages[i] == center:
    #         rumor_center = i

    return down_message

def debug_test():
    adjacency = [[] for i in range(7)]
    adjacency[0] = [1, 2]
    adjacency[1] = [0, 3, 4]
    adjacency[2] = [0, 5]
    adjacency[3] = [1]
    adjacency[4] = [1]
    adjacency[5] = [2, 6]
    adjacency[6] = [5]
    rnd.seed(2)

    source = 2  # can use any arbitrary index for the root node
    who_infected, infctn_pattern = si_model_rumor_spreading(source, adjacency, 5)
    print("who_infected:", who_infected)

    up_messages = []
    for i in range(len(who_infected)):
        up_messages.append([1, 1])
    up_messages = rumor_centrality_up(up_messages, who_infected, source, source)
    print("up_messages:", up_messages)

    rumor_centrl = rumor_centrality(who_infected)
    print("rumor_centrl:", rumor_centrl)


# def get_who_infected_plus_susceptible(who_infected, orign_adjecency):

def sqrt_max_min_prob(tree_ka:nx.Graph, unift_list,rumor_centrl):
    geo_prob_dict = {}
    print("tree_ka.number_of_nodes():",tree_ka.number_of_nodes())

    tree_ka_nodes = tree_ka.nodes
    ift_node_tree_ka = list(set(tree_ka_nodes).difference(unift_list))

    for i in ift_node_tree_ka:
        cal_max_min_ds = CalMaxMinDS(tree_ka, unift_list, i)
        max_permute_prob = cal_max_min_ds.cal_max_ds()
        min_permute_prob = cal_max_min_ds.cal_min_ds()
        geo_permute_prob = 1 / 2 * (max_permute_prob + min_permute_prob)
        geo_prob_dict[i] = geo_permute_prob + rumor_centrl[i]
    a = sorted(geo_prob_dict.items(), key=lambda item: item[1], reverse=True)
    # print("sqrt_max_min_prob:", a)
    return a


def rand_BFS_prob(tree_ka:nx.Graph, unift_list,rumor_centrl):
    BFS_rand_prob_dict = {}

    tree_ka_nodes = tree_ka.nodes
    ift_node_tree_ka = list(set(tree_ka_nodes).difference(unift_list))

    for i in ift_node_tree_ka:
        cal_bfs_ds = CalBFSRandDS(tree_ka, unift_list, i)
        BFS_rand_permute_prob = cal_bfs_ds.cal_BFS_rand_ds()
        # print("final BFS_rand_permute_prob:", BFS_rand_permute_prob)
        BFS_rand_prob_dict[i] = BFS_rand_permute_prob +rumor_centrl[i]
    a = sorted(BFS_rand_prob_dict.items(), key=lambda item: item[1], reverse=True)
    # print("rand_BFS_prob:", a)
    return a

def main():
    # rnd.seed(2)
    node_num = 3500
    ER = nx.random_graphs.random_regular_graph(6, node_num)
    # ER = nx.random_graphs.erdos_renyi_graph(node_num, 0.2)
    # pos = nx.spring_layout(ER)
    # nx.draw(ER, pos, with_labels=True, node_size=150)
    # plt.show()
    adjacency = build_adjacency_from_G(ER)

    source = generate_source(adjacency)
    all_inft_num = 130
    # spread the rumor to N people and return who_infected (the adjacency list of the infection tree)
    who_infected, infected_nodes, who_infected_plus_susceptible = si_model_rumor_spreading(source, adjacency, all_inft_num)

    # obtain rumor_centrality for each node
    rumor_centrl = rumor_centrality(who_infected)


    tree_ka = adjacency_to_graph(who_infected_plus_susceptible)
    unift_list_num = len(who_infected_plus_susceptible) - len(who_infected)
    unift_list = [len(who_infected)+i for i in range(unift_list_num)]

    # obtain sqrt_max_min_prob
    res_sqrt_prob = sqrt_max_min_prob(tree_ka, unift_list, rumor_centrl)

    # obtain rand_BFS_prob
    res_rand_BFS_prob = rand_BFS_prob(tree_ka, unift_list,rumor_centrl)

    print("res_sqrt_prob:", res_sqrt_prob)
    print("res_rand_BFS_prob:", res_rand_BFS_prob)

    hops_sqrt_prob = nx.shortest_path_length(tree_ka, source=res_sqrt_prob[0][0], target=0)
    print("hops_sqrt_prob:", hops_sqrt_prob)

    hops_rand_BFS_prob = nx.shortest_path_length(tree_ka, source=res_rand_BFS_prob[0][0], target=0)
    print("hops_rand_BFS_prob:", hops_rand_BFS_prob)



    # node_color = ["red"] * len(who_infected_plus_susceptible)
    # for i in unift_list:
    #     node_color[i] = "green"
    # print("node_color:", node_color)
    # pos = nx.spring_layout(tree_ka)
    # options = {"edgecolors": "tab:gray", "node_size": 200, "alpha": 0.9}
    # nx.draw_networkx_nodes(tree_ka, pos, node_color="tab:red", **options)
    # nx.draw_networkx_nodes(tree_ka, pos, nodelist=unift_list, node_color="tab:blue", **options)
    # nx.draw_networkx_edges(tree_ka, pos, width=1.0, alpha=0.5)
    # nx.draw_networkx_labels(tree_ka, pos, font_size=8, font_color="black")
    # plt.show()




if __name__ == '__main__':
    main()

