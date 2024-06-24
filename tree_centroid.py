import networkx as nx
import matplotlib.pyplot as plt

import random


def generate_regular_tree(node_num, node_degree):
    edge_set = []
    initial_node = 0
    add_node = 0
    for i in range(node_degree):
        add_node += 1
        edge_set.append((initial_node, add_node))
    while add_node < node_num:
        initial_node += 1
        for i in range(node_degree-1):
            add_node += 1
            if add_node < node_num:
                edge_set.append((initial_node, add_node))
    print(edge_set)
    rt = nx.Graph()
    rt.add_edges_from(edge_set)
    return rt


def generate_regular_tree_random(node_num, node_degree):
    edge_set = []
    initial_node = 0
    add_node = 0
    leaf_node_list = []
    for i in range(node_degree):
        add_node += 1
        edge_set.append((initial_node, add_node))
        leaf_node_list.append(add_node)
    initial_node += 1
    while add_node < node_num:
        parent_node = random.sample(leaf_node_list, 1)
        # print("parent:", parent_node)
        # print("leaf_node_list:", leaf_node_list)

        leaf_node_list.remove(parent_node[0])
        for i in range(node_degree-1):
            add_node += 1
            if add_node < node_num:
                edge_set.append((parent_node[0], add_node))
                leaf_node_list.append(add_node)
    # print(edge_set)
    rt = nx.Graph()
    rt.add_edges_from(edge_set)
    return rt


def cal_centroid_of_tree(tree: nx.graph):
    pass


if __name__ == '__main__':
    reguler_tree = generate_regular_tree_random(30, 4)
    pos = nx.spring_layout(reguler_tree)
    nx.draw(reguler_tree, pos, with_labels=True, node_size=150)
    plt.show()



