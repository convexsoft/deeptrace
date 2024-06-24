import networkx as nx
import matplotlib.pyplot as plt
import copy
import ast
import random
import math
from tree_centroid import generate_regular_tree
from tree_centroid import generate_regular_tree_random
import numpy as np
import csv

class NFeatureDict(dict):
    def __missing__(self, key):
        value = self[key] = type(self)()
        return value


class CalGraphDS(object):
    def __init__(self, graph: nx.Graph, unift_list: list, source: int):
        self.graph = graph
        self.unift_list = unift_list
        self.source = source
        self.nfeature_temp = self.preprocess()
        self.node_num = self.graph.number_of_nodes()

    def preprocess(self):
        nfeature_temp = NFeatureDict()

        graph_deg = nx.degree(self.graph)
        for v in graph_deg:
            nfeature_temp[v[0]]["degree"] = v[1]
            neighbor_list = self.graph.neighbors(v[0])
            uninfected_edge_num = len(set(self.unift_list).difference(set(neighbor_list)))
            nfeature_temp[v[0]]["uninft_edge_num"] = uninfected_edge_num
        return nfeature_temp

    def cal_graph_ds(self):
        no_add_select_pool = copy.copy(self.unift_list)
        no_add_select_pool.append(self.source)
        edge_prob_num = 1
        permutation_dict = {}
        permutation_dict[str([self.source])] = 1/edge_prob_num
        new_list = []

        while len(new_list) < self.node_num - len(self.unift_list):
            new_permutation_dict = {}
            for k in list(permutation_dict.keys()):
                # print("k:",k)
                temp = ast.literal_eval(k)
                temp_all_neighbors = []
                for node in temp:
                    temp_all_neighbors = temp_all_neighbors + list(self.graph.neighbors(node))
                # print("**temp_all_neighbors:", temp_all_neighbors)
                temp_all_neighbors = list(set(temp_all_neighbors).difference(set(temp)))
                # print("temp_all_neighbors:", temp_all_neighbors)
                # new_permutation_prob = permutation_dict[k]/max(1,len(temp_all_neighbors))
                temp_out_edge_num = 0
                for i in temp_all_neighbors:
                    # print("i:", i)
                    overlap_edge = set(list(self.graph.neighbors(i))).intersection(set(temp))
                    temp_out_edge_num = temp_out_edge_num + len(overlap_edge)
                    # print("temp_out_edge_num:", temp_out_edge_num)
                temp_inft_neighbors = set(temp_all_neighbors).difference(set(self.unift_list))
                for i in temp_inft_neighbors:
                    new_temp = temp + [i]
                    overlap_edge = set(list(self.graph.neighbors(i))).intersection(set(temp))
                    # print("overlap_edge:", overlap_edge)
                    # print("len(overlap_edge)/temp_out_edge_num:", len(overlap_edge)/temp_out_edge_num)

                    new_permutation_prob = permutation_dict[k]*(len(overlap_edge)/temp_out_edge_num)
                    new_permutation_dict[str(new_temp)]= new_permutation_prob
            if len(new_permutation_dict)>0:
                permutation_dict = copy.copy(new_permutation_dict)
            # print("permutation_dict:", permutation_dict)
            # if len(permutation_dict)
            new_list = list(permutation_dict.keys())[0]
            new_list = ast.literal_eval(new_list)

        # print("all_permutation_dict:", permutation_dict)
        prob = sum(list(permutation_dict.values()))
        # print("prob:", prob)
        return prob


def mle_node_get(input_graph: nx.Graph, unift_list: list):
    ift_node_list = list(set(list(input_graph.nodes)).difference(set(unift_list)))
    node_prob_dict = {}
    for node in ift_node_list:
        print("node:", node)
        calds = CalGraphDS(input_graph, unift_list, node)
        prob = calds.cal_graph_ds()
        node_prob_dict[node] = prob
    max_node = []
    max_value = max(node_prob_dict.values())
    for k,v in node_prob_dict.items():
        if v == max_value:
            max_node.append(k)
    # max_node = max(node_prob_dict, key = node_prob_dict.get)
    print("node_prob_dict:", node_prob_dict)
    print("max_node:", max_node)
    return max_node


def get_uninfected_node_list(g:nx.graph, uninft_leaf_rate):
    degree_list = nx.degree(g)
    print("degree_list:",degree_list)
    # uninft_node_list = []
    # ift_node_list = [i[0] for i in degree_list if i[1] > 1]
    # print("=======:", ift_node_list)
    # for node in ift_node_list:
    #     neighbor_list = list(nx.neighbors(g, node))
    #     # print("neighbor_list:", neighbor_list)
    #     neighbo_leaf_list = [i for i in neighbor_list if nx.degree(g, i) == 1]
    #     uninft_one_list = random.sample(neighbo_leaf_list, math.ceil(1 * len(neighbo_leaf_list)))
    #     uninft_node_list = uninft_node_list + uninft_one_list
    # uninft_node_list = list(set(uninft_node_list))
    # print("uninft_node_list:", uninft_node_list)

    leafed_index_list = [i[0] for i in degree_list if i[1] == 1]
    random.seed(10)
    uninft_size = len(leafed_index_list)
    print("leafed_index_list:", leafed_index_list)
    uninft_node_list = random.sample(leafed_index_list, k=max(math.ceil(uninft_size*uninft_leaf_rate),1))
    # uninft_node_list = random.sample(leafed_index_list, math.ceil(1 * uninft_size))
    print("uninft_node_list:",uninft_node_list)
    return uninft_node_list


def main():
    node_num = 16
    # ER = nx.random_graphs.barabasi_albert_graph(node_num, 2)

    # ER = nx.random_graphs.random_regular_graph(4, node_num)
    ER = nx.random_graphs.erdos_renyi_graph(node_num, 0.3)
    # ER = nx.minimum_spanning_tree(ER, algorithm="kruskal")

    # g = Graph.Tree(node_num, 3)
    # g1 = g.get_edgelist()
    # ER = nx.Graph(g1)

    # ER = generate_regular_tree_random(node_num, 4)
    node_list = list(range(node_num))
    # print(node_list)
    all_uninft_node_list = []
    # all_uninft_node_list = random.sample(node_list, math.ceil(node_num * 0.2))
    uninft_leaf_rate = 0.4
    all_uninft_node_list = all_uninft_node_list + get_uninfected_node_list(ER, uninft_leaf_rate)
    node_color = ["r"] * node_num
    print("all_uninft_node_list:", all_uninft_node_list)
    all_ift_node_list = list(set(list(ER.nodes)).difference(set(all_uninft_node_list)))

    for i in all_uninft_node_list:
        node_color[i] = "b"
    pos = nx.spring_layout(ER)
    nx.draw(ER, pos, with_labels=True, node_size=150, node_color=node_color)
    plt.show()

    initial_node = all_ift_node_list[0]
    # initial_node = 5
    graph_node = [initial_node]
    subg = ER.subgraph(graph_node)
    move_center_node_list = [initial_node]
    while len(graph_node) <= node_num:

        graph_node_temp = graph_node.copy()
        graph_node_temp = list(set(graph_node_temp).difference(set(all_uninft_node_list)))
        for node in graph_node_temp:
            graph_node = graph_node + (list(ER.neighbors(node)))
        graph_node = list(set(graph_node))
        subg = ER.subgraph(graph_node)
        sub_inft_node = list(set(graph_node).intersection(set(all_uninft_node_list)))
        print("graph_node:", graph_node)
        pos = nx.spring_layout(subg)
        sub_node_color = []
        for n in graph_node:
            sub_node_color.append(node_color[n])
        nx.draw(subg, pos, with_labels=True, node_size=150, node_color=sub_node_color)
        plt.show()
        center_node = mle_node_get(subg, sub_inft_node)
        move_center_node_list.append(center_node)
        print("move_center_node_list:", move_center_node_list)
        if len(graph_node) == node_num:
            break
    print("*********************")
    pos = nx.spring_layout(ER)
    nx.draw(ER, pos, with_labels=True, node_size=150, node_color=node_color)
    plt.show()


# Every leaf node in the subgraph is not infected.
def main2():
    node_num = 15
    # ER = nx.random_graphs.barabasi_albert_graph(node_num, 2)

    # ER = nx.random_graphs.random_regular_graph(4, node_num)
    ER = nx.random_graphs.erdos_renyi_graph(node_num, 0.3)
    ER = nx.minimum_spanning_tree(ER, algorithm="kruskal")

    # g = Graph.Tree(node_num, 3)
    # g1 = g.get_edgelist()
    # ER = nx.Graph(g1)

    # ER = generate_regular_tree_random(node_num, 3)
    node_list = list(range(node_num))
    # print(node_list)
    all_uninft_node_list = []
    # all_uninft_node_list = random.sample(node_list, math.ceil(node_num * 0.2))
    uninft_leaf_rate = 0.4
    all_uninft_node_list = all_uninft_node_list + get_uninfected_node_list(ER,uninft_leaf_rate=uninft_leaf_rate)
    node_color = ["r"] * node_num
    # print("all_uninft_node_list:", all_uninft_node_list)
    all_ift_node_list = list(set(list(ER.nodes)).difference(set(all_uninft_node_list)))


    initial_node = all_ift_node_list[0]
    # initial_node = 5
    for i in all_ift_node_list:
        initial_node = i
        graph_node = [initial_node]
        move_center_node_list = [initial_node]

        temp_uninft_node_list = []
        for i in range(2):
            graph_node_temp = graph_node.copy()
            graph_node_temp = list(set(graph_node_temp).difference(set(all_uninft_node_list)))
            for node in graph_node_temp:
                graph_node = graph_node + (list(ER.neighbors(node)))
                if i == 1:
                    # print("=========:", node)

                    add_leaf_node_list = list(set(list(ER.neighbors(node))).difference(set(graph_node_temp)))
                    # print("add_leaf_node_list:", add_leaf_node_list)

                    temp_uninft_node_list = temp_uninft_node_list + all_uninft_node_list + add_leaf_node_list
            graph_node = list(set(graph_node))

        # print("temp_uninft_node_list:", temp_uninft_node_list)
        subg = ER.subgraph(graph_node)
        sub_inft_node = list(set(graph_node).intersection(set(temp_uninft_node_list)))
        # print("graph_node:", graph_node)
        pos = nx.spring_layout(subg)

        for i in temp_uninft_node_list:
            node_color[i] = "b"
        sub_node_color = []
        for n in graph_node:
            sub_node_color.append(node_color[n])
        # nx.draw(subg, pos, with_labels=True, node_size=150, node_color=sub_node_color)
        # plt.show()
        center_node = mle_node_get(subg, sub_inft_node)
        move_center_node_list.append(center_node)

        while len(graph_node) <= node_num:
            temp_uninft_node_list = []

            graph_node_temp = graph_node.copy()
            graph_node_temp = list(set(graph_node_temp).difference(set(all_uninft_node_list)))
            for node in graph_node_temp:
                graph_node = graph_node + (list(ER.neighbors(node)))
                add_leaf_node_list = list(set(list(ER.neighbors(node))).difference(set(graph_node_temp)))
                # print("===node:", node)
                # print("===add_leaf_node_list:", add_leaf_node_list)

                temp_uninft_node_list = temp_uninft_node_list + add_leaf_node_list
                # print("===temp_uninft_node_list:", temp_uninft_node_list)

            graph_node = list(set(graph_node))
            subg = ER.subgraph(graph_node)
            temp_uninft_node_list = temp_uninft_node_list + all_uninft_node_list
            sub_inft_node = list(set(graph_node).intersection(set(temp_uninft_node_list)))
            # print("graph_node:", graph_node)
            pos = nx.spring_layout(subg)
            # print("!!!temp_uninft_node_list:", temp_uninft_node_list)
            node_color = ["r"] * node_num
            for i in temp_uninft_node_list:
                node_color[i] = "b"
            sub_node_color = []
            for n in graph_node:
                sub_node_color.append(node_color[n])
            # nx.draw(subg, pos, with_labels=True, node_size=150, node_color=sub_node_color)
            # plt.show()
            center_node = mle_node_get(subg, sub_inft_node)
            move_center_node_list.append(center_node)
            print("move_center_node_list:", move_center_node_list)
            if len(graph_node) == node_num:
                break
        print("*********************")
    pos = nx.spring_layout(ER)
    nx.draw(ER, pos, with_labels=True, node_size=150, node_color=node_color)
    plt.show()


def test():
    node_num = 10
    # ER = nx.random_graphs.barabasi_albert_graph(node_num, 2)
    ER = nx.random_graphs.erdos_renyi_graph(node_num, 0.3)
    ER = nx.minimum_spanning_tree(ER, algorithm="kruskal")
    node_list = list(range(7))
    print(node_list)
    uninfected_node_list = random.sample(node_list, math.ceil(node_num * 0.3))
    node_color = ["r"] * node_num
    print(uninfected_node_list)

    for i in uninfected_node_list:
        node_color[i] = "b"
    pos = nx.spring_layout(ER)
    nx.draw(ER, pos, with_labels=True, node_size=150, node_color=node_color)
    plt.show()
    center_node = mle_node_get(ER, uninfected_node_list)


def test3():
    edges_g = [(1,2),(2,3),(2,4),(2,13),(2,14),(3,7),(3,8),(3,11),(3,4),(4,5),(4,6),(4,12),(5,10),(6,9),(12,15)]
    test_g = nx.Graph()
    test_g.add_edges_from(edges_g)
    pos = nx.spring_layout(test_g)
    nx.draw(test_g, pos, with_labels=True, node_size=150)
    plt.show()
    unift_list = [7,8,9,10,11,13,14,15]
    mle_node_get(test_g, unift_list)


def test4():
    node_num = 50
    # ER = nx.random_graphs.barabasi_albert_graph(node_num, 1)
    ER = nx.random_graphs.random_regular_graph(5, node_num)
    # ER = nx.random_graphs.erdos_renyi_graph(node_num, 0.2)
    ER_tree = nx.minimum_spanning_tree(ER, algorithm="kruskal")

    with open("./edge_test4.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for i in ER_tree.edges:
            writer.writerow(list(i))


    uninft_leaf_rate = 1
    all_uninft_node_list = get_uninfected_node_list(ER_tree,uninft_leaf_rate=uninft_leaf_rate)
    print("inft_node_num:", node_num-len(all_uninft_node_list))
    mle_node_get(ER, all_uninft_node_list)


    node_color = ["red"] * node_num
    for i in all_uninft_node_list:
        node_color[i] = "green"
    print("node_color:",node_color)
    pos = nx.spring_layout(ER_tree)
    options = {"edgecolors": "tab:gray", "node_size": 200, "alpha": 0.9}
    nx.draw_networkx_nodes(ER_tree, pos, node_color="tab:red", **options)
    nx.draw_networkx_nodes(ER_tree, pos, nodelist=all_uninft_node_list, node_color="tab:blue", **options)
    nx.draw_networkx_edges(ER_tree, pos, width=1.0, alpha=0.5)
    nx.draw_networkx_labels(ER_tree, pos, font_size=8, font_color="black")
    plt.show()



if __name__ == '__main__':
    # test3()
    # test()
    test4()





