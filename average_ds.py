import networkx as nx
import matplotlib.pyplot as plt
import graph_data_process
import copy
from math import *
import numpy as np
from cal_max_min_ds_origin import CalMaxMinDS
from BFS_ds import CalBFSMaxMinDS
import ast
from numpy import mean
from networkx.algorithms.approximation.steinertree import metric_closure
import csv
import matplotlib as mpl


class NFeatureDict(dict):
    def __missing__(self, key):
        value = self[key] = type(self)()
        return value


class CalDS(object):
    def __init__(self, tree: nx.Graph, unift_list: list, source: int):
        self.tree = tree
        self.unift_list = unift_list
        self.source = source
        self.nfeature_temp = self.preprocess()
        self.node_num = self.tree.number_of_nodes()

    def preprocess(self):
        nfeature_temp = NFeatureDict()

        tree_deg = nx.degree(self.tree)
        for v in tree_deg:

            nfeature_temp[v[0]]["degree"] = v[1]
            neighbor_list = self.tree.neighbors(v[0])
            uninfected_edge_num = len(set(self.unift_list).difference(set(neighbor_list)))
            nfeature_temp[v[0]]["uninft_edge_num"] = uninfected_edge_num
        return nfeature_temp

    def cal_ds(self):
        no_add_select_pool = copy.copy(self.unift_list)
        no_add_select_pool.append(self.source)
        edge_prob_num = 1
        permutation_dict = NFeatureDict()
        permutation_dict[str([self.source])]['permutation_prob'] = 1/edge_prob_num
        permutation_dict[str([self.source])]['edge_prob_num'] = edge_prob_num
        new_list = []

        while len(new_list) < self.node_num:
            new_permutation_dict = NFeatureDict()
            for k in list(permutation_dict.keys()):
                temp = ast.literal_eval(k)
                temp_neighbors = []
                for node in temp:
                    temp_neighbors = temp_neighbors + list(self.tree.neighbors(node))
                temp_neighbors = list(set(temp_neighbors).difference(set(temp)))
                new_permutation_prob = permutation_dict[k]["permutation_prob"]/len(temp_neighbors)
                for i in temp_neighbors:
                    new_temp = temp + [i]
                    new_permutation_dict[str(new_temp)]['permutation_prob'] = new_permutation_prob
                    new_permutation_dict[str(new_temp)]['edge_prob_num'] = edge_prob_num
            permutation_dict = copy.copy(new_permutation_dict)

            new_list = list(permutation_dict.keys())[0]
            new_list = ast.literal_eval(new_list)

        # print(permutation_dict)
        return permutation_dict

    def cal_ds2(self):
        no_add_select_pool = copy.copy(self.unift_list)
        no_add_select_pool.append(self.source)
        edge_prob_num = 1
        permutation_dict = {}
        permutation_dict[str([self.source])] = 1/edge_prob_num
        new_list = []

        while len(new_list) < self.node_num:
            new_permutation_dict = {}
            for k in list(permutation_dict.keys()):
                # print("k:",k)
                temp = ast.literal_eval(k)
                temp_neighbors = []
                for node in temp:
                    temp_neighbors = temp_neighbors + list(self.tree.neighbors(node))
                # print("**temp_neighbors:", temp_neighbors)
                temp_neighbors = list(set(temp_neighbors).difference(set(temp)))
                # print("temp_neighbors:", temp_neighbors)
                new_permutation_prob = permutation_dict[k]/max(1,len(temp_neighbors))
                for i in temp_neighbors:
                    new_temp = temp + [i]
                    new_permutation_dict[str(new_temp)]= new_permutation_prob
            permutation_dict = copy.copy(new_permutation_dict)
            # print("permutation_dict:", permutation_dict)

            new_list = list(permutation_dict.keys())[0]
            new_list = ast.literal_eval(new_list)

        # print(permutation_dict)
        return permutation_dict


def run():
    node_num = 15
    ER = nx.random_graphs.barabasi_albert_graph(node_num, 3)
    tree_ka = nx.minimum_spanning_tree(ER, algorithm="kruskal")
    processed_graph = graph_data_process.TreeDataProcess(tree_ka)
    # unift_list = processed_graph.get_uninfected_node_list()
    unift_list = []
    cal = CalDS(tree_ka, unift_list, 1)
    all_ds_dict = cal.cal_ds2()


def run1():
    distance_list = []
    BFS_win_all_List = []
    all_geo_BFS_per_list = []
    all_geo_max_min_per_list = []
    all_aver_BFS_per_list = []
    all_aver_max_min_per_list = []
    all_BFS_max_per_list = []
    all_BFS_min_per_list = []
    all_max_per_list = []
    all_min_per_list = []

    for i in range(10):
        print("i:", i)
        node_num = 11
        ER = nx.random_graphs.erdos_renyi_graph(node_num, 0.4)
        tree_ka = nx.minimum_spanning_tree(ER, algorithm="kruskal")
        processed_graph = graph_data_process.TreeDataProcess(tree_ka)
        # unift_list = processed_graph.get_uninfected_node_list()
        unift_list = []
        geo_max_min_prob_dict = {}
        geo_BFS_prob_dict = {}
        aver_BFS_prob_dict = {}
        aver_max_min_prob_dict = {}
        BFS_max_prob_dict = {}
        BFS_min_prob_dict = {}
        max_prob_dict = {}
        min_prob_dict = {}
        aver_prob_dict = {}

        BFS_win = []
        max_min_win = []
        geo_BFS_per_list = []
        geo_max_min_per_list = []
        aver_BFS_per_list = []
        aver_max_min_per_list = []
        BFS_max_per_list = []
        BFS_min_per_list = []
        max_per_list = []
        min_per_list = []

        # nx.draw(tree_ka, node_size=200, with_labels=True)
        # plt.show()

        for i in np.arange(0, node_num, 1):
            # print("i:", i)

            cal_max_min_ds = CalMaxMinDS(tree_ka, unift_list, i)
            max_permute_prob = cal_max_min_ds.cal_max_ds()
            min_permute_prob = cal_max_min_ds.cal_min_ds()
            geo_min_max_prob = sqrt(max_permute_prob * min_permute_prob)
            geo_max_min_prob_dict[i] = geo_min_max_prob
            aver_min_max_prob = (min_permute_prob + min_permute_prob)/2
            aver_max_min_prob_dict[i] = aver_min_max_prob
            max_prob_dict[i] = max_permute_prob
            min_prob_dict[i] = min_permute_prob
            # print("cal_max_min_ds")

            cal_BFS_ds = CalBFSMaxMinDS(tree_ka, unift_list, i)
            max_bfs_permute_prob = cal_BFS_ds.cal_BFS_max_ds()
            min_bfs_permute_prob = cal_BFS_ds.cal_BFS_min_ds()
            geo_BFS_prob = sqrt(max_bfs_permute_prob * min_bfs_permute_prob)
            geo_BFS_prob_dict[i] = geo_BFS_prob
            aver_BFS_prob = (max_bfs_permute_prob + min_bfs_permute_prob)/2
            aver_BFS_prob_dict[i] = aver_BFS_prob
            BFS_max_prob_dict[i] = max_bfs_permute_prob
            BFS_min_prob_dict[i] = min_bfs_permute_prob
            # print("cal_BFS_ds")

            cal = CalDS(tree_ka, unift_list, i)
            all_ds_dict = cal.cal_ds2()
            aver_prob_dict[i] = mean(list(all_ds_dict.values()))
            # print("cal")

            geo_max_min_per = abs((log(geo_max_min_prob_dict[i]) - log(aver_prob_dict[i]))/log(aver_prob_dict[i]))
            geo_max_min_per_list.append(geo_max_min_per)
            aver_max_min_per = abs((log(aver_max_min_prob_dict[i]) - log(aver_prob_dict[i])) / log(aver_prob_dict[i]))
            aver_max_min_per_list.append(aver_max_min_per)
            max_per = abs((log(max_prob_dict[i]) - log(aver_prob_dict[i])) / log(aver_prob_dict[i]))
            max_per_list.append(max_per)
            min_per = abs((log(min_prob_dict[i]) - log(aver_prob_dict[i])) / log(aver_prob_dict[i]))
            min_per_list.append(min_per)

            geo_BFS_per = abs((log(geo_BFS_prob_dict[i]) - log(aver_prob_dict[i])) / log(aver_prob_dict[i]))
            geo_BFS_per_list.append(geo_BFS_per)
            aver_BFS_per = abs((log(aver_BFS_prob_dict[i]) - log(aver_prob_dict[i])) / log(aver_prob_dict[i]))
            aver_BFS_per_list.append(aver_BFS_per)
            BFS_max_per = abs((log(BFS_max_prob_dict[i]) - log(aver_prob_dict[i])) / log(aver_prob_dict[i]))
            BFS_max_per_list.append(BFS_max_per)
            BFS_min_per = abs((log(BFS_min_prob_dict[i]) - log(aver_prob_dict[i])) / log(aver_prob_dict[i]))
            BFS_min_per_list.append(BFS_min_per)


            if abs(geo_max_min_prob_dict[i] - aver_prob_dict[i]) < abs(geo_BFS_prob_dict[i] - aver_prob_dict[i]):
                max_min_win.append(1)
            elif abs(geo_max_min_prob_dict[i] - aver_prob_dict[i]) > abs(geo_BFS_prob_dict[i] - aver_prob_dict[i]):
                BFS_win.append(1)

        closure_graph = metric_closure(tree_ka, weight='weight')
        closure_graph_deg = closure_graph.degree(weight='distance')
        closure_graph_size = closure_graph.size(weight='distance')
        distance_list_per_node = []
        for k in closure_graph_deg:
            distance_list_per_node.append(k[1] / closure_graph_size)
        distance_list.append(mean(distance_list_per_node))
        BFS_win_all_List.append(sum(BFS_win))

        all_geo_BFS_per_list.append(mean(geo_BFS_per_list))
        all_geo_max_min_per_list.append(mean(geo_max_min_per_list))
        all_aver_BFS_per_list.append(mean(aver_BFS_per_list))
        all_aver_max_min_per_list.append(mean(aver_max_min_per_list))
        all_BFS_max_per_list.append(mean(BFS_max_per_list))
        all_BFS_min_per_list.append(mean(BFS_min_per_list))
        all_max_per_list.append(mean(max_per_list))
        all_min_per_list.append(mean(min_per_list))

    # print("distance_list:", distance_list)
    # print("BFS_win_all_List:", BFS_win_all_List)

    # plt.plot(distance_list, BFS_win_all_List)
    # plt.show()
    return all_geo_BFS_per_list, all_geo_max_min_per_list, all_aver_BFS_per_list, all_aver_max_min_per_list, all_BFS_max_per_list, all_BFS_min_per_list, all_max_per_list, all_min_per_list


def run2():
    distance_list = []
    BFS_win_all_List = []
    all_geo_BFS_per_list = []
    all_geo_max_min_per_list = []
    all_aver_BFS_per_list = []
    all_aver_max_min_per_list = []
    all_BFS_max_per_list = []
    all_BFS_min_per_list = []
    all_max_per_list = []
    all_min_per_list = []

    for i in range(10):
        print("i:", i)
        node_num = 10
        ER = nx.random_graphs.barabasi_albert_graph(node_num, 4)
        tree_ka = nx.minimum_spanning_tree(ER, algorithm="kruskal")
        processed_graph = graph_data_process.TreeDataProcess(tree_ka)
        # unift_list = processed_graph.get_uninfected_node_list()
        unift_list = []
        geo_max_min_prob_dict = {}
        geo_BFS_prob_dict = {}
        aver_BFS_prob_dict = {}
        aver_max_min_prob_dict = {}
        BFS_max_prob_dict = {}
        BFS_min_prob_dict = {}
        max_prob_dict = {}
        min_prob_dict = {}
        aver_prob_dict = {}

        BFS_win = []
        max_min_win = []
        geo_BFS_per_list = []
        geo_max_min_per_list = []
        aver_BFS_per_list = []
        aver_max_min_per_list = []
        BFS_max_per_list = []
        BFS_min_per_list = []
        max_per_list = []
        min_per_list = []

        # nx.draw(tree_ka, node_size=200, with_labels=True)
        # plt.show()

        for i in np.arange(0, node_num, 1):
            # print("i:", i)

            cal_max_min_ds = CalMaxMinDS(tree_ka, unift_list, i)
            max_permute_prob = cal_max_min_ds.cal_max_ds()
            min_permute_prob = cal_max_min_ds.cal_min_ds()
            geo_min_max_prob = sqrt(max_permute_prob * min_permute_prob)
            geo_max_min_prob_dict[i] = geo_min_max_prob
            aver_min_max_prob = (min_permute_prob + min_permute_prob)/2
            aver_max_min_prob_dict[i] = aver_min_max_prob
            max_prob_dict[i] = max_permute_prob
            min_prob_dict[i] = min_permute_prob
            # print("cal_max_min_ds")

            cal_BFS_ds = CalBFSMaxMinDS(tree_ka, unift_list, i)
            max_bfs_permute_prob = cal_BFS_ds.cal_BFS_max_ds()
            min_bfs_permute_prob = cal_BFS_ds.cal_BFS_min_ds()
            geo_BFS_prob = sqrt(max_bfs_permute_prob * min_bfs_permute_prob)
            geo_BFS_prob_dict[i] = geo_BFS_prob
            aver_BFS_prob = (max_bfs_permute_prob + min_bfs_permute_prob)/2
            aver_BFS_prob_dict[i] = aver_BFS_prob
            BFS_max_prob_dict[i] = max_bfs_permute_prob
            BFS_min_prob_dict[i] = min_bfs_permute_prob
            # print("cal_BFS_ds")

            cal = CalDS(tree_ka, unift_list, i)
            all_ds_dict = cal.cal_ds2()
            aver_prob_dict[i] = mean(list(all_ds_dict.values()))
            # print("cal")

            geo_max_min_per = abs((log(geo_max_min_prob_dict[i]) - log(aver_prob_dict[i]))/log(aver_prob_dict[i]))
            geo_max_min_per_list.append(geo_max_min_per)
            aver_max_min_per = abs((log(aver_max_min_prob_dict[i]) - log(aver_prob_dict[i])) / log(aver_prob_dict[i]))
            aver_max_min_per_list.append(aver_max_min_per)
            max_per = abs((log(max_prob_dict[i]) - log(aver_prob_dict[i])) / log(aver_prob_dict[i]))
            max_per_list.append(max_per)
            min_per = abs((log(min_prob_dict[i]) - log(aver_prob_dict[i])) / log(aver_prob_dict[i]))
            min_per_list.append(min_per)

            geo_BFS_per = abs((log(geo_BFS_prob_dict[i]) - log(aver_prob_dict[i])) / log(aver_prob_dict[i]))
            geo_BFS_per_list.append(geo_BFS_per)
            aver_BFS_per = abs((log(aver_BFS_prob_dict[i]) - log(aver_prob_dict[i])) / log(aver_prob_dict[i]))
            aver_BFS_per_list.append(aver_BFS_per)
            BFS_max_per = abs((log(BFS_max_prob_dict[i]) - log(aver_prob_dict[i])) / log(aver_prob_dict[i]))
            BFS_max_per_list.append(BFS_max_per)
            BFS_min_per = abs((log(BFS_min_prob_dict[i]) - log(aver_prob_dict[i])) / log(aver_prob_dict[i]))
            BFS_min_per_list.append(BFS_min_per)


            if abs(geo_max_min_prob_dict[i] - aver_prob_dict[i]) < abs(geo_BFS_prob_dict[i] - aver_prob_dict[i]):
                max_min_win.append(1)
            elif abs(geo_max_min_prob_dict[i] - aver_prob_dict[i]) > abs(geo_BFS_prob_dict[i] - aver_prob_dict[i]):
                BFS_win.append(1)

        closure_graph = metric_closure(tree_ka, weight='weight')
        closure_graph_deg = closure_graph.degree(weight='distance')
        closure_graph_size = closure_graph.size(weight='distance')
        distance_list_per_node = []
        for k in closure_graph_deg:
            distance_list_per_node.append(k[1] / closure_graph_size)
        distance_list.append(mean(distance_list_per_node))
        BFS_win_all_List.append(sum(BFS_win))

        all_geo_BFS_per_list.append(mean(geo_BFS_per_list))
        all_geo_max_min_per_list.append(mean(geo_max_min_per_list))
        all_aver_BFS_per_list.append(mean(aver_BFS_per_list))
        all_aver_max_min_per_list.append(mean(aver_max_min_per_list))
        all_BFS_max_per_list.append(mean(BFS_max_per_list))
        all_BFS_min_per_list.append(mean(BFS_min_per_list))
        all_max_per_list.append(mean(max_per_list))
        all_min_per_list.append(mean(min_per_list))

    return all_geo_BFS_per_list, all_geo_max_min_per_list, all_aver_BFS_per_list, all_aver_max_min_per_list, all_BFS_max_per_list, all_BFS_min_per_list, all_max_per_list, all_min_per_list


def run3():
    distance_list = []
    BFS_win_all_List = []
    all_geo_BFS_per_list = []
    all_geo_max_min_per_list = []
    all_aver_BFS_per_list = []
    all_aver_max_min_per_list = []
    all_BFS_max_per_list = []
    all_BFS_min_per_list = []
    all_max_per_list = []
    all_min_per_list = []

    for i in range(20):
        print("i:", i)
        node_num = 10
        degree = floor(3)
        # if degree % 2 > 0:
        #     degree -= 1
        ER = nx.random_graphs.watts_strogatz_graph(node_num, int(degree), 0.5)
        tree_ka = nx.minimum_spanning_tree(ER, algorithm="kruskal")
        processed_graph = graph_data_process.TreeDataProcess(tree_ka)
        # unift_list = processed_graph.get_uninfected_node_list()
        unift_list = []
        geo_max_min_prob_dict = {}
        geo_BFS_prob_dict = {}
        aver_BFS_prob_dict = {}
        aver_max_min_prob_dict = {}
        BFS_max_prob_dict = {}
        BFS_min_prob_dict = {}
        max_prob_dict = {}
        min_prob_dict = {}
        aver_prob_dict = {}

        BFS_win = []
        max_min_win = []
        geo_BFS_per_list = []
        geo_max_min_per_list = []
        aver_BFS_per_list = []
        aver_max_min_per_list = []
        BFS_max_per_list = []
        BFS_min_per_list = []
        max_per_list = []
        min_per_list = []

        # nx.draw(tree_ka, node_size=200, with_labels=True)
        # plt.show()

        for i in np.arange(0, node_num, 1):
            # print("i:", i)

            cal_max_min_ds = CalMaxMinDS(tree_ka, unift_list, i)
            max_permute_prob = cal_max_min_ds.cal_max_ds()
            min_permute_prob = cal_max_min_ds.cal_min_ds()
            geo_min_max_prob = sqrt(max_permute_prob * min_permute_prob)
            geo_max_min_prob_dict[i] = geo_min_max_prob
            aver_min_max_prob = (min_permute_prob + min_permute_prob)/2
            aver_max_min_prob_dict[i] = aver_min_max_prob
            max_prob_dict[i] = max_permute_prob
            min_prob_dict[i] = min_permute_prob
            # print("cal_max_min_ds")

            cal_BFS_ds = CalBFSMaxMinDS(tree_ka, unift_list, i)
            max_bfs_permute_prob = cal_BFS_ds.cal_BFS_max_ds()
            min_bfs_permute_prob = cal_BFS_ds.cal_BFS_min_ds()
            geo_BFS_prob = sqrt(max_bfs_permute_prob * min_bfs_permute_prob)
            geo_BFS_prob_dict[i] = geo_BFS_prob
            aver_BFS_prob = (max_bfs_permute_prob + min_bfs_permute_prob)/2
            aver_BFS_prob_dict[i] = aver_BFS_prob
            BFS_max_prob_dict[i] = max_bfs_permute_prob
            BFS_min_prob_dict[i] = min_bfs_permute_prob
            # print("cal_BFS_ds")

            cal = CalDS(tree_ka, unift_list, i)
            all_ds_dict = cal.cal_ds2()
            aver_prob_dict[i] = mean(list(all_ds_dict.values()))
            # print("cal")

            geo_max_min_per = abs((log(geo_max_min_prob_dict[i]) - log(aver_prob_dict[i]))/log(aver_prob_dict[i]))
            geo_max_min_per_list.append(geo_max_min_per)
            aver_max_min_per = abs((log(aver_max_min_prob_dict[i]) - log(aver_prob_dict[i])) / log(aver_prob_dict[i]))
            aver_max_min_per_list.append(aver_max_min_per)
            max_per = abs((log(max_prob_dict[i]) - log(aver_prob_dict[i])) / log(aver_prob_dict[i]))
            max_per_list.append(max_per)
            min_per = abs((log(min_prob_dict[i]) - log(aver_prob_dict[i])) / log(aver_prob_dict[i]))
            min_per_list.append(min_per)

            geo_BFS_per = abs((log(geo_BFS_prob_dict[i]) - log(aver_prob_dict[i])) / log(aver_prob_dict[i]))
            geo_BFS_per_list.append(geo_BFS_per)
            aver_BFS_per = abs((log(aver_BFS_prob_dict[i]) - log(aver_prob_dict[i])) / log(aver_prob_dict[i]))
            aver_BFS_per_list.append(aver_BFS_per)
            BFS_max_per = abs((log(BFS_max_prob_dict[i]) - log(aver_prob_dict[i])) / log(aver_prob_dict[i]))
            BFS_max_per_list.append(BFS_max_per)
            BFS_min_per = abs((log(BFS_min_prob_dict[i]) - log(aver_prob_dict[i])) / log(aver_prob_dict[i]))
            BFS_min_per_list.append(BFS_min_per)


            if abs(geo_max_min_prob_dict[i] - aver_prob_dict[i]) < abs(geo_BFS_prob_dict[i] - aver_prob_dict[i]):
                max_min_win.append(1)
            elif abs(geo_max_min_prob_dict[i] - aver_prob_dict[i]) > abs(geo_BFS_prob_dict[i] - aver_prob_dict[i]):
                BFS_win.append(1)

        closure_graph = metric_closure(tree_ka, weight='weight')
        closure_graph_deg = closure_graph.degree(weight='distance')
        closure_graph_size = closure_graph.size(weight='distance')
        distance_list_per_node = []
        for k in closure_graph_deg:
            distance_list_per_node.append(k[1] / closure_graph_size)
        distance_list.append(mean(distance_list_per_node))
        BFS_win_all_List.append(sum(BFS_win))

        all_geo_BFS_per_list.append(mean(geo_BFS_per_list))
        all_geo_max_min_per_list.append(mean(geo_max_min_per_list))
        all_aver_BFS_per_list.append(mean(aver_BFS_per_list))
        all_aver_max_min_per_list.append(mean(aver_max_min_per_list))
        all_BFS_max_per_list.append(mean(BFS_max_per_list))
        all_BFS_min_per_list.append(mean(BFS_min_per_list))
        all_max_per_list.append(mean(max_per_list))
        all_min_per_list.append(mean(min_per_list))

    return all_geo_BFS_per_list, all_geo_max_min_per_list, all_aver_BFS_per_list, all_aver_max_min_per_list, all_BFS_max_per_list, all_BFS_min_per_list, all_max_per_list, all_min_per_list


def run4():
    distance_list = []
    BFS_win_all_List = []
    all_geo_BFS_per_list = []
    all_geo_max_min_per_list = []
    all_aver_BFS_per_list = []
    all_aver_max_min_per_list = []
    all_BFS_max_per_list = []
    all_BFS_min_per_list = []
    all_max_per_list = []
    all_min_per_list = []

    for i in range(10):
        print("i:", i)
        node_num = 11
        degree = floor(5)
        if degree % 2 > 0:
            degree -= 1
        ER = nx.random_graphs.random_regular_graph(degree, node_num)
        tree_ka = nx.minimum_spanning_tree(ER, algorithm="kruskal")
        unift_list = []
        geo_max_min_prob_dict = {}
        geo_BFS_prob_dict = {}
        aver_BFS_prob_dict = {}
        aver_max_min_prob_dict = {}
        BFS_max_prob_dict = {}
        BFS_min_prob_dict = {}
        max_prob_dict = {}
        min_prob_dict = {}
        aver_prob_dict = {}

        BFS_win = []
        max_min_win = []
        geo_BFS_per_list = []
        geo_max_min_per_list = []
        aver_BFS_per_list = []
        aver_max_min_per_list = []
        BFS_max_per_list = []
        BFS_min_per_list = []
        max_per_list = []
        min_per_list = []

        for i in np.arange(0, node_num, 1):
            # print("i:", i)

            cal_max_min_ds = CalMaxMinDS(tree_ka, unift_list, i)
            max_permute_prob = cal_max_min_ds.cal_max_ds()
            min_permute_prob = cal_max_min_ds.cal_min_ds()
            geo_min_max_prob = sqrt(max_permute_prob * min_permute_prob)
            geo_max_min_prob_dict[i] = geo_min_max_prob
            aver_min_max_prob = (min_permute_prob + min_permute_prob)/2
            aver_max_min_prob_dict[i] = aver_min_max_prob
            max_prob_dict[i] = max_permute_prob
            min_prob_dict[i] = min_permute_prob
            # print("cal_max_min_ds")

            cal_BFS_ds = CalBFSMaxMinDS(tree_ka, unift_list, i)
            max_bfs_permute_prob = cal_BFS_ds.cal_BFS_max_ds()
            min_bfs_permute_prob = cal_BFS_ds.cal_BFS_min_ds()
            geo_BFS_prob = sqrt(max_bfs_permute_prob * min_bfs_permute_prob)
            geo_BFS_prob_dict[i] = geo_BFS_prob
            aver_BFS_prob = (max_bfs_permute_prob + min_bfs_permute_prob)/2
            aver_BFS_prob_dict[i] = aver_BFS_prob
            BFS_max_prob_dict[i] = max_bfs_permute_prob
            BFS_min_prob_dict[i] = min_bfs_permute_prob
            # print("cal_BFS_ds")

            cal = CalDS(tree_ka, unift_list, i)
            all_ds_dict = cal.cal_ds2()
            aver_prob_dict[i] = mean(list(all_ds_dict.values()))
            # print("cal")

            geo_max_min_per = abs((log(geo_max_min_prob_dict[i]) - log(aver_prob_dict[i]))/log(aver_prob_dict[i]))
            geo_max_min_per_list.append(geo_max_min_per)
            aver_max_min_per = abs((log(aver_max_min_prob_dict[i]) - log(aver_prob_dict[i])) / log(aver_prob_dict[i]))
            aver_max_min_per_list.append(aver_max_min_per)
            max_per = abs((log(max_prob_dict[i]) - log(aver_prob_dict[i])) / log(aver_prob_dict[i]))
            max_per_list.append(max_per)
            min_per = abs((log(min_prob_dict[i]) - log(aver_prob_dict[i])) / log(aver_prob_dict[i]))
            min_per_list.append(min_per)

            geo_BFS_per = abs((log(geo_BFS_prob_dict[i]) - log(aver_prob_dict[i])) / log(aver_prob_dict[i]))
            geo_BFS_per_list.append(geo_BFS_per)
            aver_BFS_per = abs((log(aver_BFS_prob_dict[i]) - log(aver_prob_dict[i])) / log(aver_prob_dict[i]))
            aver_BFS_per_list.append(aver_BFS_per)
            BFS_max_per = abs((log(BFS_max_prob_dict[i]) - log(aver_prob_dict[i])) / log(aver_prob_dict[i]))
            BFS_max_per_list.append(BFS_max_per)
            BFS_min_per = abs((log(BFS_min_prob_dict[i]) - log(aver_prob_dict[i])) / log(aver_prob_dict[i]))
            BFS_min_per_list.append(BFS_min_per)


            if abs(geo_max_min_prob_dict[i] - aver_prob_dict[i]) < abs(geo_BFS_prob_dict[i] - aver_prob_dict[i]):
                max_min_win.append(1)
            elif abs(geo_max_min_prob_dict[i] - aver_prob_dict[i]) > abs(geo_BFS_prob_dict[i] - aver_prob_dict[i]):
                BFS_win.append(1)

        closure_graph = metric_closure(tree_ka, weight='weight')
        closure_graph_deg = closure_graph.degree(weight='distance')
        closure_graph_size = closure_graph.size(weight='distance')
        distance_list_per_node = []
        for k in closure_graph_deg:
            distance_list_per_node.append(k[1] / closure_graph_size)
        distance_list.append(mean(distance_list_per_node))
        BFS_win_all_List.append(sum(BFS_win))

        all_geo_BFS_per_list.append(mean(geo_BFS_per_list))
        all_geo_max_min_per_list.append(mean(geo_max_min_per_list))
        all_aver_BFS_per_list.append(mean(aver_BFS_per_list))
        all_aver_max_min_per_list.append(mean(aver_max_min_per_list))
        all_BFS_max_per_list.append(mean(BFS_max_per_list))
        all_BFS_min_per_list.append(mean(BFS_min_per_list))
        all_max_per_list.append(mean(max_per_list))
        all_min_per_list.append(mean(min_per_list))

    return all_geo_BFS_per_list, all_geo_max_min_per_list, all_aver_BFS_per_list, all_aver_max_min_per_list, all_BFS_max_per_list, all_BFS_min_per_list, all_max_per_list, all_min_per_list


def histogram_plot(er_list, ba_list, ws_list, rn_olist):
    mpl.rcParams['font.sans-serif'] = ['Times New Roman']
    mpl.rcParams['axes.unicode_minus'] = False
    bar_width = 0.65
    # tick_label = ["BFSGeo", "BFSAvr", "BFSMax", "BFSMin", "DegGeo", "DegAvr", "DegMax", "DegMin"]
    tick_label = ["BFSGeo", "BFSAvr", "BFSRan", "DegGeo", "DegAvr", "DegMax", "DegMin", "DegRan"]
    Y11 = er_list
    X = np.arange(len(Y11))

    fig = plt.figure(1)

    plt.subplot(221)
    plt.bar(X, Y11, bar_width, align="center", color="blue", label="top1 acc", alpha=0.35)
    plt.title("ER Network", fontsize=13)
    plt.xticks(X, tick_label, fontsize=12, rotation=-30)
    plt.yticks(fontsize=13)

    Y21 = ba_list
    X = np.arange(len(Y11))
    plt.subplot(222)
    plt.bar(X, Y21, bar_width, align="center", color="blue", label="top1 acc", alpha=0.35)
    plt.title("BA Network", fontsize=13)
    plt.xticks(X, tick_label, fontsize=12, rotation=-30)
    plt.yticks(fontsize=13)

    Y31 = ws_list
    X = np.arange(len(Y11))
    plt.subplot(223)
    plt.bar(X, Y31, bar_width, align="center", color="blue", label="top1 acc", alpha=0.35)
    plt.title("WS Network", fontsize=13)
    plt.xticks(X, tick_label, fontsize=12, rotation=-30)
    plt.yticks(fontsize=13)

    Y41 = rn_olist
    X = np.arange(len(Y11))
    plt.subplot(224)
    plt.bar(X, Y41, bar_width, align="center", color="blue", label="top1 acc", alpha=0.35)
    plt.title("Regular Network", fontsize=13)
    plt.xticks(X, tick_label, fontsize=12, rotation=-30)
    plt.yticks(fontsize=13)

    num1 = 1
    num2 = 0
    num3 = 4
    num4 = 0
    # plt.legend(bbox_to_anchor=(num1, num2), loc=num3, fontsize=13)

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':

    with open('average_ds.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
    print("rows:",rows)
    for row in rows:
        for i in range(len(row)):
            row[i] = float(row[i])



    histogram_plot(rows[0], rows[1], rows[2], rows[3])



