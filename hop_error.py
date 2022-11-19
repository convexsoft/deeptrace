import networkx as nx
import matplotlib.pyplot as plt
import graph_data_process
import copy
import math
import numpy as np
from cal_max_min_ds_origin import CalMaxMinDS
from BFS_ds import CalBFSMaxMinDS
import ast
from numpy import mean
from networkx.algorithms.approximation.steinertree import metric_closure
import csv
import matplotlib as mpl
import pandas as pd
import seaborn as sns
import random
from math import sqrt
from math import floor


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
        ift_list = list(set(list(self.tree.nodes)).difference(set(self.unift_list)))

        while len(new_list) < len(ift_list):
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
                temp_ift_neighbors = list(set(temp_neighbors).difference(set(no_add_select_pool)))
                for i in temp_ift_neighbors:
                    new_temp = temp + [i]
                    new_permutation_dict[str(new_temp)]= new_permutation_prob
            if len(new_permutation_dict)>0:
                permutation_dict = copy.copy(new_permutation_dict)
            # print("permutation_dict:", permutation_dict)
            new_list = list(permutation_dict.keys())[0]
            new_list = ast.literal_eval(new_list)

        print("permutation_dict:", permutation_dict)
        return permutation_dict


def max_idx_list_get(dic: dict):
    max_list = []
    max_value = max(dic.values())
    for m, n in dic.items():
        if n == max_value:
            max_list.append(m)
    return max_list


def run1():
    distance_list = []
    BFS_win_List = []
    top1_geo_BFS_per_list = []
    top1_geo_max_min_per_list = []
    top1_aver_BFS_per_list = []
    top1_aver_max_min_per_list = []
    top1_BFS_max_per_list = []
    top1_BFS_min_per_list = []
    top1_max_per_list = []
    top1_min_per_list = []

    tree_num = 2

    for i in range(tree_num):
        print("i:", i)
        node_num = 11
        ER = nx.random_graphs.erdos_renyi_graph(node_num, 0.5)
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

        # nx.draw(tree_ka, node_size=200, with_labels=True)
        # plt.show()

        for i in np.arange(0, node_num, 1):
            # print("i:", i)

            cal = CalDS(tree_ka, unift_list, i)
            ds_dict = cal.cal_ds2()
            rc_num = len(list(ds_dict.values()))
            aver_prob_dict[i] = rc_num * mean(list(ds_dict.values()))
            # print("cal")

            cal_max_min_ds = CalMaxMinDS(tree_ka, unift_list, i)
            max_permute_prob = cal_max_min_ds.cal_max_ds()
            min_permute_prob = cal_max_min_ds.cal_min_ds()
            geo_min_max_prob = sqrt(max_permute_prob * min_permute_prob)
            geo_max_min_prob_dict[i] = rc_num * geo_min_max_prob
            aver_min_max_prob = (min_permute_prob + min_permute_prob)/2
            aver_max_min_prob_dict[i] = rc_num * aver_min_max_prob
            max_prob_dict[i] = rc_num * max_permute_prob
            min_prob_dict[i] = rc_num * min_permute_prob
            # print("cal_max_min_ds")

            cal_BFS_ds = CalBFSMaxMinDS(tree_ka, unift_list, i)
            max_bfs_permute_prob = cal_BFS_ds.cal_BFS_max_ds()
            min_bfs_permute_prob = cal_BFS_ds.cal_BFS_min_ds()
            geo_BFS_prob = sqrt(max_bfs_permute_prob * min_bfs_permute_prob)
            geo_BFS_prob_dict[i] = rc_num * geo_BFS_prob
            aver_BFS_prob = (max_bfs_permute_prob + min_bfs_permute_prob)/2
            aver_BFS_prob_dict[i] = rc_num * aver_BFS_prob
            BFS_max_prob_dict[i] = rc_num * max_bfs_permute_prob
            BFS_min_prob_dict[i] = rc_num * min_bfs_permute_prob
            # print("cal_BFS_ds")


        top1_aver_prob_idx_list = max_idx_list_get(aver_prob_dict)
        top1_geo_BFS_idx_list = max_idx_list_get(geo_BFS_prob_dict)
        top1_geo_max_min_idx_list = max_idx_list_get(geo_max_min_prob_dict)
        top1_aver_BFS_idx_list = max_idx_list_get(aver_BFS_prob_dict)
        top1_aver_max_min_idx_list = max_idx_list_get(aver_max_min_prob_dict)
        top1_BFS_max_idx_list = max_idx_list_get(BFS_max_prob_dict)
        top1_BFS_min_idx_list = max_idx_list_get(BFS_min_prob_dict)
        top1_max_idx_list = max_idx_list_get(max_prob_dict)
        top1_min_idx_list = max_idx_list_get(min_prob_dict)


        p = nx.shortest_path(tree_ka)
        geo_BFS_hop_err_list = []
        geo_max_min_hop_err_list = []
        aver_BFS_hop_err_list = []
        aver_max_min_hop_err_list = []
        BFS_max_hop_err_list = []
        BFS_min_hop_err_list = []
        max_hop_err_list = []
        min_hop_err_list = []

        for idx1 in top1_aver_prob_idx_list:
            for idx2 in top1_geo_BFS_idx_list:
                geo_BFS_hop_err_list.append(len(p[idx1][idx2])-1)
            for idx3 in top1_geo_max_min_idx_list:
                geo_max_min_hop_err_list.append(len(p[idx1][idx3]) - 1)
            for idx4 in top1_aver_BFS_idx_list:
                aver_BFS_hop_err_list.append(len(p[idx1][idx4]) - 1)
            for idx5 in top1_aver_max_min_idx_list:
                aver_max_min_hop_err_list.append(len(p[idx1][idx5]) - 1)
            for idx6 in top1_BFS_max_idx_list:
                BFS_max_hop_err_list.append(len(p[idx1][idx6]) - 1)
            for idx7 in top1_BFS_min_idx_list:
                BFS_min_hop_err_list.append(len(p[idx1][idx7]) - 1)
            for idx8 in top1_max_idx_list:
                max_hop_err_list.append(len(p[idx1][idx8]) - 1)
            for idx9 in top1_min_idx_list:
                min_hop_err_list.append(len(p[idx1][idx9]) - 1)

        geo_BFS_hop_err = min(geo_BFS_hop_err_list)
        geo_max_min_hop_err = min(geo_max_min_hop_err_list)
        aver_BFS_hop_err = min(aver_BFS_hop_err_list)
        aver_max_min_hop_err = min(aver_max_min_hop_err_list)
        BFS_max_hop_err = min(BFS_max_hop_err_list)
        BFS_min_hop_err = min(BFS_min_hop_err_list)
        max_hop_err = min(max_hop_err_list)
        min_hop_err = min(min_hop_err_list)


        top1_geo_BFS_per_list.append(geo_BFS_hop_err)
        top1_geo_max_min_per_list.append(geo_max_min_hop_err)
        top1_aver_BFS_per_list.append(aver_BFS_hop_err)
        top1_aver_max_min_per_list.append(aver_max_min_hop_err)
        top1_BFS_max_per_list.append(BFS_max_hop_err)
        top1_BFS_min_per_list.append(BFS_min_hop_err)
        top1_max_per_list.append(max_hop_err)
        top1_min_per_list.append(min_hop_err)

        # nx.draw(tree_ka, node_size=200, with_labels=True)
        # plt.show()

    # print("distance_list:", distance_list)
    # print("BFS_win_List:", BFS_win_List)

    # plt.plot(distance_list, BFS_win_List)
    # plt.show()["BFSGeo", "BFSAvr", "BFSRan", "DegGeo", "DegAvr", "DegMax", "DegMin", "DegRan"]
    md_list = ["BFSGeo"] * tree_num + ["BFSAvr"] * tree_num + ["BFSRan"] * tree_num + ["DegGeo"] * tree_num + [
        "DegAvr"] * tree_num + ["DegMax"] * tree_num + ["DegMin"] * tree_num + ["DegRan"] * tree_num
    print("mdlist:", md_list)
    hop_err_list = top1_geo_BFS_per_list + top1_aver_BFS_per_list + top1_BFS_max_per_list + top1_geo_max_min_per_list + top1_aver_max_min_per_list + top1_max_per_list + top1_min_per_list + top1_min_per_list
    print("hop_err_list:", hop_err_list)
    dataframe = pd.DataFrame({"mds": md_list, "hop err": hop_err_list})
    dataframe.to_csv("hop_error\\er1.csv", index=False, sep=',')
    return dataframe


def run2():
    distance_list = []
    BFS_win_List = []
    top1_geo_BFS_per_list = []
    top1_geo_max_min_per_list = []
    top1_aver_BFS_per_list = []
    top1_aver_max_min_per_list = []
    top1_BFS_max_per_list = []
    top1_BFS_min_per_list = []
    top1_max_per_list = []
    top1_min_per_list = []

    tree_num = 20

    for i in range(tree_num):
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

        # nx.draw(tree_ka, node_size=200, with_labels=True)
        # plt.show()

        for i in np.arange(0, node_num, 1):
            # print("i:", i)

            cal = CalDS(tree_ka, unift_list, i)
            ds_dict = cal.cal_ds2()
            rc_num = len(list(ds_dict.values()))
            aver_prob_dict[i] = rc_num * mean(list(ds_dict.values()))
            # print("cal")

            cal_max_min_ds = CalMaxMinDS(tree_ka, unift_list, i)
            max_permute_prob = cal_max_min_ds.cal_max_ds()
            min_permute_prob = cal_max_min_ds.cal_min_ds()
            geo_min_max_prob = sqrt(max_permute_prob * min_permute_prob)
            geo_max_min_prob_dict[i] = rc_num * geo_min_max_prob
            aver_min_max_prob = (min_permute_prob + min_permute_prob)/2
            aver_max_min_prob_dict[i] = rc_num * aver_min_max_prob
            max_prob_dict[i] = rc_num * max_permute_prob
            min_prob_dict[i] = rc_num * min_permute_prob
            # print("cal_max_min_ds")

            cal_BFS_ds = CalBFSMaxMinDS(tree_ka, unift_list, i)
            max_bfs_permute_prob = cal_BFS_ds.cal_BFS_max_ds()
            min_bfs_permute_prob = cal_BFS_ds.cal_BFS_min_ds()
            geo_BFS_prob = sqrt(max_bfs_permute_prob * min_bfs_permute_prob)
            geo_BFS_prob_dict[i] = rc_num * geo_BFS_prob
            aver_BFS_prob = (max_bfs_permute_prob + min_bfs_permute_prob)/2
            aver_BFS_prob_dict[i] = rc_num * aver_BFS_prob
            BFS_max_prob_dict[i] = rc_num * max_bfs_permute_prob
            BFS_min_prob_dict[i] = rc_num * min_bfs_permute_prob
            # print("cal_BFS_ds")

        top1_aver_prob_idx_list = max_idx_list_get(aver_prob_dict)
        top1_geo_BFS_idx_list = max_idx_list_get(geo_BFS_prob_dict)
        top1_geo_max_min_idx_list = max_idx_list_get(geo_max_min_prob_dict)
        top1_aver_BFS_idx_list = max_idx_list_get(aver_BFS_prob_dict)
        top1_aver_max_min_idx_list = max_idx_list_get(aver_max_min_prob_dict)
        top1_BFS_max_idx_list = max_idx_list_get(BFS_max_prob_dict)
        top1_BFS_min_idx_list = max_idx_list_get(BFS_min_prob_dict)
        top1_max_idx_list = max_idx_list_get(max_prob_dict)
        top1_min_idx_list = max_idx_list_get(min_prob_dict)

        p = nx.shortest_path(tree_ka)
        geo_BFS_hop_err_list = []
        geo_max_min_hop_err_list = []
        aver_BFS_hop_err_list = []
        aver_max_min_hop_err_list = []
        BFS_max_hop_err_list = []
        BFS_min_hop_err_list = []
        max_hop_err_list = []
        min_hop_err_list = []

        for idx1 in top1_aver_prob_idx_list:
            for idx2 in top1_geo_BFS_idx_list:
                geo_BFS_hop_err_list.append(len(p[idx1][idx2])-1)
            for idx3 in top1_geo_max_min_idx_list:
                geo_max_min_hop_err_list.append(len(p[idx1][idx3]) - 1)
            for idx4 in top1_aver_BFS_idx_list:
                aver_BFS_hop_err_list.append(len(p[idx1][idx4]) - 1)
            for idx5 in top1_aver_max_min_idx_list:
                aver_max_min_hop_err_list.append(len(p[idx1][idx5]) - 1)
            for idx6 in top1_BFS_max_idx_list:
                BFS_max_hop_err_list.append(len(p[idx1][idx6]) - 1)
            for idx7 in top1_BFS_min_idx_list:
                BFS_min_hop_err_list.append(len(p[idx1][idx7]) - 1)
            for idx8 in top1_max_idx_list:
                max_hop_err_list.append(len(p[idx1][idx8]) - 1)
            for idx9 in top1_min_idx_list:
                min_hop_err_list.append(len(p[idx1][idx9]) - 1)

        geo_BFS_hop_err = min(geo_BFS_hop_err_list)
        geo_max_min_hop_err = min(geo_max_min_hop_err_list)
        aver_BFS_hop_err = min(aver_BFS_hop_err_list)
        aver_max_min_hop_err = min(aver_max_min_hop_err_list)
        BFS_max_hop_err = min(BFS_max_hop_err_list)
        BFS_min_hop_err = min(BFS_min_hop_err_list)
        max_hop_err = min(max_hop_err_list)
        min_hop_err = min(min_hop_err_list)

        top1_geo_BFS_per_list.append(geo_BFS_hop_err)
        top1_geo_max_min_per_list.append(geo_max_min_hop_err)
        top1_aver_BFS_per_list.append(aver_BFS_hop_err)
        top1_aver_max_min_per_list.append(aver_max_min_hop_err)
        top1_BFS_max_per_list.append(BFS_max_hop_err)
        top1_BFS_min_per_list.append(BFS_min_hop_err)
        top1_max_per_list.append(max_hop_err)
        top1_min_per_list.append(min_hop_err)

        # nx.draw(tree_ka, node_size=200, with_labels=True)
        # plt.show()


    # print("distance_list:", distance_list)
    # print("BFS_win_List:", BFS_win_List)

    # plt.plot(distance_list, BFS_win_List)
    # plt.show()["BFSGeo", "BFSAvr", "BFSRan", "DegGeo", "DegAvr", "DegMax", "DegMin", "DegRan"]
    md_list = ["BFSGeo"] * tree_num + ["BFSAvr"] * tree_num + ["BFSRan"] * tree_num + ["DegGeo"] * tree_num + ["DegAvr"] * tree_num +["DegMax"] * tree_num + ["DegMin"] * tree_num + ["DegRan"] * tree_num
    print("mdlist:", md_list)
    hop_err_list = top1_geo_BFS_per_list + top1_aver_BFS_per_list + top1_BFS_max_per_list + top1_geo_max_min_per_list + top1_aver_max_min_per_list + top1_max_per_list + top1_min_per_list + top1_min_per_list
    print("hop_err_list:", hop_err_list)
    dataframe = pd.DataFrame({"mds": md_list, "hop err": hop_err_list})
    dataframe.to_csv("hop_error\\ba.csv", index=False, sep=',')
    return dataframe


def run3():
    distance_list = []
    BFS_win_List = []
    top1_geo_BFS_per_list = []
    top1_geo_max_min_per_list = []
    top1_aver_BFS_per_list = []
    top1_aver_max_min_per_list = []
    top1_BFS_max_per_list = []
    top1_BFS_min_per_list = []
    top1_max_per_list = []
    top1_min_per_list = []

    tree_num = 20

    for i in range(tree_num):
        print("i:", i)
        node_num = 10
        degree = floor(4)
        # if degree % 2 > 0:
        #     degree -= 1
        ER = nx.random_graphs.watts_strogatz_graph(node_num, int(degree), 0.5)
        tree_ka = nx.minimum_spanning_tree(ER, algorithm="kruskal")
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

        # nx.draw(tree_ka, node_size=200, with_labels=True)
        # plt.show()

        for i in np.arange(0, node_num, 1):
            # print("i:", i)

            cal = CalDS(tree_ka, unift_list, i)
            ds_dict = cal.cal_ds2()
            rc_num = len(list(ds_dict.values()))
            aver_prob_dict[i] = rc_num * mean(list(ds_dict.values()))
            # print("cal")

            cal_max_min_ds = CalMaxMinDS(tree_ka, unift_list, i)
            max_permute_prob = cal_max_min_ds.cal_max_ds()
            min_permute_prob = cal_max_min_ds.cal_min_ds()
            geo_min_max_prob = sqrt(max_permute_prob * min_permute_prob)
            geo_max_min_prob_dict[i] = rc_num * geo_min_max_prob
            aver_min_max_prob = (min_permute_prob + min_permute_prob)/2
            aver_max_min_prob_dict[i] = rc_num * aver_min_max_prob
            max_prob_dict[i] = rc_num * max_permute_prob
            min_prob_dict[i] = rc_num * min_permute_prob
            # print("cal_max_min_ds")

            cal_BFS_ds = CalBFSMaxMinDS(tree_ka, unift_list, i)
            max_bfs_permute_prob = cal_BFS_ds.cal_BFS_max_ds()
            min_bfs_permute_prob = cal_BFS_ds.cal_BFS_min_ds()
            geo_BFS_prob = sqrt(max_bfs_permute_prob * min_bfs_permute_prob)
            geo_BFS_prob_dict[i] = rc_num * geo_BFS_prob
            aver_BFS_prob = (max_bfs_permute_prob + min_bfs_permute_prob)/2
            aver_BFS_prob_dict[i] = rc_num * aver_BFS_prob
            BFS_max_prob_dict[i] = rc_num * max_bfs_permute_prob
            BFS_min_prob_dict[i] = rc_num * min_bfs_permute_prob
            # print("cal_BFS_ds")

        top1_aver_prob_idx_list = max_idx_list_get(aver_prob_dict)
        top1_geo_BFS_idx_list = max_idx_list_get(geo_BFS_prob_dict)
        top1_geo_max_min_idx_list = max_idx_list_get(geo_max_min_prob_dict)
        top1_aver_BFS_idx_list = max_idx_list_get(aver_BFS_prob_dict)
        top1_aver_max_min_idx_list = max_idx_list_get(aver_max_min_prob_dict)
        top1_BFS_max_idx_list = max_idx_list_get(BFS_max_prob_dict)
        top1_BFS_min_idx_list = max_idx_list_get(BFS_min_prob_dict)
        top1_max_idx_list = max_idx_list_get(max_prob_dict)
        top1_min_idx_list = max_idx_list_get(min_prob_dict)

        p = nx.shortest_path(tree_ka)
        geo_BFS_hop_err_list = []
        geo_max_min_hop_err_list = []
        aver_BFS_hop_err_list = []
        aver_max_min_hop_err_list = []
        BFS_max_hop_err_list = []
        BFS_min_hop_err_list = []
        max_hop_err_list = []
        min_hop_err_list = []

        for idx1 in top1_aver_prob_idx_list:
            for idx2 in top1_geo_BFS_idx_list:
                geo_BFS_hop_err_list.append(len(p[idx1][idx2])-1)
            for idx3 in top1_geo_max_min_idx_list:
                geo_max_min_hop_err_list.append(len(p[idx1][idx3]) - 1)
            for idx4 in top1_aver_BFS_idx_list:
                aver_BFS_hop_err_list.append(len(p[idx1][idx4]) - 1)
            for idx5 in top1_aver_max_min_idx_list:
                aver_max_min_hop_err_list.append(len(p[idx1][idx5]) - 1)
            for idx6 in top1_BFS_max_idx_list:
                BFS_max_hop_err_list.append(len(p[idx1][idx6]) - 1)
            for idx7 in top1_BFS_min_idx_list:
                BFS_min_hop_err_list.append(len(p[idx1][idx7]) - 1)
            for idx8 in top1_max_idx_list:
                max_hop_err_list.append(len(p[idx1][idx8]) - 1)
            for idx9 in top1_min_idx_list:
                min_hop_err_list.append(len(p[idx1][idx9]) - 1)

        geo_BFS_hop_err = min(geo_BFS_hop_err_list)
        geo_max_min_hop_err = min(geo_max_min_hop_err_list)
        aver_BFS_hop_err = min(aver_BFS_hop_err_list)
        aver_max_min_hop_err = min(aver_max_min_hop_err_list)
        BFS_max_hop_err = min(BFS_max_hop_err_list)
        BFS_min_hop_err = min(BFS_min_hop_err_list)
        max_hop_err = min(max_hop_err_list)
        min_hop_err = min(min_hop_err_list)

        top1_geo_BFS_per_list.append(geo_BFS_hop_err)
        top1_geo_max_min_per_list.append(geo_max_min_hop_err)
        top1_aver_BFS_per_list.append(aver_BFS_hop_err)
        top1_aver_max_min_per_list.append(aver_max_min_hop_err)
        top1_BFS_max_per_list.append(BFS_max_hop_err)
        top1_BFS_min_per_list.append(BFS_min_hop_err)
        top1_max_per_list.append(max_hop_err)
        top1_min_per_list.append(min_hop_err)

        # nx.draw(tree_ka, node_size=200, with_labels=True)
        # plt.show()

    # print("distance_list:", distance_list)
    # print("BFS_win_List:", BFS_win_List)

    # plt.plot(distance_list, BFS_win_List)
    # plt.show()["BFSGeo", "BFSAvr", "BFSRan", "DegGeo", "DegAvr", "DegMax", "DegMin", "DegRan"]
    md_list = ["BFSGeo"] * tree_num + ["BFSAvr"] * tree_num + ["BFSRan"] * tree_num + ["DegGeo"] * tree_num + [
        "DegAvr"] * tree_num + ["DegMax"] * tree_num + ["DegMin"] * tree_num + ["DegRan"] * tree_num
    print("mdlist:", md_list)
    hop_err_list = top1_geo_BFS_per_list + top1_aver_BFS_per_list + top1_BFS_max_per_list + top1_geo_max_min_per_list + top1_aver_max_min_per_list + top1_max_per_list + top1_min_per_list + top1_min_per_list

    print("hop_err_list:", hop_err_list)
    dataframe = pd.DataFrame({"mds": md_list, "hop err": hop_err_list})
    dataframe.to_csv("hop_error\\ws.csv", index=False, sep=',')
    return dataframe


def run4():
    distance_list = []
    BFS_win_List = []
    top1_geo_BFS_per_list = []
    top1_geo_max_min_per_list = []
    top1_aver_BFS_per_list = []
    top1_aver_max_min_per_list = []
    top1_BFS_max_per_list = []
    top1_BFS_min_per_list = []
    top1_max_per_list = []
    top1_min_per_list = []

    tree_num = 20

    for i in range(tree_num):
        print("i:", i)
        node_num = 10
        degree = floor(6)
        if degree % 2 > 0:
            degree -= 1
        ER = nx.random_graphs.random_regular_graph(degree, node_num)
        tree_ka = nx.minimum_spanning_tree(ER, algorithm="kruskal")
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

        # nx.draw(tree_ka, node_size=200, with_labels=True)
        # plt.show()

        for i in np.arange(0, node_num, 1):
            # print("i:", i)

            cal = CalDS(tree_ka, unift_list, i)
            ds_dict = cal.cal_ds2()
            rc_num = len(list(ds_dict.values()))
            aver_prob_dict[i] = rc_num * mean(list(ds_dict.values()))
            # print("cal")

            cal_max_min_ds = CalMaxMinDS(tree_ka, unift_list, i)
            max_permute_prob = cal_max_min_ds.cal_max_ds()
            min_permute_prob = cal_max_min_ds.cal_min_ds()
            geo_min_max_prob = sqrt(max_permute_prob * min_permute_prob)
            geo_max_min_prob_dict[i] = rc_num * geo_min_max_prob
            aver_min_max_prob = (min_permute_prob + min_permute_prob)/2
            aver_max_min_prob_dict[i] = rc_num * aver_min_max_prob
            max_prob_dict[i] = rc_num * max_permute_prob
            min_prob_dict[i] = rc_num * min_permute_prob
            # print("cal_max_min_ds")

            cal_BFS_ds = CalBFSMaxMinDS(tree_ka, unift_list, i)
            max_bfs_permute_prob = cal_BFS_ds.cal_BFS_max_ds()
            min_bfs_permute_prob = cal_BFS_ds.cal_BFS_min_ds()
            geo_BFS_prob = sqrt(max_bfs_permute_prob * min_bfs_permute_prob)
            geo_BFS_prob_dict[i] = rc_num * geo_BFS_prob
            aver_BFS_prob = (max_bfs_permute_prob + min_bfs_permute_prob)/2
            aver_BFS_prob_dict[i] = rc_num * aver_BFS_prob
            BFS_max_prob_dict[i] = rc_num * max_bfs_permute_prob
            BFS_min_prob_dict[i] = rc_num * min_bfs_permute_prob
            # print("cal_BFS_ds")

        top1_aver_prob_idx_list = max_idx_list_get(aver_prob_dict)
        top1_geo_BFS_idx_list = max_idx_list_get(geo_BFS_prob_dict)
        top1_geo_max_min_idx_list = max_idx_list_get(geo_max_min_prob_dict)
        top1_aver_BFS_idx_list = max_idx_list_get(aver_BFS_prob_dict)
        top1_aver_max_min_idx_list = max_idx_list_get(aver_max_min_prob_dict)
        top1_BFS_max_idx_list = max_idx_list_get(BFS_max_prob_dict)
        top1_BFS_min_idx_list = max_idx_list_get(BFS_min_prob_dict)
        top1_max_idx_list = max_idx_list_get(max_prob_dict)
        top1_min_idx_list = max_idx_list_get(min_prob_dict)

        p = nx.shortest_path(tree_ka)
        geo_BFS_hop_err_list = []
        geo_max_min_hop_err_list = []
        aver_BFS_hop_err_list = []
        aver_max_min_hop_err_list = []
        BFS_max_hop_err_list = []
        BFS_min_hop_err_list = []
        max_hop_err_list = []
        min_hop_err_list = []

        for idx1 in top1_aver_prob_idx_list:
            for idx2 in top1_geo_BFS_idx_list:
                geo_BFS_hop_err_list.append(len(p[idx1][idx2])-1)
            for idx3 in top1_geo_max_min_idx_list:
                geo_max_min_hop_err_list.append(len(p[idx1][idx3]) - 1)
            for idx4 in top1_aver_BFS_idx_list:
                aver_BFS_hop_err_list.append(len(p[idx1][idx4]) - 1)
            for idx5 in top1_aver_max_min_idx_list:
                aver_max_min_hop_err_list.append(len(p[idx1][idx5]) - 1)
            for idx6 in top1_BFS_max_idx_list:
                BFS_max_hop_err_list.append(len(p[idx1][idx6]) - 1)
            for idx7 in top1_BFS_min_idx_list:
                BFS_min_hop_err_list.append(len(p[idx1][idx7]) - 1)
            for idx8 in top1_max_idx_list:
                max_hop_err_list.append(len(p[idx1][idx8]) - 1)
            for idx9 in top1_min_idx_list:
                min_hop_err_list.append(len(p[idx1][idx9]) - 1)

        geo_BFS_hop_err = min(geo_BFS_hop_err_list)
        geo_max_min_hop_err = min(geo_max_min_hop_err_list)
        aver_BFS_hop_err = min(aver_BFS_hop_err_list)
        aver_max_min_hop_err = min(aver_max_min_hop_err_list)
        BFS_max_hop_err = min(BFS_max_hop_err_list)
        BFS_min_hop_err = min(BFS_min_hop_err_list)
        max_hop_err = min(max_hop_err_list)
        min_hop_err = min(min_hop_err_list)

        top1_geo_BFS_per_list.append(geo_BFS_hop_err)
        top1_geo_max_min_per_list.append(geo_max_min_hop_err)
        top1_aver_BFS_per_list.append(aver_BFS_hop_err)
        top1_aver_max_min_per_list.append(aver_max_min_hop_err)
        top1_BFS_max_per_list.append(BFS_max_hop_err)
        top1_BFS_min_per_list.append(BFS_min_hop_err)
        top1_max_per_list.append(max_hop_err)
        top1_min_per_list.append(min_hop_err)

        # nx.draw(tree_ka, node_size=200, with_labels=True)
        # plt.show()

    # print("distance_list:", distance_list)
    # print("BFS_win_List:", BFS_win_List)

    # plt.plot(distance_list, BFS_win_List)
    # plt.show()["BFSGeo", "BFSAvr", "BFSRan", "DegGeo", "DegAvr", "DegMax", "DegMin", "DegRan"]
    md_list = ["BFSGeo"] * tree_num + ["BFSAvr"] * tree_num + ["BFSRan"] * tree_num + ["DegGeo"] * tree_num + [
        "DegAvr"] * tree_num + ["DegMax"] * tree_num + ["DegMin"] * tree_num + ["DegRan"] * tree_num
    print("mdlist:", md_list)
    hop_err_list = top1_geo_BFS_per_list + top1_aver_BFS_per_list + top1_BFS_max_per_list + top1_geo_max_min_per_list + top1_aver_max_min_per_list + top1_max_per_list + top1_min_per_list + top1_min_per_list
    print("hop_err_list:", hop_err_list)
    dataframe = pd.DataFrame({"mds": md_list, "hop err": hop_err_list})
    dataframe.to_csv("hop_error\\ran.csv", index=False, sep=',')
    return dataframe


def box_plot(er_df, ba_df, ws_df, ran_df):
    mpl.rcParams['font.sans-serif'] = ['Times New Roman']
    mpl.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(2, 2)
    axes[0, 0].set_xlabel('Y Label', fontsize=0, alpha=0)
    axes[0, 1].set_xlabel('Y Label', fontsize=0, alpha=0)
    axes[1, 0].set_xlabel('Y Label', fontsize=0, alpha=0)
    axes[1, 1].set_xlabel('Y Label', fontsize=0, alpha=0)
    axes[0, 0].set_ylabel('Y Label', fontsize=13, )
    axes[0, 1].set_ylabel('Y Label', fontsize=13, )
    axes[1, 0].set_ylabel('Y Label', fontsize=13, )
    axes[1, 1].set_ylabel('Y Label', fontsize=13, )
    axes[0,0].set_xticklabels(axes[0,0].get_xticklabels(), rotation=-30, fontsize=13)
    axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=-30, fontsize=13)
    axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=-30, fontsize=13)
    axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=-30, fontsize=13)
    axes[0, 0].set_title('ER Network', )
    axes[0, 1].set_title('BA Network',)
    axes[1, 0].set_title('WS Network', )
    axes[1, 1].set_title('Regular Network', )
    sns.boxplot(x="mds", y="hop err", data=er_df,palette="Set3",  ax=axes[0,0])  # 左图
    sns.boxplot(x="mds", y="hop err", data=ba_df,palette="Set3",  ax=axes[0,1])  # 右图
    sns.boxplot(x="mds", y="hop err", data=ws_df,palette="Set3",  ax=axes[1,0])  # 右图
    sns.boxplot(x="mds", y="hop err", data=ran_df,palette="Set3",  ax=axes[1,1])  # 右图
    fig.tight_layout()
    plt.show()


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


def excel_to_df():
    er_df = pd.read_csv("hop_error\\er.csv", sep=",")
    er_df.head()
    print("df:", er_df)
    ba_df = pd.read_csv("hop_error\\ba.csv", sep=",")
    ws_df = pd.read_csv("hop_error\\ws.csv", sep=",")
    ran_df = pd.read_csv("hop_error\\ran.csv", sep=",")
    # fig, axes = plt.subplots(1, 1)
    # sns.boxplot(x="mds", y="hop err", data=df, palette="Set3")
    # plt.show()
    box_plot(er_df, ba_df, ws_df, ran_df)


def excel_to_df2():
    er_df = pd.read_csv("source_err\\er.csv", sep=",")
    print("df:", er_df)
    ba_df = pd.read_csv("source_err\\ba.csv", sep=",")
    ws_df = pd.read_csv("source_err\\ws.csv", sep=",")
    ran_df = pd.read_csv("source_err\\ran.csv", sep=",")
    # fig, axes = plt.subplots(1, 1)
    # sns.boxplot(x="mds", y="hop err", data=df, palette="Set3")
    # plt.show()
    box_plot(er_df, ba_df, ws_df, ran_df)


def get_uninfected_node_list(g:nx.graph):
    degree_list = nx.degree(g)
    leafed_index_list = [i[0] for i in degree_list if i[1] == 1]
    random.seed(10)
    uninft_size = len(leafed_index_list)
    uninft_node_list = random.sample(leafed_index_list, math.ceil(0.7 * uninft_size))
    return uninft_node_list


def compare():
    tree_num = 1
    for i in range(tree_num):
        print("i:", i)
        num = 0
        node_num = 13
        # ER = nx.random_graphs.barabasi_albert_graph(node_num, 2)
        ER = nx.random_graphs.erdos_renyi_graph(node_num, 0.25)

        tree_ka = nx.minimum_spanning_tree(ER, algorithm="kruskal")
        # processed_graph = graph_data_process.TreeDataProcess(tree_ka)
        # unift_list = processed_graph.get_uninfected_node_list()
        # pos = nx.spring_layout(tree_ka)
        # nx.draw(tree_ka, pos, with_labels = True, node_size = 100)
        # plt.show()

        unift_list = get_uninfected_node_list(tree_ka)
        ift_list = list(set(list(ER.nodes)).difference(set(unift_list)))
        max_prob_dict = {}
        min_prob_dict = {}
        aver_prob_dict = {}

        node_color = ["r"] * node_num
        for i in unift_list:
            node_color[i] = "b"
        pos = nx.spring_layout(tree_ka)
        nx.draw(tree_ka, node_size=200, with_labels=True, node_color = node_color)
        plt.show()

        for i in ift_list:
            print("i:", i)

            cal = CalDS(tree_ka, unift_list, i)
            ds_dict = cal.cal_ds2()
            rc_num = len(list(ds_dict.values()))
            aver_prob_dict[i] = max(list(ds_dict.values()))
            # print("ds_dict:", ds_dict)

            cal_max_min_ds = CalMaxMinDS(tree_ka, unift_list, i)
            max_permutation, max_permute_prob = cal_max_min_ds.cal_max_ds()
            min_permute_prob = cal_max_min_ds.cal_min_ds()
            max_prob_dict[i] = rc_num * max_permute_prob
            min_prob_dict[i] = rc_num * min_permute_prob
            true_max = max(list(ds_dict.values()))
            print("max_dict:", true_max)
            print("max:", max(ds_dict, key=ds_dict.get))
            # print("min_dict:", min(list(ds_dict.values())))
            # print("min:", min(ds_dict, key=ds_dict.get))
            print("max_appr:", max_permute_prob)
            print("max_permutation:", max_permutation)
            # print("min_appr:", min_permute_prob)
            if true_max == max_permute_prob:
                num = num + 1
        print("equal_num:", num)


if __name__ == '__main__':
    compare()




