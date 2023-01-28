import networkx as nx
import matplotlib.pyplot as plt
import graph_data_process
import copy
from math import *
import numpy as np


class NFeatureDict(dict):
    def __missing__(self, key):
        value = self[key] = type(self)()
        return value


class CalMaxMinDS(object):
    def __init__(self, tree: nx.Graph, unift_list: list, source: int):
        self.tree = tree
        self.unift_list = unift_list
        self.source = source
        self.nfeature_temp = self.preprocess()

    def preprocess(self):
        nfeature_temp = NFeatureDict()

        tree_deg = nx.degree(self.tree)
        for v in tree_deg:
            # print("v:", v)

            nfeature_temp[v[0]]["degree"] = v[1]
            neighbor_list = self.tree.neighbors(v[0])
            uninfected_edge_num = len(set(self.unift_list).difference(set(neighbor_list)))
            nfeature_temp[v[0]]["uninft_edge_num"] = uninfected_edge_num
        # print("preprocess:!!!!")
        return nfeature_temp

    def cal_max_ds(self):
        no_add_select_pool = copy.copy(self.unift_list)
        no_add_select_pool.append(self.source)
        select_pool = [node for node in self.tree.neighbors(self.source)]
        select_pool = list(set(select_pool).difference(set(self.unift_list)))
        edge_prob_num = self.nfeature_temp[self.source]["degree"]
        max_permute_prob = log(1 / edge_prob_num)

        while len(select_pool) > 0:
            # print("select_pool:", select_pool)
            select_pool_dict = {k: self.nfeature_temp[k]["degree"] for k in select_pool}
            next_inft_node = min(select_pool_dict, key=select_pool_dict.get)
            edge_prob_num = edge_prob_num + select_pool_dict[next_inft_node] - 2

            select_pool.remove(next_inft_node)
            no_add_select_pool.append(next_inft_node)
            add_to_select_pool = set(self.tree.neighbors(next_inft_node)).difference(set(no_add_select_pool))
            select_pool.extend(add_to_select_pool)
            if edge_prob_num > 0 and len(select_pool) > 0:
                max_permute_prob = max_permute_prob + log((1 / edge_prob_num))
        return max_permute_prob

    def cal_min_ds(self):
        no_add_select_pool = copy.copy(self.unift_list)
        no_add_select_pool.append(self.source)
        select_pool = [node for node in self.tree.neighbors(self.source)]

        select_pool = list(set(select_pool).difference(set(self.unift_list)))
        edge_prob_num = self.nfeature_temp[self.source]["degree"]
        min_permute_prob = log(1 / edge_prob_num)

        while len(select_pool) > 0:
            select_pool_dict = {k: self.nfeature_temp[k]["degree"] for k in select_pool}
            next_inft_node = max(select_pool_dict, key=select_pool_dict.get)
            edge_prob_num = edge_prob_num + select_pool_dict[next_inft_node] - 2

            select_pool.remove(next_inft_node)
            no_add_select_pool.append(next_inft_node)
            add_to_select_pool = set(self.tree.neighbors(next_inft_node)).difference(set(no_add_select_pool))
            select_pool.extend(add_to_select_pool)
            if edge_prob_num > 0 and len(select_pool) > 0:
                min_permute_prob = min_permute_prob + log(1 / edge_prob_num)

        return min_permute_prob


if __name__ == '__main__':
    node_num = 20
    ER = nx.random_graphs.barabasi_albert_graph(node_num, 1)
    tree_ka = nx.minimum_spanning_tree(ER, algorithm="kruskal")
    processed_graph = graph_data_process.TreeDataProcess(tree_ka)
    # unift_list = processed_graph.get_uninfected_node_list()
    unift_list = []
    geo_prob_dict = {}
    for i in np.arange(0, node_num, 1):
        print("i:", i)
        cal_max_min_ds = CalMaxMinDS(tree_ka, unift_list, i)
        max_permute_prob = cal_max_min_ds.cal_max_ds()
        min_permute_prob = cal_max_min_ds.cal_min_ds()
        print("final max_permute_prob:", max_permute_prob)
        print("final min_permute_prob:", min_permute_prob)
        geo_permute_prob = sqrt(max_permute_prob * min_permute_prob)
        print("final geo_permute_prob:", geo_permute_prob)
        geo_prob_dict[i] = geo_permute_prob
    a = sorted(geo_prob_dict.items(), key=lambda item:item[1], reverse=True)
    print("sort:", a)
