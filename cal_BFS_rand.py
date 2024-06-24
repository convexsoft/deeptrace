import networkx as nx
import matplotlib.pyplot as plt
import graph_data_process
import copy
from math import *
import numpy as np
import random


class NFeatureDict(dict):
    def __missing__(self, key):
        value = self[key] = type(self)()
        return value


class CalBFSRandDS(object):
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


    def cal_BFS_rand_ds(self):
        no_add_select_pool = copy.copy(self.unift_list)
        no_add_select_pool.append(self.source)
        select_pool = [node for node in self.tree.neighbors(self.source)]

        select_pool = list(set(select_pool).difference(set(self.unift_list)))

        edge_prob_num = self.nfeature_temp[self.source]["degree"]
        BFS_rand_permute_prob = log(1 / edge_prob_num)

        update_select_pool = copy.deepcopy(select_pool)
        bfs_trail = [self.source]
        while len(update_select_pool)>0:
            select_pool = copy.deepcopy(update_select_pool)
            while len(select_pool) > 0:
                select_pool_dict = {k: self.nfeature_temp[k]["degree"] for k in select_pool}

                # next_inft_node = max(select_pool_dict, key=select_pool_dict.get)
                next_inft_node = random.sample(select_pool,1)[0]
                bfs_trail.append(next_inft_node)
                edge_prob_num = edge_prob_num + select_pool_dict[next_inft_node] - 2
                # print("----edge_prob_num:", edge_prob_num)

                select_pool.remove(next_inft_node)
                update_select_pool.remove(next_inft_node)
                no_add_select_pool.append(next_inft_node)
                add_to_select_pool = set(self.tree.neighbors(next_inft_node)).difference(set(no_add_select_pool))

                update_select_pool.extend(add_to_select_pool)
                # print("==select_pool:", select_pool)

                if edge_prob_num > 0 and len(update_select_pool)>0:
                    # print("edge_prob_num:", edge_prob_num)
                    BFS_rand_permute_prob = BFS_rand_permute_prob + log(1 / edge_prob_num)
        # print("bfs_trail:",bfs_trail)
        return BFS_rand_permute_prob


if __name__ == '__main__':
    node_num = 10
    # random.seed(3)
    ER = nx.random_graphs.barabasi_albert_graph(node_num, 1)
    tree_ka = nx.minimum_spanning_tree(ER, algorithm="kruskal")
    processed_graph = graph_data_process.TreeDataProcess(tree_ka)
    unift_list = processed_graph.get_uninfected_node_list()
    ift_list = list(set(np.arange(0, node_num, 1)).difference(set(unift_list)))

    BFS_rand_prob_dict = {}
    for i in ift_list:
        print("i:", i)
        cal_bfs_ds = CalBFSRandDS(tree_ka, unift_list, i)
        BFS_rand_permute_prob = cal_bfs_ds.cal_BFS_rand_ds()
        print("final BFS_rand_permute_prob:", BFS_rand_permute_prob)
        BFS_rand_prob_dict[i] = BFS_rand_permute_prob
    a = sorted(BFS_rand_prob_dict.items(), key=lambda item:item[1], reverse=True)
    print("sort:", a)


    node_color = ["red"] * node_num
    for i in unift_list:
        node_color[i] = "green"
    # print("node_color:", node_color)
    pos = nx.spring_layout(tree_ka)
    options = {"edgecolors": "tab:gray", "node_size": 200, "alpha": 0.9}
    nx.draw_networkx_nodes(tree_ka, pos, node_color="tab:red", **options)
    nx.draw_networkx_nodes(tree_ka, pos, nodelist=unift_list, node_color="tab:blue", **options)
    nx.draw_networkx_edges(tree_ka, pos, width=1.0, alpha=0.5)
    nx.draw_networkx_labels(tree_ka, pos, font_size=8, font_color="black")
    plt.show()
