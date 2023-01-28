import networkx as nx  #导入networkx包，命名为nx
import matplotlib.pylab as plt #导入画图工具包，命名为plt
import numpy as np
import random
from cal_max_min_ds import CalMaxMinDS
import csv

def bfs_tracing(G):
    nodes = nx.nodes(G)
    idx_node = int(np.ceil(len(nodes)*0.3))
    # print("idx_node:",idx_node)
    tracing_G = nx.Graph()
    traced_node_list = [idx_node]
    untraced_node_list = list(nodes)
    to_trace_node_temp = [idx_node]
    G_root = finding_root(G)

    error_dist_list = []
    k = 0
    while len(untraced_node_list) > 0:
        to_trace_node_list = to_trace_node_temp
        to_trace_node_temp = []
        for n in to_trace_node_list:
            untraced_node_list.remove(n)
            print("untraced_node_list:", len(untraced_node_list))
            traced_node_list.append(n)
            neighbors_list = list(set(G.neighbors(n))-set(traced_node_list))
            to_trace_node_temp = to_trace_node_temp + neighbors_list
            for i in neighbors_list:
                new_edge = (n,i)
                tracing_G.add_edge(*new_edge)
                k = k+1
                if k%10==1:
                    tracing_G_root = finding_root(tracing_G)
                    error_dist = len(nx.shortest_path(G, tracing_G_root, G_root))-1
                    # print("error_dist:", error_dist)
                    error_dist_list.append(error_dist)
    # error_dist_list.append(G_root)
    return error_dist_list


def finding_root(tracing_G):
    degree_list = nx.degree(tracing_G)
    leafed_index_list = [i[0] for i in degree_list if i[1] == 1]
    geo_prob_dict = {}
    node_list = nx.nodes(tracing_G)
    node_list = list(set(node_list)-set(leafed_index_list))
    for i in node_list:
        # print("i:", i)
        cal_max_min_ds = CalMaxMinDS(tracing_G, leafed_index_list, i)
        max_permute_prob = cal_max_min_ds.cal_max_ds()
        min_permute_prob = cal_max_min_ds.cal_min_ds()
        # print("final max_permute_prob:", max_permute_prob)
        # print("final min_permute_prob:", min_permute_prob)
        geo_permute_prob = np.sqrt(max_permute_prob * min_permute_prob)
        # print("final geo_permute_prob:", geo_permute_prob)
        geo_prob_dict[i] = geo_permute_prob

    # a = sorted(geo_prob_dict.items(), key=lambda item: item[1], reverse=True)
    # print("sort:", a)
    res = None
    if len(geo_prob_dict)>0:
        res = max(geo_prob_dict, key=lambda x:geo_prob_dict[x])
        # print("res:", res)
    return res


def diam_graph_generate(origin_node_num, final_node_num):
    G = nx.Graph()
    edge_list = []
    for i in range(origin_node_num):
        edge_list.append((i,i+1))
    G.add_edges_from(edge_list)

    g_diam = nx.diameter(G)
    print("g_diam1:", g_diam)

    node_choise_list = list(range(origin_node_num))
    for i in range(origin_node_num + 1, final_node_num):
        next_conn_node = random.choice(node_choise_list)
        node_choise_list.append(i)
        edge = (i, next_conn_node)
        G.add_edge(*edge)

    g_diam = nx.diameter(G)
    print("g_diam:", g_diam)


    return G, g_diam




if __name__ == '__main__':
    origin_node_num = 297
    final_node_num = 1000
    def_g_diam = 300
    thresh = 3

    diam_graph_generate(origin_node_num, final_node_num)

    error_dist_all = []
    num = 30
    while num > 0:
        G, g_diam = diam_graph_generate(origin_node_num, final_node_num)
        if g_diam >def_g_diam-thresh and g_diam <def_g_diam+thresh:
            print("num:", num)
            num = num -1
            error_dist_list = bfs_tracing(G)
            error_dist_all.append(error_dist_list)

    with open("bfs_tracing_res2.csv","w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(error_dist_all)

    # bfs_tracing(G)
