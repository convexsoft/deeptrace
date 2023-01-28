import networkx as nx  #导入networkx包，命名为nx
import matplotlib.pylab as plt #导入画图工具包，命名为plt
import numpy as np
import random
from cal_max_min_ds import CalMaxMinDS
import csv

def dfs_tracing(G):
    # pos = nx.spring_layout(G)
    # nx.draw(G, pos, with_labels=True, node_size=50)
    # plt.show()

    nodes = nx.nodes(G)
    idx_node = int(np.ceil(len(nodes)*0.3))
    # print("idx_node:",idx_node)
    tracing_G = nx.Graph()
    traced_node_list = [idx_node]
    untraced_node_list = list(nodes)
    untraced_node_list.remove(idx_node)
    G_root = finding_root(G)

    error_dist_list = []
    furcation_nodes = [idx_node]

    k = 0
    while len(untraced_node_list) > 0:
        to_trace_node = traced_node_list[-1]
        neighbors = list(set(G.neighbors(to_trace_node))-set(traced_node_list))
        if len(neighbors)>0:
            next_node = neighbors[-1]
            untraced_node_list.remove(next_node)
            # print("next_node:", next_node)
            print("untraced_node_list:", len(untraced_node_list))
            next_neighbors = list(set(G.neighbors(next_node)) - set(traced_node_list))
            if len(next_neighbors)>1:
                furcation_nodes.append(next_node)
                # print("add_furcation_nodes:", furcation_nodes)
                # print("furcation_nodes:", next_node)

        else:
            flag = True
            while flag == True:
                neighbors = list(set(G.neighbors(furcation_nodes[-1])) - set(traced_node_list))
                if len(neighbors)>0:
                    to_trace_node = furcation_nodes[-1]
                    next_node = neighbors[-1]
                    untraced_node_list.remove(next_node)
                    print("untraced_node_list:", len(untraced_node_list))
                    # print("next_node_from_cha:", next_node)
                    next_neighbors = list(set(G.neighbors(next_node)) - set(traced_node_list))
                    if len(next_neighbors) > 1:
                        furcation_nodes.append(next_node)
                        # print("furcation_nodes:", next_node)
                        # print("add_furcation_nodes:", furcation_nodes)
                    flag = False
                else:
                    furcation_nodes.pop()
                    # print("furcation_nodes:", furcation_nodes)

        traced_node_list.append(next_node)
        edge = (to_trace_node, next_node)
        # print("edge:", edge)
        tracing_G.add_edge(*edge)
        k = k+1
        if k % 10 == 1:
            tracing_G_root = finding_root(tracing_G)
            error_dist = len(nx.shortest_path(G, tracing_G_root, G_root)) - 1
            error_dist_list.append(error_dist)

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
    origin_node_num = 40
    final_node_num = 1000
    def_g_diam = 50
    thresh = 3

    diam_graph_generate(origin_node_num, final_node_num)
    G, g_diam = diam_graph_generate(origin_node_num, final_node_num)

    error_dist_all = []
    num = 30
    while num > 0:
        G, g_diam = diam_graph_generate(origin_node_num, final_node_num)
        if g_diam >def_g_diam-thresh and g_diam <def_g_diam+thresh:
            print("num:", num)
            num = num -1
            error_dist_list = dfs_tracing(G)
            error_dist_all.append(error_dist_list)
    #
    with open("dfs_tracing_res_50.csv","w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(error_dist_all)

    # dfs_tracing(G)
