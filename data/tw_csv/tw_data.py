import pandas as pd
import itertools
import csv
import networkx as nx
import copy


def graph_process():
    data = pd.read_csv('tw_data_all.csv', header=None)
    edges = [edge for edge in zip(data.iloc[:,0], data.iloc[:,1])]
    G = nx.Graph()
    G.add_edges_from(edges)
    tree_ka = nx.minimum_spanning_tree(G, algorithm="kruskal")
    edges = tree_ka.edges()
    print(tree_ka.edges())
    with open("tw_data_all_tree.csv", 'wt', newline='') as f1:
        cw = csv.writer(f1)
        for edge in edges:
            cw.writerow(edge)

def graph_process_v2():
    data = pd.read_csv('tw_data_43.csv', header=None)
    edges = [edge for edge in zip(data.iloc[:,0], data.iloc[:,1])]
    G = nx.Graph()
    G.add_edges_from(edges)
    tree_ka = nx.minimum_spanning_tree(G, algorithm="kruskal")
    edges = tree_ka.edges()
    print(tree_ka.edges())
    with open("tw_data_43_tree.csv", 'wt', newline='') as f1:
        cw = csv.writer(f1)
        for edge in edges:
            cw.writerow(edge)

def graph_process_v3():
    data = pd.read_csv('tw_data_76.csv', header=None)
    edges = [edge for edge in zip(data.iloc[:,0], data.iloc[:,1])]
    G = nx.Graph()
    G.add_edges_from(edges)
    tree_ka = nx.minimum_spanning_tree(G, algorithm="kruskal")
    edges = tree_ka.edges()
    print(tree_ka.edges())
    with open("tw_data_76_tree.csv", 'wt', newline='') as f1:
        cw = csv.writer(f1)
        for edge in edges:
            cw.writerow(edge)

def find_connection():
    data = pd.read_csv('hk_csv_v2/edge_list_all.csv', header=None)
    edges = [edge for edge in zip(data.iloc[:, 0], data.iloc[:, 1])]
    G = nx.Graph()
    G.add_edges_from(edges)
    tree_ka = nx.minimum_spanning_tree(G, algorithm="kruskal")
    largest_components = max(nx.connected_components(tree_ka), key=len)
    connect_G = tree_ka.subgraph(largest_components)
    edges = connect_G.edges()
    with open("hk_csv_v2/connect_edge_list_all.csv", 'wt', newline='') as f1:
        cw = csv.writer(f1)
        for edge in edges:
            cw.writerow(edge)


def find_connection_1st_2nd():
    data = pd.read_csv('tw_data_all_tree.csv', header=None)
    edges = [edge for edge in zip(data.iloc[:, 0], data.iloc[:, 1])]
    G = nx.Graph()
    G.add_edges_from(edges)
    tree_ka = nx.minimum_spanning_tree(G, algorithm="kruskal")
    largest_components = max(nx.connected_components(tree_ka), key=len)
    connect_G = tree_ka.subgraph(largest_components)
    edges = copy.deepcopy(connect_G.edges())

    tree_ka.remove_nodes_from(list(largest_components))
    Sec_largest_components = max(nx.connected_components(tree_ka), key=len)
    connect_G_2nd = tree_ka.subgraph(Sec_largest_components)

    print(min(connect_G_2nd.nodes))

    edges_2nd = connect_G_2nd.edges()

    with open("2nd_connect_edge_list_tw.csv", 'wt', newline='') as f1:
        cw = csv.writer(f1)
        for edge in edges:
            cw.writerow(edge)
        for edge in edges_2nd:
            cw.writerow(edge)

def bfs_part_tree():
    data = pd.read_csv('hk_csv_v2/2nd_connect_edge_list_all.csv', header=None)
    edges = [edge for edge in zip(data.iloc[:, 0], data.iloc[:, 1])]
    G = nx.Graph()
    G.add_edges_from(edges)

    dfs_dept = 10

    bfs_part_G1 = nx.bfs_tree(G, 12611, reverse=False, depth_limit=dfs_dept)
    edges1 = bfs_part_G1.edges()

    bfs_part_G2 = nx.bfs_tree(G, 12667, reverse=False, depth_limit=dfs_dept)
    edges2 = bfs_part_G2.edges()
    with open("hk_csv_v3/bfs_part_tree_"+str(dfs_dept)+".csv", 'wt', newline='') as f1:
        cw = csv.writer(f1)
        for edge in edges1:
            cw.writerow(edge)
        for edge in edges2:
            cw.writerow(edge)

def dfs_part_tree():
    data = pd.read_csv('hk_csv_v2/2nd_connect_edge_list_all.csv', header=None)
    edges = [edge for edge in zip(data.iloc[:, 0], data.iloc[:, 1])]
    G = nx.Graph()
    G.add_edges_from(edges)

    dfs_dept = 6

    bfs_part_G1 = nx.dfs_tree(G, 12611, depth_limit=dfs_dept)
    edges1 = bfs_part_G1.edges()

    bfs_part_G2 = nx.dfs_tree(G, 12667, depth_limit=dfs_dept)
    edges2 = bfs_part_G2.edges()
    with open("hk_csv_v2/dfs_part_tree_"+str(dfs_dept)+".csv", 'wt', newline='') as f1:
        cw = csv.writer(f1)
        for edge in edges1:
            cw.writerow(edge)
        for edge in edges2:
            cw.writerow(edge)

def dfs_part_tree_v2():
    data = pd.read_csv('hk_csv_v2/2nd_connect_edge_list_all.csv', header=None)
    edges = [edge for edge in zip(data.iloc[:, 0], data.iloc[:, 1])]
    G = nx.Graph()
    G.add_edges_from(edges)

    dfs_dept = 10

    bfs_part_edges1 = nx.dfs_edges(G, 12611, depth_limit=dfs_dept)

    bfs_part_G2 = nx.dfs_tree(G, 12667, depth_limit=dfs_dept)
    edges2 = bfs_part_G2.edges()
    with open("hk_csv_v3/dfs_part_tree_"+str(dfs_dept)+".csv", 'wt', newline='') as f1:
        cw = csv.writer(f1)
        for edge in bfs_part_edges1:
            cw.writerow(edge)
        for edge in edges2:
            cw.writerow(edge)


if __name__ == '__main__':
    # period_data()
    # edge_list()
    # graph_process()
    # find_connection()
    # find_connection_1st_2nd()
    # bfs_part_tree()
    # dfs_part_tree_v2()

    graph_process_v2()
    graph_process_v3()