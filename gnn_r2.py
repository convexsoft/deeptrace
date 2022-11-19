import dgl
import dgl.nn as dglnn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from graph_data_process import TreeDataProcess
import networkx as nx
import numpy as np
from dgl.convert import from_networkx
from dgl.convert import to_networkx
from cal_max_min_ds import CalMaxMinDS
from math import sqrt
from math import floor
import matplotlib.pyplot as plt
import random
import pandas as pd
import itertools


class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 = dglnn.SAGEConv(
            in_feats=in_feats, out_feats=hid_feats, aggregator_type='lstm')
        self.conv2 = dglnn.SAGEConv(
            in_feats=hid_feats, out_feats=hid_feats, aggregator_type='lstm')
        self.conv3 = dglnn.SAGEConv(
            in_feats=hid_feats, out_feats=hid_feats, aggregator_type='lstm')
        self.conv4 = dglnn.SAGEConv(
            in_feats=hid_feats, out_feats=out_feats, aggregator_type='lstm')

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        h = self.conv2(graph, h)
        h = F.relu(h)
        h = self.conv3(graph, h)
        h = F.relu(h)
        h = self.conv4(graph, h)
        return h


def evaluate(model, graph, features, labels):
    model.eval()
    with th.no_grad():
        logits = model(graph, features)
        # logits = logits[mask]
        # labels = labels[mask]
        # classification
        # _, indices = th.max(logits, dim=1)
        # correct = th.sum(indices == labels)
        # return correct.item() * 1.0 / len(labels)

        # regression
        # print("logits:", logits)
        # x_axix = np.arange(1, len(labels)+1, 1)
        # print(x_axix)
        # real_val = labels.numpy()
        # val_val = logits.numpy()
        # print("real_val:",real_val)
        # plt.plot(x_axix, real_val, color='green', label='real_val')
        # plt.plot(x_axix, val_val, color='blue', label='val_val')
        # plt.legend()  # 显示图例
        # plt.xlabel('iteration times')
        # plt.ylabel('rate')
        # plt.show()
        # return

        # MSE
        loss = F.mse_loss(logits, labels)

        #r2—score
        r2_loss = R2_score(logits, labels)
        return r2_loss


def evaluate_position(model, graph, features, labels):
    model.eval()
    with th.no_grad():
        logits = model(graph, features)
        real_labels_list = labels.numpy().tolist()
        max_real_idx = real_labels_list.index(max(real_labels_list))
        # print("real_labels_list:", real_labels_list)

        eval_label_list = logits.numpy().tolist()
        pos_eval_val = eval_label_list[max_real_idx]
        eval_label_list_sort = sorted(eval_label_list, reverse=True)
        # print("eval_label_list_sort:", eval_label_list_sort)
        pos_eval_val_idx = eval_label_list_sort.index(pos_eval_val)
        # print("pos_eval_val_idx:", pos_eval_val_idx)

    # print("######position:", pos_eval_val_idx/len(real_labels_list))
    return pos_eval_val_idx


def evaluate_prob(model, graph, features, labels):
    model.eval()
    with th.no_grad():
        logits = model(graph, features)
        real_labels_list = labels.numpy().tolist()
        eval_label_list = list(itertools.chain.from_iterable(logits.numpy().tolist()))
        node_index = range(len(real_labels_list))
        real_labels_dict = dict(zip(node_index, real_labels_list))
        eval_labels_dict = dict(zip(node_index, eval_label_list))
        print("real_labels_dict:", sorted(real_labels_dict.items(), key=lambda item:item[1], reverse=True))
        print("eval_labels_dict:", sorted(eval_labels_dict.items(), key=lambda item:item[1], reverse=True))
    return real_labels_list, eval_label_list


def _sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return mask


def data_process_for_single_tree(tree: nx.Graph) -> dgl:
    graph = TreeDataProcess(tree)
    unift_node_list = graph.get_uninfected_node_list()
    graph_nfeature = graph.nfeature_process()
    tree.remove_nodes_from(list(unift_node_list))

    # label
    geo_permute_prob_list = []
    for node in nx.nodes(tree):
        if node not in set(unift_node_list):
            cal_ds = CalMaxMinDS(tree, unift_node_list, node)
            max_permute_prob = cal_ds.cal_max_ds()
            min_permute_prob = cal_ds.cal_min_ds()
            geo_permute_prob = sqrt(max_permute_prob * min_permute_prob)
            geo_permute_prob_list.append(geo_permute_prob)

    # classification
    # max_idx = geo_permute_prob_list.index(max(geo_permute_prob_list))
    # labels = [0] * len(geo_permute_prob_list)
    # labels[max_idx] = 1

    # regression
    labels = geo_permute_prob_list
    labels = th.tensor(labels)

    # feature
    graph_nfeature_arr = []
    for k, v in graph_nfeature.items():
        if k not in set(unift_node_list):
            feature_row = [v["node_num"], v["degree_per"], v["degree_per_aver"], v["inft_ndegree_per"], v["inft_alldegree_per"], v["distance_per"], v["layer_rate"], v["layer_num"]]
            # feature_row = [float(random.randint(100,900))]
            # feature_row = [ v["node_num"], v["degree_per"], v["degree_per_aver"], v["inft_ndegree_per"], v["inft_alldegree_per"], v["distance_per"], v["layer_rate"], v["layer_num"]]
            graph_nfeature_arr.append(feature_row)
    g_nfeature = th.tensor(graph_nfeature_arr)
    dgl_tree = from_networkx(tree)
    dgl_tree.ndata["labels"] = labels
    dgl_tree.ndata["feat"] = g_nfeature
    return dgl_tree


def train_data_process(tree_num: int, node_num: int):
    nxtree_list = []
    for i in range(tree_num):
        # ER = nx.random_graphs.barabasi_albert_graph(node_num, 1)
        # ER = nx.random_graphs.erdos_renyi_graph(node_num, 0.3)
        degree = floor(node_num * 3 / 20)
        ER = nx.random_graphs.watts_strogatz_graph(node_num, degree, 0.3)
        tree_ka = nx.minimum_spanning_tree(ER, algorithm="kruskal")
        nxtree_list.append(tree_ka)

    dgl_tree_list = []
    for tree in nxtree_list:
        dgl_tree = data_process_for_single_tree(tree)
        dgl_tree_list.append(dgl_tree)
    return dgl.batch(dgl_tree_list)


def test_data_process(tree_num: int, node_num: int):
    nxtree_list = []
    for i in range(tree_num):
        # ER = nx.random_graphs.erdos_renyi_graph(node_num,0.3)
        # ER = nx.random_graphs.barabasi_albert_graph(node_num, 1)
        degree = floor(node_num * 3 / 20)
        # ER = nx.random_graphs.random_regular_graph(degree, 100)
        ER = nx.random_graphs.watts_strogatz_graph(node_num, degree, 0.3)
        tree_ka = nx.minimum_spanning_tree(ER, algorithm="kruskal")
        nxtree_list.append(tree_ka)

    dgl_tree_list = []
    for tree in nxtree_list:
        dgl_tree = data_process_for_single_tree(tree)
        dgl_tree_list.append(dgl_tree)
    return dgl.batch(dgl_tree_list)


def R2_score(eval_label, real_val):
    len_label = len(eval_label)
    real_val_aver = sum(real_val)/len_label
    sum_diff1 = 0
    sum_diff2 = 0
    for i in range(len_label):
        sum_diff1 = sum_diff1 + (eval_label[i] - real_val[i])**2
        sum_diff2 = sum_diff2 + (real_val[i] - real_val_aver)**2
    res = 1 - (sum_diff1 / sum_diff2)
    return res


def gnn_test_mse(train_patch_tree: dgl, test_patch_tree: dgl):
    # train
    train_features_dim = train_patch_tree.ndata["feat"].shape[1]
    train_node_features = train_patch_tree.ndata["feat"]
    train_node_labels = train_patch_tree.ndata["labels"]

    # test
    test_node_features = test_patch_tree.ndata["feat"]
    test_node_labels = test_patch_tree.ndata["labels"]

    model = SAGE(in_feats=train_features_dim, hid_feats=50, out_feats=1)
    opt = th.optim.Adam(model.parameters())
    # 开始训练
    val_val_list = []
    epoch_num = 1
    for epoch in range(epoch_num):
        print('Epoch {}'.format(epoch))
        model.train()
        logits = model(train_patch_tree, train_node_features)
        loss = F.mse_loss(logits, train_node_labels)
        r2_lost_train = R2_score(logits, train_node_labels)
        print('r2_lost_train = {:.4f}'.format(r2_lost_train.item()))
        # 分类计算验证集accuracy
        # loss = F.cross_entropy(logits, node_labels)
        # acc = evaluate(model, patch_tree, node_features, node_labels, val_mask)
        # evaluate(model, patch_tree, node_features, node_labels, val_mask)

        # backward propagation
        opt.zero_grad()
        loss.backward()
        opt.step()

        val_r2_loss = evaluate(model, test_patch_tree, test_node_features, test_node_labels)
        print('val_r2_loss = {:.4f}'.format(val_r2_loss.item()))

        val_val_list.append(val_r2_loss.item())
        print('loss = {:.4f}'.format(loss.item()))

    x_axis = range(1, epoch_num+1, 1)
    plt.plot(x_axis, val_val_list, color='blue', label='val_val')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('R2')
    plt.show()


def gnn_test_pos(train_patch_tree: dgl, test_patch_tree):
    # train
    train_features_dim = train_patch_tree.ndata["feat"].shape[1]
    train_node_features = train_patch_tree.ndata["feat"]
    train_node_labels = train_patch_tree.ndata["labels"]

    # test
    test_node_features = test_patch_tree.ndata["feat"]
    test_node_labels = test_patch_tree.ndata["labels"]

    model = SAGE(in_feats=train_features_dim, hid_feats=100, out_feats=1)
    opt = th.optim.Adam(model.parameters())

    # 开始训练
    val_val_list = []
    epoch_num = 50
    for epoch in range(epoch_num):
        print('Epoch {}'.format(epoch))
        model.train()
        logits = model(train_patch_tree, train_node_features)
        loss = F.mse_loss(logits, train_node_labels)

        # 分类计算验证集accuracy
        # loss = F.cross_entropy(logits, node_labels)
        # acc = evaluate(model, patch_tree, node_features, node_labels, val_mask)
        # evaluate(model, patch_tree, node_features, node_labels, val_mask)

        # backward propagation
        opt.zero_grad()
        loss.backward()
        opt.step()
        print('loss = {:.4f}'.format(loss.item()))

        val_r2_loss = evaluate(model, test_patch_tree, test_node_features, test_node_labels)
        print('val_r2_loss = {:.4f}'.format(val_r2_loss.item()))

        val_val_list.append(val_r2_loss.item())
    #
    # x_axis = range(1, epoch_num + 1, 1)
    # plt.plot(x_axis, val_val_list, color='blue', label='val_val')
    # plt.legend()
    # plt.xlabel('epoch')
    # plt.ylabel('R2')
    # plt.show()

    # test
    # node_range = [50,100, 250, 500, 1000, 2500, 5000, 8000, 10000]
    node_range = [50]
    node_num_list = []
    eval_position_list = []
    for node_num in node_range:
        print("node_range:", node_num)
        for i in range(100):
            print("position test tree:", i)
            test_tree = test_data_process(1, node_num+1*i)
            node_features = test_tree.ndata["feat"]
            node_labels = test_tree.ndata["labels"]
            eval_position = evaluate_position(model, test_tree, node_features, node_labels)
            node_num_list.append(node_num+1*i)
            eval_position_list.append(eval_position)
    dataframe = pd.DataFrame({'node_num_list': node_num_list, 'eval_position_list': eval_position_list})
    dataframe.to_csv("eval_position_ER_new.csv", index=False, sep=',')


def gnn_top_k_overlap(train_patch_tree: dgl, test_patch_tree):
    # train
    train_features_dim = train_patch_tree.ndata["feat"].shape[1]
    train_node_features = train_patch_tree.ndata["feat"]
    train_node_labels = train_patch_tree.ndata["labels"]

    # test
    test_node_features = test_patch_tree.ndata["feat"]
    test_node_labels = test_patch_tree.ndata["labels"]

    model = SAGE(in_feats=train_features_dim, hid_feats=100, out_feats=1)
    opt = th.optim.Adam(model.parameters())

    # 开始训练
    val_val_list = []
    epoch_num = 50
    for epoch in range(epoch_num):
        print('Epoch {}'.format(epoch))
        model.train()
        logits = model(train_patch_tree, train_node_features)
        loss = F.mse_loss(logits, train_node_labels)

        # 分类计算验证集accuracy
        # loss = F.cross_entropy(logits, node_labels)
        # acc = evaluate(model, patch_tree, node_features, node_labels, val_mask)
        # evaluate(model, patch_tree, node_features, node_labels, val_mask)

        # backward propagation
        opt.zero_grad()
        loss.backward()
        opt.step()
        print('loss = {:.4f}'.format(loss.item()))

        val_r2_loss = evaluate(model, test_patch_tree, test_node_features, test_node_labels)
        print('val_r2_loss = {:.4f}'.format(val_r2_loss.item()))

        val_val_list.append(val_r2_loss.item())

    # test
    # node_range = [50,100, 250, 500, 1000, 2500, 5000, 8000, 10000]
    node_range = [50,100, 250, 500, 1000, 2500]
    for node_num in node_range:
        print("node_range:", node_num)
        node_num_list = []
        all_real_labels_list = []
        all_eval_label_list = []
        for i in range(100):
            print("position test tree:", i)
            test_tree = test_data_process(1, node_num+1*i)
            node_features = test_tree.ndata["feat"]
            node_labels = test_tree.ndata["labels"]
            real_labels_list, eval_label_list = evaluate_prob(model, test_tree, node_features, node_labels)
            all_real_labels_list.append(real_labels_list)
            all_eval_label_list.append(eval_label_list)
            node_num_list.append(node_num+1*i)
            # draw_tree = to_networkx(test_tree)
            # nx.draw(draw_tree, node_size=100, with_labels=True)
            # plt.show()

        dataframe = pd.DataFrame({'node_num_list': node_num_list, 'all_real_labels_list': all_real_labels_list, 'all_eval_label_list': all_eval_label_list})
        dataframe.to_csv("label_list\\label_list_SM_"+str(node_num)+".csv", index=False, sep=',')


if __name__ == '__main__':
    all_train_tree_list = []
    for i in range(1, 100, 1):
        print("construct tree:", i)
        train_patch_tree = train_data_process(2, 200+i)
        all_train_tree_list.append(train_patch_tree)
    all_train_tree = dgl.batch(all_train_tree_list)

    # train_patch_tree = train_data_process(500, 50)
    test_patch_tree = test_data_process(5, 100)

    # gnn_test_mse(all_train_tree, test_patch_tree)
    gnn_test_pos(all_train_tree, test_patch_tree)
    # gnn_top_k_overlap(all_train_tree, test_patch_tree)