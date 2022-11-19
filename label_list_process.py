import csv
import json
import copy
from math import *
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from itertools import islice


def find_max_index(ari_list, max_num):
    t = copy.deepcopy(ari_list)
    max_number = []
    max_index = []
    for _ in range(max_num):
        number = max(t)
        index = t.index(number)
        t[index] = 0
        max_number.append(number)
        max_index.append(index)
    return max_index


def find_min_index(ari_list, min_num):
    t = copy.deepcopy(ari_list)
    min_number = []
    min_index = []
    for _ in range(min_num):
        number = min(t)
        index = t.index(number)
        t[index] = float("inf")
        min_number.append(number)
        min_index.append(index)
    return min_index


def read_label_list_csv(csv_str, rate, find_len):
    inter_rate = []
    with open(csv_str)as f:
        f_csv = csv.reader(f)
        i = 0
        for row in f_csv:
            i += 1
            # print("i:", i)
            if i > 1:
                real_list = json.loads(row[1])
                eval_list = json.loads(row[2])
                len_list = len(real_list)

                max_real_index_list = find_max_index(real_list,find_len)
                max_eval_index_list = find_max_index(eval_list,find_len )

                inter_set = list(set(max_real_index_list).intersection(set(max_eval_index_list)))
                inter_rate.append(len(inter_set)/find_len)

        print("mean:", sum(inter_rate)/len(inter_rate))
    return inter_rate


def graw_all():
    x = np.arange(1, 30, 2)
    y = []
    for i in x:
        real_list = read_label_list_csv("label_list\\label_list_BA_1000.csv", i, i)
        y.append(real_list)
    plt.plot(x, y, label="sigmoid")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()



def statistics():
    node_num = [50, 100, 250, 500, 1000,2500]
    all_real_list = []
    all_node_num = []
    for num in node_num:
        real_list = read_label_list_csv("label_list\\label_list_SM_" + str(num) + ".csv", 0.1, 15)
        node_num_list = [str(num)] * len(real_list)
        all_real_list = all_real_list + real_list
        # all_real_list = [1-x for x in all_real_list]
        all_node_num = all_node_num + node_num_list
    print(all_node_num)
    dataframe = pd.DataFrame({"node num": all_node_num, "overlap rate": all_real_list})
    dataframe.to_csv("raincloud\\WS.csv", index=False, sep=',')


def all_net_stats():
    net_list = ["BA",  "WS", "ER",  "RR"]
    net_name = []
    all_satis_label = []
    for net in net_list:
        csv_str = "raincloud\\"+net+".csv"
        satis_label = []
        i = 0
        with open(csv_str)as f:
            f_csv = csv.reader(f)
            if net == "BA":
                for row in islice(f_csv, 1, None):
                    if int(row[0])==2500 and float(row[1])>0.65:
                        satis_label.append(row[1])
                net_name = net_name + ["BA"]*len(satis_label)
                all_satis_label = all_satis_label + satis_label
            elif net == "RR":
                for row in islice(f_csv, 1, None):
                    if int(row[0])==2500 and float(row[1])>0.65:
                        satis_label.append(float(row[1]))
                net_name = net_name + ["RR"] * len(satis_label)
                all_satis_label = all_satis_label + satis_label
            else:
                for row in islice(f_csv, 1, None):
                    if int(row[0])==2500 and float(row[1])>0.70:
                        satis_label.append(row[1])
                net_name = net_name + [net] * len(satis_label)
                all_satis_label = all_satis_label + satis_label
    net_name = net_name + net_name
    all_satis_label = all_satis_label + all_satis_label
    print("net_name:", net_name)
    print("satis_label", all_satis_label)
    dataframe = pd.DataFrame({"Network": net_name, "Overlap Ratio": all_satis_label})
    dataframe.to_csv("raincloud\\all_2500.csv", index=False, sep=',')


if __name__ == '__main__':
    all_net_stats()





