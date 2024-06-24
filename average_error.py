import csv
import numpy as np

def read_csv(filename):
    data_list = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)  # 跳过表头
        for row in reader:
            t = [int(i) for i in row]
            data_list.append(sum(t)/len(t))

    print(filename, sum(data_list)/len(data_list))

    return data_list

sensor_BFS_sqrt = read_csv('./data_moreG_small/2error_sensor_BFS_sqrt.csv')
sensor_BFS_rand_BFS = read_csv('./data_moreG_small/2error_sensor_BFS_rand_BFS.csv')
sensor_BFS_stc = read_csv('./data_moreG_small/2error_sensor_BFS_stc.csv')
sensor_DFS_sqrt = read_csv('./data_moreG_small/2error_sensor_DFS_sqrt.csv')
sensor_DFS_rand_BFS = read_csv('./data_moreG_small/2error_sensor_DFS_rand_BFS.csv')
sensor_DFS_stc = read_csv('./data_moreG_small/2error_sensor_DFS_stc.csv')
print("----")


complete_nary_BFS_sqrt = read_csv('./data_moreG_small/2error_complete_nary_BFS_sqrt.csv')
complete_nary_BFS_rand_BFS = read_csv('./data_moreG_small/2error_complete_nary_BFS_rand_BFS.csv')
complete_nary_BFS_stc = read_csv('./data_moreG_small/2error_complete_nary_BFS_stc.csv')
complete_nary_DFS_sqrt = read_csv('./data_moreG_small/2error_complete_nary_DFS_sqrt.csv')
complete_nary_DFS_rand_BFS = read_csv('./data_moreG_small/2error_complete_nary_DFS_rand_BFS.csv')
complete_nary_DFS_stc = read_csv('./data_moreG_small/2error_complete_nary_DFS_stc.csv')
print("----")
regular_tree_BFS_sqrt = read_csv('./data_moreG_small/2error_regular_tree_BFS_sqrt.csv')
regular_tree_BFS_rand_BFS = read_csv('./data_moreG_small/2error_regular_tree_BFS_rand_BFS.csv')
regular_tree_BFS_stc = read_csv('./data_moreG_small/2error_regular_tree_BFS_stc.csv')
regular_tree_DFS_sqrt = read_csv('./data_moreG_small/2error_regular_tree_DFS_sqrt.csv')
regular_tree_DFS_rand_BFS = read_csv('./data_moreG_small/2error_regular_tree_DFS_rand_BFS.csv')
regular_tree_DFS_stc = read_csv('./data_moreG_small/2error_regular_tree_DFS_stc.csv')
print("----")
ER_random_BFS_sqrt = read_csv('./data_moreG_small/2error_ER_random_BFS_sqrt.csv')
ER_random_BFS_rand_BFS = read_csv('./data_moreG_small/2error_ER_random_BFS_rand_BFS.csv')
ER_random_BFS_stc = read_csv('./data_moreG_small/2error_ER_random_BFS_stc.csv')
ER_random_DFS_sqrt = read_csv('./data_moreG_small/2error_ER_random_DFS_sqrt.csv')
ER_random_DFS_rand_BFS = read_csv('./data_moreG_small/2error_ER_random_DFS_rand_BFS.csv')
ER_random_DFS_stc = read_csv('./data_moreG_small/2error_ER_random_DFS_stc.csv')
print("----")
real_world_BFS_sqrt = read_csv('./data_moreG_small/2error_real_world_BFS_sqrt.csv')
real_world_BFS_rand_BFS = read_csv('./data_moreG_small/2error_real_world_BFS_rand_BFS.csv')
real_world_BFS_stc = read_csv('./data_moreG_small/2error_real_world_BFS_stc.csv')
real_world_DFS_sqrt = read_csv('./data_moreG_small/2error_real_world_DFS_sqrt.csv')
real_world_DFS_rand_BFS = read_csv('./data_moreG_small/2error_real_world_DFS_rand_BFS.csv')
real_world_DFS_stc = read_csv('./data_moreG_small/2error_real_world_DFS_stc.csv')


print("----")
SBM_BFS_sqrt = read_csv('./data_moreG_small/2error_SBM_BFS_sqrt.csv')
SBM_BFS_rand_BFS = read_csv('./data_moreG_small/2error_SBM_BFS_rand_BFS.csv')
SBM_BFS_stc = read_csv('./data_moreG_small/2error_SBM_BFS_stc.csv')
SBM_DFS_sqrt = read_csv('./data_moreG_small/2error_SBM_DFS_sqrt.csv')
SBM_DFS_rand_BFS = read_csv('./data_moreG_small/2error_SBM_DFS_rand_BFS.csv')
SBM_DFS_stc = read_csv('./data_moreG_small/2error_SBM_DFS_stc.csv')


print("----")

