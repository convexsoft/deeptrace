import csv


def read_csv(filename):
    data_list = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)  # 跳过表头
        for row in reader:
            data_list.append(row)
    return data_list


def pair_stats(all_superspreader_list):
    all_pair_type_dict = {"s1":0,"s2":0,"s3":0}

    for node_idx, superspreader_list in enumerate(all_superspreader_list):
        checked_list = [superspreader_list[0]]
        pair_type_dict={"s1":0,"s2":0,"s3":0}
        for i in range(len(superspreader_list)-1):
            if superspreader_list[i+1] == superspreader_list[i]:
                pair_type_dict["s1"]+=1
            elif superspreader_list[i+1] != superspreader_list[i] and superspreader_list[i+1] not in checked_list:
                pair_type_dict["s2"]+=1
            elif superspreader_list[i+1] != superspreader_list[i] and superspreader_list[i+1] in checked_list:
                pair_type_dict["s3"] += 1
            checked_list.append(superspreader_list[i+1])
        all_pair_type_dict["s1"] += pair_type_dict["s1"]
        all_pair_type_dict["s2"] += pair_type_dict["s2"]
        all_pair_type_dict["s3"] += pair_type_dict["s3"]
    print("====")
    all = all_pair_type_dict["s1"]+all_pair_type_dict["s2"]+all_pair_type_dict["s3"]
    all_pair_type_dict_pro = {"s1":all_pair_type_dict["s1"]/all,"s2":all_pair_type_dict["s2"]/all,"s3":all_pair_type_dict["s3"]/all}
    return all_pair_type_dict_pro

def pair_stats_v2(all_superspreader_list):
    all_pair_type_dict = {"s1":0,"s2":0,"s3":0}

    for node_idx, superspreader_list in enumerate(all_superspreader_list):
        checked_list = [superspreader_list[0]]
        pair_type_dict={"s1":0,"s2":0,"s3":0}
        for i in range(len(superspreader_list)-1):
            if superspreader_list[i+1] == superspreader_list[i]:
                pair_type_dict["s1"]+=1
            elif superspreader_list[i+1] != superspreader_list[i] and superspreader_list[i+1] not in checked_list:
                pair_type_dict["s2"]+=1
            elif superspreader_list[i+1] != superspreader_list[i] and superspreader_list[i+1] in checked_list:
                pair_type_dict["s3"] += 1
            checked_list.append(superspreader_list[i+1])
        # print("pair_type_dict:",pair_type_dict)
        all_pair_type_dict["s1"] += pair_type_dict["s1"]
        all_pair_type_dict["s2"] += pair_type_dict["s2"]
        all_pair_type_dict["s3"] += pair_type_dict["s3"]
    print("====")
    all = all_pair_type_dict["s1"]+all_pair_type_dict["s2"]+all_pair_type_dict["s3"]
    all_pair_type_dict_pro = {"s1":all_pair_type_dict["s1"]/all,"s2":all_pair_type_dict["s2"]/all,"s3":all_pair_type_dict["s3"]/all}
    return all_pair_type_dict_pro


regular_tree_BFS_sqrt = read_csv('./data_moreG_small/2debug_regular_tree_BFS_sqrt.csv')
regular_tree_BFS_rand_BFS = read_csv('./data_moreG_small/2debug_regular_tree_BFS_rand_BFS.csv')
regular_tree_BFS_stc = read_csv('./data_moreG_small/2debug_regular_tree_BFS_stc.csv')
regular_tree_DFS_sqrt = read_csv('./data_moreG_small/2debug_regular_tree_DFS_sqrt.csv')
regular_tree_DFS_rand_BFS = read_csv('./data_moreG_small/2debug_regular_tree_DFS_rand_BFS.csv')
regular_tree_DFS_stc = read_csv('./data_moreG_small/2debug_regular_tree_DFS_stc.csv')

print("regular_tree_BFS_sqrt:",pair_stats_v2(regular_tree_BFS_sqrt))
print("regular_tree_BFS_rand_BFS:",pair_stats_v2(regular_tree_BFS_rand_BFS))
print("regular_tree_BFS_stc:",pair_stats_v2(regular_tree_BFS_stc))
print("regular_tree_DFS_sqrt:",pair_stats_v2(regular_tree_DFS_sqrt))
print("regular_tree_DFS_rand_BFS:",pair_stats_v2(regular_tree_DFS_rand_BFS))
print("regular_tree_DFS_stc:",pair_stats_v2(regular_tree_DFS_stc))

print("-------------------------")


sensor_BFS_sqrt = read_csv('./data_moreG_small/2debug_sensor_BFS_sqrt.csv')
sensor_BFS_rand_BFS = read_csv('./data_moreG_small/2debug_sensor_BFS_rand_BFS.csv')
sensor_BFS_stc = read_csv('./data_moreG_small/2debug_sensor_BFS_stc.csv')
sensor_DFS_sqrt = read_csv('./data_moreG_small/2debug_sensor_DFS_sqrt.csv')
sensor_DFS_rand_BFS = read_csv('./data_moreG_small/2debug_sensor_DFS_rand_BFS.csv')
sensor_DFS_stc = read_csv('./data_moreG_small/2debug_sensor_DFS_stc.csv')


print("sensor_BFS_sqrt:",pair_stats_v2(sensor_BFS_sqrt))
print("sensor_BFS_rand_BFS:",pair_stats_v2(sensor_BFS_rand_BFS))
print("sensor_BFS_stc:",pair_stats_v2(sensor_BFS_stc))
print("sensor_DFS_sqrt:",pair_stats_v2(sensor_DFS_sqrt))
print("sensor_DFS_rand_BFS:",pair_stats_v2(sensor_DFS_rand_BFS))
print("sensor_DFS_stc:",pair_stats_v2(sensor_DFS_stc))


print("-------------------------")


complete_nary_BFS_sqrt = read_csv('./data_moreG_small/2debug_complete_nary_BFS_sqrt.csv')
complete_nary_BFS_rand_BFS = read_csv('./data_moreG_small/2debug_complete_nary_BFS_rand_BFS.csv')
complete_nary_BFS_stc = read_csv('./data_moreG_small/2debug_complete_nary_BFS_stc.csv')
complete_nary_DFS_sqrt = read_csv('./data_moreG_small/2debug_complete_nary_DFS_sqrt.csv')
complete_nary_DFS_rand_BFS = read_csv('./data_moreG_small/2debug_complete_nary_DFS_rand_BFS.csv')
complete_nary_DFS_stc = read_csv('./data_moreG_small/2debug_complete_nary_DFS_stc.csv')


print("complete_nary_BFS_sqrt:",pair_stats_v2(complete_nary_BFS_sqrt))
print("complete_nary_BFS_rand_BFS:",pair_stats_v2(complete_nary_BFS_rand_BFS))
print("complete_nary_BFS_stc:",pair_stats_v2(complete_nary_BFS_stc))
print("complete_nary_DFS_sqrt:",pair_stats_v2(complete_nary_DFS_sqrt))
print("complete_nary_DFS_rand_BFS:",pair_stats_v2(complete_nary_DFS_rand_BFS))
print("complete_nary_DFS_stc:",pair_stats_v2(complete_nary_DFS_stc))


print("-------------------------")


ER_random_BFS_sqrt = read_csv('./data_moreG_small/2debug_ER_random_BFS_sqrt.csv')
ER_random_BFS_rand_BFS = read_csv('./data_moreG_small/2debug_ER_random_BFS_rand_BFS.csv')
ER_random_BFS_stc = read_csv('./data_moreG_small/2debug_ER_random_BFS_stc.csv')
ER_random_DFS_sqrt = read_csv('./data_moreG_small/2debug_ER_random_DFS_sqrt.csv')
ER_random_DFS_rand_BFS = read_csv('./data_moreG_small/2debug_ER_random_DFS_rand_BFS.csv')
ER_random_DFS_stc = read_csv('./data_moreG_small/2debug_ER_random_DFS_stc.csv')

print("ER_random_BFS_sqrt:",pair_stats_v2(ER_random_BFS_sqrt))
print("ER_random_BFS_rand_BFS:",pair_stats_v2(ER_random_BFS_rand_BFS))
print("ER_random_BFS_stc:",pair_stats_v2(ER_random_BFS_stc))
print("ER_random_DFS_sqrt:",pair_stats_v2(ER_random_DFS_sqrt))
print("ER_random_DFS_rand_BFS:",pair_stats_v2(ER_random_DFS_rand_BFS))
print("ER_random_DFS_stc:",pair_stats_v2(ER_random_DFS_stc))

print("-------------------------")


SBM_BFS_sqrt = read_csv('./data_moreG_small/2debug_SBM_BFS_sqrt.csv')
SBM_BFS_rand_BFS = read_csv('./data_moreG_small/2debug_SBM_BFS_rand_BFS.csv')
SBM_BFS_stc = read_csv('./data_moreG_small/2debug_SBM_BFS_stc.csv')
SBM_DFS_sqrt = read_csv('./data_moreG_small/2debug_SBM_DFS_sqrt.csv')
SBM_DFS_rand_BFS = read_csv('./data_moreG_small/2debug_SBM_DFS_rand_BFS.csv')
SBM_DFS_stc = read_csv('./data_moreG_small/2debug_SBM_DFS_stc.csv')

print("SBM_BFS_sqrt:",pair_stats_v2(SBM_BFS_sqrt))
print("SBM_BFS_rand_BFS:",pair_stats_v2(SBM_BFS_rand_BFS))
print("SBM_BFS_stc:",pair_stats_v2(SBM_BFS_stc))
print("SBM_DFS_sqrt:",pair_stats_v2(SBM_DFS_sqrt))
print("SBM_DFS_rand_BFS:",pair_stats_v2(SBM_DFS_rand_BFS))
print("SBM_DFS_stc:",pair_stats_v2(SBM_DFS_stc))

print("-------------------------")

sensor_BFS_sqrt = read_csv('./data_moreG_small/2debug_sensor_BFS_sqrt.csv')
sensor_BFS_rand_BFS = read_csv('./data_moreG_small/2debug_sensor_BFS_rand_BFS.csv')
sensor_BFS_stc = read_csv('./data_moreG_small/2debug_sensor_BFS_stc.csv')
sensor_DFS_sqrt = read_csv('./data_moreG_small/2debug_sensor_DFS_sqrt.csv')
sensor_DFS_rand_BFS = read_csv('./data_moreG_small/2debug_sensor_DFS_rand_BFS.csv')
sensor_DFS_stc = read_csv('./data_moreG_small/2debug_sensor_DFS_stc.csv')

print("sensor_BFS_sqrt:",pair_stats_v2(sensor_BFS_sqrt))
print("sensor_BFS_rand_BFS:",pair_stats_v2(sensor_BFS_rand_BFS))
print("sensor_BFS_stc:",pair_stats_v2(sensor_BFS_stc))
print("sensor_DFS_sqrt:",pair_stats_v2(sensor_DFS_sqrt))
print("sensor_DFS_rand_BFS:",pair_stats_v2(sensor_DFS_rand_BFS))
print("sensor_DFS_stc:",pair_stats_v2(sensor_DFS_stc))

print("-------------------------")

real_world_BFS_sqrt = read_csv('./data_moreG/2debug_real_world_BFS_sqrt.csv')
real_world_BFS_rand_BFS = read_csv('./data_moreG/2debug_real_world_BFS_rand_BFS.csv')
real_world_BFS_stc = read_csv('./data_moreG/2debug_real_world_BFS_stc.csv')
real_world_DFS_sqrt = read_csv('./data_moreG/2debug_real_world_DFS_sqrt.csv')
real_world_DFS_rand_BFS = read_csv('./data_moreG/2debug_real_world_DFS_rand_BFS.csv')
real_world_DFS_stc = read_csv('./data_moreG/2debug_real_world_DFS_stc.csv')
print("real_world_BFS_sqrt:",pair_stats_v2(real_world_BFS_sqrt))
print("real_world_BFS_rand_BFS:",pair_stats_v2(real_world_BFS_rand_BFS))
print("real_world_BFS_stc:",pair_stats_v2(real_world_BFS_stc))
print("real_world_DFS_sqrt:",pair_stats_v2(real_world_DFS_sqrt))
print("real_world_DFS_rand_BFS:",pair_stats_v2(real_world_DFS_rand_BFS))
print("real_world_DFS_stc:",pair_stats_v2(real_world_DFS_stc))

print("-------------------------")

