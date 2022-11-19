from hop_error import CalDS
import networkx as nx
import matplotlib.pyplot as plt
from math import floor


node_num = 10


WS = nx.random_graphs.watts_strogatz_graph(node_num, floor(node_num/5), 0.3)
tree_ka = nx.minimum_spanning_tree(WS, algorithm="kruskal")
pos = nx.spring_layout(WS)
nx.draw(WS, pos, with_labels = True, node_size = 100)
plt.show()


cal = CalDS(tree_ka, [], 0)
permutation_dict = cal.cal_ds()
permutation_list = []
print(permutation_dict)
for k,v in permutation_dict.items():
    permutation_list.append(v['permutation_prob'])
print(permutation_list)


plt.plot(permutation_list, 'x')
plt.show()


