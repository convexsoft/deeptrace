## Description
This code implements a Graph Neural Network (GNN) framework, named DeepTrace, to estimate the probabilities of being superspreaders for nodes in the epidemic networks by computing the maximum-likelihood estimator by leveraging the likelihood structure to configure the training set with topological features of smaller epidemic network datasets.


## Software dependencies
- Python 3 
- Pytorch
- Deep Graph Library
- Networkx
- Other commonly used packages, such as Pandas, Numpy, etc.

The above packages can be quickly and easily installed on your laptop using pip.

## Demo

- Data processing
 
First, we need to obtain the node features of the epidemic networks used for training and prediction: the degree ratio, infected proportion and the boundary distance ratio. The relevant implementation code is in the "graph_data_process.py" file, and the class of graph data processed here is networkx.graph.

Secondly, we need to get the labels of each node of the epidemic networks used for training, that is, the probability of each node being a superspreader. The implementation code is in the "label_list_process.py" file.
 
- Training model

After we set up the environment and processed the training data, using "model.py"  we can train the GNN model in DeepTrace. A small example also in "model.py" demonstrates the predicted results of identifying superspreaders for a few randomly generated epidemic networks. The outputs of the example are the predicted positions (the second column of the outputs) of the most likely superspreader in the epidemic networks, e.g., if the predicted positions are 0, it means that the GNN model accurately identified the most likely superspreader.

## The epidemic data
The Hong Kong COVID-19 cluster data in February 2020 comes from a Nature Medicine paper  "Clustering and superspreading potential of SARS-CoV-2 infections in Hong Kong" (Adam, Dillon C, and Wu, Peng, etc., 2020).
The Hong Kong COVID-19 pandemic raw data (from January 31, 2022, to February 3, 2022) is available from the Hong Kong government's public sector open data portal (https://data.gov.hk/en-data/dataset/hk-dh-chpsebcddr-novel-infectious-agent).
The Taiwan COVID-19 pandemic raw data (from March 19, 2022, to April 1, 2022) is available from the press releases by the Taiwan Centers of Disease Control (Taiwan CDC: https://www.cdc.gov.tw/En ).
The COVID-19 pandemic processed data is in the "data" folder, and the raw data processing implementation is in the "hongkongdata.py". 
