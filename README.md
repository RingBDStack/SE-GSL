# SE-GSL
code for "SE-GSL: A General and Effective Graph Structure Learning Framework through Structural Entropy Optimization"

# Overview
- model: implement of different GNN model
- utils/max1SE.py: One-dimensional structural entropy k-selector.
- utils/coding_tree.py: Encoding tree defination and optimization.
- utils/reshape.py: node-pair sampling.
- utils/utils_data.py, utils/utils.py: data preprocessing code from [Geom-GCN](https://github.com/graphdml-uiuc-jlu/geom-gcn)

# Requirements
The implementation of PASTEL is tested under Python 3.9.12, with the following packages installed:
* `dgl-cu116==0.9.0`
* `pytorch==1.12.0`
* `numpy==1.21.5`
* `networkx==2.8.5`
* `scipy==1.8.1`
