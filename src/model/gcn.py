from dgl.nn.pytorch import GraphConv

import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, graph, in_size, n_layer, hid_size, out_size, dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        self.graph = graph
        # two-layer GCN
        for i in range(n_layer):
            self.layers.append(GraphConv(in_size, hid_size, activation=F.relu))
        self.layers.append(GraphConv(hid_size, out_size))
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.graph, h)
        return h