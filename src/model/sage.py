from dgl.nn.pytorch import SAGEConv

import torch.nn as nn
import torch.nn.functional as F


class SAGE(nn.Module):
    def __init__(self, graph, in_size, n_layer, hid_size, out_size, dropout):
        super().__init__()
        self.graph = graph
        self.layers = nn.ModuleList()
        # two-layer GraphSAGE-mean
        for i in range(n_layer):
            self.layers.append(SAGEConv(in_size, hid_size, "mean"))
        self.layers.append(SAGEConv(hid_size, out_size, "mean"))
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x):
        h = self.dropout(x)
        for l, layer in enumerate(self.layers):
            h = layer(self.graph, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h