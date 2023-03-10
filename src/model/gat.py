from dgl.nn.pytorch import GATConv

import torch.nn as nn
import torch.nn.functional as F


class GAT(nn.Module):
    def __init__(
            self,
            g,  #DGL的图对象
            n_layers,  #层数
            in_feats,  #输入特征维度
            n_hidden,  #隐层特征维度
            n_classes,  #类别数
            heads,  #多头注意力的数量
            activation,  #激活函数
            in_drop,  #输入特征的Dropout比例
            at_drop,  #注意力特征的Dropout比例
            negative_slope,  #注意力计算中Leaky ReLU的a值
    ):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = n_layers
        self.activation = activation

        self.gat_layers = nn.ModuleList()

        self.gat_layers.append(
            GATConv(in_feats,
                    n_hidden,
                    heads[0],
                    in_drop,
                    at_drop,
                    negative_slope,
                    activation=self.activation))

        for l in range(1, n_layers):
            self.gat_layers.append(
                GATConv(n_hidden * heads[l - 1],
                        n_hidden,
                        heads[l],
                        in_drop,
                        at_drop,
                        negative_slope,
                        activation=self.activation))

        self.gat_layers.append(
            GATConv(n_hidden * heads[-2],
                    n_classes,
                    heads[-1],
                    in_drop,
                    at_drop,
                    negative_slope=0.2,
                    activation=None))

    def reset_parameters(self):
        for layer in self.gat_layers:
            layer.reset_parameters()

    def set_graph(self, graph):
        self.g = graph

    def forward(self, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)
        gat_node_embed = h
        logit = self.gat_layers[-1](self.g, h).mean(1)
        return logit, gat_node_embed