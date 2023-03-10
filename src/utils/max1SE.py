import torch
import dgl


def add_knn(k, node_embed, edge_index, device):
    if device == torch.device("cpu"):
        knn_g = dgl.knn_graph(node_embed,
                              k,
                              algorithm='bruteforce',
                              dist='cosine')
    else:
        knn_g = dgl.knn_graph(node_embed,
                              k,
                              algorithm='bruteforce-sharemem',
                              dist='cosine')
    knn_g = dgl.add_reverse_edges(knn_g)
    knn_edge_index = knn_g.edges()
    knn_edge_index = torch.cat(
        (knn_edge_index[0].reshape(1, -1), knn_edge_index[1].reshape(1, -1)),
        dim=0)
    knn_edge_index = knn_edge_index.t()
    edge_index_2 = torch.concat((edge_index, knn_edge_index), dim=0)
    edge_index_2 = torch.unique(edge_index_2, dim=0)
    return edge_index_2


def calc_e1(adj: torch.Tensor):
    adj = adj - torch.diag_embed(torch.diag(adj))
    degree = adj.sum(dim=1)
    vol = adj.sum()
    idx = degree.nonzero().reshape(-1)
    g = degree[idx]
    return -((g / vol) * torch.log2(g / vol)).sum()


def get_adj_matrix(node_num, edge_index, weight) -> torch.Tensor:
    adj_matrix = torch.zeros((node_num, node_num))
    adj_matrix[edge_index.t()[0], edge_index.t()[1]] = weight
    adj_matrix = adj_matrix - torch.diag_embed(torch.diag(adj_matrix))  #去除对角线
    return adj_matrix


def get_weight(node_embedding, edge_index):
    node_num = node_embedding.shape[0]
    links = node_embedding[edge_index]
    weight = []
    for i in range(links.shape[0]):
        # print(links[i])
        weight.append(torch.corrcoef(links[i])[0, 1])
    weight = torch.tensor(weight) + 1
    weight[torch.isnan(weight)] = 0
    M = weight.mean() / (2 * node_num)
    weight = weight + M
    return weight


def knn_maxE1(edge_index: torch.Tensor, node_embedding: torch.Tensor, device):
    old_e1 = 0
    node_num = node_embedding.shape[0]
    k = 1
    while k < 50:
        edge_index_k = add_knn(k, node_embedding, edge_index, device)
        weight = get_weight(node_embedding, edge_index_k)
        # e1 = calc_e1(edge_index_k, weight)
        adj = get_adj_matrix(node_num, edge_index_k, weight)
        e1 = calc_e1(adj)
        if e1 - old_e1 > 0.1:
            k += 5
        elif e1 - old_e1 > 0.01:
            k += 3
        elif e1 - old_e1 > 0.001:
            k += 1
        else:
            break
        old_e1 = e1
    print(f'max1SE k: {k}')
    return k
