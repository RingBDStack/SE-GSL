import argparse
import pickle

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

from model.gat import GAT
from utils.max1SE import get_weight, knn_maxE1, get_adj_matrix, add_knn
from utils.reshape import reshape, get_community
from utils.code_tree import PartitionTree
from utils.utils_data import load_data


def calc_acc(logits, labels, mask):
    logits = logits[mask]
    labels = labels[mask]
    indices = logits.argmax(dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


def evaluate(features, labels, train_mask, val_mask, test_mask, k=2):
    model.eval()
    with torch.no_grad():
        logits, _ = model(features)
        train_acc = calc_acc(logits, labels, train_mask)
        val_acc = calc_acc(logits, labels, val_mask)
        test_acc = calc_acc(logits, labels, test_mask)
        return train_acc, val_acc, test_acc


def train(n_epochs=800, lr=5e-3, weight_decay=5e-4):
    # model = model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=weight_decay)
    logits_list = []
    val_acc_list = []
    test_acc_list = []
    for epoch in range(n_epochs):
        model.train()
        logits, gat_node_embed = model(features)
        logits_list.append(logits)
        loss = loss_fn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_acc, val_acc, test_acc = evaluate(features, labels, train_mask,
                                                val_mask, test_mask)
        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)
    val_acc_list = torch.tensor(val_acc_list)
    val_max_idx = val_acc_list.argmax()
    return test_acc_list[val_max_idx], val_acc_list.max(), max(test_acc_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_out_heads', type=int, default=1)
    parser.add_argument('--drop_rate', type=float)
    parser.add_argument('--n_hidden', type=int)
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--iteration', type=int, default=10)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--num_epoch', type=int, default=200)
    parser.add_argument('--se', type=int, default=2)
    parser.add_argument('--k', type=float, default=3)
    parser.add_argument('--train_percentage', type=float, default=0.6)
    parser.add_argument('--val_percentage', type=float, default=0.2)
    parser.add_argument('--random_split', action='store_true')
    parser.add_argument('--split', type=int)

    args = parser.parse_args()

    if args.dataset == 'citeseer' or args.dataset == 'cora':
        args.n_hidden = 8
        args.learning_rate = 1e-2
        args.weight_decay = 5e-5
        args.drop_rate = 0.5
    elif args.dataset == 'pubmed':
        args.n_hidden = 8
        args.n_out_heads = 8
        args.learning_rate = 1e-2
        args.weight_decay = 5e-5
        args.drop_rate = 0.5
    elif args.dataset == 'actor' or args.dataset == 'film':
        args.n_hidden = 32
        args.learning_rate = 1e-2
        args.weight_decay = 5e-5
        args.drop_rate = 0.5
    elif args.dataset == 'cornell' or args.dataset == 'texas' or args.dataset == 'wisconsin':
        args.n_hidden = 32
        args.learning_rate = 1e-2
        args.weight_decay = 5e-6
        args.drop_rate = 0.5
    else:
        # pt tw
        args.n_hidden = 32
        args.learning_rate = 1e-2
        args.weight_decay = 5e-6
        args.drop_rate = 0.5

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_heads = args.n_heads
    n_layers = args.n_layers
    n_out_heads = args.n_out_heads
    activation = F.elu
    feat_drop = args.drop_rate
    attn_drop = args.drop_rate
    negative_slope = 0.2
    n_hidden = args.n_hidden
    heads = ([n_heads] * n_layers) + [n_out_heads]

    if args.random_split:
        splits = [0]
    elif args.split is None:
        splits = range(10)
    else:
        splits = [args.split]
    for sp in splits:
        args.split = sp
        runs = args.runs
        iteration = args.iteration
        test_result = [[] for i in range(iteration)]
        highest_val_result = [[] for i in range(iteration)]
        highest_test_result = [[] for i in range(iteration)]
        for run in range(runs):
            split_path = None
            if not args.random_split:
                split_path = f'splits/{args.dataset}_split_0.6_0.2_{args.split}.npz'
            print(split_path)
            graph, in_feats, n_classes = load_data(
                args.dataset,
                splits_file_path=split_path,
                train_percentage=args.train_percentage,
                val_percentage=args.val_percentage)

            node_num = graph.num_nodes()
            graph = dgl.remove_self_loop(graph)
            graph = dgl.add_self_loop(graph)
            graph = graph.to(device)
            features = graph.ndata['feat']

            labels = graph.ndata['label']
            train_mask = graph.ndata['train_mask'].bool()
            val_mask = graph.ndata['val_mask'].bool()
            test_mask = graph.ndata['test_mask'].bool()
            edge_index = graph.edges()
            edge_index = torch.cat(
                (edge_index[0].reshape(1, -1), edge_index[1].reshape(1, -1)),
                dim=0)
            edge_index = edge_index.t()

            for i in range(iteration):
                model = GAT(graph, args.n_layers, in_feats, n_hidden,
                            n_classes, heads, activation, feat_drop, attn_drop,
                            negative_slope)
                model = model.to(device)
                model.reset_parameters()
                test_acc, highest_val_acc, highest_test_acc = train(
                    n_epochs=args.num_epoch,
                    lr=args.learning_rate,
                    weight_decay=args.weight_decay)
                print(
                    f'Split: {sp:02d} | Run: {run:02d} | Iteration: {i:02d} | test acc: {test_acc:.4f} | Highest val: {highest_val_acc:.4f} | Highest test: {highest_test_acc:.4f}'
                )
                test_result[i].append(test_acc)
                highest_val_result[i].append(highest_val_acc)
                highest_test_result[i].append(highest_test_acc)
                logits, _ = model(features)

                if i == iteration - 1:
                    break

                k = knn_maxE1(edge_index, logits, device)
                edge_index_2 = add_knn(k, logits, edge_index, device)
                weight = get_weight(logits, edge_index_2)
                adj_matrix = get_adj_matrix(node_num, edge_index_2, weight)

                code_tree = PartitionTree(adj_matrix=numpy.array(adj_matrix))
                code_tree.build_coding_tree(args.se)

                community, isleaf = get_community(code_tree)
                if args.dataset in {'cora', 'citeseer', 'pubmed'}:
                    new_edge_index = reshape(community, code_tree, isleaf,
                                             args.k)
                    new_edge_index_2 = reshape(community, code_tree, isleaf,
                                               args.k)
                    new_edge_index = torch.concat(
                        (new_edge_index.t(), new_edge_index_2.t()), dim=0)
                    new_edge_index, unique_idx = torch.unique(
                        new_edge_index, return_counts=True, dim=0)
                    new_edge_index = new_edge_index[unique_idx != 1].t()
                    add_num = int(new_edge_index.shape[1])

                    new_edge_index = torch.concat(
                        (new_edge_index.t(), edge_index.cpu()), dim=0)
                    new_edge_index = torch.unique(new_edge_index, dim=0)
                    new_edge_index = new_edge_index.t()
                    new_weight = get_weight(logits, new_edge_index.t())
                    _, delete_idx = torch.topk(new_weight,
                                               k=add_num,
                                               largest=False)
                    delete_mask = torch.ones(
                        new_edge_index.t().shape[0]).bool()
                    delete_mask[delete_idx] = False
                    new_edge_index = new_edge_index.t()[delete_mask].t()
                else:
                    new_edge_index = reshape(community, code_tree, isleaf,
                                             args.k)

                graph = dgl.graph((new_edge_index[0], new_edge_index[1]),
                                  num_nodes=node_num)
                graph = dgl.remove_self_loop(graph)
                graph = dgl.add_self_loop(graph)
                graph = graph.to(device)
                edge_index = graph.edges()
                edge_index = torch.cat(
                    (edge_index[0].reshape(1, -1), edge_index[1].reshape(
                        1, -1)),
                    dim=0)
                edge_index = edge_index.t()
                print(edge_index.shape)

        print(f'Split: {sp:02d} | final acc: ')
        test_result = torch.tensor(test_result)
        highest_val_result = torch.tensor(highest_val_result)
        highest_test_result = torch.tensor(highest_test_result)
        print(
            f'test acc: {test_result[0].mean():.4f} ± {test_result[0].std():.4f} | highest val: {highest_val_result[0].mean():.4f} ± {highest_val_result[0].std():.4f} | highest test: {highest_test_result[0].mean():.4f} ± {highest_test_result[0].std():.4f}'
        )
        test_result, _ = test_result[1:].max(dim=0)
        highest_val_result, _ = highest_val_result[1:].max(dim=0)
        highest_test_result, _ = highest_test_result[1:].max(dim=0)

        print('Our model:')
        print(
            f'test acc: {test_result.mean():.4f} ± {test_result.std():.4f} | highest val: {highest_val_result.mean():.4f} ± {highest_val_result.std():.4f} | highest test: {highest_test_result.mean():.4f} ± {highest_test_result.std():.4f}'
        )
