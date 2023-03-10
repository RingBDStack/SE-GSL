import torch

from utils.code_tree import PartitionTree, PartitionTreeNode


def get_community(code_tree: PartitionTree):
    node_dict = code_tree.tree_node
    root_id = code_tree.root_id
    tree_node_num = max(node_dict.keys()) + 1
    isleaf = torch.zeros(tree_node_num, dtype=torch.bool)
    stack = [root_id]
    while stack:
        node_id = stack.pop()
        child = node_dict[node_id].children
        if child is None:
            isleaf[node_id] = True
            continue
        child = list(child)
        for e in child[::-1]:
            stack.append(e)
        # stack.append(child[0])
    community = []
    for current_id in range(tree_node_num):
        if isleaf[current_id]:
            while True:
                parent_id = node_dict[current_id].parent
                if node_dict[parent_id].partition == node_dict[
                        current_id].partition:
                    isleaf[parent_id] = True
                    current_id = parent_id
                break
    for e in node_dict.keys():
        if not isleaf[e]:
            community.append(e)
    return community, isleaf


def get_sedict(community: list, code_tree: PartitionTree):
    node_dict = code_tree.tree_node
    se_dict = {}
    for community_id in community:
        node_list = list(node_dict[community_id].children)
        se = torch.zeros(len(node_list))
        for i, e in enumerate(node_list):
            e = node_dict[e]
            e: PartitionTreeNode
            se[i] = -(e.g / code_tree.VOL) * torch.log2(
                torch.tensor(
                    (e.vol + 1) /
                    (node_dict[e.parent].vol + 1))) + code_tree.deduct_se(
                        community_id, None)
            # se[i] = -(e.g / code_tree.VOL) * torch.log2(
            #     torch.tensor((e.vol + 1) / (node_dict[e.parent].vol + 1)))
        se = torch.softmax(se.float(), dim=0)
        se_dict[community_id] = se
    return se_dict


def select_link(community_id: int, code_tree: PartitionTree,
                isleaf: torch.Tensor, se_dict):
    node_dict = code_tree.tree_node
    node_list = list(node_dict[community_id].children)
    node_dict = code_tree.tree_node
    se = se_dict[community_id]
    id1, id2 = torch.multinomial(se, num_samples=2, replacement=True)
    link_id1 = node_list[id1]
    link_id2 = node_list[id2]
    link_id = [link_id1, link_id2]
    return link_id


def select_leaf(node_id, code_tree: PartitionTree, isleaf: torch.Tensor,
                se_dict):
    # print(node_id)
    node_dict = code_tree.tree_node
    while not isleaf[node_id]:
        node_list = list(node_dict[node_id].children)
        if len(node_list) > 1:  #避免只有一个字节点的非叶子节点
            se = se_dict[node_id]
            # print(se)
            id = torch.multinomial(se, num_samples=1, replacement=False)
            node_id = node_list[id]
        node_id = node_list[0]
    return (node_dict[node_id].partition)[0]


def reshape(community: list, code_tree: PartitionTree, isleaf: torch.Tensor,
            k):
    se_dict = {}
    edge_index = []
    node_dict = code_tree.tree_node
    # for k, v in code_tree.tree_node.items():
    #     print(k, v.__dict__)
    se_dict = get_sedict(community, code_tree)

    for community_id in community:
        node_list = list(node_dict[community_id].children)
        if len(node_list) == 1:
            continue
        prefer_edge_num = round(k * len(node_list))
        for i in range(prefer_edge_num):
            id1, id2 = select_link(community_id, code_tree, isleaf, se_dict)
            edge_index.append([
                select_leaf(id1, code_tree, isleaf, se_dict),
                select_leaf(id2, code_tree, isleaf, se_dict)
            ])
    edge_index = torch.tensor(edge_index)
    edge_index = torch.concat((edge_index, torch.flip(edge_index, dims=[1])),
                              dim=0)
    edge_index = torch.unique(edge_index, dim=0)
    return edge_index.t()
