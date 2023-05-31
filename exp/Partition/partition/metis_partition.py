import os
os.environ["METIS_DLL"] = "../pkgs/lib/libmetis.so"
os.environ["METIS_IDXTYPEWIDTH"] = "64"
os.environ["METIS_REALTYPEWIDTH"] = "64"
os.environ["OMP_NUM_THREADS"] = "1"
import torch_metis as metis
import torch
import time
# import sys, importlib
# from pathlib import Path

import sys
sys.path.append('..')
from partition.utils import show_time
from partition.utils import get_partition_label_nodes
from partition.utils import show_label_distributed
from partition.utils import get_partition_nodes
from partition.utils import get_partition_edges


def get_1d_node_weights(train_mask):
    w1 = train_mask
    return w1.view(-1).to(torch.long).contiguous()


def get_2d_node_weights(train_mask, rowptr):
    w1 = train_mask
    w4 = rowptr[1:] - rowptr[:-1]
    return torch.cat([w1.reshape(w4.size()[0], 1), w4.reshape(w1.size()[0], 1)], dim=1).view(-1).to(torch.long).contiguous()


def get_4d_node_weights(train_mask, val_mask, test_mask, rowptr):
    w1 = train_mask
    w2 = val_mask

    # SAILENT++: (https://github.com/MITIBMxGraph/SALIENT_plusplus_artifact/blob/4dfa0b6100f4572fb54fed1d4adf2fa8a9da0717/partitioners/run_4constraint_partition.py#L31)
    node_nums = rowptr.shape[0] - 1
    w3 = torch.ones(node_nums, dtype=torch.long)
    print(id(node_nums))
    w3 ^= w2 | w1
    # print((w3 | w2 | w1).sum().item())
    assert ((w3 | w2 | w1).sum().item() == node_nums)

    w4 = rowptr[1:] - rowptr[:-1]
    return torch.cat([w1.reshape(w2.size()[0], 1), w2.reshape(w1.size()[0], 1), w3.reshape(w1.size()[0], 1), w4.reshape(w1.size()[0], 1)], dim=1).view(-1).to(torch.long).contiguous()


@show_time
def metis_partition(rowptr, col, node_weights, edge_weights, nodew_dim=1, num_parts=2):
    G = metis.csr_to_metis(rowptr.contiguous(), col.contiguous(
    ), node_weights, edge_weights, nodew_dim=nodew_dim)
    print(str([1.001]*nodew_dim))
    objval, parts = metis.part_graph(
        G, nparts=num_parts, ubvec=[1.001]*nodew_dim)
    parts = torch.tensor(parts)
    print("Cost is " + str(objval))

    print("Partitions:", parts)

    bincounts = torch.bincount(parts, minlength=num_parts)
    print("Partition bin counts:", bincounts.tolist())
    return parts


# metis partition
# TODO(sanzo): 1d,2d,3d,4d
@show_time
def metis_partition_graph(num_parts, rowptr, col, train_mask, val_mask, test_mask, node_weight_dim=1):
    print("\n######## metis_partition_graph #########")
    edge_weights = torch.ones_like(
        col, dtype=torch.long, memory_format=torch.legacy_contiguous_format).share_memory_()
    
    if node_weight_dim == 4:
        node_weights = get_4d_node_weights(train_mask, val_mask, test_mask, rowptr)
    elif node_weight_dim == 2:
        node_weights = get_2d_node_weights(train_mask, rowptr)
    elif node_weight_dim == 1:
        node_weights = get_1d_node_weights(train_mask)
    else:
        assert False

    parts = metis_partition(rowptr, col, node_weights, edge_weights,
                            nodew_dim=node_weight_dim, num_parts=num_parts)
    print()

    # 每个分区node, train_nodes, val_nodes, test_nodes
    partition_nodes = get_partition_nodes(parts, num_parts)
    partition_train_nodes = get_partition_label_nodes(partition_nodes, train_mask)
    partition_val_nodes = get_partition_label_nodes(partition_nodes, val_mask)
    partition_test_nodes = get_partition_label_nodes(partition_nodes, test_mask)

    # 每个分区包含的边[[], []]
    partition_edges = get_partition_edges(partition_nodes, rowptr, col)
    print('metis partition nodes:', [len(_) for _ in partition_nodes])
    print('metis partition edges:', [len(_) for _ in partition_edges])
    show_label_distributed(parts, train_mask, val_mask, test_mask)

    return (partition_nodes, partition_edges, partition_train_nodes, partition_val_nodes, partition_test_nodes)


