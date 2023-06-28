import torch
import numpy as np
import os
import sys
sys.path.append('..')
import random
from partition.utils import show_time



@show_time
# def dgl_partition_graph(dataset, num_parts, num_nodes):
def dgl_partition_graph(dataset, num_parts, rowptr, col, train_mask, val_mask, test_mask, node_weight_dim=1):
    save_dgl_parts = f'/home/yuanh/neutron-sanzo/exp/Partition/partition/partition_result/dgl-{dataset}.pt'
    num_nodes = tarin_mask.


    exit(0)
    # if os.path.exists(save_dgl_parts):
    #     print(f'read from partition result {save_dgl_parts}.')
    #     parts = torch.load(save_dgl_parts)
    # else:
    #     parts = torch.randint(0, num_parts, (num_nodes,))
    #     torch.save(parts, save_dgl_parts)
    #     print(f'save partition result to {save_dgl_parts}.')
    return parts



if __name__ == '__main__':
    a = dgl_partition_graph('test', 4, 100000)
    print(a)
    print(torch.where(a == 0)[0].size())
    print(torch.where(a == 1)[0].size())
    print(torch.where(a == 2)[0].size())
    print(torch.where(a == 3)[0].size())
