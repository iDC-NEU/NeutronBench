import torch
import numpy as np
import os
import sys
sys.path.append('..')
import random
from partition.utils import show_time



@show_time
def hash_partition_graph(dataset, num_parts, num_nodes):
    save_hash_parts = f'/home/yuanh/neutron-sanzo/exp/Partition/partition/partition_result/hash-{dataset}-part{num_parts}.pt'
    if os.path.exists(save_hash_parts):
        print(f'read from partition result {save_hash_parts}.')
        parts = torch.load(save_hash_parts)
    else:
        parts = torch.randint(0, num_parts, (num_nodes,))
        torch.save(parts, save_hash_parts)
        print(f'save partition result to {save_hash_parts}.')
    return parts



if __name__ == '__main__':
    a = hash_partition_graph('test', 4, 100000)
    print(a)
    print(torch.where(a == 0)[0].size())
    print(torch.where(a == 1)[0].size())
    print(torch.where(a == 2)[0].size())
    print(torch.where(a == 3)[0].size())
