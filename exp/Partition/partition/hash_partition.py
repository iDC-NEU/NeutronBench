import torch
import os
import sys
sys.path.append('..')
from partition.utils import show_time


@show_time
def hash_partition_graph(dataset, num_parts, num_nodes, save_dir='./partition_result'):
    assert os.path.exists(save_dir), f'save_dir: {save_dir} not exist!'
    save_path = f'{save_dir}/hash-{dataset}-part{num_parts}.pt'

    if os.path.exists(save_path):
        print(f'read from partition result {save_path}.')
        parts = torch.load(save_path)
    else:
        parts = torch.randint(0, num_parts, (num_nodes,))
        torch.save(parts, save_path)
        print(f'save partition result to {save_path}.')
    return parts

if __name__ == '__main__':
    a = hash_partition_graph('test', 4, 100000)
    print(a)
    print(torch.where(a == 0)[0].size())
    print(torch.where(a == 1)[0].size())
    print(torch.where(a == 2)[0].size())
    print(torch.where(a == 3)[0].size())
