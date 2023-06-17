import torch
import numpy as np
import os
import sys
sys.path.append('..')
from partition.utils import show_time
import time


import random
import argparse
import numpy as np
import torch
import dgl
import dgl.sparse as dglsp
import scipy.sparse as spsp
import os

import sys
sys.path.append('..')

from partition.utils import show_time
from partition.utils import extract_dataset
from partition.utils import setup_seed
from partition.utils import generate_nts_dataset
from partition.utils import get_all_edges
from partition.utils import get_partition_result
from partition.utils import get_pagraph_partition_result
from partition.utils import show_label_distributed



from partition.metis_partition import metis_partition_graph
from partition.pagraph_partition import pagraph_partition_graph
from partition.hash_partition import hash_partition_graph





def bfs_with_time(start_time, nid, rowptr, col, hop):
    # 从nid开始进行hop次bfs，每个bfs的节点，标记{nid, time.time() - start_time}
    bfs_result = [{nid}]
    node_with_time = {nid: (nid, 0)}
    visit_nids = {nid}
    for h in range(hop):
        curr_layer = set()
        for u in bfs_result[-1]:
            for v in col[rowptr[u]: rowptr[u+1]]:
                v = v.item()
                if v in visit_nids:
                    continue
                assert v not in node_with_time
                node_with_time[v] = (nid, time.time() - start_time)
                curr_layer.add(v)
                visit_nids.add(v)
        bfs_result.append(curr_layer)
    # print(bfs_result, node_with_time, len(node_with_time))
    return node_with_time

import copy

def update_node_time(X, Y):
    X_copy = copy.copy(X)
    for k,v in Y.items():
        if k not in X or X[k][1] > v[1]:
            X[k] = v
    insert_count = 0
    update_count = 0
    for k,v in X.items():
        if k not in X_copy:
            insert_count += 1
        elif v != X_copy[k]:
            update_count += 1
    # print(f'insert {insert_count} nodes and update {update_count}')
    

def bfs_mark_node(nids, hop, rowptr, col):
    node_with_time = {}
    for u in nids:
        # print('start bfs', u)
        ret = bfs_with_time(time.time(), u, rowptr, col, hop)
        update_node_time(node_with_time, ret)
    return node_with_time


def test_lable_nids_mark_result(label_nids, node_with_time):
    for u in label_nids:
        assert u in node_with_time
        assert node_with_time[u][0] == u


def get_all_blocks(node_with_time):
    mark_block = {}
    for k,v in node_with_time.items():
        mark = v[0]
        if mark in mark_block:
            mark_block[mark].append(k)
        else:
            mark_block[mark] = [k]

    return mark_block



def get_label_blocks(all_blocks, nids):
    ret = []
    for k,v in all_blocks.items():
        if k in nids:
            ret.append(v)

    return ret


def test_blocks_info(train_blocks, val_blocks, test_blocks, all_blocks, train_nids, val_nids, test_nids):
        assert len(train_blocks) == len(train_nids)
        assert len(test_blocks) == len(test_nids)
        assert len(val_blocks) == len(val_nids)
        assert len(all_blocks) == len(train_blocks) + len(val_blocks) + len(test_blocks)
        all_blocks_list = [v for _,v in all_blocks.items()]
        all_block_nodes = [n for b in all_blocks_list for n in b]
        assert len(all_block_nodes) == len(set(all_block_nodes))
        print('all_blocks_nodes', len(all_block_nodes))
        print('train_blocks', len(train_blocks), 'train_block_node:', sum([len(x) for x in train_blocks]))
        print('val_blocks', len(val_blocks), 'val_block_node:', sum([len(x) for x in val_blocks]))
        print('test_blocks', len(test_blocks), 'test_block_node:', sum([len(x) for x in test_blocks]))


def cross_edges(block, nodes, rowptr, col):
    count = 0
    for u in block:
        count += len(set(nodes) & set(col[rowptr[u] : rowptr[u + 1]]))
    return count


def get_label_nums(nodes, train_nids, val_nids, test_nids):
    train_st = set(nodes) & set(train_nids)
    val_st = set(nodes) & set(val_nids)
    test_st = set(nodes) & set(test_nids)
    return len(train_st), len(val_st), len(test_st)



@show_time
def bytegnn_partition_graph(dataset, num_parts, num_hops, rowptr, col, train_mask, val_mask, test_mask, alpha=1.0, beta=1.0, gamma=1.0):
    print("\n######## bytegnn_partition_graph #########")
    print("num_parts {} num_hops {} ".format(num_parts, num_hops))

    save_partition_result = f'/home/yuanh/neutron-sanzo/exp/Partition/partition/partition_result/bytegnn-{dataset}-parts.pt'
    if os.path.exists(save_partition_result):
        print(f'read from partition result {save_partition_result}.')
        parts = torch.load(save_partition_result)
    else:
        train_nids = torch.where(train_mask > 0)[0].tolist()
        test_nids = torch.where(test_mask > 0)[0].tolist()
        val_nids = torch.where(val_mask > 0)[0].tolist()
        label_nids = train_nids + test_nids + val_nids
        train_num, val_num, test_num = len(train_nids), len(val_nids), len(test_nids)
        assert train_num + test_num + val_num == len(label_nids)

        node_with_time = {}
        train_mark = bfs_mark_node(train_nids, num_hops, rowptr, col)
        val_mark = bfs_mark_node(val_nids, num_hops, rowptr, col)
        test_mark = bfs_mark_node(test_nids, num_hops, rowptr, col)
        update_node_time(node_with_time, train_mark)
        update_node_time(node_with_time, val_mark)
        update_node_time(node_with_time, test_mark)
        test_lable_nids_mark_result(label_nids, node_with_time)
        print('all node_with_time:', len(node_with_time))
        
        all_blocks = get_all_blocks(node_with_time)
        assert len(all_blocks) == len(label_nids)
        train_blocks = get_label_blocks(all_blocks, train_nids)
        val_blocks = get_label_blocks(all_blocks, val_nids)
        test_blocks = get_label_blocks(all_blocks, test_nids)
        
        test_blocks_info(train_blocks, val_blocks, test_blocks, all_blocks, train_nids, val_nids, test_nids)

        train_blocks.sort(key=lambda x : len(x), reverse=True)
        val_blocks.sort(key=lambda x : len(x), reverse=True)
        test_blocks.sort(key=lambda x : len(x), reverse=True)
        node_nums = len(rowptr) - 1
        parts = -np.ones(node_nums, dtype=np.int64)
        partition_nodes = [[] for _ in range(num_parts)] 

        all_block_list = train_blocks + val_blocks + test_blocks
        all_block_list.sort(key=lambda x : len(x), reverse=True)
        # print('all_block_list', [len(x) for x in all_block_list])

        for i, block in enumerate(all_block_list):
            score = np.zeros(num_parts, dtype=np.float32)
            for j in range(num_parts):
                ce = 1
                if len(partition_nodes[j]) > 0:
                    ce += cross_edges(block, partition_nodes[j], rowptr, col) / len(partition_nodes[j])
                P_train, P_val, P_test = get_label_nums(partition_nodes[j], train_nids, val_nids, test_nids)
                bs = 1 - alpha * P_train / train_num - beta * P_val / val_num - gamma * P_test / test_num
                score[j] = ce * bs
            ids = np.argsort(score)[-1]
            parts[block] = ids
            partition_nodes[ids].extend(block)
            
        for i, nodes in enumerate(partition_nodes):
            assert (parts[nodes] == i).all()

        print(np.where(parts == -1))

        parts = torch.from_numpy(parts)
        print('parts', parts)
        torch.save(parts, save_partition_result)
        print(f'save partition result to {save_partition_result}.')
    return parts


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate Dataset')
    parser.add_argument("--dataset", type=str, default="cora", help="Dataset name (cora, citeseer, pubmed, reddit)")
    parser.add_argument("--self-loop", type=bool, default=True, help="insert self-loop (default=True)")
    parser.add_argument("--num_parts", help="Number of partitions to generate", type=int, required=True)
    parser.add_argument("--num_hops", help="Number of layer in GNN for PaGraph partition", type=int, default=2)
    parser.add_argument("--save_nts", type=bool, default=True, help="insert self-loop (default=True)")
    # parser.add_argument("--batch_size", help="batch size of gnn train", type=int, default=6000)
    # parser.add_argument("--fanout", help="Training fanouts", type=int, default=[10, 25], nargs="*", required=False)
    # parser.add_argument("--dim", help="metis dims", type=int, required=True)
    # parser.add_argument("--algo", help="partition algorithm (metis, pagraph, hash)", type=str, required=True)
    args = parser.parse_args()
    print(args)
    # assert len(args.fanout) == args.num_hops

    # args.fanout.reverse() # or reversed(args.fanout)
    setup_seed(2000)
    # graph dataset
    edges_list, features, labels, train_mask, val_mask, test_mask, graph = extract_dataset(args)
    feature_dim = features.shape[1]
    train_mask = train_mask.to(torch.long)
    val_mask = val_mask.to(torch.long)
    test_mask = test_mask.to(torch.long)

    node_nums = graph.number_of_nodes()
    edge_nums = graph.number_of_edges()
    src_nodes = edges_list[:, 0].tolist()
    dst_nodes = edges_list[:, 1].tolist()
    edges = [(x, y) for x, y in zip(src_nodes, dst_nodes)]
    assert (len(edges) == edge_nums == len(set(edges)) == edges_list.shape[0])
    indices = torch.tensor([src_nodes, dst_nodes])
    rowptr, col, value = dglsp.spmatrix(indices).csc()

    
    parts = bytegnn_partition_graph(args.dataset, args.num_parts, args.num_hops, rowptr, col, train_mask, val_mask, test_mask, alpha=1.0, beta=1.0, gamma=1.0)
    partition_nodes = []
    for i in range(args.num_parts):
        partition_nodes.append(torch.where(parts == i)[0].tolist()) 

    if args.save_nts:
        # if not os.path.exists('/home/yuanh/neutron-sanzo/exp/Partition/partition/partition_result/bytegnn'):
        #     os.makedirs('/home/yuanh/neutron-sanzo/exp/Partition/partition/partition_result/bytegnn')
        save_partition_nts = f'/home/yuanh/neutron-sanzo/exp/Partition/partition/partition_result/bytegnn/{args.dataset}-parts.txt'
        with open(save_partition_nts, 'w') as f:
            for i in range(args.num_parts):
                f.write(f'part{i} node{len(partition_nodes[i])}: ')
                f.write(' '.join([str(x) for x in partition_nodes[i]]))
                # print(save_partition[i])
                f.write('\n')


    # for _ in train_blocks:
    #     for u in _:
    #         assert u not in val_nids + test_nids

    
    
    # print(all_blocks)

