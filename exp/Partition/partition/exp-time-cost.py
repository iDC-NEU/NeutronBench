import random
import argparse
import numpy as np
import torch
import dgl
import dgl.sparse as dglsp
import scipy.sparse as spsp
import os
import sys
import argparse

sys.path.append('..')
import torch

from partition.utils import setup_seed
from partition.metis_partition import metis_partition_graph
from partition.pagraph_partition import pagraph_partition_graph
from partition.hash_partition import hash_partition_graph
from partition.bytegnn_partition import bytegnn_partition_graph
from partition.bytegnn_partition import read_bytegnn_partition_result
from partition.utils import extract_dataset
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Dataset')
    parser.add_argument("--dataset",
                        type=str,
                        default="cora",
                        help="Dataset name (cora, citeseer, pubmed, reddit)")
    parser.add_argument("--self-loop",
                        type=bool,
                        default=True,
                        help="insert self-loop (default=True)")
    parser.add_argument("--num_parts",
                        help="Number of partitions to generate",
                        type=int,
                        required=True)
    parser.add_argument("--num_hops",
                        help="Number of layer in GNN for PaGraph partition",
                        type=int,
                        default=2)
    # parser.add_argument("--batch_size", help="batch size of gnn train", type=int, default=6000)
    # parser.add_argument("--fanout", help="Training fanouts", type=int, default=[10, 25], nargs="*", required=False)
    # parser.add_argument("--dim", help="metis dims", type=int, required=True)
    # parser.add_argument("--algo", help="partition algorithm (metis, pagraph, hash)", type=str, required=True)
    args = parser.parse_args()
    print(args)

    setup_seed(2000)
    # graph dataset

    for dataset in ['ogbn-arxiv', 'ogbn-products', 'reddit', 'computer']:
        # for dataset in ['ogbn-products', 'reddit', 'computer']:
        # for dataset in ['computer']:
        # for dataset in ['reddit']:
        # for dataset in ['ogbn-arxiv']:
        args.dataset = dataset
        print('dataset', args.dataset)
        edges_list, features, labels, train_mask, val_mask, test_mask, graph = extract_dataset(
            args)
        feature_dim = features.shape[1]
        train_mask = train_mask.to(torch.long)
        val_mask = val_mask.to(torch.long)
        test_mask = test_mask.to(torch.long)

        node_nums = graph.number_of_nodes()
        edge_nums = graph.number_of_edges()
        src_nodes = edges_list[:, 0].tolist()
        dst_nodes = edges_list[:, 1].tolist()
        edges = [(x, y) for x, y in zip(src_nodes, dst_nodes)]
        assert (len(edges) == edge_nums == len(set(edges)) ==
                edges_list.shape[0])
        indices = torch.tensor([src_nodes, dst_nodes])
        rowptr, col, value = dglsp.spmatrix(indices).csc()

        mode_list = []
        time_list = []
        for dim in [1, 2, 4]:
            time_cost = -time.time()
            parts = metis_partition_graph(args.dataset,
                                          args.num_parts,
                                          rowptr,
                                          col,
                                          train_mask,
                                          val_mask,
                                          test_mask,
                                          node_weight_dim=dim)
            time_cost += time.time()
            time_list.append(round(time_cost, 3))
            mode_list.append(f'metis-dim{dim}')

        time_cost = -time.time()
        partition_nodes, partition_train_nodes = pagraph_partition_graph(
            args.dataset, args.num_parts, args.num_hops, graph, rowptr, col,
            train_mask, val_mask, test_mask)
        time_cost += time.time()
        time_list.append(round(time_cost, 3))
        mode_list.append(f'pagraph')

        time_cost = -time.time()
        parts = hash_partition_graph(args.dataset, args.num_parts, node_nums)
        time_cost += time.time()
        time_list.append(round(time_cost, 3))
        mode_list.append(f'hash')

        time_cost = -time.time()
        command = f'./bytegnn ~/neutron-sanzo/data/{args.dataset} {args.dataset} {args.num_parts} {args.num_hops} ./partition_result'
        os.system(command)
        time_cost += time.time()
        time_list.append(round(time_cost, 3))
        mode_list.append(f'bytegnn')

        print(f'####### Partition Time {args.dataset}')
        print(mode_list)
        print(time_list)

    # time_cost = -time.time()
    # parts =  read_bytegnn_partition_result(args.dataset, node_nums)
    # time_cost += time.time()
    # print(f'#### bytegnn partition {args.dataset} cost {time_cost}s.')
