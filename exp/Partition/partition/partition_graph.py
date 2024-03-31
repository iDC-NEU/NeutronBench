import argparse
import os
import sys

import torch as th
import random
import argparse
# import torch
import dgl.sparse as dglsp

sys.path.append('..')

from myPartition import mypartition_graph
from partition.utils import extract_dataset

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from partition.metis_partition import metis_partition_graph
from partition.pagraph_partition import pagraph_partition_graph
from partition.dgl_partition import dgl_partition_graph
from partition.hash_partition import hash_partition_graph

import torch


def bytegnn_partition_graph(dataset,
                            num_parts,
                            num_nodes,
                            save_dir='./partition_result'):
    assert os.path.exists(save_dir), f'save_dir: {save_dir} not exist!'
    save_path = f'{save_dir}/bytegnn-{dataset}-part{num_parts}.txt'

    if os.path.exists(save_path):
        print(f'read from partition result {save_path}.')
        with open(save_path, 'r') as f:
            partition_nodes = []
            for line in f.readlines():
                line = list(map(int, line.split(' ')))
                assert len(line) == line[1] + 2
                partition_nodes.append(line[2:])
        num_parts = len(partition_nodes)
        print('partition num', num_parts, [len(x) for x in partition_nodes])
        parts = -1 * torch.ones(num_nodes, dtype=torch.long)
        for i in range(num_parts):
            parts[partition_nodes[i]] = i
    else:
        print(
            f"bytegnn partition result not exist, please run ./bytegnn_partition.sh to generate it!"
        )
        assert False
    return parts


if __name__ == "__main__":
    argparser = argparse.ArgumentParser("Partition builtin graphs")
    argparser.add_argument(
        "--dataset",
        type=str,
        default="reddit",
        help="datasets: reddit, ogb-product, ogb-paper100M",
    )
    argparser.add_argument("--num_parts",
                           type=int,
                           default=4,
                           help="number of partitions")

    argparser.add_argument("--num_hops",
                           type=int,
                           default=1,
                           help="number of hops")

    argparser.add_argument("--mode", type=str, default=4, required=True)
    argparser.add_argument("--part_method",
                           type=str,
                           default="metis",
                           help="the partition method")
    argparser.add_argument("--self-loop",
                           type=bool,
                           default=True,
                           help="insert self-loop (default=True)")
    argparser.add_argument("--batch_size",
                           help="batch size of gnn train",
                           type=int,
                           default=6000)
    argparser.add_argument("--fanout",
                           help="Training fanouts",
                           type=int,
                           default=[10, 25],
                           nargs="*",
                           required=False)
    argparser.add_argument("--dim", help="metis dims", type=int, default=None)
    argparser.add_argument(
        "--output",
        type=str,
        default="./data",
        help="Output path of partitioned graph.",
    )
    args = argparser.parse_args()

    edges_list, features, labels, train_mask, val_mask, test_mask, graph = extract_dataset(
        args)
    # for key in graph.ndata:
    #     print(key)
    # print(graph.ndata['feature'].shape)
    print(train_mask)
    feature_dim = features.shape[1]
    train_mask = train_mask.to(th.long)
    val_mask = val_mask.to(th.long)
    test_mask = test_mask.to(th.long)

    node_nums = graph.number_of_nodes()
    edge_nums = graph.number_of_edges()
    src_nodes = edges_list[:, 0].tolist()
    dst_nodes = edges_list[:, 1].tolist()
    edges = [(x, y) for x, y in zip(src_nodes, dst_nodes)]
    assert (len(edges) == edge_nums == len(set(edges)) == edges_list.shape[0])
    indices = th.tensor([src_nodes, dst_nodes])
    rowptr, col, value = dglsp.spmatrix(indices).csc()

    # parts = read_bytegnn_partition_result(args.dataset,node_nums)
    if args.mode == 'hash':
        parts = hash_partition_graph(args.dataset, args.num_parts, node_nums)
    elif args.mode == 'metis':
        assert args.dim, 'you should specify the dim!'
        if args.dim == 2:
            parts = dgl_partition_graph(args.dataset, args.num_parts, graph,
                                        train_mask, val_mask, test_mask)
        else:
            parts = metis_partition_graph(args.dataset,
                                          args.num_parts,
                                          rowptr,
                                          col,
                                          train_mask,
                                          val_mask,
                                          test_mask,
                                          node_weight_dim=args.dim)
    elif args.mode == 'bytegnn':
        parts = bytegnn_partition_graph(args.dataset, args.num_parts,
                                        node_nums)
    elif args.mode == 'pagraph':
        partition_nodes, partition_train_nodes = pagraph_partition_graph(
            args.dataset, args.num_parts, 1, graph, rowptr, col, train_mask,
            val_mask, test_mask)
        parts = th.ones(node_nums, dtype=th.int64) * -1
        for i, nid in enumerate(partition_train_nodes):
            parts[nid] = i
        neg = th.where(parts == -1)[0].tolist()
        random.shuffle(neg)
        neg = th.tensor(neg)
        step = (len(neg) + args.num_parts - 1) // args.num_parts
        for i in range(args.num_parts):
            parts[neg[i * step:min((i + 1) * step, len(neg))]] = i
        # assert False
    else:
        assert False

    num_hops = args.num_hops
    if args.mode == 'pagraph':
        num_hops = 2

    mypartition_graph(
        graph,
        args.dataset,
        args.num_parts,
        args.output,
        parts,
        num_hops=num_hops,
    )
