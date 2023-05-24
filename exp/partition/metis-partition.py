import os
os.environ["METIS_DLL"] = "/home/yuanh/METIS-GKlib/build/lib/libmetis.so"
os.environ["METIS_IDXTYPEWIDTH"] = "64"
os.environ["METIS_REALTYPEWIDTH"] = "64"
os.environ["OMP_NUM_THREADS"] = "1"

import random
from functools import wraps
import sys
import argparse
import numpy as np
import networkx as nx
import time
import torch
import torch.nn.functional as F
import json
import scipy.sparse as sp
import dgl
import dgl.sparse as dglsp
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from dgl.data import CoraFullDataset, CoauthorCSDataset, CoauthorPhysicsDataset
from dgl.data import AmazonCoBuyComputerDataset, AmazonCoBuyPhotoDataset
from ogb.nodeproppred import DglNodePropPredDataset

import torch_metis as metis
import scipy.sparse as spsp

def extract_dataset(args):
    dataset = args.dataset

    # change dir
    if not os.path.exists(dataset):
        os.mkdir(dataset)
    os.chdir(dataset)

    if dataset in ['cora', 'citeseer', 'pubmed', 'reddit', 'reddit-small']:
        # load dataset
        reddit_small = (dataset == 'reddit-small')
        if args.dataset == 'reddit-samll':
            args.dataset = 'reddit'
        data = load_data(args)
        graph = data[0]
        features = graph.ndata['feat']
        labels = graph.ndata['label']
        train_mask = graph.ndata['train_mask']
        val_mask = graph.ndata['val_mask']
        test_mask = graph.ndata['test_mask']
        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()

        if reddit_small or args.split:
            graph, features, labels, train_mask, val_mask, test_mask = split_graph(
                graph, n_nodes, n_edges, features, labels, train_mask, val_mask, test_mask, args.frac)

        if args.self_loop:
            time_stamp = time.time()
            print('before add self loop has {} edges'.format(graph.num_edges()))
            graph = dgl.remove_self_loop(graph)
            graph = dgl.add_self_loop(graph)
            # graph = dgl.to_bidirected(graph) # simple graph
            print('after add self loop has {} edges'.format(graph.num_edges()))
            print("insert self loop cost {:.2f}s".format(
                time.time() - time_stamp))

        edges = graph.edges()
        edge_src = edges[0].numpy().reshape((-1, 1))
        edge_dst = edges[1].numpy().reshape((-1, 1))
        edges_list = np.hstack((edge_src, edge_dst))

        print("nodes: {}, edges: {}, feature dims: {}, classess: {}, label nodes: {}({}/{}/{})"
              .format(graph.number_of_nodes(), edges_list.shape,
                      list(features.shape), len(np.unique(labels)),
                      train_mask.sum() + test_mask.sum() + val_mask.sum(),
                      train_mask.sum(), val_mask.sum(), test_mask.sum()))
        return edges_list, features, labels, train_mask, val_mask, test_mask, graph

    elif dataset in ['CoraFull', 'Coauthor_cs', 'Coauthor_physics', 'AmazonCoBuy_computers', 'AmazonCoBuy_photo']:
        if dataset == 'CoraFull':
            data = CoraFullDataset()
        elif dataset == 'Coauthor_cs':
            data = CoauthorCSDataset('cs')
        elif dataset == 'Coauthor_physics':
            data = CoauthorPhysicsDataset('physics')
        elif dataset == 'AmazonCoBuy_computers':
            data = AmazonCoBuyComputerDataset('computers')
        elif dataset == 'AmazonCoBuy_photo':
            data = AmazonCoBuyPhotoDataset('photo')

        graph = data[0]
        features = torch.FloatTensor(graph.ndata['feat']).numpy()
        labels = torch.LongTensor(graph.ndata['label']).numpy()
        num_nodes = graph.number_of_nodes()

        if args.self_loop:
            time_stamp = time.time()
            print('before add self loop has {} edges'.format(
                len(graph.all_edges()[0])))
            graph = dgl.remove_self_loop(graph)
            graph = dgl.add_self_loop(graph)
            # graph = dgl.to_bidirected(graph)
            print('after add self loop has {} edges'.format(
                len(graph.all_edges()[0])))
            print("insert self loop cost {:.2f}s".format(
                time.time() - time_stamp))

        edges = graph.edges()
        edge_src = edges[0].numpy().reshape((-1, 1))
        edge_dst = edges[1].numpy().reshape((-1, 1))
        edges_list = np.hstack((edge_src, edge_dst))

        train_mask, val_mask, test_mask = split_dataset(num_nodes, 6, 3, 1)
        print("dataset: {} nodes: {} edges: {} feature dims: {} classess: {} label nodes: {}({}/{}/{})"
              .format(dataset, num_nodes, edges_list.shape,
                      list(features.shape), len(np.unique(labels)),
                      train_mask.sum() + test_mask.sum() + val_mask.sum(),
                      train_mask.sum(), val_mask.sum(), test_mask.sum()))
        return edges_list, features, labels, train_mask, val_mask, test_mask, graph

    elif dataset in ['ogbn-arxiv', 'ogbn-papers100M', 'ogbn-products']:
        # load dataset
        data = DglNodePropPredDataset(name=dataset)

        graph = data.graph[0]
        labels = data.labels
        features = graph.ndata['feat']

        split_idx = data.get_idx_split()
        train_nid, val_nid, test_nid = split_idx['train'], split_idx['valid'], split_idx['test']
        # print(len(train_nid) + len(val_nid) + len(test_nid))
        train_mask = np.zeros(graph.number_of_nodes(), dtype=bool)
        train_mask[train_nid] = True
        val_mask = np.zeros(graph.number_of_nodes(), dtype=bool)
        val_mask[val_nid] = True
        test_mask = np.zeros(graph.number_of_nodes(), dtype=bool)
        test_mask[test_nid] = True

        if args.self_loop:
            time_stamp = time.time()
            print('before add self loop has {} edges'.format(
                len(graph.all_edges()[0])))
            graph = dgl.remove_self_loop(graph)
            graph = dgl.add_self_loop(graph)
            if dataset == "ogbn-arxiv":
                graph = dgl.to_bidirected(graph)
            print('after add self loop has {} edges'.format(
                len(graph.all_edges()[0])))
            print("insert self loop cost {:.2f}s".format(
                time.time() - time_stamp))
        edges = graph.edges()
        edge_src = edges[0].numpy().reshape((-1, 1))
        edge_dst = edges[1].numpy().reshape((-1, 1))
        edges_list = np.hstack((edge_src, edge_dst))

        print("nodes: {}, edges: {}, feature dims: {}, classess: {}, label nodes: {}({}/{}/{})"
              .format(graph.number_of_nodes(), edges_list.shape,
                      list(features.shape), len(np.unique(labels)),
                      train_mask.sum() + test_mask.sum() + val_mask.sum(),
                      train_mask.sum(), val_mask.sum(), test_mask.sum()))
        return edges_list, features, labels, train_mask, val_mask, test_mask, graph

    elif dataset in ['flickr', 'yelp', 'ppi', 'ppi-large', 'amazon']:
        # prefix = os.getcwd()
        adj_full = sp.load_npz('./adj_full.npz').astype(bool)
        # adj_train = sp.load_npz('./adj_train.npz').astype(bool)
        role = json.load(open('./role.json'))
        feats = np.load('./feats.npy')
        class_map = json.load(open('./class_map.json'))
        class_map = {int(k): v for k, v in class_map.items()}
        assert len(class_map) == feats.shape[0]
        edges = adj_full.nonzero()
        graph = dgl.graph((edges[0], edges[1]))
        # print(graph)
        if args.self_loop:
            time_stamp = time.time()
            print('before add self loop has {} edges'.format(
                len(graph.all_edges()[0])))
            graph = dgl.remove_self_loop(graph)
            graph = dgl.add_self_loop(graph)
            print('after add self loop has {} edges'.format(
                len(graph.all_edges()[0])))
            print("insert self loop cost {:.2f}s".format(
                time.time() - time_stamp))

        edges = graph.edges()
        edge_src = edges[0].numpy().reshape((-1, 1))
        edge_dst = edges[1].numpy().reshape((-1, 1))
        edges_list = np.hstack((edge_src, edge_dst))
        # edges = np.vstack(adj_full.nonzero()).T
        # edges = np.hstack((edges[0].reshape((-1,1)), edges[1].reshape((-1,1))))
        # assert np.array_equal(tmp, edges)
        num_features = feats.shape[1]
        num_nodes = adj_full.shape[0]
        # num_edges = adj_full.nnz
        num_edges = len(edge_dst)

        # assert num_edges == edges.shape[0]

        train_mask = create_mask(role['tr'], num_nodes)
        val_mask = create_mask(role['va'], num_nodes)
        test_mask = create_mask(role['te'], num_nodes)

        # find onehot label if multiclass or not
        if isinstance(list(class_map.values())[0], list):
            is_multiclass = True
            num_classes = len(list(class_map.values())[0])
            class_arr = np.zeros((num_nodes, num_classes))
            for k, v in class_map.items():
                class_arr[k] = v
            labels = class_arr

            non_zero_labels = []
            for row in labels:
                non_zero_labels.append(np.nonzero(row)[0].tolist())
            labels = non_zero_labels
        else:
            num_classes = max(class_map.values()) - min(class_map.values()) + 1
            class_arr = np.zeros((num_nodes, num_classes))
            offset = min(class_map.values())
            is_multiclass = False
            for k, v in class_map.items():
                class_arr[k][v-offset] = 1
            labels = np.where(class_arr)[1]

        print("nodes: {}, edges: {}, feature dims: {}, classess: {}{}, label nodes: {}({}/{}/{})"
              .format(num_nodes, num_edges, feats.shape, num_classes, '#' if is_multiclass else '',
                      train_mask.sum() + test_mask.sum() + val_mask.sum(),
                      train_mask.sum(), val_mask.sum(), test_mask.sum()))
        return edges_list, feats, labels, train_mask, val_mask, test_mask, graph

    else:
        raise NotImplementedError


def split_graph(graph, n_nodes, n_edges, features, labels, train_mask, val_mask, test_mask, fraction):
    new_n_nodes = int(n_nodes * fraction)
    # check_type(graph, n_nodes, n_edges, features, labels, train_mask, val_mask, test_mask, fraction)
    remove_nodes_list = [x for x in range(new_n_nodes, n_nodes)]

    if isinstance(graph, nx.classes.digraph.DiGraph):
        print('graph is DiGraph')
        graph.remove_nodes_from(remove_nodes_list)
    elif isinstance(graph, DGLGraph):
        print('g is DGLGraph')
        graph.remove_nodes(remove_nodes_list)

    features = features[:new_n_nodes]
    labels = labels[:new_n_nodes]
    train_mask = train_mask[:new_n_nodes]
    val_mask = val_mask[:new_n_nodes]
    test_mask = test_mask[:new_n_nodes]

    return graph, features, labels, train_mask, val_mask, test_mask


def create_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l, dtype=bool)
    mask[idx] = True
    return mask


def split_dataset(num_nodes, x=8, y=1, z=1):
    '''
    x: train nodes, y: val nodes, z: test nodes
    '''
    train_mask = torch.tensor(
        [False for i in range(num_nodes)], dtype=torch.bool)
    val_mask = torch.tensor(
        [False for i in range(num_nodes)], dtype=torch.bool)
    test_mask = torch.tensor(
        [False for i in range(num_nodes)], dtype=torch.bool)
    step = int(num_nodes / (x + y + z))
    train_mask[: int(x * step)] = True
    val_mask[int(x * step): int((x+y) * step)] = True
    test_mask[int((x+y) * step):] = True
    assert (train_mask.sum() + val_mask.sum() + test_mask.sum() == num_nodes)
    return train_mask, val_mask, test_mask


def remask(node_num, mask_rate=None):
    train_num = int(mask_rate[0] * node_num)
    val_num = int(mask_rate[1] * node_num)
    test_num = node_num - train_num - val_num
    print('remask to', train_num, val_num, test_num)
    train_mask = np.zeros(node_num, dtype=bool)
    val_mask = np.zeros(node_num, dtype=bool)
    test_mask = np.zeros(node_num, dtype=bool)
    node_ids = np.arange(node_num)
    np.random.shuffle(node_ids)
    train_mask[node_ids[: train_num]] = True
    val_mask[node_ids[train_num: train_num + val_num]] = True
    test_mask[node_ids[train_num + val_num:]] = True
    return train_mask, val_mask, test_mask


def generate_nts_dataset(partition_nodes, partition_edges, node_num, feature_dim):
    print('node_num:', node_num, 'feature_dim:', feature_dim)
    reorder_id = {}
    curr_id = 0
    for nodes in partition_nodes:
        for u in nodes.tolist():
            reorder_id[u] = curr_id
            curr_id += 1
    assert len(reorder_id) == curr_id == sum(
        [len(nodes) for nodes in partition_nodes])

    reorder_partition_edges = [[(reorder_id[u], reorder_id[v]) for (
        u, v) in edges] for edges in partition_edges]

    assert len(reorder_partition_edges) == len(partition_edges)
    for (edges, reorder_edges) in zip(partition_edges, reorder_partition_edges):
        assert len(edges) == len(reorder_edges)
        for e1, e2 in zip(edges, reorder_edges):
            assert reorder_id[e1[0]] == e2[0] and reorder_id[e1[1]] == e2[1]

    # (dst,src) to (src,dst)
    # whole dataset file
    all_edges = get_all_edges(reorder_partition_edges)
    swap_edges = torch.tensor(all_edges)[:, [1, 0]].contiguous()
    edge2bin(f'./{args.dataset}/reorder.edge.self', swap_edges)

    # split dataset file
    for partid, edges in enumerate(reorder_partition_edges):
        swap_edges = torch.tensor(edges)[:, [1, 0]].contiguous()
        edge2bin(f'./{args.dataset}/split_edge/{partid}.edge.self', swap_edges)

    # partition info
    with open(f'./{args.dataset}/split_edge/partition.info', 'w') as f:
        for partid, nodes in enumerate(partition_nodes):
            f.writelines(f'{partid} {nodes.size()[0]}\n')

    # write mask
    reorder_train_mask = torch.zeros_like(train_mask, dtype=torch.long)
    reorder_val_mask = torch.zeros_like(val_mask, dtype=torch.long)
    reorder_test_mask = torch.zeros_like(test_mask, dtype=torch.long)
    for nodeId in range(node_num):
        reorder_train_mask[reorder_id[nodeId]] = train_mask[nodeId]
        reorder_val_mask[reorder_id[nodeId]] = val_mask[nodeId]
        reorder_test_mask[reorder_id[nodeId]] = test_mask[nodeId]
    assert reorder_train_mask.sum() == train_mask.sum()
    assert reorder_val_mask.sum() == val_mask.sum()
    assert reorder_test_mask.sum() == test_mask.sum()
    write_to_mask(f'./{args.dataset}/reorder.mask',
                  reorder_train_mask, reorder_val_mask, reorder_test_mask)

    # write faeture
    out_features = torch.ones((node_num, feature_dim))
    write_to_file(f'./{args.dataset}/reorder.featuretable',
                  out_features, '%.2f', index=True)

    # write label
    out_features = torch.ones(node_num)
    write_to_file(f'./{args.dataset}/reorder.labeltable',
                  out_features, '%.0f', index=True)


def read_edgelist_from_file(filename):
    assert os.path.exists(filename)
    edgelist = []
    with open(filename) as f:
        # for line in f.readlines():
        # print(line.strip('\n').split(' '), type(line.strip('\n')))
        # for u in line.split(' '):
        #     print(u, u)

        # edgelist = [(int(u), int(v)) for line in f.readlines() for u,v in line.strip('\n').split(' ')]
        edgelist = [tuple(int(x) for x in line.strip('\n').split(' '))
                    for line in f.readlines()]
    return edgelist


def partition_bug(partition_L, partition_nodes, partition_train_nodes, rowptr, col):
    rowptr = rowptr.tolist()
    col = col.tolist()
    reorder_id, old_id = {}, {}
    curr_id = 0
    for nodes in partition_nodes:
        for u in nodes.tolist():
            reorder_id[u] = curr_id
            old_id[curr_id] = u
            if (curr_id == 32416):
                print('32416!')
            curr_id += 1
    assert len(old_id) == len(reorder_id) == curr_id == sum(
        [len(nodes) for nodes in partition_nodes])

    # layer0
    layer0_train_nodes = partition_train_nodes[0].tolist()
    part0_layer0_neighbor = "/home/yuanh/NtsMinibatch_fzb/reordergraph/test/partition0_layer_0_full_neighbor.txt"
    part0_layer0_edges = read_edgelist_from_file(part0_layer0_neighbor)
    print('part0_layer0_edges:', len(part0_layer0_edges))
    part0_layer0_edges = [(old_id[u], old_id[v])
                          for u, v in part0_layer0_edges]

    dgl_part0_layer0_edges = partition_L[0][0]
    print('dgl_part0_layer0_edges:', len(dgl_part0_layer0_edges))

    # check layer0 dst nodes
    dst_nodes = set([u for u, _ in part0_layer0_edges])
    src_nodes = set([v for _, v in part0_layer0_edges])

    dgl_dst_nodes = set([u for u, _ in dgl_part0_layer0_edges])
    dgl_src_nodes = set([v for _, v in dgl_part0_layer0_edges])
    # print('layer0_train_nodes:', len(set(layer0_train_nodes)))
    # print('layer0 dst:', len(dst_nodes), len(dgl_dst_nodes))
    # print('dgl dst not in train nodes', len(dgl_dst_nodes - set(layer0_train_nodes)))
    assert set(layer0_train_nodes) == dgl_dst_nodes == dst_nodes
    # print('dst_nodes', dst_nodes)
    print('layer0 src:', len(src_nodes), len(dgl_src_nodes))
    assert src_nodes == dgl_src_nodes
    print('layer0 dst node:', len(dst_nodes), len(dgl_dst_nodes))
    print('layer0 src node:', len(src_nodes), len(dgl_src_nodes))

    # layer1
    part0_layer1_neighbor = "/home/yuanh/NtsMinibatch_fzb/reordergraph/test/partition0_layer_1_full_neighbor.txt"
    part0_layer1_edges = read_edgelist_from_file(part0_layer1_neighbor)
    part0_layer1_edges = [(old_id[u], old_id[v])
                          for u, v in part0_layer1_edges]
    print('part0_layer1_edges:', len(part0_layer1_edges))

    dgl_part0_layer1_edges = partition_L[0][1]
    print('dgl_part0_layer1_edges:', len(dgl_part0_layer1_edges))

    # check layer1 dst nodes
    laye1_dst_nodes = set([u for u, _ in part0_layer1_edges])
    laye1_src_nodes = set([v for _, v in part0_layer1_edges])

    laye1_dgl_dst_nodes = set([u for u, _ in dgl_part0_layer1_edges])
    laye1_dgl_src_nodes = set([v for _, v in dgl_part0_layer1_edges])

    print('layer1 dst node:', len(laye1_dst_nodes), len(laye1_dgl_dst_nodes))
    print('layer1 src node:', len(laye1_src_nodes), len(laye1_dgl_src_nodes))
    miss_nodes = [reorder_id[x] for x in src_nodes - laye1_dst_nodes]
    # print(miss_nodes)
    assert laye1_dst_nodes == laye1_dgl_dst_nodes
    assert laye1_src_nodes == laye1_dgl_src_nodes


def show_time(func):
    @wraps(func)
    def with_time(*args, **kwargs):
        time_cost = time.time()
        ret = func(*args, **kwargs)
        time_cost = time.time() - time_cost
        func_name = func.__name__
        if time_cost > 3:
            print("func {} is done, cost: {:.2f}s".format(func_name, time_cost))
        return ret
    return with_time


def create_dir(path=None):
    if path and not os.path.exists(path):
        os.makedirs(path)


@show_time
def edge2bin(name, edges):
    create_dir(os.path.dirname(name))
    edges = edges.flatten()
    with open(name, 'wb') as f:
        buf = [int(edge).to_bytes(4, byteorder=sys.byteorder)
               for edge in edges]
        f.writelines(buf)


@show_time
def write_to_mask(name, train_mask, val_mask, test_mask):
    train_mask = train_mask.tolist()
    val_mask = val_mask.tolist()
    test_mask = test_mask.tolist()
    create_dir(os.path.dirname(name))
    with open(name, 'w') as f:
        for nodeId in range(len(train_mask)):
            # if train_mask[i]:
            node_type = 'unknown'
            if train_mask[nodeId] == 1:
                node_type = 'train'
            elif val_mask[nodeId] == 1:
                node_type = 'val'
            elif test_mask[nodeId] == 1:
                node_type = 'test'
            f.write(str(nodeId) + ' ' + node_type + '\n')


@show_time
def write_to_file(name, data, format, index=False):
    if not type(data) is np.ndarray:
        data = data.numpy()
    np.savetxt(name, data, fmt=format)

    if index:
        in_file = open(name, 'r')
        out_file = open(name+'.temp', 'w')
        for i, line in enumerate(in_file):
            out_file.write(str(i) + ' ' + line)
        in_file.close()
        out_file.close()
        os.remove(name)
        os.rename(name+'.temp', name)


@show_time
def write_multi_class_to_file(name, data, format, index=False):
    with open(name, 'w') as f:
        for i, line in enumerate(data):
            line.insert(0, len(line))
            line.insert(0, i)
            # f.write(str(len(line)) + ' ' +  ' '.join(str(x) for x in line) + '\n')
            f.write(' '.join(str(x) for x in line) + '\n')


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


def show_label_distributed(parts, train_mask, val_mask, test_mask):
    train_idx = torch.nonzero(train_mask).view(-1)
    val_idx = torch.nonzero(val_mask).view(-1)
    test_idx = torch.nonzero(test_mask).view(-1)
    print('train distributed:', torch.bincount(parts[train_idx]))
    print('val distributed:', torch.bincount(parts[val_idx]))
    print('test distributed:', torch.bincount(parts[test_idx]))


def get_partition_nodes(parts):
    partition_nodes = []
    for i in range(args.num_parts):
        partition_nodes.append(torch.where(parts == i)[0])
    return partition_nodes


def get_partition_label_nodes(partition_nodes, label_mask):
    partition_label_nodes = []
    for nodes in partition_nodes:
        own_idx = torch.where(label_mask[nodes] == True)[0]
        partition_label_nodes.append(nodes[own_idx])

    # check
    # label_nodes = [x.size()[0] for x in partition_label_nodes]
    # # print(label_nodes)
    # set1 = set()
    # for x in partition_label_nodes:
    #   set1.update(x.tolist())
    # set2 = set()
    # set2.update(torch.where(label_mask > 0)[0].tolist())
    # assert(set1 == set2)

    return partition_label_nodes


@show_time
def get_partition_edges(partition_nodes, rowptr, col, edge_nums=None):
    # (dst, src)
    rowptr = rowptr.tolist()
    col = col.tolist()
    partition_edges = []
    for nodes in partition_nodes:
        edge_list = []
        for u in nodes:
            edge_list += [(u.item(), v) for v in col[rowptr[u]: rowptr[u + 1]]]
        partition_edges.append(edge_list)

    # partition_edges_ = []
    # for nodes in partition_nodes:
    #   edge_list = []
    #   src_nodes, dst_nodes = [], []
    #   for u in nodes:
    #     src_nodes += [u] * (rowptr[u + 1] - rowptr[u])
    #     dst_nodes += col[rowptr[u]: rowptr[u + 1]]
    #   assert len(src_nodes) == len(dst_nodes)
    #   partition_edges_.append(list(zip(src_nodes, dst_nodes)))

    # for x,y in zip(partition_edges, partition_edges_):
    #     assert(len(x) == len(y))
    #     print(len(x), len(y))
    # return partition_edges_

    # check
    # print([len(_) for _ in partition_edges])
    # st = set()
    # for x in partition_edges:
    #     st.update(x)
    # assert len(st) == edge_nums

    return partition_edges


@show_time
def get_L_hop_edges(partition_nodes, rowptr, col, hop=2):
    rowptr = rowptr.tolist()
    col = col.tolist()
    partition_L_hop_edges = []
    for nodes in partition_nodes:
        curr_node = nodes.tolist()
        L_hop_edges = []
        for l in range(hop):
            # get directed neighbor
            layer_edges = []
            for u in curr_node:
                layer_edges += [(u, v) for v in col[rowptr[u]: rowptr[u + 1]]]
            L_hop_edges.append(layer_edges)
            curr_node = list(set([edge[1] for edge in layer_edges]))
        partition_L_hop_edges.append(L_hop_edges)

       # check
    for i, nodes in enumerate(partition_nodes):
        nodes = nodes.tolist()
        layer0_dst_nodes = set([u for u, v in partition_L_hop_edges[i][0]])
        assert len(layer0_dst_nodes) == len(nodes)
        assert layer0_dst_nodes == set(nodes)
    return partition_L_hop_edges


@show_time
def dgl_sample_L_hop_edges(graph, nodes, batch_size, fanout):
    sampler = dgl.dataloading.NeighborSampler(fanout)
    train_dataloader = dgl.dataloading.DataLoader(
        graph,  # The graph
        nodes,
        sampler,
        device=torch.device('cpu'),  # Put the sampled MFGs on CPU or GPU
        # The following arguments are inherited from PyTorch DataLoader.
        batch_size=batch_size,  # Batch size
        shuffle=True,  # Whether to shuffle the nodes for every epoch
        drop_last=False,  # Whether to drop the last incomplete batch
        num_workers=0,  # Number of sampler processes
        use_uva=False,
    )
    L_hop_edges = []
    for step, (input_nodes, output_nodes, mfgs) in enumerate(train_dataloader):
        input_nodes = input_nodes.tolist()
        output_nodes = output_nodes.tolist()
        assert input_nodes[:len(output_nodes)] == output_nodes
        batch_L_hop_edges = []
        check_pre_src_nodes = None
        check_pre_dst_nodes = None
        for h in range(len(fanout)):
            # print(mfgs, type(mfgs))
            mfg_src_nodes, mfg_dst_nodes = mfgs[h].edges()
            mfg_src_nodes, mfg_dst_nodes = mfg_src_nodes.tolist(), mfg_dst_nodes.tolist()
            # (dst, src)
            edges = [(input_nodes[u], input_nodes[v]) for u, v in zip(mfg_dst_nodes, mfg_src_nodes)]
            ##### check ######
            tmp_src = [input_nodes[v] for v in mfg_src_nodes]
            tmp_dst = [input_nodes[v] for v in mfg_dst_nodes]
            if check_pre_src_nodes:
                tmp_src += tmp_dst
                assert set(tmp_src) == set(check_pre_dst_nodes)
            check_pre_src_nodes = tmp_src
            check_pre_dst_nodes = tmp_dst
            ##################
            assert len(edges) == mfgs[h].num_edges()
            batch_L_hop_edges.append(edges)
        batch_L_hop_edges.reverse()

        #######################
        # node_dis = []
        # for i, hop in enumerate(batch_L_hop_edges):
        #     dst_ = [u for u, _ in hop]
        #     src_ = [v for _, v in hop]
        #     print('hop {} dst {} src {} edges {}'.format(i, len(set(dst_)), len(set(src_)), len(dst_)))
        #     if i > 0:
        #         if set(dst_) != set(node_dis[-1][1]) | set(node_dis[-1][0]):
        #             print('not equal:')
        #             print(set(dst_))
        #             print(set(node_dis[-1][1]) | set(node_dis[-1][0]))
        #         assert set(dst_) == set(node_dis[-1][1]) | set(node_dis[-1][0])

        #     node_dis.append((dst_, src_))
        #######################    
        L_hop_edges.append(batch_L_hop_edges)
    return L_hop_edges


@show_time
def get_dgl_L_hop_edges(graph, partition_nodes, batch_size, fanout):
    partition_L_hop_edges = []
    for i, nodes in enumerate(partition_nodes):
        if batch_size == -1:
            batch_size = nodes.shape[0]
        print('partition', i, 'batch num:', round(nodes.shape[0] / batch_size))
        dgl_L_hop_edges = dgl_sample_L_hop_edges(graph, nodes, batch_size, fanout)
        if len(dgl_L_hop_edges) == 1: # full batch
            partition_L_hop_edges.append(dgl_L_hop_edges[0])
        else:
            partition_L_hop_edges.append(dgl_L_hop_edges)
    return partition_L_hop_edges


@show_time
def get_all_edges(partition_edges):
    assert isinstance(partition_edges, list)
    # [[[],[]],[[],[]], ...] (L_hop_edges)
    if isinstance(partition_edges[0][0], list):
        return [edge for part in partition_edges for layer in part for edge in layer]
    # [[], [], ...] (partition_edges or one_partiton_L_hop)
    elif isinstance(partition_edges[0], list):
        return [edge for part in partition_edges for edge in part]
    else:
        return partition_edges


@show_time
def get_cross_partition_edges(partition_nodes, partition_edges):
    assert len(partition_nodes) == len(partition_edges)
    partition_num = len(partition_nodes)
    cross_partition_edges = []
    for partition_id in range(partition_num):
        part_nodes = set(partition_nodes[partition_id].tolist())
        part_edges = get_all_edges(partition_edges[partition_id])
        cross_mask = [
            0 if u in part_nodes and v in part_nodes else 1 for (u, v) in part_edges]
        assert len(cross_mask) == len(part_edges)
        cross_partition_edges.append((len(cross_mask), sum(cross_mask)))
    return cross_partition_edges


@show_time
def get_L_hop_cross_edges(partition_nodes, L_hop_edges):
    # return [[local edges, remote edges], ...]
    assert len(partition_nodes) == len(L_hop_edges)
    partition_num = len(partition_nodes)
    cross_partition_edges = []
    for partition_id in range(partition_num):
        # print(type(partition_nodes[partition_id]))
        # print(type(L_hop_edges[partition_id]))
        part_nodes = set(partition_nodes[partition_id].tolist())
        part_edges = get_all_edges(L_hop_edges[partition_id])
        cross_mask = [
            0 if u in part_nodes and v in part_nodes else 1 for (u, v) in part_edges]
        # cross_mask = [0 if u in part_nodes else 1 for (u,v) in part_edges]
        assert len(cross_mask) == len(part_edges)
        cross_partition_edges.append((len(cross_mask), sum(cross_mask)))
    return cross_partition_edges


def check_two_L_hop(partition_X, partition_Y):
    assert len(partition_Y) == len(partition_X)  # same partitions
    for partx, party in zip(partition_X, partition_Y):
        assert len(partx) == len(party)  # same layers
        for layerx, layery in zip(partx, party):
            assert set(layerx) == set(layery)


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
    partition_nodes = get_partition_nodes(parts)
    partition_train_nodes = get_partition_label_nodes(partition_nodes, train_mask)
    partition_val_nodes = get_partition_label_nodes(partition_nodes, val_mask)
    partition_test_nodes = get_partition_label_nodes(partition_nodes, test_mask)

    # 每个分区包含的边[[], []]
    partition_edges = get_partition_edges(partition_nodes, rowptr, col)
    print('metis partition nodes:', [len(_) for _ in partition_nodes])
    print('metis partition edges:', [len(_) for _ in partition_edges])
    show_label_distributed(parts, train_mask, val_mask, test_mask)

    return (partition_nodes, partition_edges, partition_train_nodes, partition_val_nodes, partition_test_nodes)


# TODO(sanzo): use scipy scs_adj to instead (rowptr, col)
# def in_neighbors(csc_adj, nid):
#   return csc_adj.indices[csc_adj.indptr[nid]: csc_adj.indptr[nid+1]]

def in_neighbors(rowptr, col, nid):
    return col[rowptr[nid]: rowptr[nid + 1]].tolist()


def in_neighbors_hop(rowptr, col, nid, hops):
    if hops == 1:
        return in_neighbors(rowptr, col, nid)
    else:
        nids = []
        for depth in range(hops):
            neighs = nids[-1] if len(nids) != 0 else [nid]
            # print('neighs', neighs)
            in_neighs = []
            for n in neighs:
                in_neighs += in_neighbors(rowptr, col, n)
            nids.append(in_neighs)
            # print('nids', nids)
        return np.unique(np.hstack(nids))



def pagraph_partition_score(rowptr, col, neighbors, belongs, p_vnum, r_vnum, pnum, train_nums):
    """
    Params:
      neighbor: in-neighbor vertex set
      belongs: np array, each vertex belongings to which partition
      p_vnum: np array, each partition total vertex w/o. redundancy
      r_vnum: np array, each partition total vertex w/. redundancy
      pnum: partition number
    """
    com_neighbor = np.ones(pnum, dtype=np.int64)
    score = np.zeros(pnum, dtype=np.float32)
    # count belonged vertex
    neighbor_belong = belongs[neighbors]
    belonged = neighbor_belong[np.where(neighbor_belong != -1)]
    pid, freq = np.unique(belonged, return_counts=True)
    com_neighbor[pid] += freq
    avg_num = train_nums / pnum  # need modify to match the train vertex num
    score = com_neighbor * (-p_vnum + avg_num) / (r_vnum + 1)
    return score


def partition_partition_max_score(score, p_vnum):
    ids = np.argsort(score)[-2:]
    if score[ids[0]] != score[ids[1]]:
        return ids[1]
    else:
        return ids[0] if p_vnum[ids[0]] < p_vnum[ids[1]] else ids[1]


def pagraph_partition(num_parts, hops, rowptr, col, train_mask):
    train_nids = torch.where(train_mask == 1)[0]
    assert train_nids.shape[0] == train_mask.sum()
    vnum = rowptr.shape[0] - 1
    vtrain_num = train_nids.shape[0]
    print('total vertices: {} | train vertices: {}'.format(vnum, vtrain_num))

    belongs = -np.ones(vnum, dtype=np.int8)
    r_belongs = [-np.ones(vnum, dtype=np.int8) for _ in range(num_parts)]
    p_vnum = np.zeros(num_parts, dtype=np.int64)
    r_vnum = np.zeros(num_parts, dtype=np.int64)

    progress = 0
    for step, nid in enumerate(train_nids):
        # neighbors = in_neighbors(csc_adj, nid)
        neighbors = in_neighbors_hop(rowptr, col, nid, hops)
        # print('\n############################')
        # print('nid', nid)
        # print('neighbor', neighbors, type(neighbors))
        # print('belongs', belongs)
        # print('p_vnum', p_vnum)
        # print('r_vnum', r_vnum)
        score = pagraph_partition_score(
            rowptr, col, neighbors, belongs, p_vnum, r_vnum, num_parts, train_nids.shape[0])
        # print(score)
        ind = partition_partition_max_score(score, p_vnum)
        # print('score', score)
        # print('ind', ind)
        # print('############################')
        if belongs[nid] == -1:
            belongs[nid] = ind
            p_vnum[ind] += 1
            neighbors = np.append(neighbors, nid)
            for neigh_nid in neighbors:
                if r_belongs[ind][neigh_nid] == -1:
                    r_belongs[ind][neigh_nid] = 1
                    r_vnum[ind] += 1

        # progress
        if int(vtrain_num * progress / 100) <= step:
            #   sys.stdout.write('=>{}%\r'.format(progress))
            sys.stdout.write('{}>{}%\r'.format('=' * progress, progress))
            sys.stdout.flush()
            progress += 1

    assert np.where(belongs != -1)[0].shape[0] == train_nids.shape[0]

    sub_v = []
    sub_trainv = []
    for pid in range(num_parts):
        p_trainids = np.where(belongs == pid)[0]
        sub_trainv.append(p_trainids)
        p_v = np.where(r_belongs[pid] != -1)[0]
        sub_v.append(p_v)
        assert p_v.shape[0] == r_vnum[pid]
        print('partition', pid, 'vertex# with self-reliance: ',
              r_vnum[pid], 'w/o  self-reliance: ', p_vnum[pid])
        # print('orginal vertex: ', np.where(belongs == pid)[0])
        # print('redundancy vertex: ', np.where(r_belongs[pid] != -1)[0])
    return sub_v, sub_trainv


@show_time
def pagraph_partition_graph(num_parts, num_hops, graph, rowptr, col, train_mask, val_mask, test_mask):
    print("\n######## pagraph_partition_graph #########")
    partition_nodes, partition_train_nodes = pagraph_partition(num_parts, num_hops, rowptr, col, train_mask)
    # partition_edges = get_partition_edges(partition_nodes, rowptr, col)
    train_L_hop_edges = get_dgl_L_hop_edges(graph, partition_train_nodes, -1, [-1] * num_hops)
    partition_edges = [list(set([edge for layer in L_hop for edge in layer ])) for L_hop in train_L_hop_edges]
    # partition_edges = []
    # for L_hop in train_L_hop_edges:
    #     tmp = []
    #     for layer in L_hop:
    #         tmp += layer
    #     partition_edges.append(list(set(tmp)))
    
    # assert partition_edges == partition_edges1

    print('pagraph partition nodes:', [len(_) for _ in partition_nodes])
    print('pagraph partition edges:', [len(_) for _ in partition_edges])
    print('pargraph: partition_train_nodes', [len(_) for _ in partition_train_nodes])
    return (partition_nodes, partition_edges, partition_train_nodes)


def statistic_info(partition_nodes, partition_edges, partition_train_nodes, rowptr, col):

    corss_partition_edges = get_cross_partition_edges(partition_nodes, partition_edges)
    metis_cross_prtitiion_edge_ratio = [round(
        remote_edges / local_edges, 2) for (local_edges, remote_edges) in corss_partition_edges]
    print('cross_partition_edge_ratio:', metis_cross_prtitiion_edge_ratio)

    # get L-hop full neighbor (dgl)
    partition_dgl_L_hop_edges = get_dgl_L_hop_edges(graph, partition_train_nodes, -1, [-1, -1])
    for i, L_hop in enumerate(partition_dgl_L_hop_edges):
        layer_edge_num = [len(layer) for layer in L_hop]
        print('(dgl) partition', i, 'layer edge num', layer_edge_num)
    print('partition_L_hop', [sum([len(hop_edges) for hop_edges in part]) for part in partition_dgl_L_hop_edges])

    # check dlg full neighbor sample
    # get L-hop full neighbor (csc)
    partition_L_hop_edges = get_L_hop_edges(partition_train_nodes, rowptr, col)
    # for i, L_hop in enumerate(partition_L_hop_edges):
    #     layer_edge_num = [len(layer) for layer in L_hop]
    #     print('(csc) partition', i, 'layer edge num', layer_edge_num)
    # print('partition_L_hopp', [sum([len(hop_edges) for hop_edges in part]) for part in partition_L_hop_edges])
    check_two_L_hop(partition_dgl_L_hop_edges, partition_L_hop_edges)

    # partition_bug(partition_dgl_L_hop_edges, partition_nodes, partition_train_nodes)
    # partition_bug(partition_L_hop_edges, partition_nodes, partition_train_nodes, rowptr, col)
    # assert False

    # train_L_hop remote edges ratio
    L_hop_cross_partition_edges = get_L_hop_cross_edges(partition_nodes, partition_dgl_L_hop_edges)
    L_hop_corss_prtitiion_edge_ratio = [round(remote_edges / local_edges, 2)
                                  for (local_edges, remote_edges) in L_hop_cross_partition_edges]
    print('L_hop_cross_partition_edges:(loal, remote)', L_hop_cross_partition_edges)
    print('L_hop_corss_prtitiion_edge_ratio', L_hop_corss_prtitiion_edge_ratio)


@show_time
def exp01_metis_pagraph_l_hop_cross_edges(num_parts, rowptr, col, train_mask, val_mask, test_mask):

    # metis return:  partition_nodes, partition_edges, partition_train_nodes, partition_val_nodes, partition_test_nodes
    print("\n############ metis node_dim1 (train) ############")
    node_dim1_result = metis_partition_graph(
        num_parts, rowptr, col, train_mask, val_mask, test_mask, node_weight_dim=1)
    statistic_info(node_dim1_result[0], node_dim1_result[1], node_dim1_result[2], rowptr, col)


    print("\n############ metis node_dim2 (train degree) ############")
    node_dim2_result = metis_partition_graph(
        num_parts, rowptr, col, train_mask, val_mask, test_mask, node_weight_dim=2)
    statistic_info(node_dim2_result[0], node_dim2_result[1], node_dim2_result[2], rowptr, col)    

    print("\n############ metis node_dim4 (train val test degrees) ############")
    node_dim4_result = metis_partition_graph(
        num_parts, rowptr, col, train_mask, val_mask, test_mask, node_weight_dim=4)
    statistic_info(node_dim4_result[0], node_dim4_result[1], node_dim4_result[2], rowptr, col)

    # pargraph return: partition_nodes, partition_edges, partition_train_nodes
    print("\n############ pagraph ############")
    pargraph_result = pagraph_partition_graph(
        num_parts, args.num_hops, graph, rowptr, col, train_mask, val_mask, test_mask)
    statistic_info(pargraph_result[0], pargraph_result[1], pargraph_result[2], rowptr, col)



def setup_seed(seed):
     print('setup_seed', seed)
     dgl.seed(seed)
     dgl.random.seed(seed)
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
    #  torch.backends.cudnn.benchmark = False

def dep_cache_stastistic_info(partition_nodes, partition_dgl_L_hop_edges, hops):
    print('\n############# dep_cache_stastistic_info ##############')
    # should move to outside
    
    # partition_dgl_L_hop_edges = get_dgl_L_hop_edges(graph, partition_train_nodes, batch_size, fanout)
    
    partition_nodes = [set(nodes.tolist()) for nodes in partition_nodes]
    num_parts = len(partition_nodes)
    local_sample_count = [0] * num_parts
    receive_sample_count = [0] * num_parts
    train_count = [0] * num_parts
    recv_comm_feature_count = [0] * num_parts
    recv_comm_sample_count = [0] * num_parts
    for partid, part_L_hop in enumerate(partition_dgl_L_hop_edges):
        for batchid, batch_graph in enumerate(part_L_hop):
            train_count[partid] += (sum([len(layer) for layer in batch_graph]))
            assert len(batch_graph) == hops
            for h in range(hops):
                edge_dst_nodes = [u for u, _ in batch_graph[h]]
                edge_src_nodes = [v for _, v in batch_graph[h]]
                dst_nodes = set(edge_dst_nodes)
                src_nodes = set(edge_src_nodes)

                if h == hops - 1: # last hop nodes
                    recv_comm_feature_count[partid] += len(src_nodes - partition_nodes[partid])

                # print('edges {} src_nodes {} dst_nodes {} {} {}'.format(len(edge_src_nodes), len(src_nodes), len(dst_nodes), max(src_nodes), max(dst_nodes)))

                # to csc
                data = np.ones(len(edge_src_nodes))
                csc_adj = spsp.csc_matrix((data, (edge_src_nodes, edge_dst_nodes)))
                indptr =  csc_adj.indptr
                indices =  csc_adj.indices
                ##### check #####
                # edges_c = [(dst, src) for dst in dst_nodes for src in indices[indptr[dst]:indptr[dst+1]]]
                # assert set(edges_c) == set(batch_graph[h]) and len(set(edges_c) - set(batch_graph[h])) == 0 and len(set(batch_graph[h]) - set(edges_c)) == 0
                ################

                # print('part {} batch {} hop {} edges {} dst_nodes {} src_nodes {}'.format(partid, batchid, h, len(batch_graph[h]), len(dst_nodes), len(src_nodes)))
                
                # sample load (local and receive)
                for pid, nodes in enumerate(partition_nodes):
                    common_nodes = dst_nodes & nodes
                    if pid == partid:
                        local_sample_count[pid] += sum([indptr[u + 1] - indptr[u] for u in common_nodes])
                    else:
                        receive_sample_count[pid] += sum([indptr[u + 1] - indptr[u] for u in common_nodes])
                        recv_comm_sample_count[partid] += sum([indptr[u + 1] - indptr[u] for u in common_nodes])


    print('train_count', train_count)
    print('local_sample_count', local_sample_count)
    print('receive_sample_count', receive_sample_count)
    print('recv_comm_sample_count', recv_comm_sample_count)
    print('recv_comm_feature_count', recv_comm_feature_count)
    


def dep_comm_stastistic_info(partition_nodes, partition_dgl_L_hop_edges, hops):
    print('\n############# dep_comm_stastistic_info ##############')
    # should move to outside
    partition_nodes = [set(nodes.tolist()) for nodes in partition_nodes]
    num_parts = len(partition_nodes)
    sample_edges = [{} for _ in range(hops)]
    local_sample_nodes = [[[] for _ in range(hops)] for _ in range(num_parts) ]
    receive_sample_nodes = [[[] for _ in range(hops)] for _ in range(num_parts) ]
    assert len(local_sample_nodes) == num_parts
    assert len(local_sample_nodes[0]) == hops
    for partid, part_L_hop in enumerate(partition_dgl_L_hop_edges):
        for batchid, batch_graph in enumerate(part_L_hop):
            assert len(batch_graph) == hops
            for h in range(hops):
                edge_dst_nodes = [u for u, _ in batch_graph[h]]
                edge_src_nodes = [v for _, v in batch_graph[h]]
                dst_nodes = set(edge_dst_nodes)
                src_nodes = set(edge_src_nodes)

                # print('edges {} src_nodes {} dst_nodes {} {} {}'.format(len(edge_src_nodes), len(src_nodes), len(dst_nodes), max(src_nodes), max(dst_nodes)))

                # to csc
                data = np.ones(len(edge_src_nodes))
                csc_adj = spsp.csc_matrix((data, (edge_src_nodes, edge_dst_nodes)))
                indptr =  csc_adj.indptr
                indices =  csc_adj.indices
                ##### check #####
                # edges_c = [(dst, src) for dst in dst_nodes for src in indices[indptr[dst]:indptr[dst+1]]]
                # assert set(edges_c) == set(batch_graph[h]) and len(set(edges_c) - set(batch_graph[h])) == 0 and len(set(batch_graph[h]) - set(edges_c)) == 0
                ################

                # print('part {} batch {} hop {} edges {} dst_nodes {} src_nodes {}'.format(partid, batchid, h, len(batch_graph[h]), len(dst_nodes), len(src_nodes)))
                
                # sample result
                for dst in dst_nodes:
                    if dst not in sample_edges[h]:
                        sample_edges[h][dst] = indices[indptr[dst] : indptr[dst + 1]]

                # sample load (local and receive)
                for pid, nodes in enumerate(partition_nodes):
                    common_nodes = dst_nodes & nodes
                    if pid == partid:
                        local_sample_nodes[pid][h] += list(common_nodes)
                    else:
                        receive_sample_nodes[pid][h] += list(common_nodes)


    cross_edges = [0] * num_parts
    local_edges = [0] * num_parts
    all_sample_count = [0] * num_parts
    recv_sample_count = [0] * num_parts
    assert len(local_sample_nodes) == num_parts
    assert len(receive_sample_nodes) == num_parts
    for partid in range(num_parts):
        assert len(local_sample_nodes[partid]) == hops
        assert len(receive_sample_nodes[partid]) == hops
        for h in range(hops):
            all_sample_nodes = set(local_sample_nodes[partid][h]) | set(receive_sample_nodes[partid][h])
            
            # cross edges
            for u in all_sample_nodes:
                assert u in sample_edges[h]
                # assert isinstance(sample_edges[h][u], list)
                for v in sample_edges[h][u]:
                    if v not in partition_nodes[partid]:
                        cross_edges[partid] += 1
                    else:
                        local_edges[partid] += 1
                
            # local sample load
            all_sample_count[partid] += sum([len(sample_edges[h][u]) for u in all_sample_nodes])

            # recv sample load
            recv_sample_nodes = set(receive_sample_nodes[partid][h]) - set(local_sample_nodes[partid][h])
            recv_sample_count[partid] += sum([len(sample_edges[h][u]) for u in recv_sample_nodes])


    print('local_edges', local_edges)
    print('cross_edges', cross_edges)
    print('all_sample_count', all_sample_count)
    print('receive_sample_count', recv_sample_count)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Dataset')
    parser.add_argument("--dataset", type=str, default="cora",
                        help="Dataset name (cora, citeseer, pubmed, reddit)")
    parser.add_argument("--self-loop", type=bool, default=True,
                        help="insert self-loop (default=True)")
    parser.add_argument(
        "--num_parts", help="Number of partitions to generate", type=int, required=True)
    parser.add_argument(
        "--num_hops", help="Number of layer in GNN for PaGraph partition", type=int, default=2)
    parser.add_argument(
        "--batch_size", help="batch size of gnn train", type=int, default=6000)
    parser.add_argument("--fanout",
                        help="Training fanouts",
                        type=int, default=[10, 25], nargs="*", required=False)

    args = parser.parse_args()
    print(args)
    # assert len(args.fanout) == args.num_hops

    # args.fanout.reverse() # or reversed(args.fanout)
    setup_seed(2000)

    # graph dataset
    edges_list, features, labels, train_mask, val_mask, test_mask, graph = extract_dataset(
        args)
    train_mask = torch.tensor(train_mask, dtype=torch.long)
    val_mask = torch.tensor(val_mask, dtype=torch.long)
    test_mask = torch.tensor(test_mask, dtype=torch.long)
    node_nums = graph.number_of_nodes()
    edge_nums = graph.number_of_edges()
    src_nodes = edges_list[:, 0].tolist()
    dst_nodes = edges_list[:, 1].tolist()
    edges = [(x, y) for x, y in zip(src_nodes, dst_nodes)]
    assert (len(edges) == edge_nums == len(set(edges)) == edges_list.shape[0])
    indices = torch.tensor([src_nodes, dst_nodes])
    rowptr, col, value = dglsp.spmatrix(indices).csc()


    # metis partition result
    partition_nodes, partition_edges, partition_train_nodes, partition_val_nodes, partition_test_nodes = metis_partition_graph(
        args.num_parts, rowptr, col, train_mask, val_mask, test_mask, node_weight_dim=4)

    # partition_nodes, partition_edges, partition_train_nodes = pagraph_partition_graph(
    #     args.num_parts, args.num_hops, graph, rowptr, col, train_mask, val_mask, test_mask)

    # generate_nts_dataset(partition_nodes, partition_edges, node_nums, features.size()[1])
    
    # statistic_info(partition_nodes, partition_edges, partition_train_nodes, rowptr, col)
    
    # 1. set_seed
    # 2. dep cache compute load
    #    - depcache_compute_sample() => local_sample_count, receive_sample_count
    #    - depcache_compute_train() => train sample graph edges
    # 2. dep cache comm load
    #    - depcache_comm_sample() => send_sample_count
    #    - depcache_comm_feature() => send_feature_count
    partition_dgl_L_hop_edges = get_dgl_L_hop_edges(graph, partition_train_nodes, args.batch_size, args.fanout)
    dep_cache_stastistic_info(partition_nodes, partition_dgl_L_hop_edges, len(args.fanout))
    dep_comm_stastistic_info(partition_nodes, partition_dgl_L_hop_edges, len(args.fanout))


    exit(1)
    exp01_metis_pagraph_l_hop_cross_edges(args.num_parts, rowptr, col, train_mask, val_mask, test_mask)

    