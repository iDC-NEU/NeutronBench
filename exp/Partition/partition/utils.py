import os
import sys
import time
import torch
import json
import dgl
import numpy as np
import networkx as nx
import scipy.sparse as sp
from functools import wraps
from dgl import DGLGraph
from dgl.data import load_data
from dgl.data import CoraFullDataset, CoauthorCSDataset, CoauthorPhysicsDataset
from dgl.data import AmazonCoBuyComputerDataset, AmazonCoBuyPhotoDataset
# from load_graph import load_ogb, load_reddit
from ogb.nodeproppred import DglNodePropPredDataset
import random
import psutil


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


def show_time(func):

    @wraps(func)
    def with_time(*args, **kwargs):
        time_cost = time.time()
        ret = func(*args, **kwargs)
        time_cost = time.time() - time_cost
        func_name = func.__name__
        if time_cost > 3:
            print("func {} is done, cost: {:.2f}s".format(
                func_name, time_cost))
        return ret

    return with_time


def get_partition_nodes(parts, num_parts):
    partition_nodes = []
    for i in range(num_parts):
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


def show_label_distributed(parts, train_mask, val_mask, test_mask):
    train_idx = torch.nonzero(train_mask).view(-1)
    val_idx = torch.nonzero(val_mask).view(-1)
    test_idx = torch.nonzero(test_mask).view(-1)
    print('train distributed:', torch.bincount(parts[train_idx]).tolist())
    print('val distributed:', torch.bincount(parts[val_idx]).tolist())
    print('test distributed:', torch.bincount(parts[test_idx]).tolist())


@show_time
def get_partition_edges_inner_nodes(partition_nodes,
                                    rowptr,
                                    col,
                                    edge_nums=None):  # 仅构造在partition nodes里面的边
    # (dst, src)
    rowptr = rowptr.tolist()
    col = col.tolist()
    partition_edges = []
    for nodes in partition_nodes:
        st_nodes = set(nodes.tolist())
        edge_list = []
        for u in nodes.tolist():
            assert u in st_nodes
            for v in col[rowptr[u]:rowptr[u + 1]]:
                if v in st_nodes:
                    edge_list.append((u, v))
                # else:
                #     print(u, v)
        partition_edges.append(edge_list)
    return partition_edges


@show_time
def get_partition_edges(partition_nodes, rowptr, col, edge_nums=None):
    # (dst, src)
    rowptr = rowptr.tolist()
    col = col.tolist()
    partition_edges = []
    for nodes in partition_nodes:
        edge_list = []
        for u in nodes:
            edge_list += [(u.item(), v) for v in col[rowptr[u]:rowptr[u + 1]]]
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
def get_all_edges(partition_edges):
    assert isinstance(partition_edges, list)
    # [[[],[]],[[],[]], ...] (L_hop_edges)
    if isinstance(partition_edges[0][0], list):
        return [
            edge for part in partition_edges for layer in part
            for edge in layer
        ]
    # [[], [], ...] (partition_edges or one_partiton_L_hop)
    elif isinstance(partition_edges[0], list):
        return [edge for part in partition_edges for edge in part]
    else:
        return partition_edges


def extract_dataset(args):
    dataset = args.dataset

    # # change dir
    # if not os.path.exists(dataset):
    #     os.mkdir(dataset)
    # os.chdir(dataset)

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

        # if reddit_small or args.split:
        #     graph, features, labels, train_mask, val_mask, test_mask = split_graph(
        #         graph, n_nodes, n_edges, features, labels, train_mask, val_mask, test_mask, args.frac)

        if args.self_loop:
            time_stamp = time.time()
            print('before add self loop has {} edges'.format(
                graph.num_edges()))
            graph = dgl.remove_self_loop(graph)
            graph = dgl.add_self_loop(graph)
            # graph = dgl.to_bidirected(graph) # simple graph
            print('after add self loop has {} edges'.format(graph.num_edges()))
            print("insert self loop cost {:.2f}s".format(time.time() -
                                                         time_stamp))

        edges = graph.edges()
        edge_src = edges[0].numpy().reshape((-1, 1))
        edge_dst = edges[1].numpy().reshape((-1, 1))
        edges_list = np.hstack((edge_src, edge_dst))

        print(
            "nodes: {}, edges: {}, feature dims: {}, classess: {}, label nodes: {}({}/{}/{})"
            .format(graph.number_of_nodes(), edges_list.shape,
                    list(features.shape), len(np.unique(labels)),
                    train_mask.sum() + test_mask.sum() + val_mask.sum(),
                    train_mask.sum(), val_mask.sum(), test_mask.sum()))
        return edges_list, features, labels, train_mask, val_mask, test_mask, graph

    elif dataset in [
            'CoraFull', 'Coauthor_cs', 'Coauthor_physics',
            'AmazonCoBuy_computers', 'AmazonCoBuy_photo', 'computer'
    ]:
        if dataset == 'CoraFull':
            data = CoraFullDataset()
        elif dataset == 'Coauthor_cs':
            data = CoauthorCSDataset('cs')
        elif dataset == 'Coauthor_physics':
            data = CoauthorPhysicsDataset('physics')
        elif dataset == 'AmazonCoBuy_computers' or dataset == 'computer':
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
            print("insert self loop cost {:.2f}s".format(time.time() -
                                                         time_stamp))

        edges = graph.edges()
        edge_src = edges[0].numpy().reshape((-1, 1))
        edge_dst = edges[1].numpy().reshape((-1, 1))
        edges_list = np.hstack((edge_src, edge_dst))

        train_mask, val_mask, test_mask = split_dataset(num_nodes, 6, 3, 1)
        graph.ndata['train_mask'] = train_mask
        graph.ndata['val_mask'] = val_mask
        graph.ndata['test_mask'] = test_mask

        print(
            "dataset: {} nodes: {} edges: {} feature dims: {} classess: {} label nodes: {}({}/{}/{})"
            .format(dataset, num_nodes, edges_list.shape, list(features.shape),
                    len(np.unique(labels)),
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
        train_nid, val_nid, test_nid = split_idx['train'], split_idx[
            'valid'], split_idx['test']
        # print(len(train_nid) + len(val_nid) + len(test_nid))

        train_mask = torch.zeros(graph.number_of_nodes(), dtype=bool)
        train_mask[train_nid] = True
        val_mask = torch.zeros(graph.number_of_nodes(), dtype=bool)
        val_mask[val_nid] = True
        test_mask = torch.zeros(graph.number_of_nodes(), dtype=bool)
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
            print("insert self loop cost {:.2f}s".format(time.time() -
                                                         time_stamp))
        graph.ndata['feat'] = features
        graph.ndata['label'] = labels.reshape(-1)
        graph.ndata['train_mask'] = train_mask
        graph.ndata['val_mask'] = val_mask
        graph.ndata['test_mask'] = test_mask

        edges = graph.edges()
        edge_src = edges[0].numpy().reshape((-1, 1))
        edge_dst = edges[1].numpy().reshape((-1, 1))
        edges_list = np.hstack((edge_src, edge_dst))

        print(
            "nodes: {}, edges: {}, feature dims: {}, classess: {}, label nodes: {}({}/{}/{})"
            .format(graph.number_of_nodes(), edges_list.shape,
                    list(features.shape), len(np.unique(labels)),
                    train_mask.sum() + test_mask.sum() + val_mask.sum(),
                    train_mask.sum(), val_mask.sum(), test_mask.sum()))
        return edges_list, features, labels, train_mask, val_mask, test_mask, graph

    elif dataset in ['flickr', 'yelp', 'ppi', 'ppi-large', 'amazon']:
        # prefix = os.getcwd()
        curr_dir = os.getcwd()
        print(os.chdir(f'/home/yuanh/neutron-sanzo/data/{dataset}'))
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
            print("insert self loop cost {:.2f}s".format(time.time() -
                                                         time_stamp))

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
                class_arr[k][v - offset] = 1
            labels = np.where(class_arr)[1]

        print(
            "nodes: {}, edges: {}, feature dims: {}, classess: {}{}, label nodes: {}({}/{}/{})"
            .format(num_nodes, num_edges, feats.shape, num_classes,
                    '#' if is_multiclass else '',
                    train_mask.sum() + test_mask.sum() + val_mask.sum(),
                    train_mask.sum(), val_mask.sum(), test_mask.sum()))

        print(os.chdir(curr_dir))
        return edges_list, feats, labels, train_mask, val_mask, test_mask, graph

    else:
        raise NotImplementedError


def split_graph(graph, n_nodes, n_edges, features, labels, train_mask,
                val_mask, test_mask, fraction):
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
    # mask = np.zeros(l, dtype=bool)
    mask = torch.zeros(l, dtype=torch.bool)
    mask[idx] = True
    return mask


def split_dataset(num_nodes, x=8, y=1, z=1):
    '''
    x: train nodes, y: val nodes, z: test nodes
    '''
    train_mask = torch.tensor([False for i in range(num_nodes)],
                              dtype=torch.bool)
    val_mask = torch.tensor([False for i in range(num_nodes)],
                            dtype=torch.bool)
    test_mask = torch.tensor([False for i in range(num_nodes)],
                             dtype=torch.bool)
    step = int(num_nodes / (x + y + z))
    train_mask[:int(x * step)] = True
    val_mask[int(x * step):int((x + y) * step)] = True
    test_mask[int((x + y) * step):] = True
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
    train_mask[node_ids[:train_num]] = True
    val_mask[node_ids[train_num:train_num + val_num]] = True
    test_mask[node_ids[train_num + val_num:]] = True
    return train_mask, val_mask, test_mask


def generate_nts_dataset(dataset_name, partition_nodes, partition_edges,
                         node_num, feature_dim, train_mask, val_mask,
                         test_mask):
    print('\ngenerate nts for', dataset_name, 'node_num:', node_num,
          'feature_dim:', feature_dim)
    reorder_id = {}
    curr_id = 0
    for nodes in partition_nodes:
        for u in nodes.tolist():
            reorder_id[u] = curr_id
            curr_id += 1
    assert len(reorder_id) == curr_id == sum(
        [len(nodes) for nodes in partition_nodes])

    reorder_partition_edges = [[(reorder_id[u], reorder_id[v])
                                for (u, v) in edges]
                               for edges in partition_edges]

    assert len(reorder_partition_edges) == len(partition_edges)
    for (edges, reorder_edges) in zip(partition_edges,
                                      reorder_partition_edges):
        assert len(edges) == len(reorder_edges)
        for e1, e2 in zip(edges, reorder_edges):
            assert reorder_id[e1[0]] == e2[0] and reorder_id[e1[1]] == e2[1]

    # (dst,src) to (src,dst)
    # whole dataset file
    all_edges = get_all_edges(reorder_partition_edges)
    swap_edges = torch.tensor(all_edges)[:, [1, 0]].contiguous()
    edge2bin(f'./{dataset_name}/reorder.edge.self', swap_edges)

    # split dataset file
    for partid, edges in enumerate(reorder_partition_edges):
        swap_edges = torch.tensor(edges)[:, [1, 0]].contiguous()
        edge2bin(f'./{dataset_name}/split_edge/{partid}.edge.self', swap_edges)

    # partition info
    with open(f'./{dataset_name}/split_edge/partition.info', 'w') as f:
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
    write_to_mask(f'./{dataset_name}/reorder.mask', reorder_train_mask,
                  reorder_val_mask, reorder_test_mask)

    # write faeture
    out_features = torch.ones((node_num, feature_dim))
    write_to_file(f'./{dataset_name}/reorder.featuretable',
                  out_features,
                  '%.2f',
                  index=True)

    # write label
    out_features = torch.ones(node_num)
    write_to_file(f'./{dataset_name}/reorder.labeltable',
                  out_features,
                  '%.0f',
                  index=True)


def read_edgelist_from_file(filename):
    assert os.path.exists(filename)
    edgelist = []
    with open(filename) as f:
        # for line in f.readlines():
        # print(line.strip('\n').split(' '), type(line.strip('\n')))
        # for u in line.split(' '):
        #     print(u, u)

        # edgelist = [(int(u), int(v)) for line in f.readlines() for u,v in line.strip('\n').split(' ')]
        edgelist = [
            tuple(int(x) for x in line.strip('\n').split(' '))
            for line in f.readlines()
        ]
    return edgelist


def create_dir(path=None):
    if path and not os.path.exists(path):
        os.makedirs(path)


@show_time
def edge2bin(name, edges):
    edges = edges.flatten()
    with open(name, 'wb') as f:
        buf = [
            int(edge).to_bytes(4, byteorder=sys.byteorder) for edge in edges
        ]
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
        out_file = open(name + '.temp', 'w')
        for i, line in enumerate(in_file):
            out_file.write(str(i) + ' ' + line)
        in_file.close()
        out_file.close()
        os.remove(name)
        os.rename(name + '.temp', name)


@show_time
def write_multi_class_to_file(name, data, format, index=False):
    with open(name, 'w') as f:
        for i, line in enumerate(data):
            line.insert(0, len(line))
            line.insert(0, i)
            # f.write(str(len(line)) + ' ' +  ' '.join(str(x) for x in line) + '\n')
            f.write(' '.join(str(x) for x in line) + '\n')


def get_partition_result(parts,
                         rowptr,
                         col,
                         num_parts,
                         train_mask,
                         val_mask,
                         test_mask,
                         algo='metis'):
    print('\n####get_partition_result of', algo)
    # 每个分区node, train_nodes, val_nodes, test_nodes
    partition_nodes = get_partition_nodes(parts, num_parts)
    partition_train_nodes = get_partition_label_nodes(partition_nodes,
                                                      train_mask)
    partition_val_nodes = get_partition_label_nodes(partition_nodes, val_mask)
    partition_test_nodes = get_partition_label_nodes(partition_nodes,
                                                     test_mask)

    # 每个分区包含的边[[], []]
    partition_edges = get_partition_edges(partition_train_nodes, rowptr, col)
    # partition_edges = get_partition_edges(partition_nodes, rowptr, col)
    print(f'{algo} partition nodes:', [len(_) for _ in partition_nodes])
    print(f'{algo} partition edges:', [len(_) for _ in partition_edges])
    show_label_distributed(parts, train_mask, val_mask, test_mask)
    return (partition_nodes, partition_edges, partition_train_nodes,
            partition_val_nodes, partition_test_nodes)


def get_pagraph_partition_result(partition_nodes, rowptr, col, num_parts):
    print('\n####get_partition_result of pagraph')
    # 每个分区node, train_nodes, val_nodes, test_nodes
    # 每个分区包含的边[[], []]
    # partition_edges = get_partition_edges(partition_nodes, rowptr, col)
    partition_edges = get_partition_edges_inner_nodes(partition_nodes, rowptr,
                                                      col)
    print('pagraph partition edges:', [len(_) for _ in partition_edges])
    return partition_edges


def get_ram_usage():
    # Getting % usage of virtual_memory ( 3rd field)
    print('RAM memory used {:.1f}%, {:.2f} (GB)'.format(
        psutil.virtual_memory()[2],
        psutil.virtual_memory()[3] / 1000000000))

    # # Getting all memory using os.popen()
    # total_memory, used_memory, free_memory = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
    # # Memory usage
    # print("RAM memory {:.1f}% used".format(round((used_memory/total_memory) * 100, 2)))


def get_cpu_usage(interval=1):
    print('The CPU usage is: {:.1f}%'.format(psutil.cpu_percent(interval)))
    # # Getting loadover15 minutes
    # load1, load5, load15 = psutil.getloadavg()
    # cpu_usage15 = (load15 / os.cpu_count()) * 100
    # cpu_usage5 = (load5 / os.cpu_count()) * 100
    # cpu_usage1 = (load1 / os.cpu_count()) * 100
    # print("The CPU usage is : {:.1f}% {:.1f}% {:.1f}% (avg 1, 5, 15 min)".format(cpu_usage1, cpu_usage5, cpu_usage15))


if __name__ == '__main__':
    get_cpu_usage(interval=1)
    get_ram_usage()
