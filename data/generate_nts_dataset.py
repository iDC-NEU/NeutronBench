# -*- coding: utf-8 -*-
# @Author   : Sanzo00
# @Email    : arrangeman@163.com
# @Time     : 2022/06/17 09:00

import os
import sys
import argparse
import numpy as np
import networkx as nx
import time
import torch
import torch.nn.functional as F
import json
import numpy as np
import scipy.sparse as sp
import dgl
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from dgl.data import CoraFullDataset, CoauthorCSDataset, CoauthorPhysicsDataset
from dgl.data import AmazonCoBuyComputerDataset, AmazonCoBuyPhotoDataset
from ogb.nodeproppred import DglNodePropPredDataset


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
            graph, features, labels, train_mask, val_mask, test_mask = split_graph(graph, n_nodes, n_edges, features, labels, train_mask, val_mask, test_mask, args.frac)

        if args.self_loop:
            time_stamp = time.time()
            print('before add self loop has {} edges'.format(graph.num_edges()))
            graph = dgl.remove_self_loop(graph)
            graph = dgl.add_self_loop(graph)
            # graph = dgl.to_bidirected(graph) # simple graph
            print('after add self loop has {} edges'.format(graph.num_edges()))
            print("insert self loop cost {:.2f}s".format(time.time() - time_stamp))

        edges = graph.edges()
        edge_src = edges[0].numpy().reshape((-1,1))
        edge_dst = edges[1].numpy().reshape((-1,1))
        edges_list = np.hstack((edge_src, edge_dst))

        print("nodes: {}, edges: {}, feature dims: {}, classess: {}, label nodes: {}({}/{}/{})"
              .format(graph.number_of_nodes(), edges_list.shape, 
              list(features.shape), len(np.unique(labels)),
              train_mask.sum() + test_mask.sum() + val_mask.sum(),
              train_mask.sum(), val_mask.sum(), test_mask.sum()))
        return edges_list, features, labels, train_mask, val_mask, test_mask

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
            print('before add self loop has {} edges'.format(len(graph.all_edges()[0])))
            graph = dgl.remove_self_loop(graph)
            graph = dgl.add_self_loop(graph)
            # graph = dgl.to_bidirected(graph)
            print('after add self loop has {} edges'.format(len(graph.all_edges()[0])))
            print("insert self loop cost {:.2f}s".format(time.time() - time_stamp))

        edges = graph.edges()
        edge_src = edges[0].numpy().reshape((-1,1))
        edge_dst = edges[1].numpy().reshape((-1,1))
        edges_list = np.hstack((edge_src, edge_dst))

        train_mask, val_mask, test_mask = split_dataset(num_nodes, 6, 3, 1)
        print("dataset: {} nodes: {} edges: {} feature dims: {} classess: {} label nodes: {}({}/{}/{})"
              .format(dataset, num_nodes, edges_list.shape, 
              list(features.shape), len(np.unique(labels)),
              train_mask.sum() + test_mask.sum() + val_mask.sum(),
              train_mask.sum(), val_mask.sum(), test_mask.sum()))
        return edges_list, features, labels, train_mask, val_mask, test_mask
    
    elif dataset in ['ogbn-arxiv', 'ogbn-papers100M', 'ogbn-products']:
        #load dataset
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
            print('before add self loop has {} edges'.format(len(graph.all_edges()[0])))
            graph = dgl.remove_self_loop(graph)
            graph = dgl.add_self_loop(graph)
            if dataset == "ogbn-arxiv":
                graph = dgl.to_bidirected(graph)
            print('after add self loop has {} edges'.format(len(graph.all_edges()[0])))
            print("insert self loop cost {:.2f}s".format(time.time() - time_stamp))
        
        edges = graph.edges()
        edge_src = edges[0].numpy().reshape((-1,1))
        edge_dst = edges[1].numpy().reshape((-1,1))
        edges_list = np.hstack((edge_src, edge_dst))

        print("nodes: {}, edges: {}, feature dims: {}, classess: {}, label nodes: {}({}/{}/{})"
              .format(graph.number_of_nodes(), edges_list.shape, 
              list(features.shape), len(np.unique(labels)),
              train_mask.sum() + test_mask.sum() + val_mask.sum(),
              train_mask.sum(), val_mask.sum(), test_mask.sum()))
        return edges_list, features, labels, train_mask, val_mask, test_mask

    elif dataset in ['flickr', 'yelp', 'ppi', 'ppi-large', 'amazon']:
        # prefix = os.getcwd()
        adj_full = sp.load_npz('./adj_full.npz').astype(bool)
        # adj_train = sp.load_npz('./adj_train.npz').astype(bool)
        role = json.load(open('./role.json'))
        feats = np.load('./feats.npy')
        class_map = json.load(open('./class_map.json'))
        class_map = {int(k):v for k,v in class_map.items()}
        assert len(class_map) == feats.shape[0]
        edges = adj_full.nonzero()
        graph = dgl.graph((edges[0], edges[1]))
        # print(graph)
        if args.self_loop:
            time_stamp = time.time()
            print('before add self loop has {} edges'.format(len(graph.all_edges()[0])))
            graph = dgl.remove_self_loop(graph)
            graph = dgl.add_self_loop(graph)
            print('after add self loop has {} edges'.format(len(graph.all_edges()[0])))
            print("insert self loop cost {:.2f}s".format(time.time() - time_stamp))
        
        edges = graph.edges()
        edge_src = edges[0].numpy().reshape((-1,1))
        edge_dst = edges[1].numpy().reshape((-1,1))
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
        return edges_list, feats, labels, train_mask, val_mask, test_mask

    else:
        raise NotImplementedError


def split_graph(graph, n_nodes, n_edges, features, labels, train_mask, val_mask, test_mask, fraction):
    new_n_nodes = int(n_nodes * fraction)
    #check_type(graph, n_nodes, n_edges, features, labels, train_mask, val_mask, test_mask, fraction)
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
  train_mask = torch.tensor([False for i in range(num_nodes)], dtype=torch.bool)
  val_mask = torch.tensor([False for i in range(num_nodes)], dtype=torch.bool)
  test_mask = torch.tensor([False for i in range(num_nodes)], dtype=torch.bool)
  step = int(num_nodes / (x + y + z))
  train_mask[ : int(x * step)] = True
  val_mask[int(x * step) : int((x+y) * step)] = True
  test_mask[int((x+y) * step) : ] = True
  assert(train_mask.sum() +  val_mask.sum() + test_mask.sum() == num_nodes)
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
    train_mask[node_ids[ : train_num]] = True
    val_mask[node_ids[train_num : train_num + val_num]] = True
    test_mask[node_ids[train_num + val_num:]] = True
    return train_mask, val_mask, test_mask


def generate_nts_dataset(args, edge_list, features, labels, train_mask, val_mask, test_mask):
    dataset = args.dataset
    pre_path = os.getcwd() + '/' + dataset

    # edgelist
    # write_to_file(pre_path + '.edgelist', edge_list, "%d")

    # edge_list binary format (gemini)
    edge2bin(pre_path + '.edge', edge_list)

    # fetures
    write_to_file(pre_path + '.feat', features, "%.4f", index=True)

    # label
    if dataset in ['yelp', 'ppi', 'ppi-large', 'amazon']:
        write_multi_class_to_file(pre_path + '.label', labels, "%d", index=True)
    else:
        write_to_file(pre_path + '.label', labels, "%d", index=True)

    # mask
    mask_list = []
    for i in range(len(labels)):
        if train_mask[i] == True:
            mask_list.append('train')
        elif val_mask[i] == True:
            mask_list.append('val')
        elif test_mask[i] == True:
            mask_list.append('test')
        else:
            mask_list.append('unknown')
    write_to_mask(pre_path + '.mask', mask_list)


def show_time(func):
    def with_time(*args, **kwargs):
        time_cost = time.time()
        func(*args, **kwargs)
        time_cost = time.time() - time_cost
        name = args[0]
        print("write to {} is done, cost: {:.2f}s Throughput:{:.2f}MB/s".format(name, time_cost, os.path.getsize(name)/1024/1024/time_cost))
    return with_time


@show_time
def edge2bin(name, edges):
    edges = edges.flatten()
    with open(name, 'wb') as f:
        buf = [int(edge).to_bytes(4, byteorder=sys.byteorder) for edge in edges]
        f.writelines(buf)


@show_time
def write_to_mask(name, data):
    with open(name, 'w') as f:
        for i, node_type in enumerate(data):
            f.write(str(i) + ' ' + node_type + '\n')


@show_time
def write_to_file(name, data, format, index=False):
    if not type(data) is np.ndarray:
        data = data.numpy()
    np.savetxt(name, data, fmt=format)
    
    if index:
        in_file= open(name, 'r') 
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


# def insert_self_loop(edge_list):
#     time_stamp = time.time()
#     graph = nx.from_edgelist(edge_list)
#     print("graph {:.3f}".format(time.time() - time_stamp))

#     time_stamp = time.time()
#     graph = graph.to_directed()
#     print("to directed {:.3f}".format(time.time() - time_stamp))

#     print('before insert self loop', graph)    
#     for i in graph.nodes:
#         graph.add_edge(i, i)
#     print('after insert self loop', graph)    
#     return np.array(graph.edges)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Dataset')
    parser.add_argument("--dataset", type=str, default="cora", help="Dataset name (cora, citeseer, pubmed, reddit)")
    parser.add_argument("--self-loop", type=bool, default=True, help="insert self-loop (default=True)")
    parser.add_argument("--split", type=bool, default=False, help="wheather or not split the graph")
    parser.add_argument("--frac", type=float, default=0.8, help="the fraction of split the graph")
    parser.add_argument("--remask", type=str, default=None, help="re-mark of the graph")

    args = parser.parse_args()
    
    
    print('args: ', args)
    edges_list, features, labels, train_mask, val_mask, test_mask = extract_dataset(args)
    if args.remask is not None:
        args.remask = list(map(float, args.remask.split(',')))
        train_mask, val_mask, test_mask = remask(len(train_mask), args.remask)
    generate_nts_dataset(args, edges_list, features, labels, train_mask, val_mask, test_mask)