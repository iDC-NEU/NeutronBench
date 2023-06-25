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
from partition.bytegnn_partition import bytegnn_partition_graph
from partition.bytegnn_partition import read_bytegnn_partition_result


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
    full_batch = True if batch_size == -1 else False
    for i, nodes in enumerate(partition_nodes):
        if full_batch:
            batch_size = nodes.shape[0]
        # print('partition', i, 'batch num:', round(nodes.shape[0] / batch_size))
        dgl_L_hop_edges = dgl_sample_L_hop_edges(graph, nodes, batch_size, fanout)
        if len(dgl_L_hop_edges) == 1: # full batch
            # print(len(dgl_L_hop_edges), dgl_L_hop_edges[0][0])
            # partition_L_hop_edges.append(dgl_L_hop_edges[0])
            # partition_L_hop_edges.append(dgl_L_hop_edges[0])
            partition_L_hop_edges.append(dgl_L_hop_edges)
        else:
            partition_L_hop_edges.append(dgl_L_hop_edges)
    return partition_L_hop_edges




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


def statistic_info(graph, partition_nodes, partition_edges, partition_train_nodes, rowptr, col, batch_size, fanout):

    corss_partition_edges = get_cross_partition_edges(partition_nodes, partition_edges)
    cross_partitiion_edge_ratio = [round(remote_edges / local_edges, 2) for (local_edges, remote_edges) in corss_partition_edges]
    cross_partitiion_edge = [remote_edges for (_, remote_edges) in corss_partition_edges]
    local_partitiion_edge = [local_edges for (local_edges, _) in corss_partition_edges]
    print('cross_partition_edge_ratio:', cross_partitiion_edge_ratio)
    print('train cross_partition_edge:', cross_partitiion_edge)
    print('train local_partition_edge:', local_partitiion_edge)

    # get L-hop full neighbor (dgl)
    partition_dgl_L_hop_edges = get_dgl_L_hop_edges(graph, partition_train_nodes, -1, [-1, -1])
    # partition_dgl_1_hop_edges = get_dgl_L_hop_edges(graph, partition_train_nodes, -1, [-1])
    # print(len(partition_dgl_1_hop_edges), len(partition_dgl_1_hop_edges[0]))
    print('1-hop edges', [len(part[0][0]) for part in partition_dgl_L_hop_edges])

    # for i, part_edges in enumerate(partition_dgl_L_hop_edges):
    #     st_nodes = set(partition_nodes[i].tolist())
    #     print(len(st_nodes), type(st_nodes))
    #     part_edges = part_edges[0]
    #     for h in range(len(fanout)):
    #         for edge in part_edges[h]:
    #             if edge[0] not in st_nodes or edge[1] not in st_nodes:
    #                 print(edge)
    #             assert edge[0] in st_nodes and edge[1] in st_nodes

    # for _ in partition_dgl_L_hop_edges:
    #     assert len(_) == 1
    # partition_dgl_L_hop_edges = [_[0] for _ in partition_dgl_L_hop_edges]
    # print('partition_L_hop', [sum([len(hop_edges) for hop_edges in part]) for part in partition_dgl_L_hop_edges])
    # # check dlg full neighbor sample
    # # get L-hop full neighbor (csc)
    # partition_L_hop_edges = get_L_hop_edges(partition_train_nodes, rowptr, col)
    # check_two_L_hop(partition_dgl_L_hop_edges, partition_L_hop_edges)

    # partition_bug(partition_dgl_L_hop_edges, partition_nodes, partition_train_nodes)
    # partition_bug(partition_L_hop_edges, partition_nodes, partition_train_nodes, rowptr, col)
    # assert False

    # train_L_hop remote edges ratio
    L_hop_cross_partition_edges = get_L_hop_cross_edges(partition_nodes, partition_dgl_L_hop_edges)
    L_hop_corss_prtitiion_edge_ratio = [round(remote_edges / local_edges, 2)
                                  for (local_edges, remote_edges) in L_hop_cross_partition_edges]
    print('L_hop_cross_partition_edges:(loal, remote)', L_hop_cross_partition_edges)
    print('L_hop_corss_prtitiion_edge_ratio', L_hop_corss_prtitiion_edge_ratio)


    partition_dgl_L_hop_edges = get_dgl_L_hop_edges(graph, partition_train_nodes, batch_size, fanout)
    dep_cache_statistics_info(partition_nodes, partition_dgl_L_hop_edges, len(args.fanout))
    # dep_comm_statistics_info(partition_nodes, partition_dgl_L_hop_edges, len(args.fanout))


@show_time
def exp01_metis_pagraph_l_hop_cross_edges(dataset, graph, num_parts, batch_size, fanout, rowptr, col, train_mask, val_mask, test_mask):

    # metis return:  partition_nodes, partition_edges, partition_train_nodes, partition_val_nodes, partition_test_nodes
    print("\n############ metis node_dim1 (train) ############")
    
    parts = metis_partition_graph(dataset, num_parts, rowptr, col, train_mask, val_mask, test_mask, node_weight_dim=1)
    node_dim1_result = get_partition_result(parts, rowptr, col, num_parts, train_mask, val_mask, test_mask)    
    statistic_info(graph, node_dim1_result[0], node_dim1_result[1], node_dim1_result[2], rowptr, col, batch_size, fanout)

    print("\n############ metis node_dim2 (train degree) ############")
    
    parts = metis_partition_graph(dataset, num_parts, rowptr, col, train_mask, val_mask, test_mask, node_weight_dim=2)
    node_dim2_result = get_partition_result(parts, rowptr, col, num_parts, train_mask, val_mask, test_mask)    
    statistic_info(graph, node_dim2_result[0], node_dim2_result[1], node_dim2_result[2], rowptr, col, batch_size, fanout)    

    print("\n############ metis node_dim4 (train val test degrees) ############")
    
    parts = metis_partition_graph(dataset, num_parts, rowptr, col, train_mask, val_mask, test_mask, node_weight_dim=4)
    node_dim4_result = get_partition_result(parts, rowptr, col, num_parts, train_mask, val_mask, test_mask)    
    statistic_info(graph, node_dim4_result[0], node_dim4_result[1], node_dim4_result[2], rowptr, col, batch_size, fanout)

    # pargraph return: partition_nodes, partition_edges, partition_train_nodes
    # print("\n############ pagraph ############")
    # pargraph_result = pagraph_partition_graph(
    #     num_parts, args.num_hops, graph, rowptr, col, train_mask, val_mask, test_mask)
    # statistic_info(graph, pargraph_result[0], pargraph_result[1], pargraph_result[2], rowptr, col, batch_size, fanout)


# def dep_cache_statistics_info(partition_nodes, partition_dgl_L_hop_edges, hops):
#     print('\n############# dep_cache_statistic_info ##############')
#     partition_nodes = [set(nodes.tolist()) for nodes in partition_nodes]
#     num_parts = len(partition_nodes)
#     local_sample_count = [0] * num_parts
#     receive_sample_count = [0] * num_parts
#     train_count = [0] * num_parts
#     recv_comm_feature_count = [0] * num_parts
#     recv_comm_sample_count = [0] * num_parts
#     for partid, part_L_hop in enumerate(partition_dgl_L_hop_edges):
#         for batchid, batch_graph in enumerate(part_L_hop):
#             train_count[partid] += (sum([len(layer) for layer in batch_graph]))
#             assert len(batch_graph) == hops
#             for h in range(hops):
#                 edge_dst_nodes = [u for u, _ in batch_graph[h]]
#                 edge_src_nodes = [v for _, v in batch_graph[h]]
#                 dst_nodes = set(edge_dst_nodes)
#                 src_nodes = set(edge_src_nodes)

#                 if h == hops - 1: # last hop nodes
#                     recv_comm_feature_count[partid] += len(src_nodes - partition_nodes[partid])

#                 # print('edges {} src_nodes {} dst_nodes {} {} {}'.format(len(edge_src_nodes), len(src_nodes), len(dst_nodes), max(src_nodes), max(dst_nodes)))

#                 # to csc
#                 data = np.ones(len(edge_src_nodes))
#                 csc_adj = spsp.csc_matrix((data, (edge_src_nodes, edge_dst_nodes)))
#                 indptr =  csc_adj.indptr
#                 indices =  csc_adj.indices
#                 ##### check #####
#                 # edges_c = [(dst, src) for dst in dst_nodes for src in indices[indptr[dst]:indptr[dst+1]]]
#                 # assert set(edges_c) == set(batch_graph[h]) and len(set(edges_c) - set(batch_graph[h])) == 0 and len(set(batch_graph[h]) - set(edges_c)) == 0
#                 ################

#                 # print('part {} batch {} hop {} edges {} dst_nodes {} src_nodes {}'.format(partid, batchid, h, len(batch_graph[h]), len(dst_nodes), len(src_nodes)))
                
#                 # sample load (local and receive)
#                 for pid, nodes in enumerate(partition_nodes):
#                     common_nodes = dst_nodes & nodes
#                     if pid == partid:
#                         local_sample_count[pid] += sum([indptr[u + 1] - indptr[u] for u in common_nodes])
#                     else:
#                         receive_sample_count[pid] += sum([indptr[u + 1] - indptr[u] for u in common_nodes])
#                         recv_comm_sample_count[partid] += sum([indptr[u + 1] - indptr[u] for u in common_nodes])


#     print('train_count', train_count)
#     print('local_sample_count', local_sample_count)
#     print('receive_sample_count', receive_sample_count)
#     print('recv_comm_sample_count', recv_comm_sample_count)
#     print('recv_comm_feature_count', recv_comm_feature_count)
    



def dep_cache_statistics_info(partition_nodes, partition_dgl_L_hop_edges, hops):
    print('\n############# dep_cache_statistic_info ##############')
    partition_nodes = [set(nodes.tolist()) for nodes in partition_nodes]
    num_parts = len(partition_nodes)
    # comm_graph
    # comm_feature
    # local_sample_edges
    # recv_sample_edges
    # local_sample_nodes
    # recv_sample_nodes
    # duplicate_sample_node


    batch_num = max([len(part) for part in partition_dgl_L_hop_edges])
    print([len(part) for part in partition_dgl_L_hop_edges])
    print('max_batch_num', batch_num)

    epoch_local_sample_edges = []
    epoch_remote_sample_edges = []
    epoch_recv_sample_edges = []
    epoch_send_edges = []
    epoch_send_features = []
    epoch_sen_edges_bytes = []
    epoch_sen_features_bytes = []

    for batchid in range(batch_num):# 统计相同batchid，模拟真实训练
        sample_edge_num = [{} for _ in range(hops)]
        local_sample_nodes = [[[] for _ in range(hops)] for _ in range(num_parts) ]
        remote_sample_nodes = [[[] for _ in range(hops)] for _ in range(num_parts) ]
        receive_sample_nodes = [[[] for _ in range(hops)] for _ in range(num_parts) ]

        for partid, part_L_hop in enumerate(partition_dgl_L_hop_edges):
            if len(part_L_hop) <= batchid:
                continue
            batch_graph = part_L_hop[batchid] # 获取每个分区上第batchid个子图，模拟真实训练
            for h in range(hops): # 训练子图上的每一跳
                edge_dst_nodes = [u for u, _ in batch_graph[h]]
                edge_src_nodes = [v for _, v in batch_graph[h]]
                dst_nodes = set(edge_dst_nodes)
                src_nodes = set(edge_src_nodes)

                data = np.ones(len(edge_src_nodes))
                csc_adj = spsp.csc_matrix((data, (edge_src_nodes, edge_dst_nodes)))
                indptr =  csc_adj.indptr
                indices =  csc_adj.indices

                # count sample edges num of every node in diff hop
                for dst in dst_nodes: # 统计每个节点在不同层的采样节点数量
                    if dst not in sample_edge_num[h]:
                        sample_edge_num[h][dst] = indptr[dst + 1] - indptr[dst]
                        sample_edge_num[h][dst] = indptr[dst + 1] - indptr[dst]

                # sample nodes of every hop (local and receive)
                local_nodes = partition_nodes[partid]
                common_nodes = dst_nodes & local_nodes
                local_sample_nodes[partid][h] += list(common_nodes)
                remote_sample_nodes[partid][h] += list(dst_nodes - local_nodes)

                if len(common_nodes) == len(dst_nodes):
                    print(f'hop {h} cache all l-hop neighbors')
                else:
                    for pid, nodes in enumerate(partition_nodes):
                        if pid == partid:
                            continue
                        common_nodes = dst_nodes & nodes
                        receive_sample_nodes[pid][h] += list(common_nodes)
        # comm_graph
        # comm_feature
        # local_sample_edges
        # recv_sample_edges
        # local_sample_nodes
        # remote_sample_nodes
        # recv_sample_nodes
        # duplicate_sample_node

        # count numbers
        batch_local_sample_nodes = []
        batch_remote_sample_nodes = []
        batch_recv_sample_nodes = []
        batch_local_sample_edges = []
        batch_remote_sample_edges = []
        batch_recv_sample_edges = []
        unique_sample_nodes = [] #TODO(Sanzo): not correct, shouble count for every hop
        batch_send_edges_num = [0] * num_parts
        batch_send_features_num = [0] * num_parts

        for partid in range(num_parts):
            assert len(local_sample_nodes[partid]) == hops
            assert len(receive_sample_nodes[partid]) == hops
            batch_local_sample_nodes.append(sum([len(x) for x in local_sample_nodes[partid]]))
            batch_remote_sample_nodes.append(sum([len(x) for x in remote_sample_nodes[partid]]))
            batch_recv_sample_nodes.append(sum([len(x) for x in receive_sample_nodes[partid]]))

            st = set()
            for h in range(hops):
                st.update(local_sample_nodes[partid][h])
                st.update(receive_sample_nodes[partid][h])
            unique_sample_nodes.append(len(st))

            count = 0
            for h, nids in enumerate(local_sample_nodes[partid]):
                count += sum([sample_edge_num[h][dst] for dst in nids])
            batch_local_sample_edges.append(count)


            count = 0
            for h, nids in enumerate(remote_sample_nodes[partid]):
                count += sum([sample_edge_num[h][dst] for dst in nids])
            batch_remote_sample_edges.append(count)


            count = 0
            for h, nids in enumerate(receive_sample_nodes[partid]):
                count += sum([sample_edge_num[h][dst] for dst in nids])
            batch_recv_sample_edges.append(count)

        for partid in range(num_parts):
            for h in range(hops):
                for dst in receive_sample_nodes[partid][h]:
                    batch_send_edges_num[partid] += sample_edge_num[h][dst]
                    if h == hops - 1: # last hop
                        batch_send_features_num[partid] += sample_edge_num[h][dst]



        print('batch', batchid)
        print('local_sample_nodes', batch_local_sample_nodes)
        print('remote_sample_nodes', batch_remote_sample_nodes)
        print('recv_sample_nodes', batch_recv_sample_nodes)
        print('local_sample_edges', batch_local_sample_edges)
        print('remote_sample_edges', batch_remote_sample_edges)
        print('recv_sample_edges', batch_recv_sample_edges)
        print('unique_sample_nodes', unique_sample_nodes)
        print('send_edges_num', batch_send_edges_num)
        print('send_features_num', batch_send_features_num)

        epoch_local_sample_edges.append(batch_local_sample_edges)
        epoch_remote_sample_edges.append(batch_remote_sample_edges)
        epoch_recv_sample_edges.append(batch_recv_sample_edges)
        epoch_send_edges.append(batch_send_edges_num)
        epoch_send_features.append(batch_send_features_num)
        epoch_sen_edges_bytes.append([x * 2 * 4 for x in batch_send_edges_num])
        epoch_sen_features_bytes.append([x * feature_dim * 4 for x in batch_send_features_num])
        print('feature_dim', feature_dim)



    print('#### averge')
    print('avg_local_sample_edges', np.average(epoch_local_sample_edges, axis=0).tolist())
    print('avg_remote_sample_edges', np.average(epoch_remote_sample_edges, axis=0).tolist())
    print('avg_recv_sample_edges', np.average(epoch_recv_sample_edges, axis=0).tolist())
    print('avg_send_edges', np.average(epoch_send_edges, axis=0).tolist())
    print('avg_send_features', np.average(epoch_send_features, axis=0).tolist())
    print('avg_sen_edges_bytes', np.average(epoch_sen_edges_bytes, axis=0).tolist())
    print('avg_sen_features_bytes', np.average(epoch_sen_features_bytes, axis=0).tolist())


    print('#### sum')
    print('sum_local_sample_edges', np.sum(epoch_local_sample_edges, axis=0).tolist())
    print('sum_remote_sample_edges', np.sum(epoch_remote_sample_edges, axis=0).tolist())
    print('sum_recv_sample_edges', np.sum(epoch_recv_sample_edges, axis=0).tolist())
    print('sum_send_edges', np.sum(epoch_send_edges, axis=0).tolist())
    print('sum_send_features', np.sum(epoch_send_features, axis=0).tolist())              
    print('sum_sen_edges_bytes', np.sum(epoch_sen_edges_bytes, axis=0).tolist())
    print('sum_sen_features_bytes', np.sum(epoch_sen_features_bytes, axis=0).tolist())                    



# TODO(sanzo): optim: use big batch sample
def dep_comm_statistics_info(partition_nodes, partition_dgl_L_hop_edges, hops):
    print('\n############# dep_comm_statistic_info ##############')
    # for one epoch

    # should move to outside
    partition_nodes = [set(nodes.tolist()) for nodes in partition_nodes]
    num_parts = len(partition_nodes)
    

    batch_num = max([len(part) for part in partition_dgl_L_hop_edges])
    print([len(part) for part in partition_dgl_L_hop_edges])
    print('max_batch_num', batch_num)


    epoch_cross_edges = []
    epoch_local_edges = []
    epoch_all_sample_count = []
    epoch_recv_sample_count = []

    for batchid in range(batch_num):
        sample_edges = [{} for _ in range(hops)]
        local_sample_nodes = [[[] for _ in range(hops)] for _ in range(num_parts) ]
        receive_sample_nodes = [[[] for _ in range(hops)] for _ in range(num_parts) ]

        for partid, part_L_hop in enumerate(partition_dgl_L_hop_edges):
            if len(part_L_hop) <= batchid:
                continue
            batch_graph = part_L_hop[batchid]
            for h in range(hops): 
                edge_dst_nodes = [u for u, _ in batch_graph[h]]
                edge_src_nodes = [v for _, v in batch_graph[h]]
                dst_nodes = set(edge_dst_nodes)
                src_nodes = set(edge_src_nodes)

                data = np.ones(len(edge_src_nodes))
                csc_adj = spsp.csc_matrix((data, (edge_src_nodes, edge_dst_nodes)))
                indptr =  csc_adj.indptr
                indices =  csc_adj.indices

                # unique sample result of every hop
                for dst in dst_nodes:
                    if dst not in sample_edges[h]:
                        sample_edges[h][dst] = indices[indptr[dst] : indptr[dst + 1]]

                # sample nodes of every hop (local and receive)
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
        for partid in range(num_parts):
            for h in range(hops):
                all_sample_nodes = set(local_sample_nodes[partid][h]) | set(receive_sample_nodes[partid][h])
                # print('partion {} hop {} all_sample_nodes {}'.format(partid, h, len(all_sample_nodes)))
                # cross edges
                for u in all_sample_nodes:
                    assert u in sample_edges[h]
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
                            
        print('batch', batchid)
        print('local_edges', local_edges)
        print('cross_edges', cross_edges)
        print('all_sample_edges', all_sample_count)
        print('receive_sample_edges', recv_sample_count)

        epoch_cross_edges.append(cross_edges)
        epoch_local_edges.append(local_edges)
        epoch_all_sample_count.append(all_sample_count)
        epoch_recv_sample_count.append(recv_sample_count)

    # print('#### all')
    # print('local_edges', epoch_local_edges)
    # print('cross_edges', epoch_cross_edges)
    # print('all_sample_count', epoch_all_sample_count)
    # print('receive_sample_count', epoch_recv_sample_count)

    print('#### averge')
    print('avg_local_edges', np.average(epoch_local_edges, axis=0).tolist())
    print('avg_cross_edges', np.average(epoch_cross_edges, axis=0).tolist())
    print('avg_all_sample_edges', np.average(epoch_all_sample_count, axis=0).tolist())
    print('avg_receive_sample_edges', np.average(epoch_recv_sample_count, axis=0).tolist())


    print('#### sum')
    print('sum_local_edges', np.sum(epoch_local_edges, axis=0).tolist())
    print('sum_cross_edges', np.sum(epoch_cross_edges, axis=0).tolist())
    print('sum_all_sample_edges', np.sum(epoch_all_sample_count, axis=0).tolist())
    print('sum_receive_sample_edges', np.sum(epoch_recv_sample_count, axis=0).tolist())

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Dataset')
    parser.add_argument("--dataset", type=str, default="cora", help="Dataset name (cora, citeseer, pubmed, reddit)")
    parser.add_argument("--self-loop", type=bool, default=True, help="insert self-loop (default=True)")
    parser.add_argument("--num_parts", help="Number of partitions to generate", type=int, required=True)
    parser.add_argument("--num_hops", help="Number of layer in GNN for PaGraph partition", type=int, default=2)
    parser.add_argument("--batch_size", help="batch size of gnn train", type=int, default=6000)
    parser.add_argument("--fanout", help="Training fanouts", type=int, default=[10, 25], nargs="*", required=False)
    parser.add_argument("--dim", help="metis dims", type=int, required=True)
    parser.add_argument("--algo", help="partition algorithm (metis, pagraph, hash)", type=str, required=True)
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

    # exp01_metis_pagraph_l_hop_cross_edges(args.dataset, graph, args.num_parts, args.batch_size, args.fanout, rowptr, col, train_mask, val_mask, test_mask)
    # exit(1)
    # metis partition result
    if args.algo == 'metis':
        parts = metis_partition_graph(args.dataset, args.num_parts, rowptr, col, train_mask, val_mask, test_mask, node_weight_dim=args.dim)
        partition_nodes, partition_edges, partition_train_nodes, partition_val_nodes, partition_test_nodes = get_partition_result(parts, rowptr, col, args.num_parts, train_mask, val_mask, test_mask)
        statistic_info(graph, partition_nodes, partition_edges, partition_train_nodes, rowptr, col, args.batch_size, args.fanout)
    elif args.algo == 'pagraph':
        partition_nodes, partition_train_nodes = pagraph_partition_graph(args.dataset, args.num_parts, args.num_hops, graph, rowptr, col, train_mask, val_mask, test_mask)
        print('pagraph partition nodes:', [len(_) for _ in partition_nodes])
        print('pagraph train distributed:', [len(_) for _ in partition_train_nodes])
        partition_edges = get_pagraph_partition_result(partition_nodes, rowptr, col, args.num_parts)
        statistic_info(graph, partition_nodes, partition_edges, partition_train_nodes, rowptr, col, args.batch_size, args.fanout)
    elif args.algo == 'hash':
        parts = hash_partition_graph(args.dataset, args.num_parts, node_nums)
        partition_nodes, partition_edges, partition_train_nodes, partition_val_nodes, partition_test_nodes = get_partition_result(parts, rowptr, col, args.num_parts, train_mask, val_mask, test_mask, args.algo)
        statistic_info(graph, partition_nodes, partition_edges, partition_train_nodes, rowptr, col, args.batch_size, args.fanout)
    elif args.algo == 'bytegnn':
        # parts = bytegnn_partition_graph(args.dataset, args.num_parts, args.num_hops, rowptr, col, train_mask, val_mask, test_mask, alpha=1.0, beta=1.0, gamma=1.0)
        parts =  read_bytegnn_partition_result(args.dataset, node_nums)
        partition_nodes, partition_edges, partition_train_nodes, partition_val_nodes, partition_test_nodes = get_partition_result(parts, rowptr, col, args.num_parts, train_mask, val_mask, test_mask, args.algo)
        print('all_partition_nodes', sum([len(x) for x in partition_nodes]))
        statistic_info(graph, partition_nodes, partition_edges, partition_train_nodes, rowptr, col, args.batch_size, args.fanout)
    else:
        raise NotImplementedError
    # generate_nts_dataset(args.dataset, partition_nodes, partition_edges, node_nums, features.size()[1], train_mask, val_mask, test_mask)
    
    
    
    # partition_dgl_L_hop_edges = get_dgl_L_hop_edges(graph, partition_train_nodes, args.batch_size, args.fanout)
    # dep_cache_statistics_info(partition_nodes, partition_dgl_L_hop_edges, len(args.fanout))
    # dep_comm_statistics_info(partition_nodes, partition_dgl_L_hop_edges, len(args.fanout))
    