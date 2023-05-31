import torch
import numpy as np
import sys
sys.path.append('..')
from partition.utils import show_time


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

