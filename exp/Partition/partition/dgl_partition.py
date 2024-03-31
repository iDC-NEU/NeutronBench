import torch
import numpy as np
import os
import sys

sys.path.append('..')
import random
from partition.utils import show_time

import dgl


def get_homogeneous(g, balance_ntypes):
    assert g.is_homogeneous
    if g.is_homogeneous:
        sim_g = dgl.to_homogeneous(g)
        if isinstance(balance_ntypes, dict):
            assert len(balance_ntypes) == 1
            bal_ntypes = list(balance_ntypes.values())[0]
        else:
            bal_ntypes = balance_ntypes
    # elif isinstance(balance_ntypes, dict):
    #     # Here we assign node types for load balancing.
    #     # The new node types includes the ones provided by users.
    #     num_ntypes = 0
    #     for key in g.ntypes:
    #         if key in balance_ntypes:
    #             g.nodes[key].data["bal_ntype"] = (
    #                 F.astype(balance_ntypes[key], F.int32) + num_ntypes
    #             )
    #             uniq_ntypes = F.unique(balance_ntypes[key])
    #             assert np.all(
    #                 F.asnumpy(uniq_ntypes) == np.arange(len(uniq_ntypes))
    #             )
    #             num_ntypes += len(uniq_ntypes)
    #         else:
    #             g.nodes[key].data["bal_ntype"] = (
    #                 F.ones((g.num_nodes(key),), F.int32, F.cpu())
    #                 * num_ntypes
    #             )
    #             num_ntypes += 1
    #     sim_g = dgl.to_homogeneous(g, ndata=["bal_ntype"])
    #     bal_ntypes = sim_g.ndata["bal_ntype"]
    #     print(
    #         "The graph has {} node types and balance among {} types".format(
    #             len(g.ntypes), len(F.unique(bal_ntypes))
    #         )
    #     )
    #     # We now no longer need them.
    #     for key in g.ntypes:
    #         del g.nodes[key].data["bal_ntype"]
    #     del sim_g.ndata["bal_ntype"]
    # else:
    #     sim_g = dgl.to_homogeneous(g)
    #     bal_ntypes = sim_g.ndata[dgl.NTYPE]
    return sim_g, bal_ntypes


@show_time
def dgl_partition_graph(dataset,
                        num_parts,
                        graph,
                        train_mask,
                        val_mask,
                        test_mask,
                        save_dir='./partition_result'):
    assert os.path.exists(save_dir), f'save_dir {save_dir} not exist!'
    save_path = f'{save_dir}/dgl-{dataset}-part{num_parts}.pt'
    if os.path.exists(save_path):
        print(f'read from partition result {save_path}.')
        parts = torch.load(save_path)
    else:
        sim_g, balance_ntypes = get_homogeneous(graph, train_mask)
        parts = dgl.metis_partition_assignment(
            sim_g,
            num_parts,
            balance_ntypes=balance_ntypes,
            balance_edges=True,
            objtype="cut",
        )
        torch.save(parts, save_path)
        print(f'save partition result to {save_path}.')
    return parts


if __name__ == '__main__':
    pass
