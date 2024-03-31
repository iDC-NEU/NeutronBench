"""Functions for partitions. """

import json
import os
import time
import numpy as np
# from .metis_partition import metis_partition_graph
import dgl.backend as F
from dgl.base import NID, EID, NTYPE, ETYPE, DGLError
from dgl.convert import to_homogeneous
from dgl.random import choice as random_choice
from dgl.transforms import sort_csr_by_tag, sort_csc_by_tag
from dgl.data.utils import load_graphs, save_graphs, load_tensors, save_tensors
from dgl.partition import (
    metis_partition_assignment,
    partition_graph_with_halo,
    get_peak_mem,
)
from dgl.distributed.constants import DEFAULT_ETYPE, DEFAULT_NTYPE
from dgl.distributed.graph_partition_book import (
    RangePartitionBook,
    _etype_tuple_to_str,
    _etype_str_to_tuple,
)

RESERVED_FIELD_DTYPE = {
    'inner_node': F.uint8,    # A flag indicates whether the node is inside a partition.
    'inner_edge': F.uint8,    # A flag indicates whether the edge is inside a partition.
    NID: F.int64,
    EID: F.int64,
    NTYPE: F.int16,
    # `sort_csr_by_tag` and `sort_csc_by_tag` works on int32/64 only.
    ETYPE: F.int32
    }

def _format_part_metadata(part_metadata, formatter):
    '''Format etypes with specified formatter.
    '''
    for key in ['edge_map', 'etypes']:
        if key not in part_metadata:
            continue
        orig_data = part_metadata[key]
        if not isinstance(orig_data, dict):
            continue
        new_data = {}
        for etype, data in orig_data.items():
            etype = formatter(etype)
            new_data[etype] = data
        part_metadata[key] = new_data
    return part_metadata

def _load_part_config(part_config):
    '''Load part config and format.
    '''
    try:
        with open(part_config) as f:
            part_metadata = _format_part_metadata(json.load(f),
                _etype_str_to_tuple)
    except AssertionError as e:
        raise DGLError(f"Failed to load partition config due to {e}. "
            "Probably caused by outdated config. If so, please refer to "
            "https://github.com/dmlc/dgl/tree/master/tools#change-edge-"
            "type-to-canonical-edge-type-for-partition-configuration-json")
    return part_metadata

def _dump_part_config(part_config, part_metadata):
    '''Format and dump part config.
    '''
    part_metadata = _format_part_metadata(part_metadata, _etype_tuple_to_str)
    with open(part_config, 'w') as outfile:
        json.dump(part_metadata, outfile, sort_keys=True, indent=4)

def _save_graphs(filename, g_list, formats=None, sort_etypes=False):
    '''Preprocess partitions before saving:
    1. format data types.
    2. sort csc/csr by tag.
    '''
    for g in g_list:
        for k, dtype in RESERVED_FIELD_DTYPE.items():
            if k in g.ndata:
                g.ndata[k] = F.astype(g.ndata[k], dtype)
            if k in g.edata:
                g.edata[k] = F.astype(g.edata[k], dtype)
    for g in g_list:
        if (not sort_etypes) or (formats is None):
            continue
        if 'csr' in formats:
            g = sort_csr_by_tag(g, tag=g.edata[ETYPE], tag_type='edge')
        if 'csc' in formats:
            g = sort_csc_by_tag(g, tag=g.edata[ETYPE], tag_type='edge')
    save_graphs(filename , g_list, formats=formats)

def _get_inner_node_mask(graph, ntype_id):
    if NTYPE in graph.ndata:
        dtype = F.dtype(graph.ndata['inner_node'])
        return graph.ndata['inner_node'] * F.astype(graph.ndata[NTYPE] == ntype_id, dtype) == 1
    else:
        return graph.ndata['inner_node'] == 1

def _get_inner_edge_mask(graph, etype_id):
    if ETYPE in graph.edata:
        dtype = F.dtype(graph.edata['inner_edge'])
        return graph.edata['inner_edge'] * F.astype(graph.edata[ETYPE] == etype_id, dtype) == 1
    else:
        return graph.edata['inner_edge'] == 1

def _get_part_ranges(id_ranges):
    res = {}
    for key in id_ranges:
        # Normally, each element has two values that represent the starting ID and the ending ID
        # of the ID range in a partition.
        # If not, the data is probably still in the old format, in which only the ending ID is
        # stored. We need to convert it to the format we expect.
        if not isinstance(id_ranges[key][0], list):
            start = 0
            for i, end in enumerate(id_ranges[key]):
                id_ranges[key][i] = [start, end]
                start = end
        res[key] = np.concatenate([np.array(l) for l in id_ranges[key]]).reshape(-1, 2)
    return res

def load_partition(part_config, part_id, load_feats=True):
    ''' Load data of a partition from the data path.

    A partition data includes a graph structure of the partition, a dict of node tensors,
    a dict of edge tensors and some metadata. The partition may contain the HALO nodes,
    which are the nodes replicated from other partitions. However, the dict of node tensors
    only contains the node data that belongs to the local partition. Similarly, edge tensors
    only contains the edge data that belongs to the local partition. The metadata include
    the information of the global graph (not the local partition), which includes the number
    of nodes, the number of edges as well as the node assignment of the global graph.

    The function currently loads data through the local filesystem interface.

    Parameters
    ----------
    part_config : str
        The path of the partition config file.
    part_id : int
        The partition ID.
    load_feats : bool, optional
        Whether to load node/edge feats. If False, the returned node/edge feature
        dictionaries will be empty. Default: True.

    Returns
    -------
    DGLGraph
        The graph partition structure.
    Dict[str, Tensor]
        Node features.
    Dict[(str, str, str), Tensor]
        Edge features.
    GraphPartitionBook
        The graph partition information.
    str
        The graph name
    List[str]
        The node types
    List[(str, str, str)]
        The edge types
    '''
    config_path = os.path.dirname(part_config)
    relative_to_config = lambda path: os.path.join(config_path, path)

    with open(part_config) as conf_f:
        part_metadata = json.load(conf_f)
    assert 'part-{}'.format(part_id) in part_metadata, "part-{} does not exist".format(part_id)
    part_files = part_metadata['part-{}'.format(part_id)]
    assert 'part_graph' in part_files, "the partition does not contain graph structure."
    graph = load_graphs(relative_to_config(part_files['part_graph']))[0][0]

    assert NID in graph.ndata, "the partition graph should contain node mapping to global node ID"
    assert EID in graph.edata, "the partition graph should contain edge mapping to global edge ID"

    gpb, graph_name, ntypes, etypes = load_partition_book(part_config, part_id)
    ntypes_list = list(ntypes.keys())
    etypes_list = list(etypes.keys())
    if 'DGL_DIST_DEBUG' in os.environ:
        for ntype in ntypes:
            ntype_id = ntypes[ntype]
            # graph.ndata[NID] are global homogeneous node IDs.
            nids = F.boolean_mask(graph.ndata[NID], _get_inner_node_mask(graph, ntype_id))
            partids1 = gpb.nid2partid(nids)
            _, per_type_nids = gpb.map_to_per_ntype(nids)
            partids2 = gpb.nid2partid(per_type_nids, ntype)
            assert np.all(F.asnumpy(partids1 == part_id)), \
                'Unexpected partition IDs are found in the loaded partition ' \
                'while querying via global homogeneous node IDs.'
            assert np.all(F.asnumpy(partids2 == part_id)), \
                'Unexpected partition IDs are found in the loaded partition ' \
                'while querying via type-wise node IDs.'
        for etype in etypes:
            etype_id = etypes[etype]
            # graph.edata[EID] are global homogeneous edge IDs.
            eids = F.boolean_mask(graph.edata[EID], _get_inner_edge_mask(graph, etype_id))
            partids1 = gpb.eid2partid(eids)
            _, per_type_eids = gpb.map_to_per_etype(eids)
            partids2 = gpb.eid2partid(per_type_eids, etype)
            assert np.all(F.asnumpy(partids1 == part_id)), \
                'Unexpected partition IDs are found in the loaded partition ' \
                'while querying via global homogeneous edge IDs.'
            assert np.all(F.asnumpy(partids2 == part_id)), \
                'Unexpected partition IDs are found in the loaded partition ' \
                'while querying via type-wise edge IDs.'

    node_feats = {}
    edge_feats = {}
    if load_feats:
        node_feats, edge_feats = load_partition_feats(part_config, part_id)

    return graph, node_feats, edge_feats, gpb, graph_name, ntypes_list, etypes_list

def load_partition_feats(part_config, part_id, load_nodes=True, load_edges=True):
    '''Load node/edge feature data from a partition.

    Parameters
    ----------
    part_config : str
        The path of the partition config file.
    part_id : int
        The partition ID.
    load_nodes : bool, optional
        Whether to load node features. If ``False``, ``None`` is returned.
    load_edges : bool, optional
        Whether to load edge features. If ``False``, ``None`` is returned.

    Returns
    -------
    Dict[str, Tensor] or None
        Node features.
    Dict[str, Tensor] or None
        Edge features.
    '''
    config_path = os.path.dirname(part_config)
    relative_to_config = lambda path: os.path.join(config_path, path)

    with open(part_config) as conf_f:
        part_metadata = json.load(conf_f)
    assert 'part-{}'.format(part_id) in part_metadata, "part-{} does not exist".format(part_id)
    part_files = part_metadata['part-{}'.format(part_id)]
    assert 'node_feats' in part_files, "the partition does not contain node features."
    assert 'edge_feats' in part_files, "the partition does not contain edge feature."
    node_feats = None
    if load_nodes:
        node_feats = load_tensors(relative_to_config(part_files['node_feats']))
    edge_feats = None
    if load_edges:
        edge_feats = load_tensors(relative_to_config(part_files['edge_feats']))
    # In the old format, the feature name doesn't contain node/edge type.
    # For compatibility, let's add node/edge types to the feature names.
    if node_feats is not None:
        new_feats = {}
        for name in node_feats:
            feat = node_feats[name]
            if name.find('/') == -1:
                name = DEFAULT_NTYPE + '/' + name
            new_feats[name] = feat
        node_feats = new_feats
    if edge_feats is not None:
        new_feats = {}
        for name in edge_feats:
            feat = edge_feats[name]
            if name.find('/') == -1:
                name = _etype_tuple_to_str(DEFAULT_ETYPE) + '/' + name
            new_feats[name] = feat
        edge_feats = new_feats

    return node_feats, edge_feats

def load_partition_book(part_config, part_id):
    '''Load a graph partition book from the partition config file.

    Parameters
    ----------
    part_config : str
        The path of the partition config file.
    part_id : int
        The partition ID.

    Returns
    -------
    GraphPartitionBook
        The global partition information.
    str
        The graph name
    dict
        The node types
    dict
        The edge types
    '''
    part_metadata = _load_part_config(part_config)
    assert 'num_parts' in part_metadata, 'num_parts does not exist.'
    assert part_metadata['num_parts'] > part_id, \
            'part {} is out of range (#parts: {})'.format(part_id, part_metadata['num_parts'])
    num_parts = part_metadata['num_parts']
    assert 'num_nodes' in part_metadata, "cannot get the number of nodes of the global graph."
    assert 'num_edges' in part_metadata, "cannot get the number of edges of the global graph."
    assert 'node_map' in part_metadata, "cannot get the node map."
    assert 'edge_map' in part_metadata, "cannot get the edge map."
    assert 'graph_name' in part_metadata, "cannot get the graph name"

    # If this is a range partitioning, node_map actually stores a list, whose elements
    # indicate the boundary of range partitioning. Otherwise, node_map stores a filename
    # that contains node map in a NumPy array.
    node_map = part_metadata['node_map']
    edge_map = part_metadata['edge_map']
    if isinstance(node_map, dict):
        for key in node_map:
            is_range_part = isinstance(node_map[key], list)
            break
    elif isinstance(node_map, list):
        is_range_part = True
        node_map = {DEFAULT_NTYPE: node_map}
    else:
        is_range_part = False
    if isinstance(edge_map, list):
        edge_map = {DEFAULT_ETYPE: edge_map}

    ntypes = {DEFAULT_NTYPE: 0}
    etypes = {DEFAULT_ETYPE: 0}
    if 'ntypes' in part_metadata:
        ntypes = part_metadata['ntypes']
    if 'etypes' in part_metadata:
        etypes = part_metadata['etypes']

    if isinstance(node_map, dict):
        for key in node_map:
            assert key in ntypes, 'The node type {} is invalid'.format(key)
    if isinstance(edge_map, dict):
        for key in edge_map:
            assert key in etypes, 'The edge type {} is invalid'.format(key)

    if not is_range_part:
        raise TypeError("Only RangePartitionBook is supported currently.")

    node_map = _get_part_ranges(node_map)
    edge_map = _get_part_ranges(edge_map)
    return RangePartitionBook(part_id, num_parts, node_map, edge_map, ntypes, etypes), \
            part_metadata['graph_name'], ntypes, etypes

def _get_orig_ids(g, sim_g, orig_nids, orig_eids):
    '''Convert/construct the original node IDs and edge IDs.

    It handles multiple cases:
     * If the graph has been reshuffled and it's a homogeneous graph, we just return
       the original node IDs and edge IDs in the inputs.
     * If the graph has been reshuffled and it's a heterogeneous graph, we need to
       split the original node IDs and edge IDs in the inputs based on the node types
       and edge types.
     * If the graph is not shuffled, the original node IDs and edge IDs don't change.

    Parameters
    ----------
    g : DGLGraph
       The input graph for partitioning.
    sim_g : DGLGraph
        The homogeneous version of the input graph.
    orig_nids : tensor or None
        The original node IDs after the input graph is reshuffled.
    orig_eids : tensor or None
        The original edge IDs after the input graph is reshuffled.

    Returns
    -------
    tensor or dict of tensors, tensor or dict of tensors
    '''
    is_hetero = not g.is_homogeneous
    if is_hetero:
       # Get the type IDs
        orig_ntype = F.gather_row(sim_g.ndata[NTYPE], orig_nids)
        orig_etype = F.gather_row(sim_g.edata[ETYPE], orig_eids)
        # Mapping between shuffled global IDs to original per-type IDs
        orig_nids = F.gather_row(sim_g.ndata[NID], orig_nids)
        orig_eids = F.gather_row(sim_g.edata[EID], orig_eids)
        orig_nids = {ntype: F.boolean_mask(orig_nids, orig_ntype == g.get_ntype_id(ntype)) \
            for ntype in g.ntypes}
        orig_eids = {etype: F.boolean_mask(orig_eids, orig_etype == g.get_etype_id(etype)) \
            for etype in g.canonical_etypes}
    return orig_nids, orig_eids

def _set_trainer_ids(g, sim_g, node_parts):
    '''Set the trainer IDs for each node and edge on the input graph.

    The trainer IDs will be stored as node data and edge data in the input graph.

    Parameters
    ----------
    g : DGLGraph
       The input graph for partitioning.
    sim_g : DGLGraph
        The homogeneous version of the input graph.
    node_parts : tensor
        The node partition ID for each node in `sim_g`.
    '''
    if g.is_homogeneous:
        g.ndata['trainer_id'] = node_parts
        # An edge is assigned to a partition based on its destination node.
        g.edata['trainer_id'] = F.gather_row(node_parts, g.edges()[1])
    else:
        for ntype_id, ntype in enumerate(g.ntypes):
            type_idx = sim_g.ndata[NTYPE] == ntype_id
            orig_nid = F.boolean_mask(sim_g.ndata[NID], type_idx)
            trainer_id = F.zeros((len(orig_nid),), F.dtype(node_parts), F.cpu())
            F.scatter_row_inplace(trainer_id, orig_nid, F.boolean_mask(node_parts, type_idx))
            g.nodes[ntype].data['trainer_id'] = trainer_id
        for c_etype in g.canonical_etypes:
            # An edge is assigned to a partition based on its destination node.
            _, _, dst_type = c_etype
            trainer_id = F.gather_row(g.nodes[dst_type].data['trainer_id'],
                g.edges(etype=c_etype)[1])
            g.edges[c_etype].data['trainer_id'] = trainer_id

# rowptr, col, train_mask, val_mask, test_mask, node_weight_dim=args.dim
def mypartition_graph(g, graph_name, num_parts, out_path,node_tensor,num_hops=1, part_method="metis",
                    balance_ntypes=None, balance_edges=False, return_mapping=False,
                    num_trainers_per_machine=1, objtype='cut', graph_formats=None):
    
    # 'coo' is required for partition
    assert 'coo' in np.concatenate(list(g.formats().values())), \
        "'coo' format should be allowed for partitioning graph."
    def get_homogeneous(g, balance_ntypes):
        if g.is_homogeneous:
            sim_g = to_homogeneous(g)
            if isinstance(balance_ntypes, dict):
                assert len(balance_ntypes) == 1
                bal_ntypes = list(balance_ntypes.values())[0]
            else:
                bal_ntypes = balance_ntypes
        elif isinstance(balance_ntypes, dict):
            # Here we assign node types for load balancing.
            # The new node types includes the ones provided by users.
            num_ntypes = 0
            for key in g.ntypes:
                if key in balance_ntypes:
                    g.nodes[key].data['bal_ntype'] = F.astype(balance_ntypes[key],
                                                              F.int32) + num_ntypes
                    uniq_ntypes = F.unique(balance_ntypes[key])
                    assert np.all(F.asnumpy(uniq_ntypes) == np.arange(len(uniq_ntypes)))
                    num_ntypes += len(uniq_ntypes)
                else:
                    g.nodes[key].data['bal_ntype'] = F.ones((g.number_of_nodes(key),), F.int32,
                                                            F.cpu()) * num_ntypes
                    num_ntypes += 1
            sim_g = to_homogeneous(g, ndata=['bal_ntype'])
            bal_ntypes = sim_g.ndata['bal_ntype']
            print('The graph has {} node types and balance among {} types'.format(
                len(g.ntypes), len(F.unique(bal_ntypes))))
            # We now no longer need them.
            for key in g.ntypes:
                del g.nodes[key].data['bal_ntype']
            del sim_g.ndata['bal_ntype']
        else:
            sim_g = to_homogeneous(g)
            bal_ntypes = sim_g.ndata[NTYPE]
        return sim_g, bal_ntypes

    if objtype not in ['cut', 'vol']:
        raise ValueError

    if num_parts == 1:
        start = time.time()
        sim_g, balance_ntypes = get_homogeneous(g, balance_ntypes)
        print('Converting to homogeneous graph takes {:.3f}s, peak mem: {:.3f} GB'.format(
            time.time() - start, get_peak_mem()))
        assert num_trainers_per_machine >= 1
        if num_trainers_per_machine > 1:
            start = time.time()
            print("invoke success!")
            node_parts = node_tensor
            _set_trainer_ids(g, sim_g, node_parts)
            print('Assigning nodes to METIS partitions takes {:.3f}s, peak mem: {:.3f} GB'.format(
                time.time() - start, get_peak_mem()))

        # node_parts = F.zeros((sim_g.number_of_nodes(),), F.int64, F.cpu())
        print("invoke success!")
        node_parts = node_tensor
        parts = {0: sim_g.clone()}
        orig_nids = parts[0].ndata[NID] = F.arange(0, sim_g.number_of_nodes())
        orig_eids = parts[0].edata[EID] = F.arange(0, sim_g.number_of_edges())
        # For one partition, we don't really shuffle nodes and edges. We just need to simulate
        # it and set node data and edge data of orig_id.
        parts[0].ndata['orig_id'] = orig_nids
        parts[0].edata['orig_id'] = orig_eids
        if return_mapping:
            if g.is_homogeneous:
                orig_nids = F.arange(0, sim_g.number_of_nodes())
                orig_eids = F.arange(0, sim_g.number_of_edges())
            else:
                orig_nids = {ntype: F.arange(0, g.number_of_nodes(ntype))
                             for ntype in g.ntypes}
                orig_eids = {etype: F.arange(0, g.number_of_edges(etype))
                             for etype in g.canonical_etypes}
        parts[0].ndata['inner_node'] = F.ones((sim_g.number_of_nodes(),),
            RESERVED_FIELD_DTYPE['inner_node'], F.cpu())
        parts[0].edata['inner_edge'] = F.ones((sim_g.number_of_edges(),),
            RESERVED_FIELD_DTYPE['inner_edge'], F.cpu())
    elif part_method in ('metis', 'random'):
        start = time.time()
        sim_g, balance_ntypes = get_homogeneous(g, balance_ntypes)
        print('Converting to homogeneous graph takes {:.3f}s, peak mem: {:.3f} GB'.format(
            time.time() - start, get_peak_mem()))
        if part_method == 'metis':
            assert num_trainers_per_machine >= 1
            start = time.time()
            if num_trainers_per_machine > 1:
                assert(False)
            else:
                print("invoke success!")
                print(node_tensor[0:30])
                print(node_tensor.shape)
                node_parts = node_tensor
            print('Assigning nodes to METIS partitions takes {:.3f}s, peak mem: {:.3f} GB'.format(
                time.time() - start, get_peak_mem()))
        else:
            node_parts = random_choice(num_parts, sim_g.number_of_nodes())
        start = time.time()
        parts, orig_nids, orig_eids = partition_graph_with_halo(sim_g, node_parts, num_hops,
                                                                reshuffle=True)
        print('Splitting the graph into partitions takes {:.3f}s, peak mem: {:.3f} GB'.format(
            time.time() - start, get_peak_mem()))
        if return_mapping:
            orig_nids, orig_eids = _get_orig_ids(g, sim_g, orig_nids, orig_eids)
    else:
        raise Exception('Unknown partitioning method: ' + part_method)

    # If the input is a heterogeneous graph, get the original node types and original node IDs.
    # `part' has three types of node data at this point.
    # NTYPE: the node type.
    # orig_id: the global node IDs in the homogeneous version of input graph.
    # NID: the global node IDs in the reshuffled homogeneous version of the input graph.
    if not g.is_homogeneous:
        for name in parts:
            orig_ids = parts[name].ndata['orig_id']
            ntype = F.gather_row(sim_g.ndata[NTYPE], orig_ids)
            parts[name].ndata[NTYPE] = F.astype(ntype, RESERVED_FIELD_DTYPE[NTYPE])
            assert np.all(F.asnumpy(ntype) == F.asnumpy(parts[name].ndata[NTYPE]))
            # Get the original edge types and original edge IDs.
            orig_ids = parts[name].edata['orig_id']
            etype = F.gather_row(sim_g.edata[ETYPE], orig_ids)
            parts[name].edata[ETYPE] = F.astype(etype, RESERVED_FIELD_DTYPE[ETYPE])
            assert np.all(F.asnumpy(etype) == F.asnumpy(parts[name].edata[ETYPE]))

            # Calculate the global node IDs to per-node IDs mapping.
            inner_ntype = F.boolean_mask(parts[name].ndata[NTYPE],
                                            parts[name].ndata['inner_node'] == 1)
            inner_nids = F.boolean_mask(parts[name].ndata[NID],
                                        parts[name].ndata['inner_node'] == 1)
            for ntype in g.ntypes:
                inner_ntype_mask = inner_ntype == g.get_ntype_id(ntype)
                typed_nids = F.boolean_mask(inner_nids, inner_ntype_mask)
                # inner node IDs are in a contiguous ID range.
                expected_range = np.arange(int(F.as_scalar(typed_nids[0])),
                                            int(F.as_scalar(typed_nids[-1])) + 1)
                assert np.all(F.asnumpy(typed_nids) == expected_range)
            # Calculate the global edge IDs to per-edge IDs mapping.
            inner_etype = F.boolean_mask(parts[name].edata[ETYPE],
                                            parts[name].edata['inner_edge'] == 1)
            inner_eids = F.boolean_mask(parts[name].edata[EID],
                                        parts[name].edata['inner_edge'] == 1)
            for etype in g.canonical_etypes:
                inner_etype_mask = inner_etype == g.get_etype_id(etype)
                typed_eids = np.sort(F.asnumpy(F.boolean_mask(inner_eids, inner_etype_mask)))
                assert np.all(typed_eids == np.arange(int(typed_eids[0]),
                                                        int(typed_eids[-1]) + 1))

    os.makedirs(out_path, mode=0o775, exist_ok=True)
    tot_num_inner_edges = 0
    out_path = os.path.abspath(out_path)

    # With reshuffling, we can ensure that all nodes and edges are reshuffled
    # and are in contiguous ID space.
    if num_parts > 1:
        node_map_val = {}
        edge_map_val = {}
        for ntype in g.ntypes:
            ntype_id = g.get_ntype_id(ntype)
            val = []
            node_map_val[ntype] = []
            for i in parts:
                inner_node_mask = _get_inner_node_mask(parts[i], ntype_id)
                val.append(F.as_scalar(F.sum(F.astype(inner_node_mask, F.int64), 0)))
                inner_nids = F.boolean_mask(parts[i].ndata[NID], inner_node_mask)
                node_map_val[ntype].append([int(F.as_scalar(inner_nids[0])),
                                            int(F.as_scalar(inner_nids[-1])) + 1])
            val = np.cumsum(val).tolist()
            assert val[-1] == g.number_of_nodes(ntype)
        for etype in g.canonical_etypes:
            etype_id = g.get_etype_id(etype)
            val = []
            edge_map_val[etype] = []
            for i in parts:
                inner_edge_mask = _get_inner_edge_mask(parts[i], etype_id)
                val.append(F.as_scalar(F.sum(F.astype(inner_edge_mask, F.int64), 0)))
                inner_eids = np.sort(F.asnumpy(F.boolean_mask(parts[i].edata[EID],
                                                                inner_edge_mask)))
                edge_map_val[etype].append([int(inner_eids[0]), int(inner_eids[-1]) + 1])
            val = np.cumsum(val).tolist()
            assert val[-1] == g.number_of_edges(etype)
    else:
        node_map_val = {}
        edge_map_val = {}
        for ntype in g.ntypes:
            ntype_id = g.get_ntype_id(ntype)
            inner_node_mask = _get_inner_node_mask(parts[0], ntype_id)
            inner_nids = F.boolean_mask(parts[0].ndata[NID], inner_node_mask)
            node_map_val[ntype] = [[int(F.as_scalar(inner_nids[0])),
                                    int(F.as_scalar(inner_nids[-1])) + 1]]
        for etype in g.canonical_etypes:
            etype_id = g.get_etype_id(etype)
            inner_edge_mask = _get_inner_edge_mask(parts[0], etype_id)
            inner_eids = F.boolean_mask(parts[0].edata[EID], inner_edge_mask)
            edge_map_val[etype] = [[int(F.as_scalar(inner_eids[0])),
                                    int(F.as_scalar(inner_eids[-1])) + 1]]

        # Double check that the node IDs in the global ID space are sorted.
        for ntype in node_map_val:
            val = np.concatenate([np.array(l) for l in node_map_val[ntype]])
            assert np.all(val[:-1] <= val[1:])
        for etype in edge_map_val:
            val = np.concatenate([np.array(l) for l in edge_map_val[etype]])
            assert np.all(val[:-1] <= val[1:])

    start = time.time()
    ntypes = {ntype:g.get_ntype_id(ntype) for ntype in g.ntypes}
    etypes = {etype:g.get_etype_id(etype) for etype in g.canonical_etypes}
    part_metadata = {'graph_name': graph_name,
                     'num_nodes': g.number_of_nodes(),
                     'num_edges': g.number_of_edges(),
                     'part_method': part_method,
                     'num_parts': num_parts,
                     'halo_hops': num_hops,
                     'node_map': node_map_val,
                     'edge_map': edge_map_val,
                     'ntypes': ntypes,
                     'etypes': etypes}
    for part_id in range(num_parts):
        part = parts[part_id]

        # Get the node/edge features of each partition.
        node_feats = {}
        edge_feats = {}
        if num_parts > 1:
            for ntype in g.ntypes:
                ntype_id = g.get_ntype_id(ntype)
                # To get the edges in the input graph, we should use original node IDs.
                # Both orig_id and NID stores the per-node-type IDs.
                ndata_name = 'orig_id'
                inner_node_mask = _get_inner_node_mask(part, ntype_id)
                # This is global node IDs.
                local_nodes = F.boolean_mask(part.ndata[ndata_name], inner_node_mask)
                if len(g.ntypes) > 1:
                    # If the input is a heterogeneous graph.
                    local_nodes = F.gather_row(sim_g.ndata[NID], local_nodes)
                    print('part {} has {} nodes of type {} and {} are inside the partition'.format(
                        part_id, F.as_scalar(F.sum(part.ndata[NTYPE] == ntype_id, 0)),
                        ntype, len(local_nodes)))
                else:
                    print('part {} has {} nodes and {} are inside the partition'.format(
                        part_id, part.number_of_nodes(), len(local_nodes)))

                for name in g.nodes[ntype].data:
                    if name in [NID, 'inner_node']:
                        continue
                    node_feats[ntype + '/' + name] = F.gather_row(g.nodes[ntype].data[name],
                                                                  local_nodes)

            for etype in g.canonical_etypes:
                etype_id = g.get_etype_id(etype)
                edata_name = 'orig_id'
                inner_edge_mask = _get_inner_edge_mask(part, etype_id)
                # This is global edge IDs.
                local_edges = F.boolean_mask(part.edata[edata_name], inner_edge_mask)
                if not g.is_homogeneous:
                    local_edges = F.gather_row(sim_g.edata[EID], local_edges)
                    print('part {} has {} edges of type {} and {} are inside the partition'.format(
                        part_id, F.as_scalar(F.sum(part.edata[ETYPE] == etype_id, 0)),
                        etype, len(local_edges)))
                else:
                    print('part {} has {} edges and {} are inside the partition'.format(
                        part_id, part.number_of_edges(), len(local_edges)))
                tot_num_inner_edges += len(local_edges)

                for name in g.edges[etype].data:
                    if name in [EID, 'inner_edge']:
                        continue
                    edge_feats[_etype_tuple_to_str(etype) + '/' + name] = F.gather_row(
                        g.edges[etype].data[name], local_edges)
        else:
            for ntype in g.ntypes:
                if len(g.ntypes) > 1:
                    ndata_name = 'orig_id'
                    ntype_id = g.get_ntype_id(ntype)
                    inner_node_mask = _get_inner_node_mask(part, ntype_id)
                    # This is global node IDs.
                    local_nodes = F.boolean_mask(part.ndata[ndata_name], inner_node_mask)
                    local_nodes = F.gather_row(sim_g.ndata[NID], local_nodes)
                else:
                    local_nodes = sim_g.ndata[NID]
                for name in g.nodes[ntype].data:
                    if name in [NID, 'inner_node']:
                        continue
                    node_feats[ntype + '/' + name] = F.gather_row(g.nodes[ntype].data[name],
                                                                    local_nodes)
            for etype in g.canonical_etypes:
                if not g.is_homogeneous:
                    edata_name = 'orig_id'
                    etype_id = g.get_etype_id(etype)
                    inner_edge_mask = _get_inner_edge_mask(part, etype_id)
                    # This is global edge IDs.
                    local_edges = F.boolean_mask(part.edata[edata_name], inner_edge_mask)
                    local_edges = F.gather_row(sim_g.edata[EID], local_edges)
                else:
                    local_edges = sim_g.edata[EID]
                for name in g.edges[etype].data:
                    if name in [EID, 'inner_edge']:
                        continue
                    edge_feats[_etype_tuple_to_str(etype) + '/' + name] = F.gather_row(
                        g.edges[etype].data[name], local_edges)
        # delete `orig_id` from ndata/edata
        del part.ndata['orig_id']
        del part.edata['orig_id']

        part_dir = os.path.join(out_path, "part" + str(part_id))
        node_feat_file = os.path.join(part_dir, "node_feat.dgl")
        edge_feat_file = os.path.join(part_dir, "edge_feat.dgl")
        part_graph_file = os.path.join(part_dir, "graph.dgl")
        part_metadata['part-{}'.format(part_id)] = {
            'node_feats': os.path.relpath(node_feat_file, out_path),
            'edge_feats': os.path.relpath(edge_feat_file, out_path),
            'part_graph': os.path.relpath(part_graph_file, out_path)}
        os.makedirs(part_dir, mode=0o775, exist_ok=True)
        save_tensors(node_feat_file, node_feats)
        save_tensors(edge_feat_file, edge_feats)

        sort_etypes = len(g.etypes) > 1
        _save_graphs(part_graph_file, [part], formats=graph_formats,
            sort_etypes=sort_etypes)
    print('Save partitions: {:.3f} seconds, peak memory: {:.3f} GB'.format(
        time.time() - start, get_peak_mem()))

    _dump_part_config(f'{out_path}/{graph_name}.json', part_metadata)

    num_cuts = sim_g.number_of_edges() - tot_num_inner_edges
    if num_parts == 1:
        num_cuts = 0
    print('There are {} edges in the graph and {} edge cuts for {} partitions.'.format(
        g.number_of_edges(), num_cuts, num_parts))

    if return_mapping:
        return orig_nids, orig_eids
