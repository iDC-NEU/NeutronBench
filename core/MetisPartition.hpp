#include <metis.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cassert>
#include "comm/logger.h"
#include "FullyRepGraph.hpp"

using namespace std;

/**
 * @brief 
 * Metis Partition
 * 
 * @param whole_graph input graph
 * @param part partition result
 * @param k partition number
 * @param ojb_cut cut type(true: cut, false: vol)
 * @return the status of MetisPartition func
 */
// bool MetisPartition(vector<idx_t> &part, vector<idx_t> &xadj, vector<idx_t> &adjncy, vector<idx_t> &adjwgt, int k, bool obj_cut) {
bool MetisPartition(FullyRepGraph *whole_graph, vector<idx_t> &part, int k, bool obj_cut) {
  assert(k > 1);
  idx_t nvtxs = whole_graph->global_vertices; 
  idx_t nedges = whole_graph->global_edges;  
  // std::cout << "node " << nvtxs <<  " edge " << nedges << std::endl;
  // idx_t* xadj = reinterpret_cast<idx_t*> (whole_graph->column_offset);
  // idx_t* adjncy = reinterpret_cast<idx_t*> (whole_graph->row_indices);
  vector<idx_t> xadj_vec(whole_graph->column_offset, whole_graph->column_offset + whole_graph->global_vertices + 1);
  vector<idx_t> adjncy_vec(whole_graph->row_indices, whole_graph->row_indices + whole_graph->global_edges);
  idx_t* xadj = xadj_vec.data();
  idx_t* adjncy = adjncy_vec.data();
  // printf("xadj %d adjncy %d\n", xadj_vec.size(), adjncy_vec.size());
  for (int i = 0; i < whole_graph->global_vertices; ++i) {
    assert(xadj_vec[i] == whole_graph->column_offset[i]);
  }
  for (int i = 0; i < whole_graph->global_edges; ++i) {
    assert(adjncy_vec[i] == whole_graph->row_indices[i]);
  }
  idx_t* adjwgt = NULL; // (FIX Sanzo) get from whole_graph
  idx_t ncon = 1;                
  idx_t nparts = k;                 
  idx_t objval;                     
  // vector<idx_t> part(nvtxs, 0);
  part.resize(nvtxs, 0);
  
  idx_t options[METIS_NOPTIONS];
  METIS_SetDefaultOptions(options);
  options[METIS_OPTION_ONDISK] = 1;
  options[METIS_OPTION_NITER] = 1;
  options[METIS_OPTION_NIPARTS] = 1;
  options[METIS_OPTION_DROPEDGES] = 1;

  if (obj_cut) {
    options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;
  } else {
    options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_VOL;
  }

  // METIS_API(int) METIS_PartGraphRecursive(idx_t *nvtxs, idx_t *ncon, idx_t *xadj, 
  //                 idx_t *adjncy, idx_t *vwgt, idx_t *vsize, idx_t *adjwgt, 
  //                 idx_t *nparts, real_t *tpwgts, real_t *ubvec, idx_t *options, 
  //                 idx_t *edgecut, idx_t *part);
  auto METIS_PartFunc = (nparts > 8 || !obj_cut) ? METIS_PartGraphKway : METIS_PartGraphRecursive;
  // printf("%x %x %x\n", METIS_PartGraphKway, METIS_PartGraphRecursive, METIS_PartFunc);
  // std::cout << "start metis!!!!!" << std::endl;
  int ret;
// for (int i = 0; i < 10; ++i) {
  ret = METIS_PartFunc(
    &nvtxs,      // The number of vertices
    &ncon,       // The number of balancing constraints.
    xadj,        // indptr
    adjncy,      // indices
    
    NULL, // (vwgt) the weights of the vertices
    NULL,        // (vsize) The size of the vertices for computing
                  // the total communication volume
    NULL,        // (adjwgt) The weights of the edges
    &nparts,     // The number of partitions.
    NULL,        // (tpwgts) the desired weight for each partition and constraint
    NULL,        // (ubvec) the allowed load imbalance tolerance
    
    options,     // the array of options ##
    &objval,     // the edge-cut or the total communication volume of
                  // the partitioning solution
    part.data());       

  if (obj_cut) {
    std::cout << "Partition a graph with " << nvtxs << " nodes and "
              << nedges << " edges into " << nparts << " parts and "
              << "get " << objval << " edge cuts" << std::endl;
  } else {
    std::cout << "Partition a graph with " << nvtxs << " nodes and "
              << nedges << " edges into " << nparts << " parts and "
              << "the communication volume is " << objval << std::endl;
  }
// }

  switch (ret) {
    case METIS_OK:
      return true;
    case METIS_ERROR_INPUT:
      // LOG_INFO << "Error in Metis partitioning: input error";
      LOG_ERROR( "Error in Metis partitioning: input error");
    case METIS_ERROR_MEMORY:
      LOG_ERROR("Error in Metis partitioning: cannot allocate memory");
    default:
      LOG_ERROR("Error in Metis partitioning: other errors");
  }
  return false;
}

void MetisPartitionGraph(FullyRepGraph* whole_graph, int partition_num, std::string objtype, 
  std::vector<VertexId>& metis_partition_id, std::vector<VertexId>& metis_partition_offset) {
  bool obj_cut = objtype == "cut" ? true : false;
  vector<idx_t> partition_id;
  // vector<VertexId> offset(whole)
  // std::cout << whole_graph->column_offset.size() << " " << whole_graph->row_indices << std::endl;
  MetisPartition(whole_graph, partition_id, partition_num, obj_cut);
  
  metis_partition_id.resize(whole_graph->global_vertices);
  metis_partition_offset.resize(partition_num + 1, 0);
  // vector<int> metis_partition_edges(whole_graph->global_edges);
  // vector<int> metis_partition_edge_offset(partition_num + 1, 0);
  // get node offset of metis partition
  for (auto nodeId : partition_id) {
      metis_partition_offset[nodeId + 1]++;
  }
  for (int i = 1; i <= partition_num; ++i) {
      metis_partition_offset[i] += metis_partition_offset[i - 1];
  }
  // get node of metis partition
  vector<int> tmp(metis_partition_offset.begin(), metis_partition_offset.end());
  for (int i = 0; i < whole_graph->global_vertices; ++i) {
     metis_partition_id[tmp[partition_id[i]]++] = i; 
  }
  // vector<int> tmp_nodes(whole_graph->global_vertices);
  // std::iota(tmp_nodes.begin(), tmp_nodes.end(), 0);
  // sort(tmp_nodes.begin(), tmp_nodes.end(), [&](int x, int y) {
  //     if (partition_id[x] != partition_id[y]) 
  //         return partition_id[x] < partition_id[y];
  //     return x < y;
  // });
  // for (int i = 0; i < whole_graph->global_vertices; ++i) {
  //     assert(tmp_nodes[i] == metis_partition_id[i]);
  // }
  // std::cout << "vertices check done!" << std::endl;

  // get edge offset of metis partition
  // auto& column_offset = whole_graph->column_offset;
  // auto &node_offset = metis_partition_offset;
  // int cnt_edges = 0;
  // for (int i = 0; i < partition_num; ++i) {
  //     for (int j = node_offset[i]; j < node_offset[i + 1]; ++j) {
  //         int node_id = metis_partition_id[j];
  //         cnt_edges += column_offset[node_id + 1] - column_offset[node_id];
  //     }
  //     metis_partition_edge_offset[i + 1] = cnt_edges;
  // }

  // get edge of metis partition
  // cnt_edges = 0;
  // auto row_indices = whole_graph->row_indices;
  // for (int i = 0; i < partition_num; ++i) {
  //     for (int j = node_offset[i]; j < node_offset[i + 1]; ++j) {
  //         int node_id = metis_partition_id[j];
  //         for (int k = column_offset[node_id]; k < column_offset[node_id + 1]; ++k) {
  //             metis_partition_edges[cnt_edges++] = row_indices[k];
  //         }
  //     }
  // }

}