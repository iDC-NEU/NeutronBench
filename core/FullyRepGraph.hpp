/*
Copyright (c) 2021-2022 Qiange Wang, Northeastern University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#ifndef FULLLYREPGRAPH_HPP
#define FULLLYREPGRAPH_HPP
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <map>
#include <stack>
#include <vector>

#include "core/coocsc.hpp"
#include "core/graph.hpp"
class SampledSubgraph {
 public:
  SampledSubgraph() {
    sampled_sgs.clear();
    layers = -1;
    fanout.clear();
    sampled_sgs.clear();
    curr_layer = 0;
    threads = std::max(1, numa_num_configured_cpus());
    // threads = std::max(1, numa_num_configured_cpus() - 1);
    // threads = std::max(1, numa_num_configured_cpus() / 2);
    seeds = new unsigned[threads];
  }
  SampledSubgraph(int layers_, int batch_size_, const std::vector<int> &fanout_) {
    layers = layers_;
    batch_size = batch_size_;
    fanout = fanout_;
    sampled_sgs.clear();
    curr_layer = 0;
    curr_dst_size = batch_size;
    threads = std::max(1, numa_num_configured_cpus());
    // threads = std::max(1, numa_num_configured_cpus() - 1);
    // threads = std::max(1, numa_num_configured_cpus() / 2);
    seeds = new unsigned[threads];
  }

  SampledSubgraph(int layers_, const std::vector<int> &fanout_) {
    layers = layers_;
    fanout = fanout_;
    sampled_sgs.clear();
    curr_layer = 0;
    threads = std::max(1, numa_num_configured_cpus());
    // threads = std::max(1, numa_num_configured_cpus() - 1);
    // threads = std::max(1, numa_num_configured_cpus() / 2);
    // seeds = new unsigned[threads];
  }

  ~SampledSubgraph() {
    fanout.clear();
    for (int i = 0; i < sampled_sgs.size(); i++) {
      delete sampled_sgs[i];
    }
    sampled_sgs.clear();
  }

  void alloc_memory(VertexId v_size) {
    for (int i = 0; i < layers; ++i) {
      int e_size = v_size * fanout[i];
      sampled_sgs.push_back(new sampCSC(v_size, e_size));
      v_size = e_size;
    }
  }

  void update_degrees(Graph<Empty> *graph, int layer) {
    // LOG_DEBUG("udpate degrees");
    VertexId *outs = graph->out_degree_for_backward;
    VertexId *ins = graph->in_degree_for_backward;
    memset(outs, 0, sizeof(outs));
    memset(ins, 0, sizeof(ins));
    // printf("vertex %d outs elememt %d ins element %d\n", graph->vertices, sizeof(outs)/sizeof(VertexId),
    // sizeof(ins)/sizeof(VertexId)); assert(sizeof(outs) / sizeof(VertexId) == graph->vertices); assert(sizeof(ins) /
    // sizeof(VertexId) == graph->vertices);
    for (int i = 0; i < graph->vertices; ++i) {
      outs[i] = 0;
      ins[i] = 0;
      // assert(outs[i] == 0);
      // assert(ins[i] == 0);
    }
    for (auto src : sampled_sgs[layer]->src()) {
      outs[src]++;
    }
    for (auto dst : sampled_sgs[layer]->dst()) {
      ins[dst]++;
    }

    // for (int i = 0; i < graph->vertices; ++i) {
    //     assert(graph->in_degree_for_backward[i] == ins[i]);
    //     assert(graph->out_degree_for_backward[i] == outs[i]);
    // }
  }

  void random_gen_seed() {
    for (int i = 0; i < threads; ++i) {
      seeds[i] = rand();
    }
  }

  void sample_preprocessing(VertexId layer) {
    curr_layer = layer;
    if (0 == layer) {
      sampCSC *sampled_sg = new sampCSC(0);
      sampled_sgs.push_back(sampled_sg);
    } else {
      sampCSC *sampled_sg = new sampCSC(curr_dst_size);
      // sampled_sg->allocate_all();
      sampled_sg->allocate_vertex();
      sampled_sgs.push_back(sampled_sg);
    }
    // assert(layer==sampled_sgs.size()-1);
  }
  void sample_load_destination(std::function<void(std::vector<VertexId> &destination)> dst_select, VertexId layer) {
    dst_select(sampled_sgs[layer]->dst());  // init destination;
  }

  void init_co(std::function<VertexId(VertexId dst)> get_nbr_size, VertexId layer) {
    VertexId offset = 0;
    for (VertexId i = 0; i < curr_dst_size; i++) {
      sampled_sgs[layer]->c_o()[i] = offset;
      offset += get_nbr_size(sampled_sgs[layer]->dst()[i]);  // init destination;
    }
    // std::cout << std::endl;
    sampled_sgs[layer]->c_o()[curr_dst_size] = offset;
    sampled_sgs[layer]->update_edges(offset);
  }

  void sample_load_destination(VertexId layer) {
    assert(layer > 0);
    for (VertexId i_id = 0; i_id < curr_dst_size; i_id++) {
      sampled_sgs[layer]->dst()[i_id] = sampled_sgs[layer - 1]->src()[i_id];
    }
  }
  // int random_uniform_int(const int min = 0, const int max = 1) {
  //     thread_local std::default_random_engine generator;
  //     std::uniform_int_distribution<int> distribution(min, max);
  //     return distribution(generator);
  // }
  void sample_processing(std::function<void(VertexId fanout_i, VertexId dst, std::vector<VertexId> &column_offset,
                                            std::vector<VertexId> &row_indices, VertexId id)>
                             vertex_sample) {
    {
      // random_gen_seed();
      // threads=30;
      omp_set_num_threads(threads);
      // LOG_DEBUG("thrads %d", threads);
      // LOG_DEBUG("processing %d %d layer %d, fanout %d", 0, curr_dst_size, curr_layer, fanout[curr_layer]);
#pragma omp parallel for num_threads(threads)
      for (VertexId begin_v_i = 0; begin_v_i < curr_dst_size; begin_v_i += 1) {
        // for every vertex, apply the sparse_slot at the partition
        // corresponding to the step
        vertex_sample(fanout[curr_layer], sampled_sgs[curr_layer]->dst()[begin_v_i], sampled_sgs[curr_layer]->c_o(),
                      sampled_sgs[curr_layer]->r_i(), begin_v_i);
      }
    }
  }

  // void sample_postprocessing(){
  //     sampled_sgs[sampled_sgs.size()-1]->postprocessing();
  //     curr_dst_size=sampled_sgs[sampled_sgs.size()-1]->get_distinct_src_size();
  //     curr_layer++;
  // }

  // void sample_postprocessing(Bitmap* bits){
  //     sampled_sgs[sampled_sgs.size()-1]->postprocessing(bits);
  //     curr_dst_size=sampled_sgs[sampled_sgs.size()-1]->get_distinct_src_size();
  //     curr_layer++;
  // }

  void sample_postprocessing(Bitmap *bits, int layer) {
    sampled_sgs[layer]->postprocessing(bits);
    curr_dst_size = sampled_sgs[layer]->get_distinct_src_size();
    // curr_layer++;
  }

  void compute_one_layer(
      std::function<void(VertexId local_dst, std::vector<VertexId> &column_offset, std::vector<VertexId> &row_indices)>
          sparse_slot,
      VertexId layer) {
    //  void compute_one_layer(std::function<void(VertexId local_dst,
    //                 VertexId* column_offset, VertexId* row_indices)>sparse_slot,VertexId layer){

    // threads = 30;
    omp_set_num_threads(threads);
#pragma omp parallel for num_threads(threads)
    for (VertexId begin_v_i = 0; begin_v_i < sampled_sgs[layer]->v_size; begin_v_i += 1) {
      sparse_slot(begin_v_i, sampled_sgs[layer]->c_o(), sampled_sgs[layer]->r_i());
    }
  }

  void compute_one_layer_backward(
      std::function<void(VertexId local_dst, std::vector<VertexId> &column_offset, std::vector<VertexId> &row_indices)>
          sparse_slot,
      VertexId layer) {
    //   void compute_one_layer_backward(std::function<void(VertexId local_dst,
    //                     VertexId* column_offset, VertexId* row_indices)>sparse_slot,VertexId layer){
    // LOG_DEBUG("start backward");
    omp_set_num_threads(threads);
#pragma omp parallel for num_threads(threads)
    for (VertexId begin_v_i = 0; begin_v_i < sampled_sgs[layer]->src_size; begin_v_i += 1) {
      sparse_slot(begin_v_i, sampled_sgs[layer]->r_o(), sampled_sgs[layer]->c_i());
    }
    // LOG_DEBUG("end backward");
  }

  std::vector<sampCSC *> sampled_sgs;
  int layers;
  int batch_size;
  std::vector<int> fanout;
  int curr_layer;
  int curr_dst_size;
  int threads;
  unsigned *seeds;
};
class FullyRepGraph {
 public:
  // topo:
  VertexId *dstList;
  VertexId *srcList;
  // meta info
  Graph<Empty> *graph_;
  VertexId *partition_offset;
  VertexId partitions;
  VertexId partition_id;
  VertexId global_vertices;
  VertexId global_edges;
  // vertex range for this chunk
  VertexId owned_vertices;
  VertexId owned_edges;
  VertexId owned_mirrors;

  // global graph;
  VertexId *column_offset;
  VertexId *row_indices;

  FullyRepGraph() {}
  FullyRepGraph(Graph<Empty> *graph) {
    global_vertices = graph->vertices;
    global_edges = graph->edges;
    owned_vertices = graph->owned_vertices;
    partitions = graph->partitions;
    partition_id = graph->partition_id;
    partition_offset = graph->partition_offset;
    graph_ = graph;
  }
  void SyncAndLog(const char *data) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (partition_id == 0) std::cout << data << std::endl;
  }
  void GenerateAll() {
    ReadRepGraphFromRawFile();
    SyncAndLog("NeutronStar::Preprocessing[Generate Full Replicated Graph Topo]");
    SyncAndLog("------------------finish graph preprocessing--------------\n");
  }
  void ReadRepGraphFromRawFile() {
    column_offset = new VertexId[global_vertices + 1];
    row_indices = new VertexId[global_edges];
    memset(column_offset, 0, sizeof(VertexId) * (global_vertices + 1));
    memset(row_indices, 0, sizeof(VertexId) * global_edges);
    VertexId *tmp_offset = new VertexId[global_vertices + 1];
    memset(tmp_offset, 0, sizeof(VertexId) * (global_vertices + 1));
    long total_bytes = file_size(graph_->filename.c_str());
#ifdef PRINT_DEBUG_MESSAGES
    if (partition_id == 0) {
      printf("|V| = %u, |E| = %lu\n", vertices, edges);
    }
#endif
    int edge_unit_size = sizeof(VertexId) * 2;
    EdgeId read_edges = global_edges;
    long bytes_to_read = edge_unit_size * read_edges;
    long read_offset = 0;
    long read_bytes;
    int fin = open(graph_->filename.c_str(), O_RDONLY);
    EdgeUnit<Empty> *read_edge_buffer = new EdgeUnit<Empty>[CHUNKSIZE];

    assert(lseek(fin, read_offset, SEEK_SET) == read_offset);
    read_bytes = 0;
    while (read_bytes < bytes_to_read) {
      long curr_read_bytes;
      if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE) {
        curr_read_bytes = read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
      } else {
        curr_read_bytes = read(fin, read_edge_buffer, bytes_to_read - read_bytes);
      }
      assert(curr_read_bytes >= 0);
      read_bytes += curr_read_bytes;
      EdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
      // #pragma omp parallel for
      for (EdgeId e_i = 0; e_i < curr_read_edges; e_i++) {
        VertexId src = read_edge_buffer[e_i].src;
        VertexId dst = read_edge_buffer[e_i].dst;
        tmp_offset[dst + 1]++;
      }
    }
    for (int i = 0; i < global_vertices; i++) {
      tmp_offset[i + 1] += tmp_offset[i];
    }

    memcpy(column_offset, tmp_offset, sizeof(VertexId) * (global_vertices + 1));
    // printf("%d\n", column_offset[vertices]);
    assert(lseek(fin, read_offset, SEEK_SET) == read_offset);
    read_bytes = 0;
    while (read_bytes < bytes_to_read) {
      long curr_read_bytes;
      if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE) {
        curr_read_bytes = read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
      } else {
        curr_read_bytes = read(fin, read_edge_buffer, bytes_to_read - read_bytes);
      }
      assert(curr_read_bytes >= 0);
      read_bytes += curr_read_bytes;
      EdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
      // #pragma omp parallel for
      for (EdgeId e_i = 0; e_i < curr_read_edges; e_i++) {
        VertexId src = read_edge_buffer[e_i].src;
        VertexId dst = read_edge_buffer[e_i].dst;
        //        if(dst==875710)
        //            printf("%d",read_edge_buffer[e_i].src);
        row_indices[tmp_offset[dst]++] = src;
      }
    }
    delete[] read_edge_buffer;
    delete[] tmp_offset;
  }
};

#endif