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
    threads = std::max(1, numa_num_configured_cpus() / 2 - 1);
    // LOG_DEBUG("SamplerSubgraph thraeds %d", threads);
    seeds = new unsigned[threads];
  }
  
  void update_fanout(const std::vector<int> fanout_) {
    assert (fanout.size() == fanout_.size());
    for (int i = 0; i < fanout.size(); ++i) {
      fanout[i] = fanout_[i];
    }
  }

  void show_fanout(std::string info) {
    std::cout << info << " ";
    for (auto& it : fanout) {
      printf("%d, ", it);
    }
    printf("\n");
  }

  void update_fanout(int nums) {
    for (auto& it : fanout) {
      it = nums;
    }
  }
  
  SampledSubgraph(int layers_, int batch_size_, const std::vector<int> &fanout_) {
    layers = layers_;
    batch_size = batch_size_;
    fanout = fanout_;
    sampled_sgs.clear();
    curr_layer = 0;
    curr_dst_size = batch_size;
    threads = std::max(1, numa_num_configured_cpus() / 2 - 1);
    // LOG_DEBUG("SamplerSubgraph thraeds %d", threads);
    seeds = new unsigned[threads];
  }

  SampledSubgraph(int layers_, const std::vector<int> &fanout_) {
    layers = layers_;
    fanout = fanout_;
    sampled_sgs.clear();
    curr_layer = 0;
    threads = std::max(1, numa_num_configured_cpus() / 2 - 1);
    // LOG_DEBUG("SamplerSubgraph thraeds %d", threads);
    // seeds = new unsigned[threads];
    for (int i = 0; i < layers; ++i) {
      sampled_sgs.push_back(new sampCSC(0));
    }
  }

  // void copy_to_ssg(SampledSubgraph* ssg) {
  //   ssg->layers = layers;
  //   ssg->batch_size = batch_size;
  //   ssg->fanout = fanout;
  //   ssg->curr_layer = curr_layer;
  //   ssg->curr_dst_size = curr_dst_size;
  //   ssg->threads = threads;

  //   std::vector<sampCSC *> sampled_sgs;
  //   int layers;
  //   int batch_size;
  //   std::vector<int> fanout;
  //   int curr_layer;
  //   int curr_dst_size;
  //   int threads;
  //   unsigned *seeds;

  //   ssg->sampled_sgs.resize(sampled_sgs.size());
  //   for (int i = 0; i < layers; ++i) {
  //     // ssg->sampled_sgs[i] = new sampCSC
  //   }
  // }

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

  void compute_weight(Graph<Empty> *graph) {
  // std::pair<double, double> compute_weight(Graph<Empty> *graph) {
    // double update_degree_time = 0;
    // double compute_weight_time = 0;
    for (size_t i = 0; i < layers; ++i) {
      // update_degree_time -= get_time();
      sampled_sgs[i]->update_degree(graph);
      // update_degree_time += get_time();

      // compute_weight_time -= get_time();
      sampled_sgs[i]->compute_weight_forward(graph);
      sampled_sgs[i]->compute_weight_backward(graph);
      // compute_weight_time += get_time();
    }
    // return {update_degree_time, compute_weight_time};
  }

  void alloc_dev_array(bool pull = true) {
    for (int i = 0; i < layers; ++i) {
      sampled_sgs[i]->alloc_dev_array(pull);
    }
  }

  void random_gen_seed() {
    for (int i = 0; i < threads; ++i) {
      seeds[i] = rand();
    }
  }

  void trans_graph_to_gpu(bool pull = true) {
    for (int i = 0; i < layers; ++i) {
      // TODO(sanzo): not alloc memory fo csr in push version
      // sampled_sgs[i]->alloc_dev_array(pull);
      // LOG_DEBUG("alloc dev arry done");
      sampled_sgs[i]->copy_data_to_device(pull);
      sampled_sgs[i]->copy_ewb_to_device();
      sampled_sgs[i]->copy_ewf_to_device();
      // LOG_DEBUG("copy_data_to device done");
    }
  }

  void trans_graph_to_gpu_async(cudaStream_t cs, bool pull = true) {
    for (int i = 0; i < layers; ++i) {
      // TODO(sanzo): not alloc memory fo csr in push version
      // sampled_sgs[i]->alloc_dev_array(pull);
      // LOG_DEBUG("alloc dev arry done");
      sampled_sgs[i]->copy_data_to_device_async(cs, pull);
      sampled_sgs[i]->copy_ewb_to_device_async(cs);
      sampled_sgs[i]->copy_ewf_to_device_async(cs);
      // LOG_DEBUG("copy_data_to device done");
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
    assert(false);
    // dst_select(sampled_sgs[layer]->dst());  // init destination;
  }

  void init_co(std::function<VertexId(VertexId dst)> get_nbr_size, VertexId layer) {
    VertexId offset = 0;
    for (VertexId i = 0; i < curr_dst_size; i++) {
      sampled_sgs[layer]->c_o()[i] = offset;
      offset += get_nbr_size(sampled_sgs[layer]->dst()[i]);  // init destination;
    }
    // std::cout << std::endl;
    sampled_sgs[layer]->c_o()[curr_dst_size] = offset;
    // sampled_sgs[layer]->update_edges(offset);
    sampled_sgs[layer]->alloc_edges(offset);
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
  void sample_processing(
      std::function<void(VertexId fanout_i, VertexId dst, VertexId *column_offset, VertexId *row_indices, VertexId id)>
          vertex_sample) {
    // random_gen_seed();
    // threads=30;
    // omp_set_num_threads(threads);f
    // LOG_DEBUG("thrads %d", threads);
    // LOG_DEBUG("processing %d %d layer %d, fanout %d", 0, curr_dst_size, curr_layer, fanout[curr_layer]);
// #pragma omp parallel for num_threads(threads)
omp_set_num_threads(threads);
#pragma omp parallel for num_threads(threads)
    for (VertexId begin_v_i = 0; begin_v_i < curr_dst_size; begin_v_i += 1) {
      // for every vertex, apply the sparse_slot at the partition
      // corresponding to the step
      vertex_sample(fanout[curr_layer], sampled_sgs[curr_layer]->dst()[begin_v_i],
                    sampled_sgs[curr_layer]->c_o().data(), sampled_sgs[curr_layer]->r_i().data(), begin_v_i);
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

  void sample_postprocessing(Bitmap *bits, int layer, VertexId* node_idx) {
    sampled_sgs[layer]->postprocessing(bits, node_idx);
    curr_dst_size = sampled_sgs[layer]->src_size;
    // curr_layer++;
  }


  // void add_pre_layer_edges() {
  //   std::cout << "layers " << layers << std::endl;
  //   for (int i = 0; i < layers; ++i) {
  //     std::cout << "layer " << i << std::endl;
  //     auto one_layer = sampled_sgs[i];
  //     int dst_size = one_layer->dst().size();
  //     int src_size = one_layer->src().size();
  //     std::cout << dst_size << " " << src_size << std::endl;
  //     std::cout << one_layer->c_o().size() << " " << one_layer->r_i().size() << std::endl;
  //   }

  //   for (int i = 1; i < layers; ++i) {
  //     auto curr_layer = sampled_sgs[i];
  //     auto pre_layer = sampled_sgs[i - 1];
  //     std::set<VertexId> tmp;
  //     std::vector<VertexId> curr_source;       // global id
  //     std::vector<VertexId> curr_destination;  // global id

  //     tmp.insert(pre_layer->dst().begin(), pre_layer->dst().end());
  //     tmp.insert(curr_layer->dst().begin(), curr_layer->dst().end());
  //     std::copy(tmp.begin(), tmp.end(), std::back_inserter(curr_destination));
  //     VertexId curr_dst_size = curr_destination.size();

  //     tmp.clear();
  //     tmp.insert(pre_layer->src().begin(), pre_layer->src().end());
  //     tmp.insert(curr_layer->src().begin(), curr_layer->src().end());
  //     std::copy(tmp.begin(), tmp.end(), std::back_inserter(curr_source));
  //     VertexId curr_src_size = curr_source.size();

  //     std::unordered_map<VertexId, VertexId> src_idx;
  //     for (int i = 0; i < curr_src_size; ++i) {
  //       src_idx[curr_source[i]] - i;
  //     }
      

  //     std::vector<VertexId> curr_row_offset(curr_dst_size + 1, 0);
  //     std::vector<VertexId> curr_row_indices;  // local id

      
  //     std::unordered_map<VertexId, std::unordered_set<VertexId>> count_offset;
  //     // add pre layer edges
  //     for (int i = 0; i < pre_layer->dst().size(); ++i) {
  //       // std::cout << i << " " << pre_layer->c_o(i) <<  " " << pre_layer->c_o(i + 1) << std::endl;
  //       for (int j = pre_layer->c_o(i); j < pre_layer->c_o(i + 1); ++j) {
  //         // if (j >= pre_layer->src_size) {
  //         //   std::cout << pre_layer->src_size << " " << i << " " << pre_layer->dst(i) << " " << j << " " << pre_layer->c_o(i) << " " << pre_layer->c_o(i + 1) << std::endl;
  //         // }
  //         count_offset[pre_layer->dst(i)].insert(pre_layer->src(pre_layer->r_i(j)));
  //       }
  //     }
  //     std::cout << "add pre done" << std::endl;
      
  //     // add curr layer edges
  //     for (int i = 0; i < curr_layer->dst().size(); ++i) {
  //       for (int j = curr_layer->c_o(i); j < curr_layer->c_o(i + 1); ++j) {
  //         count_offset[curr_layer->dst(i)].insert(curr_layer->src(curr_layer->r_i(j)));
  //       }
  //     }
  //     std::cout << "add curr done" << std::endl;

  //     // get curr layer column offset
  //     for (int i = 0; i < curr_dst_size; ++i) {
  //       curr_row_offset[i + 1] = count_offset[curr_destination[i]].size() + curr_row_offset[i];
  //       // printf("%d %d\n", i, curr_row_offset[i + 1]);
  //     }
  //     std::cout << "curr row offset done" << std::endl;

  //     curr_layer->alloc_vertices(curr_dst_size);
  //     for (int i = 0; i < curr_dst_size + 1; ++i) {
  //       curr_layer->c_o()[i] = curr_row_offset[i];
  //     }
  //     std::cout << "curr offset setdone" << std::endl;

  //     curr_layer->dst().clear();
  //     std::copy(curr_destination.begin(), curr_destination.end(), std::back_inserter(curr_layer->dst()));
  //     curr_layer->src().clear();
  //     std::copy(curr_source.begin(), curr_source.end(), std::back_inserter(curr_layer->src()));
  //     std::cout << "copy src dst done" << std::endl;

  //     int tmp_e_size = 0;
  //     for (int i = 0; i < curr_dst_size; ++i) {
  //       int dst = curr_destination[i];
  //       tmp_e_size += count_offset[dst].size();
  //     }
  //     curr_layer->alloc_edges(tmp_e_size);
  //     std::cout << "alloc edges done" << std::endl;

  //     for (int i = 0; i < curr_dst_size; ++i) {
  //       int dst = curr_destination[i];
  //       for (const auto& v : count_offset[dst]) {
  //         curr_layer->r_i()[curr_row_offset[i]++] = src_idx[v];
  //       }
  //     }
  //     std::cout << "r_i done" << std::endl;
  //   }
  //   // exit(0);


  //   for (int i = 0; i < layers; ++i) {
  //     std::cout << "layer " << i << std::endl;
  //     auto one_layer = sampled_sgs[i];
  //     int dst_size = one_layer->dst().size();
  //     int src_size = one_layer->src().size();
  //     std::cout << dst_size << " " << src_size << std::endl;
  //     std::cout << one_layer->c_o().size() << " " << one_layer->r_i().size() << std::endl;
  //   }
    
  // }

  void compute_one_layer(
      std::function<void(VertexId local_dst, VertexId *column_offset, VertexId *row_indices)> sparse_slot,
      VertexId layer) {
    //  void compute_one_layer(std::function<void(VertexId local_dst,
    //                 VertexId* column_offset, VertexId* row_indices)>sparse_slot,VertexId layer){

    // omp_set_num_threads(threads);
// #pragma omp parallel for num_threads(threads)
omp_set_num_threads(threads);
#pragma omp parallel for num_threads(threads)
    for (VertexId begin_v_i = 0; begin_v_i < sampled_sgs[layer]->v_size; begin_v_i += 1) {
      sparse_slot(begin_v_i, sampled_sgs[layer]->c_o().data(), sampled_sgs[layer]->r_i().data());
    }
  }

  void compute_one_layer_backward(
      std::function<void(VertexId local_dst, VertexId *column_offset, VertexId *row_indices)> sparse_slot,
      VertexId layer) {
    //   void compute_one_layer_backward(std::function<void(VertexId local_dst,
    //                     VertexId* column_offset, VertexId* row_indices)>sparse_slot,VertexId layer){
    // LOG_DEBUG("start backward");
    // omp_set_num_threads(threads);
// #pragma omp parallel for num_threads(threads)
omp_set_num_threads(threads);
#pragma omp parallel for num_threads(threads)
    for (VertexId begin_v_i = 0; begin_v_i < sampled_sgs[layer]->src_size; begin_v_i += 1) {
      sparse_slot(begin_v_i, sampled_sgs[layer]->r_o().data(), sampled_sgs[layer]->c_i().data());
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

  VertexId *column_offset_bak;
  VertexId *row_indices_bak;
  int threads;

  FullyRepGraph() {}
  FullyRepGraph(Graph<Empty> *graph) {
    global_vertices = graph->vertices;
    global_edges = graph->edges;
    owned_vertices = graph->owned_vertices;
    partitions = graph->partitions;
    partition_id = graph->partition_id;
    partition_offset = graph->partition_offset;
    graph_ = graph;
    threads = std::max(1, numa_num_configured_cpus() / 2 - 1);
    // LOG_DEBUG("SamplerSubgraph thraeds %d", threads);
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

  void update_graph(std::vector<VertexId> &sample_nids) {
    VertexId node_num = sample_nids.size();
    VertexId edge_num = 0;
    for (auto id : sample_nids) {
      edge_num += column_offset_bak[id + 1] - column_offset_bak[id];
    }
    // VertexId *ri = new VertexId[edge_num];
    // VertexId *co = new VertexId[node_num + 1];
    // LOG_DEBUG("node %d edges %d", node_num, edge_num);
    column_offset[0] = 0;
    for (int i = 0; i < node_num; ++i) {
      int dst = sample_nids[i];
      int edges = column_offset_bak[dst + 1] - column_offset_bak[dst];
      column_offset[i + 1] = column_offset[i] + edges;
// omp_set_num_threads(threads);
// omp_set_num_threads(threads);
//     #pragma omp parallel for num_threads(threads)

      for (int j = column_offset[i]; j < column_offset[i + 1]; ++j) {
        row_indices[j] = row_indices_bak[column_offset_bak[dst] + j - column_offset[i]];
      }
    }
    assert(column_offset[sample_nids.size()] == edge_num);
    global_vertices = node_num;
    global_edges = edge_num;
  }

  void back_to_global() {
    memcpy(column_offset, column_offset_bak, sizeof(VertexId) * (global_vertices + 1));
    memcpy(row_indices, row_indices_bak, sizeof(VertexId) * global_edges);
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

    column_offset_bak = new VertexId[global_vertices + 1];
    row_indices_bak = new VertexId[global_edges];
    memcpy(column_offset_bak, column_offset, sizeof(VertexId) * (global_vertices + 1));
    memcpy(row_indices_bak, row_indices, sizeof(VertexId) * global_edges);

    delete[] read_edge_buffer;
    delete[] tmp_offset;
  }
};

#endif