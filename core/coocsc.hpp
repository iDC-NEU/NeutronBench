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
#if CUDA_ENABLE
#include "ntsCUDA.hpp"
#endif

#ifndef COOCSC_HPP
#define COOCSC_HPP
#include <algorithm>
#include <map>
#include <vector>

#include "utils/rand.hpp"
class sampCSC {
 public:
  sampCSC() {
    v_size = 0;
    e_size = 0;
    size_dev_src = 0;
    size_dev_dst = 0;
    size_dev_src_max = 0;
    size_dev_dst_max = 0;
    size_dev_edge = 0;
    size_dev_edge_max = 0;
    // column_offset.clear();
    // row_indices.clear();
    // src_index.clear();
    // destination.clear();
    // source.clear();
    // column_indices = nullptr;
    // row_offset = nullptr;
    // row_indices = nullptr;
    // column_offset = nullptr;
    // source = nullptr;
    // destination = nullptr;
    // node_idx = nullptr;
    threads = std::max(1, numa_num_configured_cpus() / 2 - 1);
    // LOG_DEBUG("sampCSC thraeds %d", threads);
  }
  sampCSC(VertexId v_) {
    v_size = v_;
    e_size = 0;
    size_dev_src = 0;
    size_dev_dst = 0;
    size_dev_src_max = 0;
    size_dev_dst_max = 0;
    size_dev_edge = 0;
    size_dev_edge_max = 0;
    column_offset.resize(v_ + 1);
    destination.resize(v_);
    // node_idx = nullptr;
    threads = std::max(1, numa_num_configured_cpus() / 2 - 1);
    // LOG_DEBUG("sampCSC thraeds %d", threads);
  }

  sampCSC(VertexId v_, VertexId e_) {
    v_size = v_;
    e_size = e_;
    size_dev_src = 0;
    size_dev_dst = 0;
    size_dev_src_max = 0;
    size_dev_dst_max = 0;
    size_dev_edge = 0;
    size_dev_edge_max = 0;
    // source.resize(e_);
    destination.resize(v_);
    column_offset.resize(v_ + 1);
    column_indices.resize(e_);
    // row_offset.resize(e_);
    row_indices.resize(e_);
    edge_weight_backward.resize(e_);
    edge_weight_forward.resize(e_);

    // source = new VertexId[e_];
    // destination = new VertexId[v_];
    // column_offset = new VertexId[v_ + 1];
    // column_indices = new VertexId[e_];
    // row_offset = new VertexId[e_ + 1];
    // row_indices = new VertexId[e_];
    // edge_weight_forward = new ValueType[e_];
    // edge_weight_backward = new ValueType[e_];
    // node_idx = nullptr;
    threads = std::max(1, numa_num_configured_cpus() / 2 - 1);
    // LOG_DEBUG("sampCSC thraeds %d", threads);
  }

  ~sampCSC() {
    column_offset.clear();
    row_indices.clear();
    destination.clear();
    source.clear();
    row_offset.clear();
    column_indices.clear();
    FreeEdge(dev_row_indices);
    FreeEdge(dev_column_indices);
    FreeBuffer(dev_edge_weight_backward);
    FreeBuffer(dev_edge_weight_forward);
    FreeEdge(dev_destination);
    FreeEdge(dev_column_offset);
    FreeEdge(dev_row_offset);
    FreeEdge(dev_source);
    // delete[] node_idx;
  }

  // void init(std::vector<VertexId>& column_offset, std::vector<VertexId>& row_indices, std::vector<VertexId>& source,
  //           std::vector<VertexId>& destination) {
  //   assert(false);
  //   // this->column_offset = column_offset;
  //   // this->row_indices = row_indices;
  //   // this->source = source;
  //   // this->destination = destination;
  // }

  // TODO(sanzo): use memcpy, need arr size arg
  void init(std::vector<VertexId>& column_offset, std::vector<VertexId>& row_indices, std::vector<VertexId>& source,
            std::vector<VertexId>& destination) {
    this->column_offset = column_offset;
    this->row_indices = row_indices;
    this->source = source;
    this->destination = destination;
  }

  void compute_weight_forward(Graph<Empty>* graph) {
    edge_weight_forward.resize(e_size);
    assert(e_size == column_offset.back());
    assert(v_size + 1 == column_offset.size());
    omp_set_num_threads(threads);
#pragma omp parallel for
    for (VertexId i = 0; i < v_size; ++i) {
      for (VertexId j = column_offset[i]; j < column_offset[i + 1]; ++j) {
        VertexId src_id = source[row_indices[j]];
        VertexId dst_id = destination[i];
        edge_weight_forward[j] = nts::op::nts_norm_degree(graph, src_id, dst_id);
      }
    }
  }

  void compute_weight_backward(Graph<Empty>* graph) {
    edge_weight_backward.resize(e_size);
    assert(src_size + 1 == row_offset.size());
    omp_set_num_threads(threads);
#pragma omp parallel for
    for (VertexId i = 0; i < src_size; ++i) {
      for (VertexId j = row_offset[i]; j < row_offset[i + 1]; ++j) {
        VertexId src_id = source[i];
        VertexId dst_id = destination[column_indices[j]];
        edge_weight_backward[j] = nts::op::nts_norm_degree(graph, src_id, dst_id);
      }
    }
  }

  // void update_degree_of_csc(Graph<Empty>* graph) {
  void update_degree(Graph<Empty>* graph) {
    VertexId* outs = graph->out_degree_for_backward;
    VertexId* ins = graph->in_degree_for_backward;
    // omp_set_num_threads(threads);
    // #pragma omp parallel for num_threads(threads)
    // for (int i = 0; i < graph->vertices; ++i) {
    //   outs[i] = 0;
    //   ins[i] = 0;
    // }

    // int dst_size = v_size;
    // for (int i = 0; i < dst_size; ++i) {
    //   ins[destination[i]] += column_offset[i + 1] - column_offset[i];
    //   // #pragma omp parallel for num_threads(threads)
    //   for (int j = column_offset[i]; j < column_offset[i + 1]; ++j) {
    //     int local_src = row_indices[j];
    //     outs[source[local_src]]++;
    //   }
    // }
    int dst_size = v_size;
    omp_set_num_threads(threads);
#pragma omp parallel for
    for (int i = 0; i < dst_size; ++i) {
      ins[destination[i]] = 0;
    }

    omp_set_num_threads(threads);
#pragma omp parallel for
    for (int i = 0; i < dst_size; ++i) {
      ins[destination[i]] += column_offset[i + 1] - column_offset[i];
    }

    omp_set_num_threads(threads);
#pragma omp parallel for
    for (int i = 0; i < src_size; ++i) {
      outs[source[i]] = 0;
    }

    omp_set_num_threads(threads);
#pragma omp parallel for
    for (int i = 0; i < src_size; ++i) {
      outs[source[i]] += row_offset[i + 1] - row_offset[i];
    }

    // long sum_ins = 0, sum_outs = 0;
    // for (int i = 0; i < graph->vertices; ++i) {
    //     sum_ins += ins[i];
    //     sum_outs += outs[i];
    // }
    // assert(sum_ins == sum_outs);
  }

  //   void update_degree_of_csr(Graph<Empty>* graph) {
  // LOG_DEBUG("update_degree_of_csr");
  //     VertexId* outs = graph->out_degree_for_backward;
  //     VertexId* ins = graph->in_degree_for_backward;
  // #pragma omp parallel for
  //     for (int i = 0; i < graph->vertices; ++i) {
  //       outs[i] = 0;
  //       ins[i] = 0;
  //     }
  //     for (int i = 0; i < src_size; ++i) {
  //       ins[source[i]] += row_offset[i + 1] - row_offset[i];
  //       // #pragma omp parallel for
  //       for (int j = row_offset[i]; j < row_offset[i + 1]; ++j) {
  //         int local_dst = column_indices[j];
  //         outs[destination[local_dst]]++;
  //       }
  //     }
  //     // long sum_ins = 0, sum_outs = 0;
  //     // for (int i = 0; i < graph->vertices; ++i) {
  //     //     sum_ins += ins[i];
  //     //     sum_outs += outs[i];
  //     // }
  //     // assert(sum_ins == sum_outs);
  //   }

  void generate_csr_from_csc() {
    // assert(source.size() == destination.size());

    int dst_size = v_size;
    int edge_size = e_size;
    // row_offset.resize(src_size + 1);
    // memset(row_offset.data(), 0, sizeof(VertexId) * (src_size + 1));

    row_offset = std::vector<VertexId>(src_size + 1, 0);
    // assert(dst_size + 1 == column_offset.size());

    // omp_set_num_threads(threads);
    // #pragma omp parallel for num_threads(threads)
    //     for (int i = 0; i < src_size + 1; ++i) {
    //       assert(row_offset[i] == 0);
    //     }

    //     for (int i = 0; i < dst_size; ++i) {
    //       // #pragma omp parallel for
    // // #pragma omp parallel for num_threads(4)
    //       for (int j = column_offset[i]; j < column_offset[i + 1]; ++j) {
    //         int local_src = row_indices[j];
    //         row_offset[local_src + 1]++;
    //       }
    //     }

    // #pragma omp parallel for num_threads(threads)
    for (VertexId i = 0; i < edge_size; i++) {
      row_offset[row_indices[i] + 1]++;
    }

    for (int i = 1; i <= src_size; ++i) {
      row_offset[i] += row_offset[i - 1];
    }
    assert(row_offset[src_size] == column_offset[v_size]);
    assert(row_offset[src_size] == e_size);
    assert(row_offset.size() == src_size + 1);

    column_indices.resize(e_size);

    // std::vector<int> tmp_row_offset(row_offset.begin(), row_offset.end());
    // // #pragma omp parallel for
    // for (int i = 0; i < dst_size; ++i) {
    //   // #pragma omp parallel for
    //   for (int j = column_offset[i]; j < column_offset[i + 1]; ++j) {
    //     int local_src = row_indices[j];
    //     column_indices[tmp_row_offset[local_src]++] = i;
    //   }
    // }

    for (int i = 0; i < dst_size; ++i) {
      for (int j = column_offset[i]; j < column_offset[i + 1]; ++j) {
        int local_src = row_indices[j];
        column_indices[row_offset[local_src]++] = i;
      }
    }

    for (int i = src_size; i > 0; --i) {
      row_offset[i] = row_offset[i - 1];
    }
    row_offset[0] = 0;

    // LOG_DEBUG("afrt dome");
    // for (int i= 0; i < src_size; ++i) {
    //     assert(tmp_row_offset[i] == row_offset[i + 1]);
    // }
  }

  // void postprocessing() {
  //   src_size = 0;
  //   row_indices_debug.resize(e_size, 0);
  //   for (VertexId i_src = 0; i_src < e_size; i_src++) {
  //     //  printf("debug %d\n",i_src);
  //     row_indices_debug[i_src] = row_indices[i_src];
  //     if (0xFFFFFFFF == row_indices[i_src]) {
  //       continue;
  //     }
  //     auto iter = node_idx.find(row_indices[i_src]);
  //     // printf("%d\n",iter == node_idx.end());
  //     if (iter == node_idx.end()) {
  //       //    printf("debug %d\n",i_src);
  //       node_idx.insert(std::make_pair(row_indices[i_src], src_size));
  //       src_size++;
  //       //     printf("debug %d\n",i_src);
  //       source.push_back(row_indices[i_src]);
  //       row_indices[i_src] = src_size - 1;
  //       // reset src for computation
  //     } else {
  //       // redundant continue;
  //       assert(row_indices[i_src] == iter->first);
  //       row_indices[i_src] = iter->second;  // reset src for computation
  //     }
  //   }
  // }

  void postprocessing(Bitmap* bits, VertexId* node_idx) {
    // std::unordered_set<int> st;
    // for (VertexId i = 0; i < e_size; ++i) {
    //     VertexId node_id = row_indices[i];
    //     st.insert(node_id);
    //     assert(bits->get_bit(node_id) > 0);
    // }

    // assert(st.size() == bits->get_ones());
    //////////////////////////////
    VertexId all_node_num = bits->get_size();

    // if (!node_idx) {
    //   node_idx = new VertexId[all_node_num]{};
    // }
    // omp_set_num_threads(threads);
    // #pragma omp parallel for num_threads(threads)
    // for (int i = 0; i < all_node_num; ++i) {
    //   node_idx[i] = -1;
    // }

    // std::vector<int> node_idx(bits->size, -1);
    // VertexId unique_node_num = bits->get_ones();
    // source.resize(unique_node_num);
    source.clear();
    src_size = 0;
    // for (int i = 0; i < all_node_num; ++i) {
    //   if (bits->get_bit(i) > 0) {
    //     // source.push_back(i); // TODO(pre-alloc)
    //     source[src_size] = i;
    //     node_idx[i] = src_size++;
    //   }
    // }

    int length = WORD_OFFSET(all_node_num) + 1;
    for (VertexId i_src = 0; i_src < all_node_num; i_src += 64) {
      unsigned long word = bits->data[WORD_OFFSET(i_src)];
      VertexId vtx = i_src;
      VertexId offset = 0;
      while (word != 0) {
        if (word & 1) {
          // printf("#dst %d %d\n",vtx+offset, src_size);
          // ssg->sampled_sgs[i]->src_index.insert(std::make_pair(vtx+offset, src_size));
          // src_index_array[vtx+offset]=ssg->sampled_sgs[i]->src_size;
          // ssg->sampled_sgs[i]->source.push_back(vtx+offset);
          // ssg->sampled_sgs[i]->src_size++;
          // source[src_size] = vtx + offset;
          source.push_back(vtx + offset);
          node_idx[vtx + offset] = src_size++;
        }
        offset++;
        word = word >> 1;
      }
    }
    assert(src_size == source.size());
    /////////// check
    //     VertexId *temp = new VertexId[e_size];
    // #pragma omp parallel for
    //   for (int i = 0; i < e_size; ++i) {
    //     temp[i] = row_indices[i];
    //   }

    //     method1 -= get_time();
    // #pragma omp parallel for
    //     // for (size_t i = 0; i < row_indices.size(); ++i) {
    //     for (VertexId i = 0; i < e_size; ++i) {
    //       int src = row_indices[i];
    //       assert(node_idx[src] != -1);
    //       row_indices[i] = node_idx[src];
    //       // row_indices[i] = node_idx[row_indices[i]];
    //     }
    //     method1 += get_time();

    // #pragma omp parallel for
    //   for (int i = 0; i < e_size; ++i) {
    //     row_indices[i] = temp[i];
    //   }
    // method2-= get_time();
    // for (size_t i = 0; i < row_indices.size(); ++i) {
    omp_set_num_threads(threads);
#pragma omp parallel for
    for (VertexId i = 0; i < e_size; ++i) {
      int src = row_indices[i];
      // assert(node_idx[src] != -1);
      row_indices[i] = node_idx[src];
      // row_indices[i] = node_idx[row_indices[i]];
    }
    // method2 += get_time();

    // #pragma omp parallel for
    // for (int i = 0; i < e_size; ++i) {
    //   row_indices[i] = temp[i];
    // }

    // VertexId *temp2 = new VertexId[e_size];
    //   method3 -= get_time();
    //   // for (size_t i = 0; i < row_indices.size(); ++i) {
    // #pragma omp parallel for
    //   for (VertexId i = 0; i < e_size; ++i) {
    //     temp2[i] = node_idx[row_indices[i]];
    //         // row_indices[i] = node_idx[row_indices[i]];
    //   }
    // #pragma omp parallel for
    //   for (VertexId i = 0; i < e_size; ++i) {
    //     row_indices[i] = temp2[i];
    //         // row_indices[i] = node_idx[row_indices[i]];
    //   }
    //   method3 += get_time();
    // LOG_DEBUG("e_size %d single thread %.3f, read-write %.3f, temp %.3f", e_size, method2, method1, method3);

    // LOG_DEBUG("v_size %d e_size %d unique %d src-size %d dst.size %d src.size %d", v_size, e_size, cnt, src_size,
    // dst().size(), src().size());
  }

  void zero_debug_time() {
    method1 = 0;
    method2 = 0;
    method3 = 0;
  }
  void print_debug_time() {
    LOG_DEBUG("e_size %d single thread %.3f, read-write %.3f, temp %.3f", e_size, method2, method1, method3);
  }

  void allocate_vertex() {
    // destination = new VertexId[v_size]{};
    // column_offset = new VertexId[v_size + 1]{};
    destination.resize(v_size);
    column_offset.resize(v_size + 1);
  }

  void init_dst(VertexId* dst) {
    // #pragma omp parallel for num_threads(threads)
    omp_set_num_threads(threads);
#pragma omp parallel for
    for (int i = 0; i < v_size; ++i) {
      destination[i] = dst[i];
    }

    // memcpy(destination.data(), dst, sizeof(VertexId) * v_size);
  }

  void init_dst(std::vector<VertexId>& dst) {
    omp_set_num_threads(threads);
#pragma omp parallel for
    for (int i = 0; i < v_size; ++i) {
      destination[i] = dst[i];
    }
    // memcpy(destination.data(), dst.data(), sizeof(VertexId) * v_size);
  }

  void random_batch(std::vector<VertexId>& sampled_nids) {
    int all_node_length = sampled_nids.size();
    std::unordered_set<int> st;
    while (st.size() < v_size) {
      st.insert(rand_int(all_node_length));
    }
    assert(v_size == st.size());
    int idx = 0;
    for (auto& id : st) {
      destination[idx++] = sampled_nids[id];
    }
  }

  void allocate_co_from_dst() {
    assert(false);
    // v_size = destination.size();
    // column_offset.resize(v_size + 1, 0);
  }
  void allocate_edge() {
    assert(0);
    // row_indices.resize(e_size, 0);
  }
  void allocate_edge(VertexId e_size) {
    assert(false);
    this->e_size = e_size;
    // row_indices = new VertexId[e_size]{};
    row_indices.resize(e_size);
  }
  void update_edges(VertexId e_size) { this->e_size = e_size; }
  void alloc_edges(VertexId e_size) {
    this->e_size = e_size;
    row_indices.resize(e_size);
  }
  void update_vertices(VertexId v_size) { this->v_size = v_size; }
  void alloc_vertices(VertexId v_size) {
    this->v_size = v_size;
    destination.resize(v_size);
    column_offset.resize(v_size + 1);
  }
  void allocate_all() {
    allocate_vertex();
    allocate_edge();
  }

  VertexId c_o(VertexId vid) { return column_offset[vid]; }
  VertexId r_i(VertexId vid) { return row_indices[vid]; }
  VertexId c_i(VertexId vid) { return column_indices[vid]; }
  VertexId r_o(VertexId vid) { return row_offset[vid]; }
  std::vector<VertexId>& dst() { return destination; }
  std::vector<VertexId>& src() { return source; }
  VertexId dst(VertexId idx) {
    assert(idx < v_size);
    return destination[idx];
  }
  VertexId src(VertexId idx) {
    assert(idx < src_size);
    return source[idx];
  }
  std::vector<VertexId>& c_o() { return column_offset; }
  std::vector<VertexId>& r_i() { return row_indices; }
  std::vector<VertexId>& r_o() { return row_offset; }
  std::vector<VertexId>& c_i() { return column_indices; }
  ValueType* dev_ewf() { return dev_edge_weight_forward; }
  ValueType* dev_ewb() { return dev_edge_weight_backward; }
  VertexId* dev_c_o() { return dev_column_offset; }
  VertexId* dev_r_o() { return dev_row_offset; }
  VertexId* dev_c_i() { return dev_column_indices; }
  VertexId* dev_r_i() { return dev_row_indices; }

  VertexId get_distinct_src_size() { return src_size; }
  void debug() {
    assert(false);
    // printf("print one layer:\ndst:\t");
    // for (int i = 0; i < destination.size(); i++) {
    //   printf("%d\t", destination[i]);
    // }
    // printf("\nc_o:\t");
    // for (int i = 0; i < column_offset.size(); i++) {
    //   printf("%d\t", column_offset[i]);
    // }
    // printf("\nr_i:\t");
    // for (int i = 0; i < row_indices.size(); i++) {
    //   printf("%d\t", row_indices[i]);
    // }
    // printf("\nrid:\t");
    // for (int i = 0; i < row_indices_debug.size(); i++) {
    //   printf("%d\t", row_indices_debug[i]);
    // }
    // printf("\nsrc:\t");
    // for (int i = 0; i < source.size(); i++) {
    //   printf("%d\t", source[i]);
    // }
    // printf("\n\n");
  }

  void debug_generate_csr_from_csc() {
    // LOG_DEBUG("start debgug_generate");
    std::vector<std::pair<int, int>> edge_csc, edge_csr;
    // LOG_DEBUG("after vector");
    for (int i = 0; i < src_size; ++i) {
      for (int j = row_offset[i]; j < row_offset[i + 1]; ++j) {
        edge_csr.push_back(std::make_pair(i, column_indices[j]));
      }
    }
    // LOG_DEBUG("after push edge_csr");

    for (int i = 0; i < v_size; ++i) {
      for (int j = column_offset[i]; j < column_offset[i + 1]; ++j) {
        edge_csc.push_back(std::make_pair(row_indices[j], i));
      }
    }
    // LOG_DEBUG("csc.size() %d csr.size() %d", edge_csc.size(), edge_csr.size());
    assert(edge_csc.size() == edge_csr.size());

    auto cmp = [](const auto& l, const auto& r) {
      if (l.first != r.first) {
        return l.first < r.first;
      }
      return l.second < r.second;
    };

    sort(edge_csr.begin(), edge_csr.end(), cmp);
    sort(edge_csc.begin(), edge_csc.end(), cmp);
    // LOG_DEBUG("fater sort");
    int edge_size = edge_csc.size();
    for (int i = 0; i < edge_size; ++i) {
      // printf("(%d %d) - (%d %d)\n", edge_csr[i].first, edge_csr[i].second, edge_csc[i].first, edge_csc[i].second);
      assert(edge_csc[i].first == edge_csr[i].first);
      assert(edge_csc[i].second == edge_csr[i].second);
    }
    // assert(false);
  }

  void alloc_dev_array(bool pull = true) {
    //////// TODO(Sanzo): (realloc)
    const float mem_factor = 1.2;
    if (v_size + 1 > size_dev_dst_max) {
      if (size_dev_dst_max > 0) {
        // free_gpu_data(dev_column_offset);
        // free_gpu_data(dev_destination);
        FreeEdge(dev_destination);
        FreeEdge(dev_column_offset);
      }
      size_dev_dst_max = (v_size + 1) * mem_factor;
      // alloc_gpu_data(&dev_destination, size_dev_dst_max);
      // alloc_gpu_data(&dev_column_offset, size_dev_dst_max);
      allocate_gpu_edge(&dev_destination, size_dev_dst_max);
      allocate_gpu_edge(&dev_column_offset, size_dev_dst_max);

      size_dev_dst = v_size;
    } else {
      size_dev_dst = v_size;
    }

    if ((src_size + 1) > size_dev_src_max) {
      if (size_dev_src_max > 0) {
        FreeEdge(dev_row_offset);
        FreeEdge(dev_source);
      }
      size_dev_src_max = (src_size + 1) * mem_factor;
      allocate_gpu_edge(&dev_source, size_dev_src_max);
      allocate_gpu_edge(&dev_row_offset, size_dev_src_max);
      size_dev_src = src_size;
    } else {
      size_dev_src = src_size;
    }

    if (e_size > size_dev_edge_max) {
      if (size_dev_edge_max > 0) {
        FreeEdge(dev_row_indices);
        FreeEdge(dev_column_indices);
        FreeBuffer(dev_edge_weight_backward);
        FreeBuffer(dev_edge_weight_forward);
      }
      size_dev_edge_max = e_size * mem_factor;
      allocate_gpu_edge(&dev_row_indices, size_dev_edge_max);
      allocate_gpu_edge(&dev_column_indices, size_dev_edge_max);
      allocate_gpu_buffer(&dev_edge_weight_forward, size_dev_edge_max);
      allocate_gpu_buffer(&dev_edge_weight_backward, size_dev_edge_max);
      size_dev_edge = e_size;
    } else {
      size_dev_edge = e_size;
    }
  }

  void alloc_dev_array_async(cudaStream_t stream, bool pull = true) {
    //////// TODO(Sanzo): (realloc)
    const float mem_factor = 1.2;
    if (v_size + 1 > size_dev_dst_max) {
      if (size_dev_dst_max > 0) {
        // free_gpu_data(dev_column_offset);
        // free_gpu_data(dev_destination);
        FreeEdgeAsync(dev_destination, stream);
        FreeEdgeAsync(dev_column_offset, stream);
      }
      size_dev_dst_max = (v_size + 1) * mem_factor;
      // alloc_gpu_data(&dev_destination, size_dev_dst_max);
      // alloc_gpu_data(&dev_column_offset, size_dev_dst_max);
      allocate_gpu_edge_async(&dev_destination, size_dev_dst_max, stream);
      allocate_gpu_edge_async(&dev_column_offset, size_dev_dst_max, stream);

      size_dev_dst = v_size;
    } else {
      size_dev_dst = v_size;
    }

    if ((src_size + 1) > size_dev_src_max) {
      if (size_dev_src_max > 0) {
        FreeEdgeAsync(dev_row_offset, stream);
        FreeEdgeAsync(dev_source, stream);
      }
      size_dev_src_max = (src_size + 1) * mem_factor;
      allocate_gpu_edge_async(&dev_source, size_dev_src_max, stream);
      allocate_gpu_edge_async(&dev_row_offset, size_dev_src_max, stream);
      size_dev_src = src_size;
    } else {
      size_dev_src = src_size;
    }

    if (e_size > size_dev_edge_max) {
      if (size_dev_edge_max > 0) {
        FreeEdgeAsync(dev_row_indices, stream);
        FreeEdgeAsync(dev_column_indices, stream);
        FreeBufferAsync(dev_edge_weight_backward, stream);
        FreeBufferAsync(dev_edge_weight_forward, stream);
      }
      size_dev_edge_max = e_size * mem_factor;
      allocate_gpu_edge_async(&dev_row_indices, size_dev_edge_max, stream);
      allocate_gpu_edge_async(&dev_column_indices, size_dev_edge_max, stream);
      allocate_gpu_buffer_async(&dev_edge_weight_forward, size_dev_edge_max, stream);
      allocate_gpu_buffer_async(&dev_edge_weight_backward, size_dev_edge_max, stream);
      size_dev_edge = e_size;
    } else {
      size_dev_edge = e_size;
    }
  }

  void copy_data_to_device(bool pull = true) {
    move_bytes_in(dev_column_offset, column_offset.data(), (v_size + 1) * sizeof(VertexId));
    move_bytes_in(dev_row_indices, row_indices.data(), e_size * sizeof(VertexId));

    move_bytes_in(dev_source, source.data(), src_size * sizeof(VertexId));
    move_bytes_in(dev_destination, destination.data(), v_size * sizeof(VertexId));

    if (pull) {
      copy_csr_to_device();
    }
  }

  void copy_data_to_device_async(cudaStream_t cs, bool pull = true) {
    move_bytes_in_async(dev_column_offset, column_offset.data(), (v_size + 1) * sizeof(VertexId), cs);
    move_bytes_in_async(dev_row_indices, row_indices.data(), e_size * sizeof(VertexId), cs);

    move_bytes_in_async(dev_source, source.data(), src_size * sizeof(VertexId), cs);
    move_bytes_in_async(dev_destination, destination.data(), v_size * sizeof(VertexId), cs);

    if (pull) {
      copy_csr_to_device_async(cs);
    }
  }

  void copy_ewf_to_device() {
    move_bytes_in(dev_edge_weight_forward, edge_weight_forward.data(), e_size * sizeof(ValueType));
  }

  void copy_ewb_to_device() {
    move_bytes_in(dev_edge_weight_backward, edge_weight_backward.data(), e_size * sizeof(ValueType));
  }

  void copy_csr_to_device() {
    move_bytes_in(dev_row_offset, row_offset.data(), (src_size + 1) * sizeof(VertexId));
    move_bytes_in(dev_column_indices, column_indices.data(), e_size * sizeof(VertexId));
  }

  void copy_ewf_to_device_async(cudaStream_t cs) {
    move_bytes_in_async(dev_edge_weight_forward, edge_weight_forward.data(), e_size * sizeof(ValueType), cs);
  }

  void copy_ewb_to_device_async(cudaStream_t cs) {
    move_bytes_in_async(dev_edge_weight_backward, edge_weight_backward.data(), e_size * sizeof(ValueType), cs);
  }

  void copy_csr_to_device_async(cudaStream_t cs) {
    move_bytes_in_async(dev_row_offset, row_offset.data(), (src_size + 1) * sizeof(VertexId), cs);
    move_bytes_in_async(dev_column_indices, column_indices.data(), e_size * sizeof(VertexId), cs);
  }

  // private:
  // std::vector<VertexId> source;          // global id
  // std::vector<VertexId> destination;     // global id
  // std::vector<VertexId> row_offset;
  // std::vector<VertexId> row_indices;     // local id
  // std::vector<VertexId> column_offset;
  // std::vector<VertexId> column_indices;  // local id
  // std::vector<VertexId> edge_weight_forward;   // local id
  // std::vector<VertexId> edge_weight_backward;  // local id
  // std::vector<VertexId> row_indices_debug;  // local id
  // VertexId* source;
  // VertexId* destination;

  // std::unordered_map<VertexId, VertexId> src_index;  // set
  int threads;
  // VertexId* node_idx;

  VertexId v_size;    // dst_size
  VertexId e_size;    // edge size
  VertexId src_size;  // distinct src size

  std::vector<VertexId> source;       // global id
  std::vector<VertexId> destination;  // global id
  std::vector<VertexId> row_offset;
  std::vector<VertexId> row_indices;  // local id
  std::vector<VertexId> column_offset;
  std::vector<VertexId> column_indices;         // local id
  std::vector<ValueType> edge_weight_forward;   // local id
  std::vector<ValueType> edge_weight_backward;  // local id
  std::vector<VertexId> row_indices_debug;      // local id

  VertexId* dev_source;
  VertexId* dev_destination;
  VertexId* dev_row_offset;
  VertexId* dev_row_indices;
  VertexId* dev_column_offset;
  VertexId* dev_column_indices;
  ValueType* dev_edge_weight_forward;
  ValueType* dev_edge_weight_backward;

  VertexId size_dev_src, size_dev_dst, size_dev_src_max, size_dev_dst_max;
  VertexId size_dev_co, size_dev_ri, size_dev_ewf;
  VertexId size_dev_ci, size_dev_ro, size_dev_ewb;
  VertexId size_dev_edge, size_dev_edge_max = 0;
  double method1, method2, method3;

  void copy_to(sampCSC* csc) {
    csc->threads = threads;
    csc->v_size = v_size;
    csc->e_size = e_size;
    csc->src_size = src_size;
    csc->threads = threads;
    csc->threads = threads;
    csc->threads = threads;
    csc->threads = threads;
  }
};

#endif