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

#include <algorithm>
#include <map>
#include <vector>
#ifndef COOCSC_HPP
#define COOCSC_HPP

class sampCSC {
 public:
  sampCSC() {
    v_size = 0;
    e_size = 0;
    column_indices = nullptr;
    row_offset = nullptr;
    row_indices = nullptr;
    column_offset = nullptr;
    source = nullptr;
    destination = nullptr;
    node_idx = nullptr;
  }
  sampCSC(VertexId v_, VertexId e_) {
    v_size = v_;
    e_size = e_;
    // column_offset.resize(v_ + 1);
    // row_indices.resize(e_);
    // row_offset.resize(e_);
    // column_indices.resize(e_);
    // destination.resize(v_);
    // source.resize(e_);
    source = new VertexId[e_];
    destination = new VertexId[v_];
    column_offset = new VertexId[v_ + 1];
    column_indices = new VertexId[e_];
    row_offset = new VertexId[e_ + 1];
    row_indices = new VertexId[e_];
    edge_weight_forward = new ValueType[e_];
    edge_weight_backward = new ValueType[e_];
    node_idx = nullptr;
  }

  void init(std::vector<VertexId>& column_offset, std::vector<VertexId>& row_indices, std::vector<VertexId>& source,
            std::vector<VertexId>& destination) {
    assert(false);
    // this->column_offset = column_offset;
    // this->row_indices = row_indices;
    // this->source = source;
    // this->destination = destination;
  }

  // TODO(sanzo): use memcpy, need arr size arg
  void init(VertexId* column_offset, VertexId* row_indices, VertexId* source, VertexId* destination) {
    this->column_offset = column_offset;
    this->row_indices = row_indices;
    this->source = source;
    this->destination = destination;
  }

  void compute_weight_forward(Graph<Empty>* graph) {
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
#pragma omp parallel for
    for (VertexId i = 0; i < src_size; ++i) {
      for (VertexId j = row_offset[i]; j < row_offset[i + 1]; ++j) {
        VertexId src_id = source[i];
        VertexId dst_id = destination[column_indices[j]];
        edge_weight_backward[j] = nts::op::nts_norm_degree(graph, src_id, dst_id);
      }
    }
  }

  sampCSC(VertexId v_) {
    v_size = v_;
    e_size = 0;
    column_offset = new VertexId[v_ + 1]{};
    row_indices = nullptr;
    row_offset = nullptr;
    column_indices = nullptr;
    node_idx = nullptr;
    destination = new VertexId[v_];
    source = nullptr;
  }
  // void alloc_index_table(VertexId size) {
  //     node_idx[]
  // }
  ~sampCSC() {
    delete[] column_offset;
    delete[] row_indices;
    delete[] node_idx;
    delete[] destination;
    delete[] source;
    delete[] row_offset;
    delete[] column_indices;
  }

  // void update_degree_of_csc(Graph<Empty>* graph) {
  void update_degree(Graph<Empty>* graph) {
    VertexId* outs = graph->out_degree_for_backward;
    VertexId* ins = graph->in_degree_for_backward;
#pragma omp parallel for
    for (int i = 0; i < graph->vertices; ++i) {
      outs[i] = 0;
      ins[i] = 0;
    }
    int dst_size = v_size;
    for (int i = 0; i < dst_size; ++i) {
      ins[destination[i]] += column_offset[i + 1] - column_offset[i];
      // #pragma omp parallel for
      for (int j = column_offset[i]; j < column_offset[i + 1]; ++j) {
        int local_src = row_indices[j];
        outs[source[local_src]]++;
      }
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
    // assert(row_offset.size() >= src_size + 1);
    memset(row_offset, 0, sizeof(VertexId) * (src_size + 1));
#pragma omp parallel for
    for (int i = 0; i < src_size + 1; ++i) {
      assert(row_offset[i] == 0);
    }
    for (int i = 0; i < dst_size; ++i) {
      // #pragma omp parallel for
      for (int j = column_offset[i]; j < column_offset[i + 1]; ++j) {
        int local_src = row_indices[j];
        row_offset[local_src + 1]++;
      }
    }
    for (int i = 1; i <= src_size; ++i) {
      row_offset[i] += row_offset[i - 1];
    }
    assert(row_offset[src_size] == column_offset[v_size]);
    assert(row_offset[src_size] == e_size);
    std::vector<int> tmp_row_offset(row_offset, row_offset + src_size + 1);

    // #pragma omp parallel for
    for (int i = 0; i < dst_size; ++i) {
      // #pragma omp parallel for
      for (int j = column_offset[i]; j < column_offset[i + 1]; ++j) {
        int local_src = row_indices[j];
        column_indices[tmp_row_offset[local_src]++] = i;
      }
    }
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

  void postprocessing(Bitmap* bits) {
    // std::unordered_set<int> st;
    // for (VertexId i = 0; i < e_size; ++i) {
    //     VertexId node_id = row_indices[i];
    //     st.insert(node_id);
    //     assert(bits->get_bit(node_id) > 0);
    // }

    // assert(st.size() == bits->get_ones());
    //////////////////////////////

    if (!node_idx) {
      node_idx = new VertexId[bits->get_size()]{};
      // LOG_DEBUG("alloc node_idx in first call, size %d", bits->get_size());
    }
#pragma omp parallel for
    for (int i = 0; i < bits->get_size(); ++i) {
      node_idx[i] = -1;
    }

    // std::vector<int> node_idx(bits->size, -1);

    src_size = 0;
    for (int i = 0; i < bits->size; ++i) {
      if (bits->get_bit(i) > 0) {
        // source.push_back(i); // TODO(pre-alloc)
        source[src_size] = i;
        node_idx[i] = src_size++;
      }
    }

#pragma omp parallel for
    // for (size_t i = 0; i < row_indices.size(); ++i) {
    for (VertexId i = 0; i < e_size; ++i) {
      int src = row_indices[i];
      assert(node_idx[src] != -1);
      row_indices[i] = node_idx[src];
    }
    // LOG_DEBUG("v_size %d e_size %d unique %d src-size %d dst.size %d src.size %d", v_size, e_size, cnt, src_size,
    // dst().size(), src().size());
  }

  void allocate_vertex() {
    destination = new VertexId[v_size]{};
    column_offset = new VertexId[v_size + 1]{};
  }

  void init_dst(VertexId* dst) { memcpy(destination, dst, sizeof(VertexId) * v_size); }
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
    // row_indices.resize(e_size, 0);
    row_indices = new VertexId[e_size]{};
  }
  void update_edges(VertexId e_size) { this->e_size = e_size; }
  void update_vertices(VertexId v_size) { this->v_size = v_size; }
  void allocate_all() {
    allocate_vertex();
    allocate_edge();
  }
  VertexId c_o(VertexId vid) { return column_offset[vid]; }
  VertexId r_i(VertexId vid) { return row_indices[vid]; }
  VertexId c_i(VertexId vid) { return column_indices[vid]; }
  VertexId r_o(VertexId vid) { return row_offset[vid]; }
  VertexId* dst() { return destination; }
  VertexId* src() { return source; }
  VertexId* c_o() { return column_offset; }
  VertexId* r_i() { return row_indices; }
  VertexId* r_o() { return row_offset; }
  VertexId* c_i() { return column_indices; }
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

  void allocate_dev_array(VertexId vtx_size, VertexId edge_size) {
    // column_offset = (VertexId *)cudaMallocPinned((vtx_size + 1) * sizeof(VertexId));
    // dev_column_offset = (VertexId *)cudaMallocGPU((vtx_size + 1) * sizeof(VertexId));
    // row_offset = (VertexId *)cudaMallocPinned((edge_size + 1) * sizeof(VertexId));
    // dev_row_offset = (VertexId *)cudaMallocGPU((edge_size + 1) * sizeof(VertexId));

    // row_indices = (VertexId *)cudaMallocPinned((edge_size + 1) * sizeof(VertexId));
    // edge_weight_forward = (ValueType *)cudaMallocPinned((edge_size + 1) * sizeof(ValueType));
    // dev_row_indices = (VertexId *)cudaMallocGPU((edge_size + 1) * sizeof(VertexId));
    // dev_edge_weight_forward = (ValueType *)cudaMallocGPU((edge_size + 1) * sizeof(ValueType));

    // column_indices = (VertexId *)cudaMallocPinned((edge_size + 1) * sizeof(VertexId));
    // edge_weight_backward = (ValueType *)cudaMallocPinned((edge_size + 1) * sizeof(ValueType));
    // dev_column_indices = (VertexId *)cudaMallocGPU((edge_size + 1) * sizeof(VertexId));
    // dev_edge_weight_backward = (ValueType *)cudaMallocGPU((edge_size + 1) * sizeof(ValueType));
  }

  // private:
  VertexId* column_offset;
  VertexId* row_offset;
  VertexId* row_indices;     // local id
  VertexId* column_indices;  // local id
  VertexId* source;          // global id
  VertexId* destination;     // global id

  std::vector<VertexId> row_indices_debug;  // local id

  // VertexId* source;
  // VertexId* destination;

  // std::unordered_map<VertexId, VertexId> src_index;  // set
  VertexId* node_idx;

  VertexId v_size;    // dst_size
  VertexId e_size;    // edge size
  VertexId src_size;  // distinct src size

  ValueType* edge_weight_forward;   // local id
  ValueType* edge_weight_backward;  // local id
  ValueType* dev_edge_weight_forward;
  ValueType* dev_edge_weight_backward;

  VertexId* dev_source;
  VertexId* dev_destination;

  VertexId* dev_column_offset;
  VertexId* dev_row_indices;
  VertexId* dev_column_indices;
  VertexId* dev_row_offset;

  VertexId size_dev_co, size_dev_ri, size_dev_ewf;
  VertexId size_dev_ci, size_dev_ro, size_dev_ewb;
};

#endif