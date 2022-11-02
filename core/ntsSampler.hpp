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
#ifndef NTSSAMPLER_HPP
#define NTSSAMPLER_HPP
#include <stdlib.h>

#include <cmath>
#include <mutex>
#include <random>

#include "FullyRepGraph.hpp"
#include "core/MetisPartition.hpp"

enum Device { CPU, GPU };
class Sampler {
 public:
  std::vector<SampledSubgraph*> work_queue;  // excepted to be single write multi read
  SampledSubgraph* subgraph;
  std::mutex queue_start_lock;
  int queue_start;
  std::mutex queue_end_lock;
  int queue_end;
  FullyRepGraph* whole_graph;
  VertexId start_vid, end_vid;
  VertexId work_range[2];
  VertexId work_offset;
  std::vector<VertexId> sample_nids;
  std::vector<int> fanout;
  std::vector<VertexId> metis_partition_offset;
  std::vector<VertexId> metis_partition_id;
  VertexId batch_size;
  VertexId batch_nums;
  VertexId layers;
  Cuda_Stream* cs;

  double sample_pre_time = 0;
  double sample_load_dst = 0;
  double sample_init_co = 0;
  double sample_post_time = 0;
  double sample_processing_time = 0;
  double layer_time = 0;
  // int batch_size;
  // int layers;

  Bitmap* sample_bits;

  int gpu_id = 0;
  Device device = CPU;

  void zero_debug_time() {
    sample_pre_time = 0;
    sample_load_dst = 0;
    sample_init_co = 0;
    sample_post_time = 0;
    sample_processing_time = 0;
    layer_time = 0;
  }

  // template<typename T>
  // T RandInt(T lower, T upper) {
  //     std::uniform_int_distribution<T> dist(lower, upper - 1);
  //     return dist(rng_);
  // }
  // std::default_random_engine rng_;

  Sampler(FullyRepGraph* whole_graph_, VertexId work_start, VertexId work_end) {
    cs = new Cuda_Stream();
    whole_graph = whole_graph_;
    queue_start = -1;
    queue_end = 0;
    work_range[0] = work_start;
    work_range[1] = work_end;
    work_offset = work_start;
    work_queue.clear();
    sample_bits = new Bitmap(whole_graph->global_vertices);
  }
  Sampler(FullyRepGraph* whole_graph_, std::vector<VertexId>& index) {
    // Sampler(FullyRepGraph* whole_graph_, std::vector<VertexId>& index, Device dev = CPU, int gpu_id = 0) {
    // this->device = dev;
    // this->gpu_id = gpu_id;
    cs = new Cuda_Stream();
    // assert(index.size() > 0);
    sample_nids.assign(index.begin(), index.end());
    assert(sample_nids.size() == index.size());
    whole_graph = whole_graph_;
    queue_start = -1;
    queue_end = 0;
    work_range[0] = 0;
    work_range[1] = sample_nids.size();
    work_offset = 0;
    // LOG_DEBUG("vertices %d", whole_graph->global_vertices);
    sample_bits = new Bitmap(whole_graph->global_vertices);

    fanout = whole_graph->graph_->gnnctx->fanout;
    batch_size = whole_graph->graph_->config->batch_size;
    if (work_range[1] < batch_size) batch_size = work_range[1];
    batch_nums = (work_range[1] + batch_size - 1) / batch_size;
    layers = whole_graph->graph_->gnnctx->layer_size.size() - 1;
    // all_nodes = sample_nids.size();
    // assert(layers = 2);
    // for (int i = 0; i < work_range[1]; i += batch_size) {
    //     VertexId actl_size = std::min(batch_size, work_range[1] - i);
    //     work_queue.push_back(new SampledSubgraph(layers, fanout));
    //     work_queue.back()->allocate_memory(actl_size);
    // }
    work_queue.clear();
    subgraph = new SampledSubgraph(layers, fanout);
    // pre_alloc_one();
    // assert (false);
  }

  void trans_to_gpu() {
    auto sub_graph = subgraph;
    for (int i = 0; i < layers; ++i) {
      sub_graph->sampled_sgs[i]->alloc_dev_array();
      sub_graph->sampled_sgs[i]->copy_data_to_device();
    }
  }

  void pre_alloc_one() {
    work_queue.push_back(new SampledSubgraph(layers, fanout));
    work_queue.back()->alloc_memory(batch_size);
  }

  void load_feature_gpu(NtsVar& local_feature, ValueType* global_feature_buffer) {
    auto csc_layer = subgraph->sampled_sgs[0];
    if (local_feature.size(0) < csc_layer->src_size) {
      local_feature.resize_({csc_layer->src_size, local_feature.size(1)});
    }
    ValueType* local_feature_buffer =
        whole_graph->graph_->Nts->getWritableBuffer(local_feature, torch::DeviceType::CUDA);
    cs->zero_copy_feature_move_gpu(local_feature_buffer, global_feature_buffer, csc_layer->dev_source,
                                   local_feature.size(1), csc_layer->src_size);
  }

  void load_label_gpu(NtsVar& local_label, long* global_label_buffer) {
    auto csc_layer = subgraph->sampled_sgs[layers - 1];
    auto classes = whole_graph->graph_->config->classes;
    assert(classes > 0);
    if (classes > 1 &&
        (local_label.dim() != 2 || local_label.size(0) != csc_layer->v_size || local_label.size(1) != classes)) {
      local_label.resize_({csc_layer->v_size, classes});
    }

    if (classes == 1 && local_label.size(0) != csc_layer->v_size) {
      local_label.resize_({csc_layer->v_size});
    }

    long* local_label_buffer = nullptr;
    if (classes > 1) {
      local_label_buffer = whole_graph->graph_->Nts->getWritableBuffer2d<long>(local_label, torch::DeviceType::CUDA);
    } else {
      local_label_buffer = whole_graph->graph_->Nts->getWritableBuffer1d<long>(local_label, torch::DeviceType::CUDA);
    }
    assert(local_label_buffer != nullptr);

    if (classes > 1) {
      cs->global_copy_mulilabel_move_gpu(local_label_buffer, global_label_buffer, csc_layer->dev_destination,
                                         csc_layer->v_size, classes);
    } else {
      cs->global_copy_label_move_gpu(local_label_buffer, global_label_buffer, csc_layer->dev_destination,
                                     csc_layer->v_size);
    }
  }

  NtsVar get_label(VertexId* dst, VertexId dst_size, NtsVar& whole, Graph<Empty>* graph) {
    NtsVar f_output;
    if (graph->config->classes > 1) {
      f_output = graph->Nts->NewLeafKLongTensor({dst_size, graph->config->classes});
    } else {
      f_output = graph->Nts->NewLeafKLongTensor({dst_size});
    }

#pragma omp parallel for
    for (int i = 0; i < dst_size; i++) {
      f_output[i] = whole[dst[i] - graph->partition_offset[graph->partition_id]];
    }
    return f_output;
  }

  ~Sampler() { clear_queue(); }
  bool has_rest_safe() {
    bool condition = false;
    int cond_start = 0;
    queue_start_lock.lock();
    cond_start = queue_start;
    queue_start_lock.unlock();

    int cond_end = 0;
    queue_end_lock.lock();
    cond_end = queue_end;
    queue_end_lock.unlock();

    condition = cond_start < cond_end && cond_start >= 0;
    return condition;
  }

  bool has_rest() {
    return queue_start < queue_end && queue_start >= 0;
    ;
  }
  //    bool has_rest(){
  //        bool condition=false;
  //        condition=queue_start<queue_end&&queue_start>=0;
  //        return condition;
  //    }
  SampledSubgraph* get_one_safe() {
    //        while(true){
    //            bool condition=queue_start<queue_end;
    //            if(condition){
    //                break;
    //            }
    //         __asm volatile("pause" ::: "memory");
    //        }
    queue_start_lock.lock();
    VertexId id = queue_start++;
    queue_start_lock.unlock();
    assert(id < work_queue.size());
    return work_queue[id];
  }

  SampledSubgraph* get_one() {
    VertexId id = queue_start++;
    assert(id < work_queue.size());
    return work_queue[id];
  }

  int size() { return work_queue.size(); }

  void push_one_safe(SampledSubgraph* ssg) {
    work_queue.push_back(ssg);
    queue_end_lock.lock();
    queue_end++;
    queue_end_lock.unlock();
    if (work_queue.size() == 1) {
      queue_start_lock.lock();
      queue_start = 0;
      queue_start_lock.unlock();
    }
  }

  void push_one(SampledSubgraph* ssg) {
    work_queue.push_back(ssg);
    queue_end++;
    if (work_queue.size() == 1) {
      queue_start = 0;
    }
  }

  void clear_queue() {
    for (VertexId i = 0; i < work_queue.size(); i++) {
      delete work_queue[i];
    }
    work_queue.clear();
  }
  bool sample_not_finished() { return work_offset < work_range[1]; }
  void restart() {
    work_offset = work_range[0];
    queue_start = -1;
    queue_end = 0;
  }

  int random_uniform_int(const int min = 0, const int max = 1) {
    // thread_local std::default_random_engine generator;
    // unsigned seed = 2000;
    // static thread_local std::mt19937 generator(seed);
    static thread_local std::mt19937 generator;
    std::uniform_int_distribution<int> distribution(min, max);
    return distribution(generator);
  }

  template <typename T>
  T rand_int(T upper) {
    return rand_int<T>(0, upper);
  }

  // random from [lower, upper)
  template <typename T>
  T rand_int(T lower, T upper) {
    assert(lower < upper);
    // unsigned seed = 2000;
    // static thread_local std::mt19937 generator(seed);
    static thread_local std::mt19937 generator;
    std::uniform_int_distribution<T> distribution(lower, upper - 1);
    return distribution(generator);
  }

  void sample_one(int type = 0, bool phase = true) {
    // zero_debug_time();
    // void reservoir_sample(int layers, int batch_size_, const
    // std::vector<int>& fanout_, int type = 0){ LOG_DEBUG("layers %d batch_size
    // %d fanout %d-%d", layers, batch_size_, fanout_[0], fanout_[1]);
    assert(work_offset < work_range[1]);
    // assert(batch_size == batch_size_);
    VertexId actl_batch_size = std::min(batch_size, work_range[1] - work_offset);
    // LOG_DEBUG("actl_batch %d", actl_batch_size);
    // SampledSubgraph* ssg=new SampledSubgraph(layers,fanout_);
    // auto ssg = work_queue[work_offset / batch_size_];
    auto ssg = subgraph;
    // LOG_DEBUG("fuck batch_size %d", actl_batch_size);
    ssg->curr_dst_size = actl_batch_size;
    for (int i = 0; i < layers; i++) {
      layer_time -= get_time();
      ssg->curr_layer = i;
      sample_load_dst -= get_time();
      auto csc_layer = ssg->sampled_sgs[i];
      // LOG_DEBUG("sample_one layer %d update_vertices done, dst_size %d", i, ssg->curr_dst_size);
      // csc_layer->update_vertices(ssg->curr_dst_size);
      csc_layer->alloc_vertices(ssg->curr_dst_size);
      if (i == 0) {
        csc_layer->init_dst(sample_nids.data() + work_offset);
        // LOG_DEBUG("dst size %d", csc_layer->v_size);
        /////////////// check
        // for (int j = 0; j < csc_layer->v_size; ++j) {
        //   assert(csc_layer->dst()[j] == sample_nids[j + work_offset]);
        // }
        // LOG_DEBUG("layer 0 src size %d", csc_layer->src().size());
      } else {
        csc_layer->init_dst(ssg->sampled_sgs[i - 1]->src());
        // LOG_DEBUG("dst size %d", csc_layer->v_size);
        // LOG_DEBUG("after init_dst csc v_size %d, pre src size %d", csc_layer->v_size, ssg->sampled_sgs[i -
        // 1]->src_size);
        for (int j = 0; j < csc_layer->v_size; ++j) {
          // std::cout << "dst size " << csc_layer->dst().size() << std::endl;
          // std::cout << "pre src size " << ssg->sampled_sgs[i - 1]->src().size() << std::endl;
          // std::cout << j << " " << csc_layer->dst()[j] <<  " " << ssg->sampled_sgs[i - 1]->src()[j] << std::endl;
          assert(csc_layer->dst()[j] == ssg->sampled_sgs[i - 1]->src()[j]);
        }
      }
      // LOG_DEBUG("sample_one layer %d init_dst done", i);
      // LOG_DEBUG("after init dst");
      // ssg->sampled_sgs.push_back(csc_layer);
      sample_load_dst += get_time();
      // LOG_DEBUG("sample_load_dst cost %.3f", sample_load_dst);

      sample_init_co -= get_time();
      ssg->init_co(
          [&](VertexId dst) {
            int nbrs = whole_graph->column_offset[dst + 1] - whole_graph->column_offset[dst];
            if (fanout[i] < 0) return nbrs;
            return std::min(nbrs, fanout[i]);
          },
          i);
      sample_init_co += get_time();
      // LOG_DEBUG("sample_one layer %d init_column_offset done", i);

      // LOG_DEBUG("sample_init_co cost %.3f", sample_init_co);

      sample_bits->clear();
      sample_processing_time -= get_time();
      // LOG_DEBUG("fanout_i %d\n", fanout_i[i]);
      // ssg->sample_processing(std::bind(&Sampler::NeighborUniformSample, this, std::placeholders::_1,
      // std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5));
      ssg->sample_processing(
          [&](VertexId fanout_i, VertexId dst, VertexId* column_offset, VertexId* row_indices, VertexId id) {
            this->NeighborUniformSample(fanout_i, dst, column_offset, row_indices, id);
            // this->NeighborUniformSample_reservoir(fanout_i, dst, column_offset, row_indices, id);
          });
      sample_processing_time += get_time();
      // LOG_DEBUG("sample_one layer %d processing done", i);

      // whole_graph->SyncAndLog("sample_processing");
      sample_post_time -= get_time();
      ssg->sample_postprocessing(sample_bits, i);
      // ssg->sample_postprocessing();
      sample_post_time += get_time();
      // LOG_DEBUG("sample_one layer %d post_processing done", i);
      // LOG_DEBUG("sample_post %.3f", sample_post);
      layer_time += get_time();
    }
    std::reverse(ssg->sampled_sgs.begin(), ssg->sampled_sgs.end());
    work_offset += actl_batch_size;
    // LOG_DEBUG("layer %.3f, pre_time %.3f, load_dst_time %.3f, init_co %.3f, processing %.3f, post_time %.3f,",
    // layer_time, sample_pre_time, sample_load_dst, sample_init_co, sample_processing_time, sample_post_time);
    // push_one(ssg);
    // printf("debug: sample one done!\n");
  }

  void reverse_sgs() {
    for (auto ssg : work_queue) {
      std::reverse(ssg->sampled_sgs.begin(), ssg->sampled_sgs.end());
    }
  }

  void RandomSample(size_t set_size, size_t num, std::vector<size_t>& out) {
    if (num < set_size) {
      std::unordered_set<size_t> sampled_idxs;
      while (sampled_idxs.size() < num) {
        sampled_idxs.insert(rand_int(set_size));
      }
      out.insert(out.end(), sampled_idxs.begin(), sampled_idxs.end());
    } else {
      for (size_t i = 0; i < set_size; ++i) {
        out.push_back(i);
      }
    }
  }

  /*
   * For a sparse array whose non-zeros are represented by nz_idxs,
   * negate the sparse array and outputs the non-zeros in the negated array.
   */
  void NegateArray(const std::vector<size_t>& nz_idxs, size_t arr_size, std::vector<size_t>& out) {
    // nz_idxs must have been sorted.
    auto it = nz_idxs.begin();
    size_t i = 0;
    // CHECK_GT(arr_size, nz_idxs.back());
    assert(arr_size > nz_idxs.back());
    for (; i < arr_size && it != nz_idxs.end(); i++) {
      if (*it == i) {
        it++;
        continue;
      }
      out.push_back(i);
    }
    for (; i < arr_size; i++) {
      out.push_back(i);
    }
  }

  void NeighborUniformSample_reservoir(VertexId fanout_i, VertexId dst, VertexId* column_offset, VertexId* row_indices,
                                       VertexId id) {
    for (VertexId src_idx = whole_graph->column_offset[dst]; src_idx < whole_graph->column_offset[dst + 1]; src_idx++) {
      // ReservoirSampling
      VertexId write_pos = (src_idx - whole_graph->column_offset[dst]);
      if (write_pos < fanout_i) {
        write_pos += column_offset[id];
        row_indices[write_pos] = whole_graph->row_indices[src_idx];
      } else {
        // VertexId random=rand()%write_pos;
        // VertexId random=rand_r(&seeds[omp_get_thread_num()])%write_pos;
        VertexId random = random_uniform_int(0, write_pos - 1);
        if (random < fanout_i) {
          row_indices[random + column_offset[id]] = whole_graph->row_indices[src_idx];
          // sample_bits->set_bit(whole_graph->row_indices[src_idx]);
        }
      }
    }
    for (int i = 0; i < std::min(fanout_i, whole_graph->column_offset[dst + 1] - whole_graph->column_offset[dst]);
         ++i) {
      sample_bits->set_bit(row_indices[i + column_offset[id]]);
    }
  }

  void NeighborUniformSample(VertexId fanout_i, VertexId dst, VertexId* column_offset, VertexId* row_indices,
                             VertexId id) {
    auto whole_offset = whole_graph->column_offset;
    auto whole_indices = whole_graph->row_indices;
    VertexId edge_nums = whole_offset[dst + 1] - whole_offset[dst];
    ////////////////////////////////////////////////
    // LOG_DEBUG("edge_nusm %d, fanout %d", edge_nums, fanout_i);
    if (edge_nums <= fanout_i) {
      // LOG_DEBUG("  just return");
      // double small_time = -get_time();
      int pos = column_offset[id];
      for (int i = 0; i < edge_nums; ++i) {
        row_indices[pos++] = whole_indices[whole_offset[dst] + i];
        sample_bits->set_bit(whole_indices[whole_offset[dst] + i]);
      }
      // LOG_DEBUG("edge_nupm %d fanout %d offset [%d,%d] id %d", edge_nums, fanout_i, column_offset[id],
      // column_offset[id + 1], id);
      assert(pos == column_offset[id + 1]);
      // small_time += get_time();
      // LOG_DEBUG("small time %.3f", small_time);
      return;
    }

    assert(fanout_i < edge_nums);
    // LOG_DEBUG("  try to reallocated");

    // int pos1 = column_offset[id];
    // for (int i = 0; i < fanout_i; ++i) {
    //     row_indices[pos1++] = whole_indices[whole_offset[dst] + i];
    //     sample_bits->set_bit(whole_indices[whole_offset[dst] + i]);
    // }
    // return;

    std::vector<size_t> sorted_idxs;
    double random_time = -get_time();
    // LOG_DEBUG("saorted_idx size %d", sorted_idxs.size());
    // assert(sorted_idxs.size() == fanout_i);
    // sorted_idxs.reserve(fanout_i);
    if (edge_nums > 2 * fanout_i) {
      // sorted_idxs.reserve(fanout_i);
      RandomSample(edge_nums, fanout_i, sorted_idxs);
      std::sort(sorted_idxs.begin(), sorted_idxs.end());
    } else {
      std::vector<size_t> negate;
      // negate.reserve(edge_nums - fanout_i);
      RandomSample(edge_nums, edge_nums - fanout_i, negate);
      // LOG_DEBUG("after RandomSample");
      std::sort(negate.begin(), negate.end());
      NegateArray(negate, edge_nums, sorted_idxs);
      // LOG_DEBUG("after NegateArray");
    }
    random_time = -get_time();
    // LOG_DEBUG("random time %.3f", random_time);
    // LOG_DEBUG("after random");
#pragma omp parallel for
    for (size_t i = 1; i < sorted_idxs.size(); ++i) {
      assert(sorted_idxs[i] > sorted_idxs[i - 1]);
    }
    assert(sorted_idxs.size() == fanout_i);
    ///////////////////////////////////////////////////

    // std::unordered_set<size_t> sorted_idxs;
    // int actl_fanout = min(fanout_i, edge_nums);
    // while (sorted_idxs.size() < actl_fanout) {
    //   sorted_idxs.insert(rand_int(edge_nums));
    // }

    int pos = column_offset[id];
    for (auto& idx : sorted_idxs) {
      row_indices[pos++] = whole_indices[whole_offset[dst] + idx];
      sample_bits->set_bit(whole_indices[whole_offset[dst] + idx]);
    }
    assert(pos == column_offset[id + 1]);
  }

  void LayerUniformSample(int layers, int batch_size, std::vector<int> fanout) {
    // construct layer
    assert(work_offset < work_range[1]);
    int actl_batch_size = std::min((VertexId)batch_size, work_range[1] - work_offset);
    // VertexId* indices = whole_graph->srcList;
    VertexId* indices = whole_graph->row_indices;
    VertexId* indptr = whole_graph->column_offset;

    // std::vector<VertexId> layer_offset;
    std::vector<VertexId> node_mapping;
    std::vector<VertexId> layer_sizes;
    // std::vector<float> probabilities;

    std::copy(sample_nids.begin() + work_offset, sample_nids.begin() + work_offset + actl_batch_size,
              std::back_inserter(node_mapping));
    // LOG_DEBUG("copy %d %d len %d", work_offset, work_offset + actl_batch_size, sample_nids.size());
    work_offset += actl_batch_size;
    layer_sizes.push_back(node_mapping.size());
    VertexId curr = 0;
    VertexId next = node_mapping.size();
    // LOG_DEBUG("start construct layer");
    for (int i = layers - 1; i >= 0; --i) {
      // LOG_DEBUG("layer %d layer_size %d %d %d", i, layer_size, curr, next);
      std::unordered_set<VertexId> candidate_set;
      for (int j = curr; j < next; ++j) {
        VertexId dst = node_mapping[j];
        candidate_set.insert(indices + indptr[dst], indices + indptr[dst + 1]);
      }
      // printf("layer %d layer size %d\n", i, candidate_set.size());
      // LOG_DEBUG("candidate_set is done");
      std::vector<VertexId> candidate_vector;
      copy(candidate_set.begin(), candidate_set.end(), std::back_inserter(candidate_vector));
      // LOG_DEBUG("candidate_vector is done");
      int layer_size = std::min(fanout[i], (int)candidate_vector.size());
      if (layer_size == -1) layer_size = candidate_vector.size();

      std::unordered_map<VertexId, size_t> n_occurrences;
      size_t n_candidates = candidate_vector.size();
      unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
      std::shuffle(candidate_vector.begin(), candidate_vector.end(), std::default_random_engine(seed));

      for (int j = 0; j < layer_size; ++j) {
        // TODO(sanzo): thread safe
        // VertexId dst = candidate_vector[RandInt(0, n_candidates)];
        // VertexId dst = candidate_vector[rand() % n_candidates];
        VertexId dst = candidate_vector[j];
        if (!n_occurrences.insert(std::make_pair(dst, 1)).second) {
          ++n_occurrences[dst];
        }
      }
      // LOG_DEBUG("layer node is done");
      for (auto const& pair : n_occurrences) {
        node_mapping.push_back(pair.first);
      }
      // LOG_DEBUG("add node to node_mapping");

      layer_sizes.push_back(node_mapping.size() - next);
      curr = next;
      next = node_mapping.size();
    }

    std::reverse(node_mapping.begin(), node_mapping.end());
    std::reverse(layer_sizes.begin(), layer_sizes.end());
    // std::vector<int64_t> layer_offset;
    // layer_offset.push_back(0);
    // for (auto const &size : layer_sizes) {
    //     layer_offset.push_back(layer_offset.back() + size);
    // }
    // LOG_DEBUG("consruct layer done");

    SampledSubgraph* ssg = ConstructSampledSubgraph(layers, layer_sizes, node_mapping);
    push_one(ssg);
  }

  // construct subgraph
  SampledSubgraph* ConstructSampledSubgraph(int layers, std::vector<VertexId>& layer_sizes,
                                            std::vector<VertexId>& node_mapping) {
    auto indptr = whole_graph->column_offset;
    auto indices = whole_graph->row_indices;
    SampledSubgraph* ssg = new SampledSubgraph();
    VertexId curr = 0;
    // LOG_DEBUG("start construct subgraph");
    for (int i = 0; i < layers; ++i) {
      // LOG_DEBUG("layer %d", i);
      // size_t src_size = layer_sizes[i];
      size_t src_size = layer_sizes[i];
      std::unordered_map<VertexId, VertexId> source_map;
      std::vector<VertexId> sources;
      // TODO(sanzo): redundancy copy
      std::copy(node_mapping.begin() + curr, node_mapping.begin() + curr + src_size, std::back_inserter(sources));
      for (int j = 0; j < src_size; ++j) {
        source_map.insert(std::make_pair(node_mapping[curr + j], j));
      }
      // printf("source_map size %d\n", source_map.size());
      // LOG_DEBUG("source_map is done");

      std::vector<VertexId> sub_edges;
      size_t dst_size = layer_sizes[i + 1];
      std::vector<VertexId> destination;
      // TODO(sanzo): redundancy copy
      std::copy(node_mapping.begin() + curr + src_size, node_mapping.begin() + curr + src_size + dst_size,
                std::back_inserter(destination));
      std::vector<VertexId> column_offset;
      column_offset.push_back(0);
      // LOG_DEBUG("start select src node dst_size %d", dst_size);
      // printf("layer %d src_size %d dst_size %d\n", i, src_size, dst_size);
      int tmp_cnt = 0;
      for (int j = 0; j < dst_size; ++j) {
        VertexId dst = node_mapping[curr + src_size + j];
        // LOG_DEBUG("j %d dst %d", j, dst);
        // std::pair<VertexId, VertexId> id_pair;
        // std::vertex<VertexId> neighbor_indices;
        for (int k = indptr[dst]; k < indptr[dst + 1]; ++k) {
          tmp_cnt++;
          if (source_map.find(indices[k]) != source_map.end()) {
            // source_map.push_back(indices[k]);
            sub_edges.push_back(source_map[indices[k]]);
            // } else {
            //     printf("layer %d %d is not in source_map\n", i, indices[k]);
          }
        }
        // printf("sub_edges.size = %d\n", sub_edges.size());
        column_offset.push_back(sub_edges.size());
      }
      // printf("sub_edges  size %d tmp_cnt %d\n", sub_edges.size(), tmp_cnt);

      sampCSC* sample_sg = new sampCSC(dst_size, sub_edges.size());
      sample_sg->init(column_offset, sub_edges, sources, destination);
      // sample_sg->generate_csr_from_csc();
      ssg->sampled_sgs.push_back(sample_sg);
      // LOG_DEBUG("subgraph is done");
      curr += src_size;
    }
    return ssg;
  }

  void ClusterGCNSample(int layers, int batch_size, int partition_num, std::string objtype = "cut") {
    if (metis_partition_id.empty()) {
      double metis_time = -get_time();
      // (FIXME Sanzo) store metis partion result to file
      MetisPartitionGraph(whole_graph, partition_num, objtype, metis_partition_id, metis_partition_offset);
      metis_time += get_time();
      printf("metis partition cost %.3f\n", metis_time);
    }

    std::vector<int> random_partition(partition_num);
    std::iota(random_partition.begin(), random_partition.end(), 0);
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::shuffle(random_partition.begin(), random_partition.end(), std::default_random_engine(seed));
    std::unordered_set<VertexId> node_set;
    // std::set<VertexId> node_set;
    // vector<VertexId> node_ids_check;
    auto& offset = metis_partition_offset;
    auto& ids = metis_partition_id;
    int actl_batch_size = std::min(batch_size, partition_num);
    work_offset += actl_batch_size;
    // std::cout << "actl batch size " << actl_batch_size << std::endl;
    for (int i = 0; i < actl_batch_size; ++i) {
      auto select_part = random_partition[i];
      // node_set.insert(select_part);
      node_set.insert(ids.begin() + offset[select_part], ids.begin() + offset[select_part + 1]);
      // for (int j = offset[i]; j < offset[i + 1]; ++j) {
      //     node_ids_check.push_back(ids[j]);
      // }
    }
    // std::cout << "node set " << node_set.size() << std::endl;
    // int compare_ret = std::equal(node_ids.begin(), node_ids.end(), node_ids_check.begin());
    // assert(compare_ret);
    // std::cout << "node_ids.size() " << node_ids.size() << std::endl;
    vector<VertexId> node_ids(node_set.begin(), node_set.end());
    // std::cout << "select nodes node_ids " << node_ids.size() << std::endl;

    // std::vector<VertexId> column_offset;
    // std::vector<VertexId> row_indices;
    // column_offset.push_back(0);
    // for (auto dst : node_ids) {
    //     for (int i = whole_graph->column_offset[dst]; i < whole_graph->column_offset[dst + 1]; ++i) {
    //         int src = whole_graph->row_indices[i];
    //         if (node_set.find(src) == node_set.end()) continue;
    //         row_indices.push_back(src);
    //     }
    //     column_offset.push_back(row_indices.size());
    // }

    std::vector<VertexId> layer_sizes(layers + 1, node_ids.size());
    std::vector<VertexId> node_mapping;
    for (int i = 0; i < layers + 1; ++i) std::copy(node_ids.begin(), node_ids.end(), std::back_inserter(node_mapping));

    // for (int i = 0; i < layers; ++i) {
    //     std::cout << "layer " << i << " " << layer_sizes[i] << std::endl;
    // }
    // std::cout << "node mapping size " << node_mapping.size() << std::endl;
    // std::cout << "start const graph " << std::endl;
    // std::cout << layers << " " << layer_sizes.size() << " " << node_mapping.size() << std::endl;
    SampledSubgraph* ssg = ConstructSampledSubgraph(layers, layer_sizes, node_mapping);
    // std::cout << "end const graph " << std::endl;
    push_one(ssg);
  }

  void isValidSampleGraph() {
    // set batch size to whole graph vertices num, thus sample graph equal to full graph
    // auto sample_graph = work_queue.front();
    // unordered_set<int> nodes(sample_nids.begin(), sample_nids.end());
    // for (int i = 1; i >= 0; --i) { // default two layer
    //     auto layer_graph = sample_graph->sampled_sgs[i];
    //     auto src_nodes = layer_graph->source;
    //     auto dst_nodes = layer_graph->destination;
    //     auto column_offset = layer_graph->column_offset;
    //     auto row_indices = layer_graph->row_indices;
    //     for (int i = 0; i < dst_nodes.size(); ++i) {
    //         auto dst = dst_nodes[i];
    //         std::cout << dst << " " << std::endl;
    //         assert(nodes.find(dst) != nodes.end());
    //         int sample_edges_num = column_offset[i + 1] - column_offset[i];
    //         int actl_edges_num = whole_graph->column_offset[dst + 1] - whole_graph->column_offset[dst];
    //         printf("edges %d %d\n", sample_edges_num, actl_edges_num);
    //         assert(sample_edges_num == actl_edges_num);
    //     }

    //     nodes.clear();
    //     for (auto& dst : dst_nodes) {
    //         nodes.insert(&row_indices[column_offset[dst]], &row_indices[column_offset[dst + 1]]);
    //     }
    //     printf("next layer node size %d\n", nodes.size());
    // }
    auto sample_graph = work_queue.front();
    unordered_set<int> nodes[2];
    nodes[1].insert(sample_nids.begin(), sample_nids.end());
    vector<int> edge_cnt(2, 0);
    for (int i = 1; i >= 0; --i) {
      for (auto& dst : nodes[i]) {
        // printf("dst %d\n", dst);
        int start = whole_graph->column_offset[dst];
        int end = whole_graph->column_offset[dst + 1];
        // printf("%d start %d end %d\n", dst, start, end);
        edge_cnt[i] += end - start;
        if (i > 0) {
          nodes[i - 1].insert(&whole_graph->row_indices[start], &whole_graph->row_indices[end]);
        }
        // printf("insert done!\n");
      }
      // printf("check layer %d nodes %d edges %d\n", i, nodes[i].size(), edge_cnt[i]);
      // assert(nodes[i].size() == sample_graph->sampled_sgs[i]->destination);
      // printf("layer %d sample (%d %d), actl (%d %d)\n", i, sample_graph->sampled_sgs[i]->v_size
      //       , sample_graph->sampled_sgs[i]->e_size, nodes[i].size(), edge_cnt[i]);
      assert(nodes[i].size() == sample_graph->sampled_sgs[i]->v_size);
      assert(edge_cnt[i] == sample_graph->sampled_sgs[i]->e_size);

      // printf("next layer nodes sizse %d\n", nodes[i - 1].size());
    }
  }
};

#endif