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
#include "utils/rand.hpp"

enum Device { CPU, GPU };
class Sampler {
 public:
  std::vector<SampledSubgraph*> work_queue;  // excepted to be single write multi read
  SampledSubgraph* subgraph;
  SampledSubgraph** subgraph_list;
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
  std::vector<std::vector<VertexId>> batch_nodes;
  VertexId batch_size;
  VertexId batch_nums;
  VertexId layers;
  VertexId metis_batch_id;
  Cuda_Stream* cs;
  bool full_batch = false;
  int batch_size_switch_idx = 0;
  std::vector<int> batch_size_vec;

  int sample_rate_switch_idx = 0;
  std::vector<float> sample_rate_vec;

  double sample_pre_time = 0;
  double sample_load_dst = 0;
  double sample_init_co = 0;
  double sample_post_time = 0;
  double sample_processing_time = 0;
  double sample_convert_graph_time = 0;
  double sample_compute_weight = 0;

  double layer_time = 0;
  int best_acc_epoch = -1;
  float best_val_acc = 0.0;

  int threads;
  // int batch_size;
  // int layers;

  Bitmap* sample_bits;
  VertexId* node_idx;

  int gpu_id = 0;
  Device device = CPU;

  void zero_debug_time() {
    sample_pre_time = 0;
    sample_load_dst = 0;
    sample_init_co = 0;
    sample_post_time = 0;
    sample_processing_time = 0;
    layer_time = 0;
    sample_convert_graph_time = 0;
    sample_compute_weight = 0;
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
    update_threads();
    // LOG_DEBUG("Sampeler thraeds %d", threads);
  }

  Sampler(FullyRepGraph* whole_graph_, std::vector<VertexId>& index, bool full_batch = false) {
    this->full_batch = full_batch;
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
    node_idx = new VertexId[whole_graph->global_vertices];

    fanout = whole_graph->graph_->gnnctx->fanout;

    // if (whole_graph_->graph_->config->batch_switch_time <= 0) {

    ////////////////////////////////////////////////////////
    update_batch_size(whole_graph->graph_->config->batch_size);

    ////////////////////////////////////////////////////////
    // batch_size = whole_graph->graph_->config->batch_size;
    // if (work_range[1] < batch_size || full_batch) batch_size = work_range[1];
    // batch_nums = (work_range[1] + batch_size - 1) / batch_size;
    ////////////////////////////////////////////////

    // if (whole_graph->graph_->config->batch_type == METIS) {
    //   batch_nums == batch_nodes.size();
    // }
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

    ////////////////////////////////////////////////
    batch_size_vec = whole_graph_->graph_->config->batch_size_vec;
    batch_size_switch_idx = -1;
    // LOG_DEBUG("show batch_size_vec in Samper constructor functioni:");
    // for (auto it : batch_size_vec) {
    //   std::cout << it << " ";
    // }
    // std::cout << std::endl;

    sample_rate_vec = whole_graph_->graph_->config->sample_rate_vec;
    sample_rate_switch_idx = -1;
    // LOG_DEBUG("show sampel_rate_vec in Samper constructor functioni:");
    // for (auto it : sample_rate_vec) {
    //   std::cout << it << " ";
    // }
    // std::cout << std::endl;
    // assert(false);
    update_threads();
    // LOG_DEBUG("Sampeler thraeds %d", threads);
  }

  Sampler(FullyRepGraph* whole_graph_, std::vector<VertexId>& index, int pipelines, bool full_batch = false) {
    this->full_batch = full_batch;
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
    node_idx = new VertexId[whole_graph->global_vertices];

    fanout = whole_graph->graph_->gnnctx->fanout;

    update_batch_size(whole_graph->graph_->config->batch_size);

    layers = whole_graph->graph_->gnnctx->layer_size.size() - 1;
    assert(layers == fanout.size());
    // }
    work_queue.clear();
    subgraph = new SampledSubgraph(layers, fanout);
    subgraph_list = new SampledSubgraph*[pipelines];
    for (int i = 0; i < pipelines; ++i) {
      subgraph_list[i] = new SampledSubgraph(layers, fanout);
    }

    ////////////////////////////////////////////////
    batch_size_vec = whole_graph_->graph_->config->batch_size_vec;
    batch_size_switch_idx = -1;
    sample_rate_vec = whole_graph_->graph_->config->sample_rate_vec;
    sample_rate_switch_idx = -1;
    update_threads();
    // LOG_DEBUG("Sampeler thraeds %d", threads);
  }

  void update_threads() {
    if (whole_graph->graph_->config->threads > 0 && whole_graph->graph_->config->threads <= numa_num_configured_cpus()) {
      threads = whole_graph->graph_->config->threads;
    } else {
      threads = std::max(1, numa_num_configured_cpus() / 2 - 1);
    }
  }

  void update_metis_data(std::vector<VertexId>& part_ids, std::vector<VertexId>& offsets) {
    metis_partition_id = std::move(part_ids);
    metis_partition_offset = std::move(offsets);
    std::unordered_set<VertexId> all_train_nodes;
    all_train_nodes.insert(sample_nids.begin(), sample_nids.end());
    assert(all_train_nodes.size() == sample_nids.size());
    int partition_nums = metis_partition_offset.size() - 1;
    assert(partition_nums > 0);

    int debug_cnt = 0;
    for (int i = 0; i < partition_nums; ++i) {
      std::vector<VertexId> tmp;
      for (int j = metis_partition_offset[i]; j < metis_partition_offset[i + 1]; ++j) {
        if (all_train_nodes.find(metis_partition_id[j]) != all_train_nodes.end()) {
          tmp.push_back(metis_partition_id[j]);
          debug_cnt++;
        }
      }
      if (!tmp.empty()) {
        batch_nodes.push_back(std::move(tmp));
      }
    }
    assert(debug_cnt == sample_nids.size());
    LOG_DEBUG("metis batch num %d", batch_nodes.size());
    for (int i = 0; i < partition_nums; ++i) {
      printf("metis batch id %d has %d nodes\n", i, batch_nodes[i].size());
    }
  }

  void update_batch_size(VertexId batch_size_) {
    // batch_size = whole_graph->graph_->config->batch_size;
    LOG_DEBUG("batch_size switch to %d", batch_size_);
    batch_size = batch_size_;
    if (work_range[1] < batch_size || full_batch) batch_size = work_range[1];
    batch_nums = (work_range[1] + batch_size - 1) / batch_size;
  }

  bool update_batch_size_from_time(float gcn_run_time) {
    int idx = gcn_run_time / whole_graph->graph_->config->batch_switch_time;
    if (idx != batch_size_switch_idx && idx < batch_size_vec.size()) {
      batch_size_switch_idx = idx;
      update_batch_size(batch_size_vec[batch_size_switch_idx]);
      return true;
    }
    return false;
  }

  bool update_sample_rate_from_time(float gcn_run_time) {
    int idx = gcn_run_time / whole_graph->graph_->config->sample_switch_time;
    if (idx != sample_rate_switch_idx && idx < sample_rate_vec.size()) {
      sample_rate_switch_idx = idx;
      whole_graph->graph_->config->sample_rate = sample_rate_vec[sample_rate_switch_idx];
      LOG_DEBUG("sample_rate switch to %.3f", whole_graph->graph_->config->sample_rate);
      return true;
    }
    return false;
  }

  bool update_batch_size_from_acc(int epoch, float val_acc, float gcn_run_time) {
    // LOG_DEBUG("epoch %d val_acc %.3f best_val_acc %.3f best_epoch %d", epoch, val_acc, best_val_acc, best_acc_epoch);
    if (best_acc_epoch == -1) {
      best_acc_epoch = epoch;
      best_val_acc = val_acc;
    } else if (val_acc > best_val_acc) {
      best_acc_epoch = epoch;
      best_val_acc = val_acc;
    } else if (epoch - best_acc_epoch >= whole_graph->graph_->config->batch_switch_acc) {
      if (batch_size_switch_idx + 1 < batch_size_vec.size()) {
        whole_graph->graph_->config->batch_size = batch_size_vec[++batch_size_switch_idx];
        LOG_DEBUG("epoch %d best_acc_epoch %d gcn_run_time %.3f, batch_size switch to %d", epoch, best_acc_epoch,
                  gcn_run_time, whole_graph->graph_->config->batch_size);
        best_acc_epoch = epoch;
        // LOG_DEBUG("epoch %d val_acc %.3f best_val_acc %.3f best_epoch %d", epoch, val_acc, best_val_acc,
        // best_acc_epoch);
        return true;
      }
    }
    return false;
  }

  void update_fanout(int nums) {
    for (auto& it : fanout) {
      it = nums;
    }
    for (auto it : fanout) {
      assert(it == nums);
    }
    subgraph->update_fanout(nums);
  }

  void update_fanout(std::vector<int> nums) {
    assert(fanout.size() == nums.size());
    for (int i = 0; i < nums.size(); ++i) {
      fanout[i] = nums[i];
    }
    subgraph->update_fanout(nums);
  }

  void show_fanout(std::string info) {
    std::cout << info << " ";
    for (auto& it : fanout) {
      printf("%d, ", it);
    }
    printf("\n");
  }

  void update_batch_nums(VertexId nums) { this->batch_nums = nums; }

  void update_batch_nums() { this->batch_nums = batch_nodes.size(); }

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

  void load_feature_gpu(Cuda_Stream* cuda_stream, SampledSubgraph* ssg, NtsVar& local_feature,
                        ValueType* global_feature_buffer) {
    auto csc_layer = ssg->sampled_sgs[0];
    // LOG_DEBUG("loacl_feature (%d %d) src_size %u", local_feature.size(0), local_feature.size(1),
    // csc_layer->src_size);
    if (local_feature.size(0) < csc_layer->src_size) {
      local_feature.resize_({csc_layer->src_size, local_feature.size(1)});
    }
    ValueType* local_feature_buffer =
        whole_graph->graph_->Nts->getWritableBuffer(local_feature, torch::DeviceType::CUDA);
    cuda_stream->zero_copy_feature_move_gpu(local_feature_buffer, global_feature_buffer, csc_layer->dev_source,
                                            local_feature.size(1), csc_layer->src_size);
  }

  std::pair<double, double> load_feature_gpu_cache(NtsVar& local_feature, ValueType* global_feature_buffer,
                                                   ValueType* dev_cache_feature, VertexId* local_idx,
                                                   VertexId* local_idx_cache, VertexId* cache_node_hashmap,
                                                   VertexId* dev_local_idx, VertexId* dev_local_idx_cache,
                                                   VertexId* dev_cache_node_hashmap) {
    // std::vector<int> &cache_node_hashmap) {

    auto csc_layer = subgraph->sampled_sgs[0];
    if (local_feature.size(0) < csc_layer->src_size) {
      local_feature.resize_({csc_layer->src_size, local_feature.size(1)});
    }
    ValueType* local_feature_buffer =
        whole_graph->graph_->Nts->getWritableBuffer(local_feature, torch::DeviceType::CUDA);

    // std::vector<int> local_idx_cache, local_idx;
    int local_idx_cnt = 0;
    int local_idx_cache_cnt = 0;
    // std::vector<int> local_idx_cache, global_idx_cache, local_idx, global_idx;
    // LOG_DEBUG("src_size %d vertices %d", csc_layer->src_size, whole_graph->graph_->vertices);
    for (int i = 0; i < csc_layer->src_size; ++i) {
      int node_id = csc_layer->src()[i];
      // LOG_DEBUG("node_id %d ", node_id);
      // LOG_DEBUG("cache_node_hashmap[node_id] %d", cache_node_hashmap[node_id]);
      if (cache_node_hashmap[node_id] != -1) {
        local_idx_cache[local_idx_cache_cnt++] = i;
        // local_idx_cache.push_back(cache_node_hashmap[node_id]);
        // global_idx_cache.push_back(csc_layer->src[i]);
      } else {
        local_idx[local_idx_cnt++] = i;
        // global_idx.push_back(csc_layer->src[i]);
      }
    }
    // LOG_DEBUG("start zero_copy_feature_move_gpU_cache");
    double trans_feature_cost = -get_time();
    cs->zero_copy_feature_move_gpu_cache(local_feature_buffer, global_feature_buffer, csc_layer->dev_source,
                                         local_feature.size(1), local_idx_cnt, dev_local_idx);
    trans_feature_cost += get_time();
    // LOG_DEBUG("gather_fature_from_gpu_cache");
    double gather_gpu_cache_cost = -get_time();
    cs->gather_feature_from_gpu_cache(local_feature_buffer, dev_cache_feature, csc_layer->dev_source,
                                      local_feature.size(1), local_idx_cache_cnt, dev_local_idx_cache,
                                      dev_cache_node_hashmap);
    gather_gpu_cache_cost += get_time();
    return {trans_feature_cost, gather_gpu_cache_cost};
  }

  std::pair<double, double> load_feature_gpu_cache(Cuda_Stream* cuda_stream, SampledSubgraph* ssg,
                                                   NtsVar& local_feature, ValueType* global_feature_buffer,
                                                   ValueType* dev_cache_feature, VertexId* local_idx,
                                                   VertexId* local_idx_cache, VertexId* cache_node_hashmap,
                                                   VertexId* dev_local_idx, VertexId* dev_local_idx_cache,
                                                   VertexId* dev_cache_node_hashmap) {
    // std::vector<int> &cache_node_hashmap) {

    auto csc_layer = ssg->sampled_sgs[0];
    if (local_feature.size(0) < csc_layer->src_size) {
      local_feature.resize_({csc_layer->src_size, local_feature.size(1)});
    }
    ValueType* local_feature_buffer =
        whole_graph->graph_->Nts->getWritableBuffer(local_feature, torch::DeviceType::CUDA);

    // std::vector<int> local_idx_cache, local_idx;
    int local_idx_cnt = 0;
    int local_idx_cache_cnt = 0;
    // std::vector<int> local_idx_cache, global_idx_cache, local_idx, global_idx;
    // LOG_DEBUG("src_size %d vertices %d", csc_layer->src_size, whole_graph->graph_->vertices);
    for (int i = 0; i < csc_layer->src_size; ++i) {
      int node_id = csc_layer->src()[i];
      // LOG_DEBUG("node_id %d ", node_id);
      // LOG_DEBUG("cache_node_hashmap[node_id] %d", cache_node_hashmap[node_id]);
      if (cache_node_hashmap[node_id] != -1) {
        local_idx_cache[local_idx_cache_cnt++] = i;
        // local_idx_cache.push_back(cache_node_hashmap[node_id]);
        // global_idx_cache.push_back(csc_layer->src[i]);
      } else {
        local_idx[local_idx_cnt++] = i;
        // global_idx.push_back(csc_layer->src[i]);
      }
    }
    // LOG_DEBUG("start zero_copy_feature_move_gpU_cache");
    double trans_feature_cost = -get_time();
    cuda_stream->zero_copy_feature_move_gpu_cache(local_feature_buffer, global_feature_buffer, csc_layer->dev_source,
                                                  local_feature.size(1), local_idx_cnt, dev_local_idx);
    trans_feature_cost += get_time();
    // LOG_DEBUG("gather_fature_from_gpu_cache");
    double gather_gpu_cache_cost = -get_time();
    cuda_stream->gather_feature_from_gpu_cache(local_feature_buffer, dev_cache_feature, csc_layer->dev_source,
                                               local_feature.size(1), local_idx_cache_cnt, dev_local_idx_cache,
                                               dev_cache_node_hashmap);
    gather_gpu_cache_cost += get_time();
    return {trans_feature_cost, gather_gpu_cache_cost};
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

  void load_label_gpu(Cuda_Stream* cuda_stream, SampledSubgraph* ssg, NtsVar& local_label, long* global_label_buffer) {
    auto csc_layer = ssg->sampled_sgs[layers - 1];
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
      cuda_stream->global_copy_mulilabel_move_gpu(local_label_buffer, global_label_buffer, csc_layer->dev_destination,
                                                  csc_layer->v_size, classes);
    } else {
      cuda_stream->global_copy_label_move_gpu(local_label_buffer, global_label_buffer, csc_layer->dev_destination,
                                              csc_layer->v_size);
    }
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

  void print_batch_nodes() {
    std::cout << "(v_size=" << subgraph->sampled_sgs.back()->v_size << ") ";
    auto ssg = subgraph;
    for (int i = 0; i < subgraph->sampled_sgs.back()->v_size; ++i) {
      std::cout << subgraph->sampled_sgs.back()->dst(i) << " ";
    }
    std::cout << std::endl;
  }

  void insert_batch_nodes(std::unordered_set<VertexId>& st) {
    auto ssg = subgraph;
    for (int i = 0; i < subgraph->sampled_sgs.back()->v_size; ++i) {
      st.insert(subgraph->sampled_sgs.back()->dst(i));
    }
  }

  uint64_t get_compute_cnt() {
    uint64_t ret = 0;
    for (int i = 0; i < layers; ++i) {
      ret += subgraph->sampled_sgs[i]->e_size;
    }
    return ret;
  }

  void sample_one(int type = 0, bool phase = true) { sample_one(subgraph, type, phase); }


  void sample_one_with_dst(SampledSubgraph* ssg, int type = 0, bool phase = true) {
    assert(work_offset < work_range[1]);
    // assert(batch_size == batch_size_);
    VertexId actl_batch_size = std::min(batch_size, work_range[1] - work_offset);
    if (phase && whole_graph->graph_->config->batch_type == METIS) {
      actl_batch_size = batch_nodes[metis_batch_id].size();
    }
    assert(actl_batch_size > 0);
    // LOG_DEBUG("actl_batch %d", actl_batch_size);
    // SampledSubgraph* ssg=new SampledSubgraph(layers,fanout_);
    // auto ssg = work_queue[work_offset / batch_size_];
    // auto ssg = subgraph;
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
        if (phase && whole_graph->graph_->config->batch_type == RANDOM) {
          // LOG_DEBUG("batch: random_batch");
          csc_layer->random_batch(sample_nids);
        } else if (phase && whole_graph->graph_->config->batch_type == METIS) {
          // LOG_DEBUG("init dst batch_nodes %d", metis_batch_id);
          csc_layer->init_dst(batch_nodes[metis_batch_id++]);
        } else {
          // LOG_DEBUG("batch: work_offset");
          csc_layer->init_dst(sample_nids.data() + work_offset);
        }
        work_offset += actl_batch_size;
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

        // #pragma omp parallel for num_threads(threads)
        //         for (int j = 0; j < csc_layer->v_size; ++j) {
        //           assert(csc_layer->dst()[j] == ssg->sampled_sgs[i - 1]->src()[j]);
        //         }
      }
      // LOG_DEBUG("sample_one layer %d init_dst done", i);
      // LOG_DEBUG("after init dst");
      // ssg->sampled_sgs.push_back(csc_layer);
      sample_load_dst += get_time();
      // LOG_DEBUG("sample_load_dst cost %.3f", sample_load_dst);

      sample_init_co -= get_time();
      if (whole_graph->graph_->config->sample_rate > 0) {  // use sample rate
        float sample_rate = whole_graph->graph_->config->sample_rate;
        // LOG_DEBUG("sample_rate %.3f", sample_rate);
        ssg->init_co(
            [&](VertexId dst) {
              int nbrs = whole_graph->column_offset[dst + 1] - whole_graph->column_offset[dst];
              // if (fanout[i] < 0) return nbrs;
              int lower_edge = std::min(whole_graph->graph_->config->lower_fanout, nbrs);
              return std::max(int(nbrs * sample_rate), lower_edge) + 1;
            },
            i);
      } else {
        ssg->init_co(
            [&](VertexId dst) {
              int nbrs = whole_graph->column_offset[dst + 1] - whole_graph->column_offset[dst];
              // if (fanout[i] < 0 && full_batch) {
              //   assert(false);
              // }
              if (fanout[i] < 0) return nbrs;
              return std::min(nbrs, fanout[i]) + 1;
            },
            i);
      }

      sample_init_co += get_time();
      // LOG_DEBUG("sample_one layer %d init_column_offset done", i);

      // LOG_DEBUG("sample_init_co cost %.3f", sample_init_co);

      sample_processing_time -= get_time();
      sample_bits->clear();
      // LOG_DEBUG("fanout_i %d\n", fanout_i[i]);
      // ssg->sample_processing(std::bind(&Sampler::NeighborUniformSample, this, std::placeholders::_1,
      // std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5));
      ssg->sample_processing(
          [&](VertexId fanout_i, VertexId dst, VertexId* column_offset, VertexId* row_indices, VertexId id) {
            // this->NeighborUniformSample(fanout_i, dst, column_offset, row_indices, id);

            auto whole_offset = whole_graph->column_offset;
            auto whole_indices = whole_graph->row_indices;
            VertexId edge_nums = whole_offset[dst + 1] - whole_offset[dst];
            // fanout_i = column_offset[id + 1] - column_offset[id];

            if (whole_graph->graph_->config->sample_rate > 0) {
              VertexId tmp_fanout = std::max(whole_graph->graph_->config->lower_fanout,
                                             int(edge_nums * whole_graph->graph_->config->sample_rate));
              // LOG_DEBUG("neightbor sample fanout %d, edge %d sample rate %d", fanout_i, edge_nums, tmp_fanout);
              fanout_i = tmp_fanout;
            }
            VertexId actl_fanout = min(fanout_i, edge_nums);
            assert(column_offset[id + 1] - column_offset[id] == actl_fanout + 1);

            ///////////// no sorted_idxs copy //////////////////
            int pos = column_offset[id];
            if (fanout_i < edge_nums) {
              std::unordered_set<size_t> sampled_idxs;
              while (sampled_idxs.size() < fanout_i) {
                // sampled_idxs.insert(rand_int(fanout_i));
                // sampled_idxs.insert(rand_int_seed(fanout_i));
                sampled_idxs.insert(random_uniform_int(0, edge_nums - 1));
              }

              for (auto& idx : sampled_idxs) {
                row_indices[pos++] = whole_indices[whole_offset[dst] + idx];
                sample_bits->set_bit(whole_indices[whole_offset[dst] + idx]);
              }

              // random replace
              // include dst

              // sorted_idxs.insert(sorted_idxs.end(), sampled_idxs.begin(), sampled_idxs.end());
            } else {
              for (size_t i = 0; i < edge_nums; ++i) {
                // sorted_idxs.push_back(i);

                row_indices[pos++] = whole_indices[whole_offset[dst] + i];
                sample_bits->set_bit(whole_indices[whole_offset[dst] + i]);
              }
            }
            row_indices[pos++] = dst;
            sample_bits->set_bit(dst);
            assert(pos == column_offset[id+1]);
          });

          std::unordered_set<VertexId> check_sample_result;
          VertexId all_node_num = sample_bits->get_size();
          for(VertexId i_src=0;i_src<all_node_num;i_src+=64){
              unsigned long word= sample_bits->data[WORD_OFFSET(i_src)];
              VertexId vtx=i_src;
              VertexId offset=0;
              while(word != 0){
                  if(word & 1){
                      check_sample_result.insert(vtx + offset);
                  }
                  offset++;
                  word = word >> 1;
              }
          }

          bool all_dst_in = true;
          for (auto u : csc_layer->dst()) {
            if (check_sample_result.find(u) == check_sample_result.end()) {
              all_dst_in = false;
              break;
            }
          }
          if (!all_dst_in)  {
            std::cout << "all_dst_in false!" << std::endl;;
            assert(false);
          }




      sample_processing_time += get_time();
      // LOG_DEBUG("sample_one layer %d processing done", i);

      // whole_graph->SyncAndLog("sample_processing");
      sample_post_time -= get_time();
      // ssg->sample_postprocessing();
      ssg->sample_postprocessing(sample_bits, i, node_idx);
    
      sample_post_time += get_time();
      layer_time += get_time();
    }

    // ssg->add_pre_layer_edges();
    // std::cout << "add pre layer done" << std::endl;

    std::reverse(ssg->sampled_sgs.begin(), ssg->sampled_sgs.end());

    sample_convert_graph_time -= get_time();
    if (whole_graph->graph_->config->mini_pull > 0) {  // generate csr structure for backward of pull mode
      for (auto p : ssg->sampled_sgs) {
        p->generate_csr_from_csc();
      }
    }
    sample_convert_graph_time += get_time();

    ssg->alloc_dev_array(whole_graph->graph_->config->mini_pull > 0);
    sample_compute_weight -= get_time();
    ssg->compute_weight(whole_graph->graph_);

    // auto [degree_time, weight_time] = ssg->compute_weight(whole_graph->graph_);
    // sample_degree_time += degree_time;
    // sample_weight_time += weight_time;
    sample_compute_weight += get_time();
  }


  void sample_one(SampledSubgraph* ssg, int type = 0, bool phase = true) {
    assert(work_offset < work_range[1]);
    // assert(batch_size == batch_size_);
    VertexId actl_batch_size = std::min(batch_size, work_range[1] - work_offset);
    if (phase && whole_graph->graph_->config->batch_type == METIS) {
      actl_batch_size = batch_nodes[metis_batch_id].size();
    }
    assert(actl_batch_size > 0);
    // LOG_DEBUG("actl_batch %d", actl_batch_size);
    // SampledSubgraph* ssg=new SampledSubgraph(layers,fanout_);
    // auto ssg = work_queue[work_offset / batch_size_];
    // auto ssg = subgraph;
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
        if (phase && whole_graph->graph_->config->batch_type == RANDOM) {
          // LOG_DEBUG("batch: random_batch");
          csc_layer->random_batch(sample_nids);
        } else if (phase && whole_graph->graph_->config->batch_type == METIS) {
          // LOG_DEBUG("init dst batch_nodes %d", metis_batch_id);
          csc_layer->init_dst(batch_nodes[metis_batch_id++]);
        } else {
          // LOG_DEBUG("batch: work_offset");
          csc_layer->init_dst(sample_nids.data() + work_offset);
        }
        work_offset += actl_batch_size;
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

        // #pragma omp parallel for num_threads(threads)
        //         for (int j = 0; j < csc_layer->v_size; ++j) {
        //           assert(csc_layer->dst()[j] == ssg->sampled_sgs[i - 1]->src()[j]);
        //         }
      }
      // LOG_DEBUG("sample_one layer %d init_dst done", i);
      // LOG_DEBUG("after init dst");
      // ssg->sampled_sgs.push_back(csc_layer);
      sample_load_dst += get_time();
      // LOG_DEBUG("sample_load_dst cost %.3f", sample_load_dst);

      sample_init_co -= get_time();
      if (phase && whole_graph->graph_->config->sample_rate > 0) {  // use sample rate
      // LOG_DEBUG("phase %d use sample rate", phase);
        float sample_rate = whole_graph->graph_->config->sample_rate;
        // LOG_DEBUG("sample_rate %.3f", sample_rate);
        ssg->init_co(
            [&](VertexId dst) {
              int nbrs = whole_graph->column_offset[dst + 1] - whole_graph->column_offset[dst];
              // if (fanout[i] < 0) return nbrs;
              int lower_edge = std::min(whole_graph->graph_->config->lower_fanout, nbrs);
              return std::max(int(nbrs * sample_rate), lower_edge);
            },
            i);
      } else {
        ssg->init_co(
            [&](VertexId dst) {
              int nbrs = whole_graph->column_offset[dst + 1] - whole_graph->column_offset[dst];
              // if (fanout[i] < 0 && full_batch) {
              //   assert(false);
              // }
              if (fanout[i] < 0) return nbrs;
              return std::min(nbrs, fanout[i]);
            },
            i);
      }

      sample_init_co += get_time();
      // LOG_DEBUG("sample_one layer %d init_column_offset done", i);

      // LOG_DEBUG("sample_init_co cost %.3f", sample_init_co);

      sample_processing_time -= get_time();
      sample_bits->clear();
      // LOG_DEBUG("fanout_i %d\n", fanout_i[i]);
      // ssg->sample_processing(std::bind(&Sampler::NeighborUniformSample, this, std::placeholders::_1,
      // std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5));
      ssg->sample_processing(
          [&](VertexId fanout_i, VertexId dst, VertexId* column_offset, VertexId* row_indices, VertexId id) {
            // this->NeighborUniformSample(fanout_i, dst, column_offset, row_indices, id);

            auto whole_offset = whole_graph->column_offset;
            auto whole_indices = whole_graph->row_indices;
            VertexId edge_nums = whole_offset[dst + 1] - whole_offset[dst];
            // fanout_i = column_offset[id + 1] - column_offset[id];

            if (phase && whole_graph->graph_->config->sample_rate > 0) {
              VertexId tmp_fanout = std::max(whole_graph->graph_->config->lower_fanout,
                                             int(edge_nums * whole_graph->graph_->config->sample_rate));
              // LOG_DEBUG("neightbor sample fanout %d, edge %d sample rate %d", fanout_i, edge_nums, tmp_fanout);
              fanout_i = tmp_fanout;
            }

            VertexId actl_fanout = min(fanout_i, edge_nums);
            assert(column_offset[id + 1] - column_offset[id] == actl_fanout);

            // std::vector<size_t> sorted_idxs;
            // double random_time = -get_time();
            // sorted_idxs.reserve(fanout_i);
            // RandomSample(edge_nums, fanout_i, sorted_idxs);

            // if (edge_nums < fanout_i) {
            //   std::unordered_set<size_t> sampled_idxs;
            //   while (sampled_idxs.size() < edge_nums) {
            //     // sampled_idxs.insert(rand_int(fanout_i));
            //     // sampled_idxs.insert(rand_int_seed(fanout_i));
            //     sampled_idxs.insert(random_uniform_int(0, fanout_i - 1));
            //   }
            //   sorted_idxs.insert(sorted_idxs.end(), sampled_idxs.begin(), sampled_idxs.end());
            // } else {
            //   for (size_t i = 0; i < fanout_i; ++i) {
            //     sorted_idxs.push_back(i);
            //   }
            // }

            // // random_time = -get_time();
            // assert(sorted_idxs.size() == fanout_i);
            // int pos = column_offset[id];
            // for (auto& idx : sorted_idxs) {
            //   row_indices[pos++] = whole_indices[whole_offset[dst] + idx];
            //   sample_bits->set_bit(whole_indices[whole_offset[dst] + idx]);
            // }
            // assert(pos == column_offset[id + 1]);

            ///////////// no sorted_idxs copy //////////////////
            int pos = column_offset[id];
            if (fanout_i < edge_nums) {
            // whrong version
            // if (edge_nums < fanout_i) {
            //   std::unordered_set<size_t> sampled_idxs;
            //   while (sampled_idxs.size() < edge_nums) {
            //     sampled_idxs.insert(random_uniform_int(0, fanout_i - 1));
            //   }  
            //   for (auto& idx : sampled_idxs) {
            //     row_indices[pos++] = whole_indices[whole_offset[dst] + idx];
            //     sample_bits->set_bit(whole_indices[whole_offset[dst] + idx]);
            //   }
              ////////////////////////////////////////////////////////////////


              std::unordered_set<size_t> sampled_idxs;
              while (sampled_idxs.size() < fanout_i) {
                sampled_idxs.insert(random_uniform_int(0, edge_nums - 1));
              }
              for (auto& idx : sampled_idxs) {
                row_indices[pos++] = whole_indices[whole_offset[dst] + idx];
                sample_bits->set_bit(whole_indices[whole_offset[dst] + idx]);
              }


              // speed sample rate when use sample rate
              // std::unordered_set<size_t> sampled_idxs;
              // if (fanout_i * 2 < edge_nums) {
              //   while (sampled_idxs.size() < fanout_i) {
              //     sampled_idxs.insert(random_uniform_int(0, edge_nums - 1));
              //   }
              //   for (auto& idx : sampled_idxs) {
              //     row_indices[pos++] = whole_indices[whole_offset[dst] + idx];
              //     sample_bits->set_bit(whole_indices[whole_offset[dst] + idx]);
              //   }
              // } else {
              //   while (sampled_idxs.size() < edge_nums - fanout_i) {
              //     // sampled_idxs.insert(rand_int(fanout_i));
              //     // sampled_idxs.insert(rand_int_seed(fanout_i));
              //     sampled_idxs.insert(random_uniform_int(0, edge_nums - 1));
              //   }
              //   for (int idx = 0; idx < edge_nums; ++idx) {
              //     if (sampled_idxs.find(idx) != sampled_idxs.end()) continue;
              //     row_indices[pos++] = whole_indices[whole_offset[dst] + idx];
              //     sample_bits->set_bit(whole_indices[whole_offset[dst] + idx]);
              //   }
              // }
              ////////////////////////////////////////////////////////////////////////

              // sorted_idxs.insert(sorted_idxs.end(), sampled_idxs.begin(), sampled_idxs.end());
            } else {
              for (size_t i = 0; i < edge_nums; ++i) {


            // } else {
            //   for (size_t i = 0; i < fanout_i; ++i) {
            //     // sorted_idxs.push_back(i);

                row_indices[pos++] = whole_indices[whole_offset[dst] + i];
                sample_bits->set_bit(whole_indices[whole_offset[dst] + i]);
              }
            }
          });

      // ssg->sample_processing(
      //     [&](VertexId fanout_i, VertexId dst, VertexId* column_offset, VertexId* row_indices, VertexId id) {
      //       // this->NeighborUniformSample(fanout_i, dst, column_offset, row_indices, id);
      //         VertexId nbr_size = whole_graph->column_offset[dst+1] - whole_graph->column_offset[dst];
      //         int num = column_offset[id + 1] - column_offset[id];
      //         std::unordered_map<VertexId, int> sampled_idxs;
      //         int pos = 0;
      //         if(num > fanout_i){
      //             while (sampled_idxs.size() < num) {
      //                 VertexId rand = random_uniform_int(0,nbr_size - 1);
      //                 sampled_idxs.insert(std::pair<size_t, int>(rand, 1));
      //             }
      //             VertexId src_idx = whole_graph->column_offset[dst];
      //             for (auto it = sampled_idxs.begin(); it != sampled_idxs.end(); it++) {
      //                 // ssg->sampled_sgs[i]->sample_ans[column_offset[id] + pos] = whole_graph->row_indices[src_idx
      //                 + it->first]; row_indices[column_offset[id] + pos] = whole_graph->row_indices[src_idx +
      //                 it->first]; pos++; sample_bits->set_bit(whole_graph->row_indices[src_idx + it->first]);
      //             }
      //         }
      //         else{
      //             VertexId src_idx = whole_graph->column_offset[dst];
      //             for(VertexId idx = 0; idx < num; idx++){
      //                 // ssg->sampled_sgs[i]->sample_ans[column_offset[id] + pos] = whole_graph->row_indices[src_idx
      //                 + idx]; row_indices[column_offset[id] + pos] = whole_graph->row_indices[src_idx + idx]; pos++;
      //                 sample_bits->set_bit(whole_graph->row_indices[src_idx + idx]);
      //             }
      //         }
      //     });

      sample_processing_time += get_time();
      // LOG_DEBUG("sample_one layer %d processing done", i);

      // whole_graph->SyncAndLog("sample_processing");
      sample_post_time -= get_time();
      // ssg->sample_postprocessing();
      ssg->sample_postprocessing(sample_bits, i, node_idx);
    
      sample_post_time += get_time();
      layer_time += get_time();
    }

    // ssg->add_pre_layer_edges();
    // std::cout << "add pre layer done" << std::endl;

    std::reverse(ssg->sampled_sgs.begin(), ssg->sampled_sgs.end());

    sample_convert_graph_time -= get_time();
    if (whole_graph->graph_->config->mini_pull > 0) {  // generate csr structure for backward of pull mode
      for (auto p : ssg->sampled_sgs) {
        p->generate_csr_from_csc();
      }
    }
    sample_convert_graph_time += get_time();

    ssg->alloc_dev_array(whole_graph->graph_->config->mini_pull > 0);
    sample_compute_weight -= get_time();
    ssg->compute_weight(whole_graph->graph_);

    // auto [degree_time, weight_time] = ssg->compute_weight(whole_graph->graph_);
    // sample_degree_time += degree_time;
    // sample_weight_time += weight_time;
    sample_compute_weight += get_time();
  }

  void reverse_sgs() {
    for (auto ssg : work_queue) {
      std::reverse(ssg->sampled_sgs.begin(), ssg->sampled_sgs.end());
    }
  }

  void reverse_sgs(SampledSubgraph* ssg) { std::reverse(ssg->sampled_sgs.begin(), ssg->sampled_sgs.end()); }

  void RandomSample(size_t set_size, size_t num, std::vector<size_t>& out) {
    if (num < set_size) {
      std::unordered_set<size_t> sampled_idxs;
      while (sampled_idxs.size() < num) {
        // sampled_idxs.insert(rand_int(set_size));
        // sampled_idxs.insert(rand_int_seed(set_size));
        sampled_idxs.insert(random_uniform_int(0, set_size - 1));
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
    fanout_i = column_offset[id + 1] - column_offset[id];

    if (whole_graph->graph_->config->sample_rate > 0) {
      VertexId tmp_fanout = std::max(whole_graph->graph_->config->lower_fanout,
                                     int(edge_nums * whole_graph->graph_->config->sample_rate));
      // LOG_DEBUG("neightbor sample fanout %d, edge %d sample rate %d", fanout_i, edge_nums, tmp_fanout);
      fanout_i = tmp_fanout;
    }

    VertexId actl_fanout = min(fanout_i, edge_nums);
    assert(column_offset[id + 1] - column_offset[id] == actl_fanout);

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
      // if (edge_nums > 2 * fanout_i || true) {
      sorted_idxs.reserve(fanout_i);
      RandomSample(edge_nums, fanout_i, sorted_idxs);
      ///////////////////////////
      std::sort(sorted_idxs.begin(), sorted_idxs.end());
      ///////////////////////////
    } else {
      std::vector<size_t> negate;
      negate.reserve(edge_nums - fanout_i);
      RandomSample(edge_nums, edge_nums - fanout_i, negate);
      // LOG_DEBUG("after RandomSample");
      ///////////////////////////
      std::sort(negate.begin(), negate.end());
      NegateArray(negate, edge_nums, sorted_idxs);
      ///////////////////////////
    }
    random_time = -get_time();
    // LOG_DEBUG("random time %.3f", random_time);
    // LOG_DEBUG("after random");
    // #pragma omp parallel for
    //     for (size_t i = 1; i < sorted_idxs.size(); ++i) {
    //       assert(sorted_idxs[i] > sorted_idxs[i - 1]);
    //     }
    assert(sorted_idxs.size() == fanout_i);
    ///////////////////////////////////////////////////

    // std::unordered_set<size_t> sorted_idxs;
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