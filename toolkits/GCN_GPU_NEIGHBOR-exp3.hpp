#include <c10/cuda/CUDACachingAllocator.h>

#include "core/neutronstar.hpp"
#include "core/ntsPeerRPC.hpp"
#include "utils/torch_func.hpp"
#include "utils/utils.hpp"

class GCN_GPU_NEIGHBOR_EXP3_impl {
 public:
  int iterations;
  ValueType learn_rate;
  ValueType weight_decay;
  ValueType drop_rate;
  ValueType alpha;
  ValueType beta1;
  ValueType beta2;
  ValueType epsilon;
  ValueType decay_rate;
  ValueType decay_epoch;
  ValueType best_val_acc;
  double start_time;
  int layers;
  // graph
  VertexSubset* active;
  // graph with no edge data
  Graph<Empty>* graph;
  // std::vector<CSC_segment_pinned *> subgraphs;
  // NN
  GNNDatum* gnndatum;
  NtsVar L_GT_C;
  NtsVar L_GT_G;
  NtsVar MASK;
  NtsVar MASK_gpu;
  // GraphOperation *gt;
  PartitionedGraph* partitioned_graph;
  // Variables
  std::vector<Parameter*> P;
  std::vector<NtsVar> X;
  nts::ctx::NtsContext* ctx;
  FullyRepGraph* fully_rep_graph;
  double train_sample_time = 0;
  double train_compute_time = 0;
  // double mpi_comm_time = 0;
  // double rpc_comm_time = 0;
  // double rpc_wait_time = 0;
  float loss_epoch = 0;
  float f1_epoch = 0;
  Sampler* train_sampler = nullptr;
  Sampler* eval_sampler = nullptr;
  Sampler* test_sampler = nullptr;

  std::vector<VertexId> train_nids, val_nids, test_nids;

  // double gcn_start_time = 0;
  double gcn_run_time;
  double gcn_gather_time;
  double gcn_sample_time;
  double gcn_trans_time;
  double gcn_train_time;
  double gcn_cache_hit_rate;
  double gcn_trans_memory;

  double epoch_sample_time = 0;
  double epoch_gather_label_time = 0;
  double epoch_gather_feat_time = 0;
  double epoch_transfer_graph_time = 0;
  double epoch_transfer_feat_time = 0;
  double epoch_transfer_label_time = 0;
  double epoch_train_time = 0;
  int epoch_cache_hit = 0;
  int epoch_all_node = 0;
  double debug_time = 0;
  vector<float> explicit_rate;

  int threads;
  float* dev_cache_feature;

  VertexId *local_idx, *local_idx_cache, *dev_local_idx, *dev_local_idx_cache;
  Cuda_Stream* cuda_stream;

  std::mutex sample_mutex;
  std::mutex transfer_mutex;
  std::mutex train_mutex;
  int pipelines;
  Cuda_Stream* cuda_stream_list;
  std::vector<at::cuda::CUDAStream> torch_stream;

  // int batch_size_switch_idx = 0;

  NtsVar F;
  NtsVar loss;
  NtsVar tt;
  float acc;
  int batch;
  long correct;
  long train_nodes;
  int max_batch_num;
  int min_batch_num;
  std::string dataset_name;
  torch::nn::Dropout drpmodel;
  // double sample_cost = 0;
  std::vector<torch::nn::BatchNorm1d> bn1d;

  ntsPeerRPC<ValueType, VertexId>* rpc;
  int hosts;
  std::vector<int> cache_node_idx_seq;
  // std::unordered_set<int> cache_node_hashmap;
  // std::vector<int> cache_node_hashmap;
  VertexId* cache_node_hashmap;
  VertexId* dev_cache_node_hashmap;
  int cache_node_num = 0;
  double used_gpu_mem, total_gpu_mem;
  // std::unordered_map<std::string, std::vector<int>> batch_size_mp;
  // std::vector<int> batch_size_vec;
  ~GCN_GPU_NEIGHBOR_EXP3_impl() { delete active; }

  GCN_GPU_NEIGHBOR_EXP3_impl(Graph<Empty>* graph_, int iterations_, bool process_local = false,
                             bool process_overlap = false) {
    graph = graph_;
    iterations = iterations_;

    active = graph->alloc_vertex_subset();
    active->fill();

    graph->init_gnnctx(graph->config->layer_string);
    graph->init_gnnctx_fanout(graph->config->fanout_string);
    reverse(graph->gnnctx->fanout.begin(), graph->gnnctx->fanout.end());
    graph->init_rtminfo();
    graph->rtminfo->process_local = graph->config->process_local;
    graph->rtminfo->reduce_comm = graph->config->process_local;
    graph->rtminfo->copy_data = false;
    graph->rtminfo->process_overlap = graph->config->overlap;
    graph->rtminfo->with_weight = true;
    graph->rtminfo->with_cuda = false;
    graph->rtminfo->lock_free = graph->config->lock_free;
    hosts = graph->partitions;
    if (hosts > 1) {
      rpc = new ntsPeerRPC<ValueType, VertexId>();
    } else {
      rpc = nullptr;
    }
    best_val_acc = 0;
    cuda_stream = new Cuda_Stream();
    pipelines = graph->config->pipelines;
    pipelines = std::max(1, pipelines);
    torch_stream.clear();

    gcn_run_time = 0;
    gcn_gather_time = 0;
    gcn_sample_time = 0;
    gcn_trans_time = 0;
    gcn_train_time = 0;
    gcn_cache_hit_rate = 0;
    gcn_trans_memory = 0;
    threads = std::max(1, numa_num_configured_cpus() / 2 - 1);

    // batch_size_mp["ppi"] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 2048, 4096, 9716};
    // batch_size_mp["ppi-large"] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 44906};
    // batch_size_mp["flickr"] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 44625};
    // batch_size_mp["AmazonCoBuy_computers"] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8250};
    // batch_size_mp["ogbn-arxiv"] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
    // 65536, 90941}; batch_size_mp["AmazonCoBuy_photo"] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4590};

    // batch_size_switch_idx = 0;
    // batch_size_vec = graph->config->batch_size_vec;
  }

  void init_nids() {
    for (int i = 0; i < graph->gnnctx->l_v_num; ++i) {
      // for (int i = graph->partition_offset[graph->partition_id]; i < graph->partition_offset[graph->partition_id +
      // 1]; ++i) {
      int type = gnndatum->local_mask[i];
      // std::cout << i << " " << type << " " << i + graph->partition_offset[graph->partition_id] << std::endl;
      if (type == 0) {
        train_nids.push_back(i + graph->partition_offset[graph->partition_id]);
      } else if (type == 1) {
        // std::cout << "type = 1" << std::endl;
        val_nids.push_back(i + graph->partition_offset[graph->partition_id]);
      } else if (type == 2) {
        // std::cout << "type = 3" << std::endl;
        test_nids.push_back(i + graph->partition_offset[graph->partition_id]);
      }
    }
  }

  void gater_cpu_cache_feature_and_trans_to_gpu() {
    long feat_dim = graph->gnnctx->layer_size[0];
    // LOG_DEBUG("feat_dim %d", feat_dim);
    dev_cache_feature = (float*)cudaMallocGPU(cache_node_num * sizeof(float) * feat_dim);
    // gather_cache_feature, prepare trans to gpu
    // LOG_DEBUG("start gather_cpu_cache_feature");
    float* local_cache_feature_gather = new float[cache_node_num * feat_dim];
// std::cout << "###" << cache_node_num * sizeof(float) * feat_dim << " " << cache_node_num * feat_dim << std::endl;
// std::cout << "###" << graph->vertices * feat_dim << "l_v_num " << graph->gnnctx->l_v_num << " " << graph->vertices <<
// std::endl;
// #pragma omp parallel for
// omp_set_num_threads(threads);
#pragma omp parallel for num_threads(threads)
    for (int i = 0; i < cache_node_num; ++i) {
      int node_id = cache_node_idx_seq[i];
      // assert(node_id < graph->vertices);
      // assert(node_id < graph->gnnctx->l_v_num);
      // LOG_DEBUG("copy node_id %d to", node_id);
      // LOG_DEBUG("local_id %d", cache_node_hashmap[node_id]);

      for (int j = 0; j < feat_dim; ++j) {
        assert(cache_node_hashmap[node_id] < cache_node_num);
        local_cache_feature_gather[cache_node_hashmap[node_id] * feat_dim + j] =
            gnndatum->local_feature[node_id * feat_dim + j];
      }
    }
    LOG_DEBUG("start trans to gpu");
    move_data_in(dev_cache_feature, local_cache_feature_gather, 0, cache_node_num, feat_dim);
    local_idx = (VertexId*)cudaMallocPinned(1ll * graph->vertices * sizeof(VertexId));
    local_idx_cache = (VertexId*)cudaMallocPinned(1ll * graph->vertices * sizeof(VertexId));
    dev_local_idx = (VertexId*)getDevicePointer(local_idx);
    dev_local_idx_cache = (VertexId*)getDevicePointer(local_idx_cache);
  }

  void mark_cache_node(std::vector<int>& cache_nodes) {
    // init mask
    // #pragma omp parallel for
    // #pragma omp parallel for num_threads(threads)
    for (int i = 0; i < graph->vertices; ++i) {
      cache_node_hashmap[i] = -1;
      // assert(cache_node_hashmap[i] == -1);
    }

    // mark cache nodes
    int tmp_idx = 0;
    for (int i = 0; i < cache_node_num; ++i) {
      // LOG_DEBUG("cache_nodes[%d] = %d", i, cache_nodes[i]);
      cache_node_hashmap[cache_nodes[i]] = tmp_idx++;
    }
    LOG_DEBUG("cache_node_num %d tmp_idx %d", cache_node_num, tmp_idx);
    assert(cache_node_num == tmp_idx);

    // // debug
    // int cache_node_hashmap_num = 0;
    // for (int i = 0; i < graph->vertices; ++i) {
    //   // LOG_DEBUG("cache_node_hashmap[%d] = %d", i, cache_node_hashmap[i]);
    //   cache_node_hashmap_num += cache_node_hashmap[i] != -1; // unsigned
    // }
    // LOG_DEBUG("cache_node_hashmap_num %d", cache_node_hashmap_num);
    // assert(cache_node_hashmap_num == cache_node_num);
  }

  void cache_high_degree(std::vector<int>& node_idx) {
    std::sort(node_idx.begin(), node_idx.end(), [&](const int x, const int y) {
      return graph->out_degree_for_backward[x] > graph->out_degree_for_backward[y];
    });
    // #pragma omp parallel for num_threads(threads)
    for (int i = 1; i < graph->vertices; ++i) {
      assert(graph->out_degree_for_backward[node_idx[i]] <= graph->out_degree_for_backward[node_idx[i - 1]]);
    }
    mark_cache_node(node_idx);
  }

  void cache_random_node(std::vector<int>& node_idx) {
    shuffle_vec_seed(node_idx);
    mark_cache_node(node_idx);
  }

  void cache_sample(std::vector<int>& node_idx) {
    std::vector<int> node_sample_cnt(node_idx.size(), 0);
    int epochs = 1;
    auto ssg = train_sampler->subgraph;
    for (int i = 0; i < epochs; ++i) {
      while (train_sampler->work_offset < train_sampler->work_range[1]) {
        train_sampler->sample_one(ssg, graph->config->batch_type, ctx->is_train());
        // train_sampler->sample_one_with_dst(ssg, graph->config->batch_type, ctx->is_train());
        for (int i = 0; i < layers; ++i) {
          auto p = ssg->sampled_sgs[i]->src();
          for (const auto& v : p) {
            node_sample_cnt[v]++;
          }
          // LOG_DEBUG("layer %d %d", i, p.size());
        }
        train_sampler->reverse_sgs();
      }
      train_sampler->restart();
    }
    sort(node_idx.begin(), node_idx.end(),
         [&](const int x, const int y) { return node_sample_cnt[x] > node_sample_cnt[y]; });
    // for (int i = 1; i < node_idx.size(); ++i) {
    //   assert(node_sample_cnt[node_idx[i - 1]] >= node_sample_cnt[node_idx[i]]);
    // }
    mark_cache_node(node_idx);
  }

  // pre train some epochs to get idle memory of GPU when training
  double get_gpu_idle_mem() {
    // store degree
    VertexId* outs_bak = new VertexId[graph->vertices];
    VertexId* ins_bak = new VertexId[graph->vertices];
    for (int i = 0; i < graph->vertices; ++i) {
      outs_bak[i] = graph->out_degree_for_backward[i];
      ins_bak[i] = graph->in_degree_for_backward[i];
    }

    X[0] = graph->Nts->NewLeafTensor({1000, F.size(1)}, torch::DeviceType::CUDA);
    NtsVar target_lab;
    if (graph->config->classes > 1) {
      target_lab =
          graph->Nts->NewLabelTensor({graph->config->batch_size, graph->config->classes}, torch::DeviceType::CUDA);
    } else {
      target_lab = graph->Nts->NewLabelTensor({graph->config->batch_size}, torch::DeviceType::CUDA);
    }
    double max_gpu_used = 0;

    int epochs = 1;
    for (int i = 0; i < epochs; ++i) {
      auto ssg = train_sampler->subgraph;
      while (train_sampler->work_offset < train_sampler->work_range[1]) {
        if (ctx->training == true) zero_grad();  // should zero grad after every mini batch compute
        train_sampler->sample_one(ssg, graph->config->batch_type, ctx->is_train());
        // train_sampler->sample_one_with_dst(ssg, graph->config->batch_type, ctx->is_train());
        ssg->trans_graph_to_gpu_async(cuda_stream_list[0].stream,
                                      graph->config->mini_pull > 0);  // trans subgraph to gpu
        if (hosts > 1) {
          X[0] = nts::op::get_feature_from_global(*rpc, ssg->sampled_sgs[0]->src().data(),
                                                  ssg->sampled_sgs[0]->src_size, F, graph);
        } else {
          X[0] = nts::op::get_feature(ssg->sampled_sgs[0]->src().data(), ssg->sampled_sgs[0]->src_size, F, graph);
        }
        X[0] = X[0].cuda().set_requires_grad(true);

        if (hosts > 1) {
          target_lab = nts::op::get_label_from_global(ssg->sampled_sgs.back()->dst().data(),
                                                      ssg->sampled_sgs.back()->v_size, L_GT_C, graph);
        } else {
          target_lab =
              nts::op::get_label(ssg->sampled_sgs.back()->dst().data(), ssg->sampled_sgs.back()->v_size, L_GT_C, graph);
        }
        target_lab = target_lab.cuda();

        for (int l = 0; l < layers; l++) {
          graph->rtminfo->curr_layer = l;
          NtsVar Y_i = ctx->runGraphOp<nts::op::SingleGPUSampleGraphOp>(ssg, graph, l, X[l], &cuda_stream_list[0]);
          X[l + 1] = ctx->runVertexForward([&](NtsVar n_i) { return vertexForward(n_i); }, Y_i);
        }

        auto loss_ = Loss(X[layers], target_lab, graph->config->classes == 1);
        if (ctx->training == true) {
          ctx->appendNNOp(X[layers], loss_);
          ctx->self_backward(false);
          // Update();
        }

        if (graph->config->classes == 1) {
          get_correct(X[layers], target_lab, graph->config->classes == 1);
          // target_lab.size(0);
        } else {
          f1_score(X[layers], target_lab, graph->config->classes == 1);
        }
        train_sampler->reverse_sgs();
      }
      train_sampler->restart();
      get_gpu_mem(used_gpu_mem, total_gpu_mem);
      max_gpu_used = std::max(used_gpu_mem, max_gpu_used);
      LOG_DEBUG("get_gpu_idle_mem(): used %.3f max_used %.3f total %.3f", used_gpu_mem, max_gpu_used, total_gpu_mem);
    }

    // restore degree
    for (int i = 0; i < graph->vertices; ++i) {
      graph->out_degree_for_backward[i] = outs_bak[i];
      graph->in_degree_for_backward[i] = ins_bak[i];
    }
    delete[] outs_bak;
    delete[] ins_bak;

    return max_gpu_used;
  }

  // pre train some epochs to get idle memory of GPU when training
  double get_gpu_idle_mem_pipe() {
    // store degree
    VertexId* outs_bak = new VertexId[graph->vertices];
    VertexId* ins_bak = new VertexId[graph->vertices];
    for (int i = 0; i < graph->vertices; ++i) {
      outs_bak[i] = graph->out_degree_for_backward[i];
      ins_bak[i] = graph->in_degree_for_backward[i];
    }

    double max_gpu_used = 0;

    NtsVar tmp_X0[pipelines];
    NtsVar tmp_target_lab[pipelines];

    for (int i = 0; i < pipelines; i++) {
      tmp_X0[i] = graph->Nts->NewLeafTensor({1000, F.size(1)}, torch::DeviceType::CUDA);
      if (graph->config->classes > 1) {
        tmp_target_lab[i] =
            graph->Nts->NewLabelTensor({graph->config->batch_size, graph->config->classes}, torch::DeviceType::CUDA);
      } else {
        tmp_target_lab[i] = graph->Nts->NewLabelTensor({graph->config->batch_size}, torch::DeviceType::CUDA);
      }
    }

    auto sampler = train_sampler;

    for (int i = 0; i < 1; ++i) {
      std::thread threads[pipelines];
      for (int tid = 0; tid < pipelines; ++tid) {
        threads[tid] = std::thread(
            [&](int thread_id) {
              ////////////////////////////////// sample //////////////////////////////////
              std::unique_lock<std::mutex> sample_lock(sample_mutex, std::defer_lock);
              sample_lock.lock();
              while (sampler->work_offset < sampler->work_range[1]) {
                auto ssg = sampler->subgraph_list[thread_id];
                sampler->sample_one(ssg, graph->config->batch_type, ctx->is_train());
                // sampler->sample_one_with_dst(ssg, graph->config->batch_type, ctx->is_train());
                cudaStreamSynchronize(cuda_stream_list[thread_id].stream);
                sample_lock.unlock();
                // get_gpu_mem(used_gpu_mem, total_gpu_mem);

                ////////////////////////////////// transfer //////////////////////////////////
                std::unique_lock<std::mutex> transfer_lock(transfer_mutex, std::defer_lock);
                transfer_lock.lock();
                ssg->trans_graph_to_gpu_async(cuda_stream_list[thread_id].stream, graph->config->mini_pull > 0);
                sampler->load_feature_gpu(&cuda_stream_list[thread_id], ssg, tmp_X0[thread_id],
                                          gnndatum->dev_local_feature);
                sampler->load_label_gpu(&cuda_stream_list[thread_id], ssg, tmp_target_lab[thread_id],
                                        gnndatum->dev_local_label);
                cudaStreamSynchronize(cuda_stream_list[thread_id].stream);
                transfer_lock.unlock();

                ////////////////////////////////// train //////////////////////////////////
                std::unique_lock<std::mutex> train_lock(train_mutex, std::defer_lock);
                train_lock.lock();
                at::cuda::setCurrentCUDAStream(torch_stream[thread_id]);
                if (ctx->training == true) zero_grad();  // should zero grad after every mini batch compute
                for (int l = 0; l < layers; l++) {       // forward
                  graph->rtminfo->curr_layer = l;
                  if (l == 0) {
                    NtsVar Y_i = ctx->runGraphOp<nts::op::SingleGPUSampleGraphOp>(ssg, graph, l, tmp_X0[thread_id],
                                                                                  &cuda_stream_list[thread_id]);
                    X[l + 1] = ctx->runVertexForward([&](NtsVar n_i) { return vertexForward(n_i); }, Y_i);
                  } else {
                    NtsVar Y_i = ctx->runGraphOp<nts::op::SingleGPUSampleGraphOp>(ssg, graph, l, X[l],
                                                                                  &cuda_stream_list[thread_id]);
                    X[l + 1] = ctx->runVertexForward([&](NtsVar n_i) { return vertexForward(n_i); }, Y_i);
                  }
                }

                auto loss_ = Loss(X[layers], tmp_target_lab[thread_id], graph->config->classes == 1);
                if (ctx->training == true) {
                  ctx->appendNNOp(X[layers], loss_);
                  ctx->self_backward(false);
                  // Update();
                }

                if (graph->config->classes == 1) {
                  get_correct(X[layers], tmp_target_lab[thread_id], graph->config->classes == 1);
                } else {
                  f1_score(X[layers], tmp_target_lab[thread_id], graph->config->classes == 1);
                }
                cudaStreamSynchronize(cuda_stream_list[thread_id].stream);
                train_lock.unlock();

                sample_lock.lock();
                sampler->reverse_sgs(ssg);
              }
              sample_lock.unlock();
            },
            tid);
      }

      for (int tid = 0; tid < pipelines; ++tid) {
        threads[tid].join();
      }
      sampler->restart();

      get_gpu_mem(used_gpu_mem, total_gpu_mem);
      max_gpu_used = std::max(used_gpu_mem, max_gpu_used);
      LOG_DEBUG("get_gpu_idle_mem_pipe(): epoch %d used %.3f max_used %.3f total %.3f", i, used_gpu_mem, max_gpu_used,
                total_gpu_mem);
    }

    // restore degree
    for (int i = 0; i < graph->vertices; ++i) {
      graph->out_degree_for_backward[i] = outs_bak[i];
      graph->in_degree_for_backward[i] = ins_bak[i];
    }
    delete[] outs_bak;
    delete[] ins_bak;

    return max_gpu_used;
  }

  void determine_cache_node_idx(int node_nums) {
    if (node_nums > graph->vertices) node_nums = graph->vertices;
    cache_node_num = node_nums;
    LOG_DEBUG("cache_node_num %d (%.3f)", cache_node_num, 1.0 * cache_node_num / graph->vertices);

    cache_node_idx_seq.resize(graph->vertices);
    std::iota(cache_node_idx_seq.begin(), cache_node_idx_seq.end(), 0);
    // cache_node_hashmap.resize(graph->vertices);
    cache_node_hashmap = (VertexId*)cudaMallocPinned(1ll * graph->vertices * sizeof(VertexId));
    dev_cache_node_hashmap = (VertexId*)getDevicePointer(cache_node_hashmap);

    // #pragma omp parallel for
    // #pragma omp parallel for num_threads(threads)
    for (int i = 0; i < graph->vertices; ++i) {
      cache_node_hashmap[i] = -1;
      // assert(cache_node_hashmap[i] == -1);
    }

    if (graph->config->cache_policy == "sample") {
      LOG_DEBUG("cache_sample");
      cache_sample(cache_node_idx_seq);
    } else if (graph->config->cache_policy == "degree") {  // default cache high degree
      LOG_DEBUG("cache_high_degree");
      cache_high_degree(cache_node_idx_seq);
    } else if (graph->config->cache_policy == "random") {  // default cache high degree
      LOG_DEBUG("cache_random_node");
      cache_random_node(cache_node_idx_seq);
    }
    gater_cpu_cache_feature_and_trans_to_gpu();
  }

  void init_active() {
    active = graph->alloc_vertex_subset();
    active->fill();
  }

  void init_graph() {
    fully_rep_graph = new FullyRepGraph(graph);
    fully_rep_graph->GenerateAll();
    fully_rep_graph->SyncAndLog("read_finish");

    // cp = new nts::autodiff::ComputionPath(gt, subgraphs);
    ctx = new nts::ctx::NtsContext();
  }

  void get_batch_num() {
    VertexId max_vertex = 0;
    VertexId min_vertex = std::numeric_limits<VertexId>::max();
    for (int i = 0; i < graph->partitions; i++) {
      max_vertex = std::max(graph->partition_offset[i + 1] - graph->partition_offset[i], max_vertex);
      min_vertex = std::min(graph->partition_offset[i + 1] - graph->partition_offset[i], min_vertex);
    }
    max_batch_num = max_vertex / graph->config->batch_size;
    min_batch_num = min_vertex / graph->config->batch_size;
    if (max_vertex % graph->config->batch_size != 0) {
      max_batch_num++;
    }
    if (min_vertex % graph->config->batch_size != 0) {
      min_batch_num++;
    }
  }

  void init_nn() {
    // get_gpu_mem(used_gpu_mem, total_gpu_mem);
    // LOG_DEBUG("init_nn(): gcn_gpu_mem %.3fM", used_gpu_mem);

    // const uint64_t seed = 2000;
    // torch::manual_seed(seed);
    // torch::cuda::manual_seed_all(seed);

    learn_rate = graph->config->learn_rate;
    weight_decay = graph->config->weight_decay;
    drop_rate = graph->config->drop_rate;
    alpha = graph->config->learn_rate;
    decay_rate = graph->config->decay_rate;
    decay_epoch = graph->config->decay_epoch;
    layers = graph->gnnctx->layer_size.size() - 1;
    beta1 = 0.9;
    beta2 = 0.999;
    epsilon = 1e-9;
    gnndatum = new GNNDatum(graph->gnnctx, graph);
    if (0 == graph->config->feature_file.compare("random")) {
      gnndatum->random_generate();
      LOG_DEBUG("generate feat_label_mask is done");
    } else {
      gnndatum->readFeature_Label_Mask(graph->config->feature_file, graph->config->label_file,
                                       graph->config->mask_file);
      LOG_DEBUG("read feature_label_mask is done");
    }
    int val_cnt = 0;
    for (int i = 0; i < graph->gnnctx->l_v_num; ++i) {
      if (gnndatum->local_mask[i] == 1) val_cnt++;
    }

    // get_gpu_mem(used_gpu_mem, total_gpu_mem);
    // LOG_DEBUG("init_nn() after read_feature: gcn_gpu_mem %.3fM", used_gpu_mem);

    // creating tensor to save Label and Mask
    if (graph->config->classes > 1) {
      gnndatum->registLabel(L_GT_C, gnndatum->local_label, gnndatum->gnnctx->l_v_num, graph->config->classes);
    } else {
      gnndatum->registLabel(L_GT_C);
    }
    gnndatum->registMask(MASK);
    // get_gpu_mem(used_gpu_mem, total_gpu_mem);
    // LOG_DEBUG("init_nn() after registlabel: gcn_gpu_mem %.3fM", used_gpu_mem);

    MASK_gpu = MASK.cuda();
    // get_gpu_mem(used_gpu_mem, total_gpu_mem);
    // LOG_DEBUG("init_nn() after maks.cuda(): gcn_gpu_mem %.3fM", used_gpu_mem);
    gnndatum->generate_gpu_data();
    // get_gpu_mem(used_gpu_mem, total_gpu_mem);
    // LOG_DEBUG("init_nn() after generate_gpu_data(): gcn_gpu_mem %.3fM", used_gpu_mem);

    torch::Device GPU(torch::kCUDA, 0);

    for (int i = 0; i < layers; i++) {
      P.push_back(new Parameter(graph->gnnctx->layer_size[i], graph->gnnctx->layer_size[i + 1], alpha, beta1, beta2,
                                epsilon, weight_decay));
      if (graph->config->batch_norm && i < layers - 1) {
        bn1d.push_back(torch::nn::BatchNorm1d(graph->gnnctx->layer_size[i]));
        // bn1d.back().to(GPU);
        // bn1d.back().cuda();
      }
    }

    for (int i = 0; i < P.size(); i++) {
      P[i]->init_parameter();
      P[i]->set_decay(decay_rate, decay_epoch);
      P[i]->to(GPU);
      P[i]->Adam_to_GPU();
    }
    // get_gpu_mem(used_gpu_mem, total_gpu_mem);
    // LOG_DEBUG("init_nn() after P->to(GPU): gcn_gpu_mem %.3fM", used_gpu_mem);

    // drpmodel = torch::nn::Dropout(torch::nn::DropoutOptions().p(drop_rate).inplace(true));

    F = graph->Nts->NewLeafTensor(gnndatum->local_feature, {graph->gnnctx->l_v_num, graph->gnnctx->layer_size[0]},
                                  torch::DeviceType::CPU);

    // X[i] is vertex representation at layer i
    for (int i = 0; i < layers + 1; i++) {
      NtsVar d;
      X.push_back(d);
    }

    X[0] = F.set_requires_grad(true);

    if (hosts > 1) {
      rpc->set_comm_num(graph->partitions - 1);
      rpc->register_function("get_feature", [&](std::vector<VertexId> vertexs) {
        int start = graph->partition_offset[graph->partition_id];
        int feature_size = F.size(1);
        ValueType* ntsVarBuffer = graph->Nts->getWritableBuffer(F, torch::DeviceType::CPU);
        std::vector<std::vector<ValueType>> result_vector;
        result_vector.resize(vertexs.size());

// #pragma omp parallel for
// omp_set_num_threads(threads);
#pragma omp parallel for num_threads(threads)
        for (int i = 0; i < vertexs.size(); i++) {
          result_vector[i].resize(feature_size);
          memcpy(result_vector[i].data(), ntsVarBuffer + (vertexs[i] - start) * feature_size,
                 feature_size * sizeof(ValueType));
        }
        return result_vector;
      });
    }
    // get_gpu_mem(used_gpu_mem, total_gpu_mem);
    // LOG_DEBUG("init_nn() done: gcn_gpu_mem %.3fM", used_gpu_mem);
  }

  void Update() {
    for (int i = 0; i < P.size(); i++) {
      // accumulate the gradient using all_reduce
      // if (ctx->is_train() && graph->rtminfo->epoch >= graph->config->time_skip) mpi_comm_time -= get_time();
      if (graph->gnnctx->l_v_num == 0) {
        P[i]->all_reduce_to_gradient(torch::zeros_like(P[i]->W));
      } else {
        P[i]->all_reduce_to_gradient(P[i]->W.grad().cpu());
      }
      // if (ctx->is_train() && graph->rtminfo->epoch >= graph->config->time_skip) mpi_comm_time += get_time();
      // update parameters with Adam optimizer
      P[i]->learnC2G_with_decay_Adam();
      // P[i]->learnC2C_with_Adam();
      P[i]->next();
    }
  }

  void UpdateZero() {
    for (int l = 0; l < layers; l++) {
      P[l]->all_reduce_to_gradient(torch::zeros({P[l]->row, P[l]->col}, torch::kFloat));
    }
  }

  void empty_gpu_cache() {
    for (int ti = 0; ti < 5; ++ti) {  // clear gpu cache memory
      c10::cuda::CUDACachingAllocator::emptyCache();
    }
  }

  void debug_batch_sample_info(Sampler* sampler, SampledSubgraph* ssg) {
    // if (sampler->work_offset == graph->config->batch_size) {
    std::cout << "batch nodes: ";
    for (int i = 0; i < 10; ++i) {
      std::cout << ssg->sampled_sgs.back()->dst()[i] << " ";
    }
    std::cout << "\n";

    std::cout << "1-hop nodes: ";
    for (int i = 0; i < 10; ++i) {
      std::cout << ssg->sampled_sgs.back()->src()[i] << " ";
    }
    size_t sum = 0;
    std::cout << "1-hop nodes: ";
    for (auto v : ssg->sampled_sgs.back()->src()) {
      sum += v;
    }
    std::cout << ", sum = " << sum << std::endl;
    // }
  }

  NtsVar vertexForward(NtsVar& n_i) {
    int l = graph->rtminfo->curr_layer;
    if (l == layers - 1) {  // last layer
      return P[l]->forward(n_i);
    } else {
      if (graph->config->batch_norm) {
        // std::cout << n_i.device() << std::endl;
        // std::cout << this->bn1d[l].device() << std::endl;
        n_i = this->bn1d[l](n_i);  // for arxiv dataset
        // n_i = torch::batch_norm(n_i);
      }
      return torch::dropout(torch::relu(P[l]->forward(n_i)), drop_rate, ctx->is_train());
    }
  }

  void pipeline_version(Sampler* sampler) {
    NtsVar tmp_X0[pipelines];
    NtsVar tmp_target_lab[pipelines];
    LOG_DEBUG("pipeline %d", pipelines);
    for (int i = 0; i < pipelines; i++) {
      tmp_X0[i] = graph->Nts->NewLeafTensor({1000, F.size(1)}, torch::DeviceType::CUDA);
      if (graph->config->classes > 1) {
        tmp_target_lab[i] =
            graph->Nts->NewLabelTensor({graph->config->batch_size, graph->config->classes}, torch::DeviceType::CUDA);
      } else {
        tmp_target_lab[i] = graph->Nts->NewLabelTensor({graph->config->batch_size}, torch::DeviceType::CUDA);
      }
    }

    std::thread threads[pipelines];
    for (int tid = 0; tid < pipelines; ++tid) {
      threads[tid] = std::thread(
          [&](int thread_id) {
            ////////////////////////////////// sample //////////////////////////////////
            std::unique_lock<std::mutex> sample_lock(sample_mutex, std::defer_lock);
            std::unique_lock<std::mutex> transfer_lock(transfer_mutex, std::defer_lock);
            std::unique_lock<std::mutex> train_lock(train_mutex, std::defer_lock);
            sample_lock.lock();
            while (sampler->work_offset < sampler->work_range[1]) {
              auto ssg = sampler->subgraph_list[thread_id];
              epoch_sample_time -= get_time();
              sampler->sample_one(ssg, graph->config->batch_type, ctx->is_train());
              // sampler->sample_one_with_dst(ssg, graph->config->batch_type, ctx->is_train());
              epoch_sample_time += get_time();
              cudaStreamSynchronize(cuda_stream_list[thread_id].stream);
              sample_lock.unlock();

              ////////////////////////////////// transfer //////////////////////////////////
              transfer_lock.lock();
              epoch_transfer_graph_time -= get_time();
              ssg->trans_graph_to_gpu_async(cuda_stream_list[thread_id].stream, graph->config->mini_pull > 0);
              epoch_transfer_graph_time += get_time();
              if (graph->config->cache_type == "none") {  // trans feature use zero copy (omit gather feature)
                epoch_transfer_feat_time -= get_time();
                sampler->load_feature_gpu(&cuda_stream_list[thread_id], ssg, tmp_X0[thread_id],
                                          gnndatum->dev_local_feature);
                epoch_transfer_feat_time += get_time();
                // get_gpu_mem(used_gpu_mem, total_gpu_mem);
              } else if (graph->config->cache_type == "gpu_memory" ||
                         graph->config->cache_type == "rate") {  // trans freature which is not cache in gpu
                // epoch_transfer_feat_time -= get_time();
                auto [trans_feature_tmp, gather_gpu_cache_tmp] = sampler->load_feature_gpu_cache(
                    &cuda_stream_list[thread_id], ssg, tmp_X0[thread_id], gnndatum->dev_local_feature,
                    dev_cache_feature, local_idx, local_idx_cache, cache_node_hashmap, dev_local_idx,
                    dev_local_idx_cache, dev_cache_node_hashmap);
                // epoch_transfer_feat_time += get_time();
                epoch_transfer_feat_time += trans_feature_tmp;
                epoch_gather_feat_time += gather_gpu_cache_tmp;

                debug_time -= get_time();
                epoch_all_node += ssg->sampled_sgs[0]->src().size();
                for (auto& it : ssg->sampled_sgs[0]->src()) {
                  if (cache_node_hashmap[it] != -1) {
                    epoch_cache_hit++;
                  }
                }
                debug_time += get_time();
              } else {
                std::cout << "cache_type: " << graph->config->cache_type << " is not support!" << std::endl;
                assert(false);
              }
              epoch_transfer_label_time -= get_time();
              sampler->load_label_gpu(&cuda_stream_list[thread_id], ssg, tmp_target_lab[thread_id],
                                      gnndatum->dev_local_label);

              epoch_transfer_label_time += get_time();
              cudaStreamSynchronize(cuda_stream_list[thread_id].stream);
              transfer_lock.unlock();

              ////////////////////////////////// train //////////////////////////////////
              train_lock.lock();
              at::cuda::setCurrentCUDAStream(torch_stream[thread_id]);
              if (ctx->training == true) zero_grad();  // should zero grad after every mini batch compute
              epoch_train_time -= get_time();
              for (int l = 0; l < layers; l++) {  // forward
                graph->rtminfo->curr_layer = l;
                if (l == 0) {
                  NtsVar Y_i = ctx->runGraphOp<nts::op::SingleGPUSampleGraphOp>(ssg, graph, l, tmp_X0[thread_id],
                                                                                &cuda_stream_list[thread_id]);
                  X[l + 1] = ctx->runVertexForward([&](NtsVar n_i) { return vertexForward(n_i); }, Y_i);
                } else {
                  NtsVar Y_i = ctx->runGraphOp<nts::op::SingleGPUSampleGraphOp>(ssg, graph, l, X[l],
                                                                                &cuda_stream_list[thread_id]);
                  X[l + 1] = ctx->runVertexForward([&](NtsVar n_i) { return vertexForward(n_i); }, Y_i);
                }
              }

              auto loss_ = Loss(X[layers], tmp_target_lab[thread_id], graph->config->classes == 1);
              loss_epoch += loss_.item<float>();

              if (ctx->training == true) {
                ctx->appendNNOp(X[layers], loss_);
                ctx->self_backward(false);
                Update();
              }

              if (graph->config->classes == 1) {
                correct += get_correct(X[layers], tmp_target_lab[thread_id], graph->config->classes == 1);
                train_nodes += tmp_target_lab[thread_id].size(0);
              } else {
                f1_epoch += f1_score(X[layers], tmp_target_lab[thread_id], graph->config->classes == 1);
              }
              epoch_train_time += get_time();

              cudaStreamSynchronize(cuda_stream_list[thread_id].stream);
              train_lock.unlock();

              sample_lock.lock();
              sampler->reverse_sgs(ssg);
            }
            sample_lock.unlock();
            /////// disable thread
            return;
          },
          tid);
    }
    for (int tid = 0; tid < pipelines; ++tid) {
      threads[tid].join();
    }
    assert(sampler->work_offset == sampler->work_range[1]);
    sampler->restart();
  }

  void explicit_version(Sampler* sampler) {
    X[0] = graph->Nts->NewLeafTensor({1000, F.size(1)}, torch::DeviceType::CUDA);
    NtsVar target_lab;
    if (graph->config->classes > 1) {
      target_lab =
          graph->Nts->NewLabelTensor({graph->config->batch_size, graph->config->classes}, torch::DeviceType::CUDA);
    } else {
      target_lab = graph->Nts->NewLabelTensor({graph->config->batch_size}, torch::DeviceType::CUDA);
    }
    sampler->metis_batch_id = 0;
    while (sampler->work_offset < sampler->work_range[1]) {
      if (ctx->training == true) zero_grad();  // should zero grad after every mini batch compute
      auto ssg = sampler->subgraph;
      epoch_sample_time -= get_time();
      sampler->sample_one(ssg, graph->config->batch_type, ctx->is_train());
      // sampler->sample_one_with_dst(ssg, graph->config->batch_type, ctx->is_train());
      epoch_sample_time += get_time();

      epoch_transfer_graph_time -= get_time();
      ssg->trans_graph_to_gpu_async(cuda_stream_list[0].stream, graph->config->mini_pull > 0);  // trans subgraph to gpu
      // ssg->trans_graph_to_gpu_async(cuda_stream->stream, graph->config->mini_pull > 0);  // trans subgraph to gpu
      epoch_transfer_graph_time += get_time();

      ///////////start gather and trans feature (explicit version) /////////////
      epoch_gather_feat_time -= get_time();
      // if (hosts > 1) {
      //   X[0] = nts::op::get_feature_from_global(*rpc, ssg->sampled_sgs[0]->src().data(),
      //   ssg->sampled_sgs[0]->src_size,
      //                                           F, graph);
      //   // if (type == 0 && graph->rtminfo->epoch >= graph->config->time_skip) rpc_comm_time += tmp_time;
      // } else {
      X[0] = nts::op::get_feature(ssg->sampled_sgs[0]->src().data(), ssg->sampled_sgs[0]->src_size, F, graph);
      // X[0] = nts::op::get_feature(ssg->sampled_sgs[0]->src(), F, graph);
      // }
      epoch_gather_feat_time += get_time();

      epoch_transfer_feat_time -= get_time();
      X[0] = X[0].cuda().set_requires_grad(true);
      epoch_transfer_feat_time += get_time();

      ////////start trans target_lab (explicit)//////////
      epoch_gather_label_time -= get_time();
      // if (hosts > 1) {
      //   target_lab = nts::op::get_label_from_global(ssg->sampled_sgs.back()->dst().data(),
      //                                               ssg->sampled_sgs.back()->v_size, L_GT_C, graph);
      //   // if (type == 0 && graph->rtminfo->epoch >= graph->config->time_skip) rpc_comm_time += tmp_time;
      // } else {
      target_lab =
          nts::op::get_label(ssg->sampled_sgs.back()->dst().data(), ssg->sampled_sgs.back()->v_size, L_GT_C, graph);

      // target_lab = nts::op::get_label(ssg->sampled_sgs.back()->dst(), L_GT_C, graph);
      // }
      epoch_gather_label_time += get_time();

      // double trans_label_gpu_cost = -get_time();
      epoch_transfer_label_time -= get_time();
      target_lab = target_lab.cuda();
      epoch_transfer_label_time += get_time();
      ///////end trans target_lab (explicit)///////

      epoch_train_time -= get_time();
      at::cuda::setCurrentCUDAStream(torch_stream[0]);
      for (int l = 0; l < layers; l++) {  // forward
        graph->rtminfo->curr_layer = l;
        // NtsVar Y_i = ctx->runGraphOp<nts::op::SingleGPUSampleGraphOp>(ssg, graph, l, X[l]);
        NtsVar Y_i = ctx->runGraphOp<nts::op::SingleGPUSampleGraphOp>(ssg, graph, l, X[l], &cuda_stream_list[0]);
        X[l + 1] = ctx->runVertexForward([&](NtsVar n_i) { return vertexForward(n_i); }, Y_i);
      }

      auto loss_ = Loss(X[layers], target_lab, graph->config->classes == 1);
      loss_epoch += loss_.item<float>();

      if (ctx->training == true) {
        ctx->appendNNOp(X[layers], loss_);
        ctx->self_backward(false);
        Update();
      }
      if (graph->config->classes == 1) {
        correct += get_correct(X[layers], target_lab, graph->config->classes == 1);
        train_nodes += target_lab.size(0);
      } else {
        f1_epoch += f1_score(X[layers], target_lab, graph->config->classes == 1);
      }
      epoch_train_time += get_time();

      sampler->reverse_sgs();
    }
    assert(sampler->work_offset == sampler->work_range[1]);
    sampler->restart();
  }

  void zerocopy_version(Sampler* sampler) {
    X[0] = graph->Nts->NewLeafTensor({1000, F.size(1)}, torch::DeviceType::CUDA);
    NtsVar target_lab;
    if (graph->config->classes > 1) {
      target_lab =
          graph->Nts->NewLabelTensor({graph->config->batch_size, graph->config->classes}, torch::DeviceType::CUDA);
    } else {
      target_lab = graph->Nts->NewLabelTensor({graph->config->batch_size}, torch::DeviceType::CUDA);
    }
    sampler->metis_batch_id = 0;
    while (sampler->work_offset < sampler->work_range[1]) {
      if (ctx->training == true) zero_grad();  // should zero grad after every mini batch compute
      auto ssg = sampler->subgraph;
      epoch_sample_time -= get_time();
      sampler->sample_one(ssg, graph->config->batch_type, ctx->is_train());
      // sampler->sample_one_with_dst(ssg, graph->config->batch_type, ctx->is_train());
      epoch_sample_time += get_time();

      epoch_transfer_graph_time -= get_time();
      ssg->trans_graph_to_gpu_async(cuda_stream_list[0].stream, graph->config->mini_pull > 0);  // trans subgraph to gpu
      // ssg->trans_graph_to_gpu_async(cuda_stream->stream, graph->config->mini_pull > 0);  // trans subgraph to gpu
      epoch_transfer_graph_time += get_time();

      ///////// trans feature (zero copy or cache  version)//////////////////
      // if (graph->config->cache_rate <= 0) { // trans feature use zero
      // graph->config->cache_type = "none";
      if (graph->config->cache_type == "none") {  // trans feature use zero copy (omit gather feature)
        // sampler->load_feature_gpu(X[0], gnndatum->dev_local_feature);
        // sampler->load_feature_gpu(cuda_stream, ssg, X[0], gnndatum->dev_local_feature);
        epoch_transfer_feat_time -= get_time();
        sampler->load_feature_gpu(&cuda_stream_list[0], ssg, X[0], gnndatum->dev_local_feature);
        epoch_transfer_feat_time += get_time();
        // trans freature which is not cache in gpu
        // } else if (graph->config->cache_type == "gpu_memory" && graph->rtminfo->epoch >= 5){
        if (graph->config->threshold_trans > 0) explicit_rate.push_back(cnt_suit_explicit_block(ssg));
      } else if (graph->config->cache_type == "gpu_memory" ||
                 graph->config->cache_type == "rate") {  // trans freature which is not cache in gpu
                                                         // trans_feature_cost -= get_time();
        // auto [trans_feature_tmp, gather_gpu_cache_tmp] = sampler->load_feature_gpu_cache(
        //     X[0], gnndatum->dev_local_feature, dev_cache_feature, local_idx, local_idx_cache, cache_node_hashmap,
        //     dev_local_idx, dev_local_idx_cache, dev_cache_node_hashmap);

        epoch_transfer_feat_time -= get_time();
        auto [trans_feature_tmp, gather_gpu_cache_tmp] = sampler->load_feature_gpu_cache(
            &cuda_stream_list[0], ssg, X[0], gnndatum->dev_local_feature, dev_cache_feature, local_idx, local_idx_cache,
            cache_node_hashmap, dev_local_idx, dev_local_idx_cache, dev_cache_node_hashmap);
        epoch_transfer_feat_time += get_time();

        epoch_all_node += ssg->sampled_sgs[0]->src().size();
        for (auto& it : ssg->sampled_sgs[0]->src()) {
          if (cache_node_hashmap[it] != -1) {
            epoch_cache_hit++;
          }
        }
        if (graph->config->threshold_trans > 0)
          explicit_rate.push_back(cnt_suit_explicit_block(ssg, cache_node_hashmap));
      } else {
        std::cout << "cache_type: " << graph->config->cache_type << " is not support!" << std::endl;
        assert(false);
      }
      // /####/end trans feature (zero copy or cache version) ############/

      ///////start trans target_lab (zero copy) //////
      // sampler->load_label_gpu(target_lab, gnndatum->dev_local_label);
      epoch_transfer_label_time -= get_time();
      sampler->load_label_gpu(&cuda_stream_list[0], ssg, target_lab, gnndatum->dev_local_label);
      epoch_transfer_label_time += get_time();
      // /end trans target_lab (zero  copy)////////

      epoch_train_time -= get_time();
      at::cuda::setCurrentCUDAStream(torch_stream[0]);
      for (int l = 0; l < layers; l++) {  // forward
        graph->rtminfo->curr_layer = l;
        // NtsVar Y_i = ctx->runGraphOp<nts::op::SingleGPUSampleGraphOp>(ssg, graph, l, X[l]);
        NtsVar Y_i = ctx->runGraphOp<nts::op::SingleGPUSampleGraphOp>(ssg, graph, l, X[l], &cuda_stream_list[0]);
        X[l + 1] = ctx->runVertexForward([&](NtsVar n_i) { return vertexForward(n_i); }, Y_i);
      }

      auto loss_ = Loss(X[layers], target_lab, graph->config->classes == 1);
      loss_epoch += loss_.item<float>();

      if (ctx->training == true) {
        ctx->appendNNOp(X[layers], loss_);
        ctx->self_backward(false);
        Update();
      }
      if (graph->config->classes == 1) {
        correct += get_correct(X[layers], target_lab, graph->config->classes == 1);
        train_nodes += target_lab.size(0);
      } else {
        f1_epoch += f1_score(X[layers], target_lab, graph->config->classes == 1);
      }
      epoch_train_time += get_time();
      sampler->reverse_sgs();
    }
    assert(sampler->work_offset == sampler->work_range[1]);
    sampler->restart();
    if (graph->config->threshold_trans > 0) {
      LOG_DEBUG("epoch suit explicit trans block rate %.3f(%.3f)", get_mean(explicit_rate), get_var(explicit_rate));
    }
  }

  void count_sample_hop_nodes(Sampler* sampler) {
    std::vector<std::vector<int>> all_batch_hop_nodes;
    while (sampler->work_offset < sampler->work_range[1]) {
      auto ssg = sampler->subgraph;
      sampler->sample_one(ssg, graph->config->batch_type, ctx->is_train());
      // sampler->sample_one_with_dst(ssg, graph->config->batch_type, ctx->is_train());
      sampler->reverse_sgs();
      std::vector<int> tmp_hop_nodes;
      for (auto sg : ssg->sampled_sgs) {
        tmp_hop_nodes.push_back(sg->src_size);
      }
      tmp_hop_nodes.push_back(ssg->sampled_sgs.back()->v_size);
      all_batch_hop_nodes.push_back(tmp_hop_nodes);
    }
    std::vector<float> ret = get_mean(all_batch_hop_nodes);
    printf("dataset_name %s", graph->config->dataset_name.c_str());
    for (auto x : ret) {
      printf(" %.3f", x);
    } printf("\n");
    assert(sampler->work_offset == sampler->work_range[1]);
    sampler->restart();
  }


  // std::pair<int,int>
  float cnt_suit_explicit_block(SampledSubgraph* ssg, VertexId* cache_node_hashmap = nullptr) {
    int node_num_block = 256 * 1024 / 4 / graph->gnnctx->layer_size[0];
    int threshold_node_num_block = node_num_block * graph->config->threshold_trans;
    auto csc_layer = ssg->sampled_sgs[0];
    std::vector<int> need_transfer(graph->vertices, 0);
    for (auto& v : csc_layer->src()) {
      need_transfer[v] = 1;
    }
    std::unordered_map<int, int> freq;
    for (int i = 0; i < graph->vertices; i += node_num_block) {
      int cnt = 0;
      for (int j = i; j < std::min(i + node_num_block, (int)graph->vertices); ++j) {
        if (!cache_node_hashmap) {
          if (need_transfer[j] == 1) cnt++;
        } else {
          if (need_transfer[j] == 1 && cache_node_hashmap[j] == -1) cnt++;
        }
      }
      if (freq.find(cnt) == freq.end()) freq[cnt] = 0;
      freq[cnt]++;
    }

    ////////////// check freq record all the src nodes
    int all_node_cnt = 0, block_cnt = 0;
    for (auto& v : freq) {
      all_node_cnt += v.first * v.second;
      block_cnt += v.second;
    }
    assert(block_cnt == (graph->vertices + node_num_block - 1) / node_num_block);
    if (!cache_node_hashmap) {
      assert(all_node_cnt == csc_layer->src().size());
    } else {
      int tmp = 0;
      for (int j = 0; j < (int)graph->vertices; ++j) {
        if (need_transfer[j] == 1 && cache_node_hashmap[j] == -1) tmp++;
      }
      assert(tmp == all_node_cnt);
    }
    //////////////////////////////

    std::vector<std::pair<int, int>> all(freq.begin(), freq.end());
    sort(all.begin(), all.end(), [&](auto& x, auto& y) { return x.first > y.first; });
    // // check sort is correct
    // for (int i = 1; i < all.size(); ++i) {
    //   assert(all[i - 1].first > all[i].first);
    // }

    int rate_cnt = 0;
    int rate_all = 0;
    for (auto& v : all) {
      if (v.first > threshold_node_num_block) {
        // std::cout << v.first << " " << v.second << std::endl;
        rate_cnt += v.second;
      }
      if (v.first > 0) rate_all += v.second;
    }
    float rate_trans = rate_cnt > 0 ? rate_cnt * 1.0 / rate_all : 0;
    LOG_DEBUG("nodes_in_one_block %d threshold %d (%.2f), suit_explicit_trans_rate %.2f (%d/%d)", node_num_block, threshold_node_num_block, graph->config->threshold_trans,rate_trans,
              rate_cnt, rate_all);
    // return {rate_cnt, rate_all};
    return rate_trans;
  }

  // std::tuple<float, double>
  float Forward(Sampler* sampler, int type = 0) {
    correct = 0;
    train_nodes = 0;
    loss_epoch = 0;
    f1_epoch = 0;

    // enum BatchType { SHUFFLE, SEQUENCE, RANDOM, DELLOW, DELHIGH, METIS};
    if (graph->config->batch_type == SHUFFLE || graph->config->batch_type == RANDOM ||
        graph->config->batch_type == DELLOW || graph->config->batch_type == DELHIGH) {
      shuffle_vec(sampler->sample_nids);
    }

    // sampler->zero_debug_time();
    epoch_sample_time = 0;
    epoch_gather_label_time = 0;
    epoch_gather_feat_time = 0;
    epoch_transfer_graph_time = 0;
    epoch_transfer_feat_time = 0;
    epoch_transfer_label_time = 0;
    epoch_train_time = 0;
    epoch_cache_hit = 0;
    epoch_all_node = 0;
    debug_time = 0;

    int batch_num = sampler->batch_nums;

    if (graph->config->mode == "pipeline") {
      LOG_DEBUG("pipeline version");
      pipeline_version(sampler);
    } else if (graph->config->mode == "explicit") {
      LOG_DEBUG("explicit version");
      explicit_version(sampler);
    } else if (graph->config->mode == "zerocopy") {
      LOG_DEBUG("zerocopy version");
      zerocopy_version(sampler);
    } else {
      LOG_DEBUG("not support");
      assert(false);
    }

    double post_time = -get_time();
    loss_epoch /= sampler->batch_nums;

    if (graph->config->classes > 1) {
      f1_epoch /= sampler->batch_nums;
      MPI_Allreduce(MPI_IN_PLACE, &f1_epoch, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
      acc = f1_epoch / hosts;
    } else {
      // if (type == 0 && graph->rtminfo->epoch >= graph->config->time_skip) mpi_comm_time -= get_time();
      MPI_Allreduce(MPI_IN_PLACE, &correct, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &train_nodes, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
      // if (type == 0 && graph->rtminfo->epoch >= graph->config->time_skip) mpi_comm_time += get_time();
      acc = 1.0 * correct / train_nodes;
    }
    LOG_DEBUG("Train ACC %d/%d %.3f", correct, train_nodes, acc);

    double epoch_transfer_time = epoch_transfer_feat_time + epoch_transfer_label_time + epoch_transfer_graph_time;
    double epoch_gather_time = epoch_gather_feat_time + epoch_gather_label_time;
    double epoch_run_time = epoch_sample_time + epoch_transfer_time + epoch_gather_time + epoch_train_time;
    double epoch_cache_miss = (epoch_all_node - epoch_cache_hit);
    double epoch_trans_memory = epoch_cache_miss * graph->gnnctx->layer_size[0] * sizeof(ValueType) / 1024 / 1024;
    double epoch_cache_hit_rate = epoch_all_node > 0 ? 1.0 * epoch_cache_hit / epoch_all_node : 0;
    if (graph->rtminfo->epoch >= graph->config->time_skip) {
      gcn_run_time += epoch_run_time;
      gcn_sample_time += epoch_sample_time;
      gcn_gather_time += epoch_gather_time;
      gcn_trans_time += epoch_transfer_time;
      gcn_train_time += epoch_train_time;
      gcn_cache_hit_rate += epoch_cache_hit_rate;
      gcn_trans_memory += epoch_trans_memory;
    }
    get_gpu_mem(used_gpu_mem, total_gpu_mem);
    LOG_INFO(
        "Epoch %03d epoch_time %.3f sample_time %.3f gather_time %.3f trans_time %.3f train_time %.3f "
         "cache_rate %.3f cache_hit_rate %.3f trans_memory %.3fM gpu_mem %.3fM",
        graph->rtminfo->epoch, epoch_run_time, epoch_sample_time, epoch_gather_time, epoch_transfer_time,
        epoch_train_time, graph->config->cache_rate, epoch_cache_hit_rate, epoch_trans_memory, used_gpu_mem);
    post_time += get_time();
    // LOG_DEBUG("pre_time %.3f post_time %.3f debug_time %.3f", pre_time, post_time, debug_time);
    return acc;
  }

  void zero_grad() {
    // LOG_DEBUG("P.sizes %d", P.size());
    for (int i = 0; i < P.size(); i++) {
      // LOG_DEBUG("zero_grad(%d) addr %p", i, P[i]);
      P[i]->zero_grad();
    }
  }


  void run_cache_exp() {
    float cache_rate_end = graph->config->cache_rate_end;
    float cache_rate_num = graph->config->cache_rate_num;

    init_nids();
    LOG_INFO("label rate: %.3f, train/val/test: (%d/%d/%d) (%.3f/%.3f/%.3f)",
             1.0 * (train_nids.size() + val_nids.size() + test_nids.size()) / graph->vertices, train_nids.size(),
             val_nids.size(), test_nids.size(), train_nids.size() * 1.0 / graph->vertices,
             val_nids.size() * 1.0 / graph->vertices, test_nids.size() * 1.0 / graph->vertices);

    cuda_stream_list = new Cuda_Stream[pipelines];
    auto default_stream = at::cuda::getDefaultCUDAStream();
    for (int i = 0; i < pipelines; i++) {
      torch_stream.push_back(at::cuda::getStreamFromPool(true));
      auto stream = torch_stream[i].stream();
      cuda_stream_list[i].setNewStream(stream);
      for (int j = 0; j < i; j++) {
        if (cuda_stream_list[j].stream == stream || stream == default_stream) {
          LOG_DEBUG("stream i:%p is repeat with j: %p, default: %p\n", stream, cuda_stream_list[j].stream,
                    default_stream);
          exit(3);
        }
      }
    }

    // train_sampler = new Sampler(fully_rep_graph, train_nids);
    train_sampler = new Sampler(fully_rep_graph, train_nids, pipelines, false);
    eval_sampler = new Sampler(fully_rep_graph, val_nids, true);  // true mean full batch
    test_sampler = new Sampler(fully_rep_graph, test_nids, true);  // true mean full batch
    // eval_sampler->update_fanout(-1);                            // val not sample

    for (int i = 0; i < cache_rate_num + 1; ++i) {

      gcn_cache_hit_rate = 0;
      float curr_cache_rate = cache_rate_end / cache_rate_num * i;
      graph->config->cache_rate = curr_cache_rate;
      // std::cout << i << " " << curr_cache_rate << std::endl;
      assert(curr_cache_rate >= 0 && curr_cache_rate <= 1);
      determine_cache_node_idx(graph->vertices * curr_cache_rate);
      assert(iterations == graph->config->epochs);
      for (int i_i = 0; i_i < iterations; i_i++) {
        graph->rtminfo->epoch = i_i;
        ctx->train();
        graph->rtminfo->forward = true;
        float train_acc = Forward(train_sampler, 0);
        float train_loss = loss_epoch;
      }
      gcn_cache_hit_rate /= (graph->config->epochs - graph->config->time_skip);

      LOG_DEBUG("dataset %s gcn_cache_rate %.3f gcn_cache_type %s batch_size %u gcn_cache_hit_rate %.3f",
          graph->config->dataset_name.c_str(), curr_cache_rate, graph->config->cache_type.c_str(), graph->config->batch_size, gcn_cache_hit_rate);
      cudaFreeHost(cache_node_hashmap);
    }
  
  }


  float run() {
    // get_gpu_mem(used_gpu_mem, total_gpu_mem);
    // LOG_DEBUG("run(): gcn_gpu_mem %.3fM", used_gpu_mem);
    if (graph->partition_id == 0) {
      LOG_INFO("GNNmini::[Dist.GPU.GCNimpl] running [%d] Epoches\n", iterations);
    }
    // get train/val/test node index. (may be move this to GNNDatum)
    BatchType batch_type = graph->config->batch_type;
    assert(best_val_acc == 0);
    init_nids();
    LOG_INFO("label rate: %.3f, train/val/test: (%d/%d/%d) (%.3f/%.3f/%.3f)",
             1.0 * (train_nids.size() + val_nids.size() + test_nids.size()) / graph->vertices, train_nids.size(),
             val_nids.size(), test_nids.size(), train_nids.size() * 1.0 / graph->vertices,
             val_nids.size() * 1.0 / graph->vertices, test_nids.size() * 1.0 / graph->vertices);

    cuda_stream_list = new Cuda_Stream[pipelines];
    auto default_stream = at::cuda::getDefaultCUDAStream();
    for (int i = 0; i < pipelines; i++) {
      torch_stream.push_back(at::cuda::getStreamFromPool(true));
      auto stream = torch_stream[i].stream();
      cuda_stream_list[i].setNewStream(stream);
      for (int j = 0; j < i; j++) {
        if (cuda_stream_list[j].stream == stream || stream == default_stream) {
          LOG_DEBUG("stream i:%p is repeat with j: %p, default: %p\n", stream, cuda_stream_list[j].stream,
                    default_stream);
          exit(3);
        }
      }
    }

    // train_sampler = new Sampler(fully_rep_graph, train_nids);
    train_sampler = new Sampler(fully_rep_graph, train_nids, pipelines, false);
    eval_sampler = new Sampler(fully_rep_graph, val_nids, true);  // true mean full batch
    test_sampler = new Sampler(fully_rep_graph, test_nids, true);  // true mean full batch
    // eval_sampler->update_fanout(-1);                            // val not sample

    // aix exp
    // std::vector<vector<int>> test_vector_mean{{1, 2, 3}, {1, 2, 10}};
    // std::vector<float> ret = get_mean(test_vector_mean);
    // std::cout << "test get_mean vector ";
    // for (auto x : ret) {
    //   std::cout << x << " ";
    // } std::cout << std::endl;
    // count_sample_hop_nodes(train_sampler);
    // assert(false);


    double run_time = -get_time();
    float config_run_time = graph->config->run_time;
    if (config_run_time > 0) {
      start_time = get_time();
      iterations = INT_MAX;
      LOG_DEBUG("iterations %d config_run_time %.3f", iterations, config_run_time);
    }

    double cache_init_time = -get_time();
    if (graph->config->cache_type == "gpu_memory") {
      // LOG_DEBUG("start get_gpu_idle_mem()");
      // double max_gpu_mem = get_gpu_idle_mem();
      LOG_DEBUG("start get_gpu_idle_mem_pipe()");
      double max_gpu_mem = get_gpu_idle_mem_pipe();
      LOG_DEBUG("release gpu memory");
      empty_gpu_cache();
      get_gpu_mem(used_gpu_mem, total_gpu_mem);
      LOG_DEBUG("used %.3f total %.3f (after emptyCache)", used_gpu_mem, total_gpu_mem);
      // double free_memory = total_gpu_mem - max_gpu_mem - 200;
      double free_memory = total_gpu_mem - max_gpu_mem - 100;
      int memory_nodes = free_memory * 1024 * 1024 / sizeof(ValueType) / graph->gnnctx->layer_size[0];
      determine_cache_node_idx(memory_nodes);
      get_gpu_mem(used_gpu_mem, total_gpu_mem);
      LOG_DEBUG("used %.3f total %.3f (after cache feature)", used_gpu_mem, total_gpu_mem);
    } else if (graph->config->cache_type == "rate") {
      assert(graph->config->cache_rate >= 0 && graph->config->cache_rate <= 1);
      determine_cache_node_idx(graph->vertices * graph->config->cache_rate);
    } else if (graph->config->cache_type == "random") {
      assert(graph->config->cache_rate >= 0 && graph->config->cache_rate <= 1);
      determine_cache_node_idx(graph->vertices * graph->config->cache_rate);
    } else if (graph->config->cache_type == "none") {
      LOG_DEBUG("There is no cache_type!");
    } else {
      std::cout << "cache_type: " << graph->config->cache_type << " is not support!" << std::endl;
      assert(false);
    }
    cache_init_time += get_time();

    // move to constructor

    double fractional_seconds_since_epoch
    = std::chrono::duration_cast<std::chrono::duration<double>>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    LOG_DEBUG("gcn_start_run_at %.3f", fractional_seconds_since_epoch);

    double iteration_time = 0;
    for (int i_i = 0; i_i < iterations; i_i++) {
      if (config_run_time > 0 && iteration_time >= config_run_time) {
        graph->config->epochs = i_i;
        iterations = i_i;
        break;
      }
      graph->rtminfo->epoch = i_i;
      ctx->train();
      graph->rtminfo->forward = true;
      double one_iteration_time = -get_time();
      float train_acc = Forward(train_sampler, 0);
      float train_loss = loss_epoch;
      one_iteration_time += get_time();
      if (i_i >= graph->config->time_skip) iteration_time += one_iteration_time;
      // get_gpu_mem(used_gpu_mem, total_gpu_mem);
      // LOG_DEBUG("Forward after empty_gpu_cache() %.3f", used_gpu_mem);
      // empty_gpu_cache();
    }

    LOG_DEBUG("average %d epoch", graph->config->epochs - graph->config->time_skip);
    iteration_time /= (graph->config->epochs - graph->config->time_skip);
    LOG_DEBUG("one_epoch_time %.3f, cache_init_time %.3f", iteration_time, cache_init_time);

    gcn_run_time /= (graph->config->epochs - graph->config->time_skip);
    gcn_sample_time /= (graph->config->epochs - graph->config->time_skip);
    gcn_gather_time /= (graph->config->epochs - graph->config->time_skip);
    gcn_trans_time /= (graph->config->epochs - graph->config->time_skip);
    gcn_train_time /= (graph->config->epochs - graph->config->time_skip);
    gcn_trans_memory /= (graph->config->epochs - graph->config->time_skip);
    gcn_cache_hit_rate /= (graph->config->epochs - graph->config->time_skip);

    LOG_DEBUG("gcn_run_time - sum_4_time %.3f",
              gcn_run_time - gcn_sample_time - gcn_gather_time - gcn_train_time - gcn_trans_time);
    gcn_run_time = gcn_sample_time + gcn_gather_time + gcn_train_time + gcn_trans_time;
    get_gpu_mem(used_gpu_mem, total_gpu_mem);
    if (graph->config->threshold_trans > 0) {
      LOG_DEBUG("avg suit explicit trans block rate %.3f(%.3f)", get_mean(explicit_rate), get_var(explicit_rate));
    }

    LOG_DEBUG(
        "dataset %s cache_rate %.3f batch_size %d used_memmory %.3fM gcn_run_time %.3f gcn_sample_time %.3f (%.3f)  "
        "gcn_gather_time %.3f "
        "(%.3f) gcn_trans_time %.3f (%.3f) gcn_train_time %.3f (%.3f); prepare_time %.3f (%.3f) compute_time %.3f "
        "(%.3f) cache_hit_rate %.3f trans_memory %.3fM",
        graph->config->dataset_name.c_str(), graph->config->cache_rate, graph->config->batch_size, used_gpu_mem,
        gcn_run_time, gcn_sample_time, gcn_sample_time / gcn_run_time, gcn_gather_time, gcn_gather_time / gcn_run_time,
        gcn_trans_time, gcn_trans_time / gcn_run_time, gcn_train_time, gcn_train_time / gcn_run_time,
        (gcn_sample_time + gcn_gather_time + gcn_trans_time),
        (gcn_sample_time + gcn_gather_time + gcn_trans_time) / gcn_run_time, (gcn_train_time),
        (gcn_train_time) / gcn_run_time, gcn_cache_hit_rate, gcn_trans_memory);
    return best_val_acc;
  }
};
